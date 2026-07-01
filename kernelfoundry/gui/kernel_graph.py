"""Page for showing a graph of kernel evolution for a given job/task/user, with interactive node selection to view details."""

from nicegui import ui
from typing import Optional
import networkx as nx
import math

from kernelfoundry.gui.utils import (
    get_kernels_by_job_id,
    get_kernels_by_op_and_run,
    get_op_names,
    get_jobs_by_op,
    get_id_by_uuid,
)
from kernelfoundry.gui.kernel_detail import kernel_detail_page


def size_from_runtime(rt, min_rt, max_rt, node_count):
    """Calculate node size from runtime using log scaling, with adaptive sizes based on node count"""
    # Calculate adaptive min/max sizes based on node count
    if node_count <= 20:
        min_size, max_size = 20, 50
    elif node_count >= 200:
        min_size, max_size = 4, 8
    else:
        # Linear interpolation between (20, 50) and (4, 8)
        progress = (node_count - 20) / (200 - 20)  # 0.0 to 1.0
        min_size = 20 - progress * (20 - 4)  # 20 -> 4
        max_size = 50 - progress * (50 - 8)  # 50 -> 8

    # Handle missing values
    if rt < 0:
        return min_size

    # Log scaling to soften extremes
    log_rt = math.log1p(rt)  # log(1+rt)
    log_min = math.log1p(min_rt)
    log_max = math.log1p(max_rt)

    # Normalize to [0, 1]
    norm = (log_rt - log_min) / (log_max - log_min + 1e-9)

    # Map to [min_size, max_size]
    return max_size - norm * (max_size - min_size)


def _derive_status(kernel):
    eval_log = (getattr(kernel, "eval_log", "") or "").lower()
    if kernel.improve_over_native > 1:
        return "improved"
    elif kernel.status == "correct":
        return "correct"
    elif kernel.status == "compiled":
        if "timed out" in eval_log:
            return "test_timeout"
        else:
            return "test_error"
    elif kernel.status == "error":
        if "timed out" in eval_log:
            return "build_timeout"
        else:
            return "build_error"


STATUS_TO_CATEGORY = {
    "improved": 0,
    "correct": 1,
    "test_error": 2,
    "build_error": 3,
    "test_timeout": 4,
    "build_timeout": 5,
}


def _get_best_kernel_criteria(kernels):
    """Choose best-kernel metric: speedup when available, else minimum runtime."""
    valid_speedups = [
        k.improve_over_native for k in kernels if k.improve_over_native is not None and k.improve_over_native >= 0
    ]
    if valid_speedups:
        return "speedup", max(valid_speedups)

    valid_runtimes = [k.runtime for k in kernels if k.runtime >= 0]
    return "runtime", min(valid_runtimes, default=None)


def _is_best_kernel(node_attrs, best_metric, best_value):
    if best_value is None:
        return False

    if best_metric == "speedup":
        speedup = node_attrs.get("improve_over_native")
        return speedup is not None and math.isclose(speedup, best_value, rel_tol=1e-9, abs_tol=1e-12)

    runtime = node_attrs.get("runtime", -1)
    return runtime >= 0 and runtime == best_value


def build_graph_with_layout(kernels):
    """Build graph data and compute tree layout using networkx and pygraphviz"""
    if len(kernels) == 0:
        return [], []
    G = nx.DiGraph()

    min_runtime = min((k.runtime for k in kernels if k.runtime >= 0), default=0)
    max_runtime = max((k.runtime for k in kernels if k.runtime >= 0), default=1800)
    best_metric, best_value = _get_best_kernel_criteria(kernels)

    # First collect all valid UUIDs
    valid_uuids = set(kernel.uuid for kernel in kernels if kernel.uuid)

    # Add nodes and edges to the graph
    for kernel in kernels:
        if kernel.uuid:
            G.add_node(
                kernel.uuid,
                status=_derive_status(kernel),
                trial=kernel.trial,
                version=kernel.version,
                runtime=kernel.runtime,
                timestamp=kernel.timestamp,
                improve_over_native=kernel.improve_over_native,
            )
            # Only add edge if parent also exists in our kernel set
            if kernel.parent_uuid is not None and kernel.parent_uuid in valid_uuids:
                G.add_edge(kernel.parent_uuid, kernel.uuid)

    # Compute tree layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # Normalize coordinates to reasonable range instead of raw scaling
    x_coords = [p[0] for p in pos.values()]
    y_coords = [p[1] for p in pos.values()]
    # Get current ranges
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    # Target coordinate ranges (adjust these to control spacing)
    target_width, target_height = 800, 600
    # Normalize coordinates to target ranges
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    normalized_pos = {}
    for node, (x, y) in pos.items():
        # Normalize to 0-1, then scale to target range
        norm_x = (x - x_min) / x_range * target_width
        norm_y = -((y - y_min) / y_range * target_height)  # Negative to flip Y axis
        normalized_pos[node] = (norm_x, norm_y)

    def get_runtime_status(node_attrs):
        runtime = node_attrs.get("runtime", -1)
        if runtime < 0:
            return node_attrs.get("status")
        return f"{runtime}ms"

    nodes = [
        {
            "name": f"UUID: {node}",
            "description": f"Iter: {G.nodes[node].get('trial')} - Branch: {G.nodes[node].get('version')} - Runtime: {get_runtime_status(G.nodes[node])}",
            "x": normalized_pos[node][0],
            "y": normalized_pos[node][1],
            "trial": G.nodes[node].get("trial"),
            "version": G.nodes[node].get("version"),
            "timestamp": G.nodes[node].get("timestamp").isoformat(timespec="microseconds"),
            "category": STATUS_TO_CATEGORY.get(G.nodes[node].get("status"), 2),
            "symbolSize": size_from_runtime(G.nodes[node].get("runtime", -1), min_runtime, max_runtime, len(G.nodes)),
            "itemStyle": {
                "borderWidth": (4 if _is_best_kernel(G.nodes[node], best_metric, best_value) else 1),
                "borderColor": ("#000000" if _is_best_kernel(G.nodes[node], best_metric, best_value) else "#FFFFFF"),
            },
        }
        for node in G.nodes
    ]

    links = [
        {
            "source": f"UUID: {source}",
            "target": f"UUID: {target}",
            "lineStyle": {"curveness": 0.2},
        }
        for source, target in G.edges
    ]

    return nodes, links


def build_graph_with_only_nodes(kernels):
    """Build graph data without edges, only nodes based on kernel IDs"""
    G = nx.DiGraph()

    min_runtime = min((k.runtime for k in kernels if k.runtime >= 0), default=0)
    max_runtime = max((k.runtime for k in kernels if k.runtime >= 0), default=1800)
    best_metric, best_value = _get_best_kernel_criteria(kernels)

    # Add nodes to the graph
    for kernel in kernels:
        if kernel.id:
            G.add_node(
                kernel.id,
                status=_derive_status(kernel),
                trial=kernel.trial,
                version=kernel.version,
                runtime=kernel.runtime,
                improve_over_native=kernel.improve_over_native,
            )

    # Compute tree layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    nodes = [
        {
            "name": f"ID: {node}",
            "x": pos[node][0],
            "y": -8 * pos[node][1],
            "trial": G.nodes[node].get("trial"),
            "version": G.nodes[node].get("version"),
            "category": STATUS_TO_CATEGORY.get(G.nodes[node].get("status"), 2),
            "symbolSize": size_from_runtime(G.nodes[node].get("runtime", -1), min_runtime, max_runtime, len(G.nodes)),
            "itemStyle": {
                "borderWidth": (4 if _is_best_kernel(G.nodes[node], best_metric, best_value) else 1),
                "borderColor": ("#000000" if _is_best_kernel(G.nodes[node], best_metric, best_value) else "#FFFFFF"),
            },
        }
        for node in G.nodes
    ]

    links = []

    return nodes, links


def kernel_graph_page(
    job_name: Optional[str] = None,
    task_name: Optional[str] = None,
    user_id: Optional[str] = None,
    job_id: Optional[str] = None,
):
    """Page showing a graph of kernel evolution for a given job/task/user, with interactive node selection to view details."""
    if hasattr(ui, "traceviewer") and hasattr(ui.traceviewer, "patch_html"):
        ui.traceviewer.patch_html()

    print(f"Rendering graph page with job_name={job_name}, task_name={task_name}, user_id={user_id}, job_id={job_id}")

    selected_job_id = None
    if job_id is not None and str(job_id).strip() != "":
        try:
            selected_job_id = int(job_id)
        except ValueError:
            ui.notify(f"Invalid job_id '{job_id}'. Falling back to job/task filters.", type="warning")

    valid_op_names = get_op_names(user_id)
    valid_runs = []
    if task_name and task_name not in valid_op_names:
        ui.notify(f"Task Name '{task_name}' not found.", type="negative")
        task_name = None
        job_name = None

    if task_name:
        valid_runs = get_jobs_by_op(task_name, user_id)
        if job_name and job_name not in valid_runs:
            ui.notify(f"Job '{job_name}' not found for task '{task_name}'.", type="negative")
            job_name = None

    # Main side-by-side layout
    with ui.row().classes("w-full gap-0"):
        # Left side: Controls and Graph (fixed width)
        with ui.column().style("width: 480px; flex-shrink: 0;"):
            # State for show_all checkbox
            show_all_state = {"value": False}

            # Controls row
            with ui.row().classes("gap-4 mb-4"):
                op_name_input = (
                    ui.select(
                        label="Task Name",
                        value=task_name,
                        with_input=True,
                        options=valid_op_names,
                    )
                    .props("clearable")
                    .style("width: 300px;")
                )

                def on_op_input_change(e):
                    run_input.value = ""
                    # Use None if show_all is checked, otherwise use user_id
                    current_user_id = None if show_all_state["value"] else user_id
                    run_input.options = get_jobs_by_op(e.value, current_user_id)
                    run_input.update()

                op_name_input.on_value_change(on_op_input_change)

                # Show All checkbox (only visible when user_id is provided)
                if user_id:
                    show_all_checkbox = ui.checkbox("Show All", value=False).classes("text-xs")

                    def on_show_all_change(e):
                        show_all_state["value"] = e.value
                        # Update op names based on checkbox state
                        current_user_id = None if e.value else user_id
                        op_name_input.options = get_op_names(current_user_id)
                        op_name_input.value = None
                        run_input.value = None
                        run_input.options = []
                        run_input.update()

                    show_all_checkbox.on_value_change(on_show_all_change)

            with ui.row().classes("gap-4 mb-4"):
                run_input = (
                    ui.select(
                        label="Job Name",
                        value=job_name,
                        with_input=True,
                        options=valid_runs,
                    )
                    .props("clearable")
                    .style("width: 300px;")
                )

                def on_plot():
                    if op_name_input.value is not None and run_input.value is not None:
                        url = f"/graph?job_name={run_input.value}&task_name={op_name_input.value}"
                        if user_id:
                            url += f"&user_id={user_id}"
                        ui.navigate.to(url)
                    else:
                        ui.notify("Please select both task name and job name", color="warning")

                ui.button("Plot", on_click=on_plot).classes("mt-2")

            if selected_job_id is not None:
                ui.label(f"Showing results for Job ID: {selected_job_id}").classes("text-sm text-gray-600 mb-2")

            # Graph container
            if selected_job_id is not None:
                kernels = get_kernels_by_job_id(selected_job_id)
            else:
                kernels = get_kernels_by_op_and_run(op_name_input.value, run_input.value, user_id)
            nodes, links = build_graph_with_layout(kernels)
            if len(nodes) == 0 and len(links) == 0:
                # uuid is probably missing, fallback to id-based graph
                nodes, links = build_graph_with_only_nodes(kernels)

            ui.html('<div id="kernel-graph" style="width: 480px; height: 600px;"></div>', sanitize=False)

        # Right side: Kernel Detail (takes remaining space)
        detail_container = ui.column().classes("flex-1 min-w-0")
        detail_container.style("display: none;")

    # Note: due to bugs with nicegui ui.echarts click event handling, we will inject raw javascript
    def graph_javascript_code(nodes, links):
        return f"""
        const ensureEcharts = () => new Promise((resolve, reject) => {{
            if (window.echarts) {{
                resolve();
                return;
            }}

            const timeoutMs = 10000;
            const startTime = Date.now();
            const timer = setInterval(() => {{
                if (window.echarts) {{
                    clearInterval(timer);
                    resolve();
                    return;
                }}

                if (Date.now() - startTime > timeoutMs) {{
                    clearInterval(timer);
                    reject(new Error('ECharts failed to load in time'));
                }}
            }}, 50);
        }});

        ensureEcharts().then(() => {{
        const chartDom = document.getElementById('kernel-graph');
        const myChart = window.echarts.init(chartDom);
        const options = {{
            tooltip: {{
                position: 'right',
                formatter: function (params) {{
                    return params.data.description;
                }}
            }},
            animation: false,
            legend: [{{
                data: ['improved', 'correct', 'test_error', 'build_error', 'test_timeout', 'build_timeout', 'best kernel'],
                top: 'bottom',
                itemGap: 20,
                width: '100%'
            }}],
            series: [{{
                type: 'graph',
                layout: 'none',
                symbolSize: 50,
                roam: true,
                label: {{
                    show: true,
                    formatter: function (params) {{
                        if (params.data.trial == null || params.data.version == null) {{
                            return 'val';
                        }}
                        return "i" + params.data.trial + "-b" + params.data.version;
                    }}
                }},
                edgeSymbol: ['circle', 'arrow'],
                edgeSymbolSize: [4, 10],
                data: {nodes},
                links: {links},
                lineStyle: {{
                    opacity: 0.9,
                    width: 2,
                    curveness: 0,
                }},
                categories: [
                    {{ name: 'improved',      itemStyle: {{ color: '#2ECC71' }} }},  // green
                    {{ name: 'correct',       itemStyle: {{ color: '#5BA4CF' }} }},  // blue
                    {{ name: 'test_error',    itemStyle: {{ color: '#F5924E' }} }},  // orange
                    {{ name: 'build_error',   itemStyle: {{ color: '#E05A5A' }} }},  // Red
                    {{ name: 'test_timeout',  itemStyle: {{ color: '#F5C842' }} }},  // yellow
                    {{ name: 'build_timeout', itemStyle: {{ color: '#db70b8' }} }},  // pink
                    {{ name: 'best kernel', itemStyle: {{ color: '#FFFFFF', borderWidth: 4, borderColor: '#000000' }} }},
                ],
            }}],
        }};
        myChart.setOption(options);
        console.log(options)


        // Add click event listener
        myChart.on('click', function (params) {{
            if (params.dataType === 'node') {{
                const kernelName = params.data.name;
                // Create custom event
                emitEvent('kernel_clicked', kernelName);
            }}
        }});
        }}).catch((error) => {{
            console.error('Unable to initialize kernel graph:', error);
        }});
        """

    def on_kernel_clicked(e):
        parts = str(e.args).split(" ", 1)
        if len(parts) != 2:
            ui.notify("Unable to open kernel details for selected node", type="warning")
            return

        prefix, value = parts
        kernel_id = None
        if prefix == "ID:":
            kernel_id = value
        elif prefix == "UUID:":
            uuid = value
            kernel_id = get_id_by_uuid(uuid)

        if kernel_id is None:
            ui.notify("Kernel details could not be resolved", type="warning")
            return

        detail_container.clear()
        detail_container.style("display: block;")

        with detail_container:
            kernel_detail_page(kernel_id)

    ui.on("kernel_clicked", on_kernel_clicked)

    # Delay JavaScript execution to ensure DOM and echarts library are ready
    ui.run_javascript(graph_javascript_code(nodes, links))

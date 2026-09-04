from nicegui import ui
import statistics

from kernelfoundry.gui.utils import get_kernels_by_job_id, get_kernels_by_op_and_run, normalize_runtime_stats


def _extract_speedup_map(runtime_stats, gpu_arch=None):
    """Flatten runtime_stats into {(gpu_arch, shape): speedup} with numeric values only."""
    speedup_map = {}
    runtime_stats = normalize_runtime_stats(runtime_stats, gpu_arch)
    for gpu_name, shape_stats in runtime_stats.items():
        for shape, metrics in shape_stats.items():
            if not isinstance(metrics, dict):
                continue
            speedup_value = metrics.get("speedup", metrics.get("Speedup"))
            if speedup_value is None:
                continue
            speedup_map[(gpu_name, shape)] = float(speedup_value)

    return speedup_map


def _dominates(a_map, b_map, rel_margin=0.025):
    """Return True if kernel A dominates kernel B with epsilon margin."""
    b_dims = set(b_map.keys())
    if not b_dims:
        return False
    if not b_dims.issubset(a_map.keys()):
        return False

    strictly_better = False
    lower_factor = 1.0 - rel_margin
    upper_factor = 1.0 + rel_margin

    for dim in b_dims:
        a_val = a_map[dim]
        b_val = b_map[dim]
        if a_val < b_val * lower_factor:
            return False
        if a_val > b_val * upper_factor:
            strictly_better = True

    return strictly_better


def _find_pareto_optimal_kernels(kernels, rel_margin=0.025):
    """Return list of kernel objects that are not dominated by any other kernel."""
    candidates = []
    for kernel in kernels:
        runtime_stats = kernel.runtime_stats
        if runtime_stats is None:
            continue
        speedup_map = _extract_speedup_map(runtime_stats, getattr(kernel, "gpu_arch", None))
        if speedup_map:
            candidates.append((kernel, speedup_map))

    pareto = []
    for i, (kernel_i, speedup_i) in enumerate(candidates):
        dominated = False
        for j, (_, speedup_j) in enumerate(candidates):
            if i == j:
                continue
            if _dominates(speedup_j, speedup_i, rel_margin=rel_margin):
                dominated = True
                break
        if not dominated:
            pareto.append(kernel_i)

    return pareto


def _short_shape_name(name):
    if not isinstance(name, str):
        return str(name)
    if "[" in name:
        return name.split("[")[-1].rstrip("]")
    return name


def _format_duration(delta_seconds):
    """Format a duration in seconds as e.g. '1h 5m', '5m 30s', or '12s'."""
    if delta_seconds is None or delta_seconds < 0:
        return "N/A"
    delta_seconds = int(delta_seconds)
    hours, remainder = divmod(delta_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _build_speedup_line_chart_options(kernels, rel_margin=0.025, pareto_kernels=None):
    """Build ECharts line-plot options for pareto kernels across all arch/shape dimensions."""
    if pareto_kernels is None:
        pareto_kernels = _find_pareto_optimal_kernels(kernels, rel_margin=rel_margin)
    if not pareto_kernels:
        return None

    # Sort for stable legend ordering by mean speedup descending.
    def mean_speedup(kernel):
        speedup_map = _extract_speedup_map(getattr(kernel, "runtime_stats", None), getattr(kernel, "gpu_arch", None))
        return statistics.fmean(speedup_map.values()) if speedup_map else 0.0

    pareto_kernels = sorted(pareto_kernels, key=mean_speedup, reverse=True)

    kernel_speedup_maps = {
        kernel.id: _extract_speedup_map(getattr(kernel, "runtime_stats", None), getattr(kernel, "gpu_arch", None))
        for kernel in pareto_kernels
    }

    x_labels = []
    for speedup_map in kernel_speedup_maps.values():
        for gpu_arch, shape in speedup_map.keys():
            label = f"{gpu_arch}:{_short_shape_name(shape)}"
            if label not in x_labels:
                x_labels.append(label)

    x_labels.append("mean speedup")

    series = []
    for kernel in pareto_kernels:
        speedup_map = kernel_speedup_maps[kernel.id]
        label_map = {
            f"{gpu_arch}:{_short_shape_name(shape)}": value for (gpu_arch, shape), value in speedup_map.items()
        }

        data = [label_map.get(label, None) for label in x_labels[:-1]]
        improve_over_native = getattr(kernel, "improve_over_native", None)
        try:
            mean_val = float(improve_over_native) if improve_over_native is not None else None
        except (TypeError, ValueError):
            mean_val = None
        data.append(mean_val)

        series.append(
            {
                "name": f"kernel {kernel.id}",
                "type": "line",
                "connectNulls": False,
                "showSymbol": True,
                "symbolSize": 7,
                "data": data,
                "lineStyle": {"width": 2},
            }
        )

    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"type": "scroll", "top": 8},
        "grid": {"left": 40, "right": 20, "top": 55, "bottom": 95},
        "xAxis": {
            "type": "category",
            "data": x_labels,
            "axisLabel": {"interval": 0, "rotate": 60, "fontSize": 10},
        },
        "yAxis": {
            "type": "value",
            "name": "Speedup",
        },
        "series": series,
        "dataZoom": [
            {"type": "inside", "xAxisIndex": 0},
            {"type": "slider", "xAxisIndex": 0, "height": 16, "bottom": 30},
        ],
        "markLine": {
            "silent": True,
            "lineStyle": {"type": "dashed", "color": "#444"},
            "data": [{"yAxis": 1.0}],
        },
    }


def _build_speedup_over_iterations_chart_options(kernels):
    """Build ECharts scatter-plot options for speedup vs. iteration (trial)."""
    correct_data = []
    incorrect_data = []

    for kernel in kernels:
        trial = getattr(kernel, "trial", None)
        if trial is None:
            continue
        status = getattr(kernel, "status", None)
        improve_over_native = getattr(kernel, "improve_over_native", None)

        is_incorrect = status != "correct" or (improve_over_native is not None and float(improve_over_native) < 0)

        if is_incorrect:
            incorrect_data.append([int(trial), 0])
        else:
            if improve_over_native is not None:
                try:
                    correct_data.append([int(trial), float(improve_over_native)])
                except (TypeError, ValueError):
                    pass

    if not correct_data and not incorrect_data:
        return None

    series = []
    if correct_data:
        series.append(
            {
                "name": "Correct",
                "type": "scatter",
                "symbolSize": 8,
                "data": correct_data,
                "itemStyle": {"color": "#1976d2", "opacity": 0.75},
            }
        )
    if incorrect_data:
        series.append(
            {
                "name": "Incorrect, no speedup",
                "type": "scatter",
                "symbolSize": 8,
                "data": incorrect_data,
                "itemStyle": {"color": "#e53935", "opacity": 0.65},
            }
        )

    return {
        "tooltip": {"trigger": "item"},
        "legend": {"type": "scroll", "top": 8},
        "grid": {"left": 55, "right": 20, "top": 55, "bottom": 60},
        "xAxis": {
            "type": "value",
            "name": "Iteration",
            "nameLocation": "middle",
            "nameGap": 30,
            "minInterval": 1,
        },
        "yAxis": {
            "type": "value",
            "name": "Speedup",
        },
        "series": series,
        "dataZoom": [
            {"type": "inside"},
            {"type": "slider", "height": 16, "bottom": 10},
        ],
    }


def render_kernel_report(container, kernels, rel_margin=0.025, job=None):
    """Render summary stats and speedup analysis into the given UI container."""
    container.clear()
    container.style("display: block;")

    # Validation runs re-test existing kernels rather than generating new ones; such kernels
    # have no trial index, so exclude them from the "generated" count.
    generated_kernels = [k for k in kernels if getattr(k, "trial", None) is not None]
    total_kernels = len(generated_kernels)
    correct_count = sum(1 for k in generated_kernels if getattr(k, "status", None) == "correct")
    correct_rate = (100.0 * correct_count / total_kernels) if total_kernels > 0 else 0.0
    speedups = [
        float(k.improve_over_native)
        for k in generated_kernels
        if getattr(k, "improve_over_native", None) is not None and getattr(k, "improve_over_native", None) >= 0
    ]
    min_speedup = min(speedups) if speedups else None
    max_speedup = max(speedups) if speedups else None
    platform = next((k.gpu_arch for k in kernels if getattr(k, "gpu_arch", None)), None)

    duration_str = "N/A"
    if (
        job is not None
        and getattr(job, "started_at", None) is not None
        and getattr(job, "finished_at", None) is not None
    ):
        duration_str = _format_duration((job.finished_at - job.started_at).total_seconds())

    with container:
        ui.label("Job report").classes("text-h5 mb-3")

        with ui.row().classes("w-full gap-4 mb-4 items-stretch"):
            with ui.card().classes("flex-1 min-w-0"):
                ui.label("Info").classes("text-subtitle1")
                ui.separator().classes("my-2")
                with ui.column().classes("gap-1"):
                    ui.label(f"Duration: {duration_str}").classes("text-body1")
                    ui.label(f"Generated kernels: {total_kernels}").classes("text-body1")
                    ui.label(f"Platform: {platform or 'N/A'}").classes("text-body1")

            with ui.card().classes("flex-1 min-w-0"):
                ui.label("Metrics").classes("text-subtitle1")
                ui.separator().classes("my-2")
                with ui.column().classes("gap-1"):
                    ui.label(f"Correctness rate: {correct_count}/{total_kernels} ({correct_rate:.1f}%)").classes(
                        "text-body1"
                    )
                    ui.label("Min speedup: " + (f"{min_speedup:.4f}" if min_speedup is not None else "N/A")).classes(
                        "text-body1"
                    )
                    ui.label("Max speedup: " + (f"{max_speedup:.4f}" if max_speedup is not None else "N/A")).classes(
                        "text-body1"
                    )

            if job is not None:

                def _fmt_tokens(value):
                    return "N/A" if value is None else f"{value:,}"

                input_tokens = job.input_tokens
                cached_input_tokens = job.cached_input_tokens
                output_tokens = job.output_tokens
                total_input_tokens = (
                    None
                    if input_tokens is None and cached_input_tokens is None
                    else (input_tokens or 0) + (cached_input_tokens or 0)
                )

                with ui.card().classes("flex-1 min-w-0"):
                    ui.label("Token usage").classes("text-subtitle1")
                    ui.separator().classes("my-2")
                    with ui.column().classes("gap-1"):
                        ui.label(f"New input tokens: {_fmt_tokens(input_tokens)}").classes("text-body1")
                        ui.label(f"Cached input tokens: {_fmt_tokens(cached_input_tokens)}").classes("text-body1")
                        ui.label(f"Total input tokens: {_fmt_tokens(total_input_tokens)}").classes("text-body1")
                        ui.label(f"Output tokens: {_fmt_tokens(output_tokens)}").classes("text-body1")

        with ui.card().classes("w-full"):
            ui.label("Speedup over iterations").classes("text-subtitle1")
            ui.label("Each dot is one kernel. Incorrect kernels are shown in red at speedup 0.").classes(
                "text-caption text-gray-600"
            )
            ui.separator().classes("my-2")
            iter_chart_options = _build_speedup_over_iterations_chart_options(kernels)
            if iter_chart_options is None:
                ui.label("No iteration data available to draw the plot.").classes("text-gray-700")
            else:
                ui.echart(iter_chart_options).classes("w-full").style("height: 400px;")

        with ui.card().classes("w-full"):
            ui.label("Speedup analysis").classes("text-subtitle1")
            ui.label(
                f"Showing epsilon-Pareto kernels, i.e. all kernels that are not dominated by any other kernel by a margin of at least {rel_margin * 100:.1f}%."
            ).classes("text-caption text-gray-600")
            ui.separator().classes("my-2")

            pareto_section = ui.column().classes("w-full gap-2")

            def _render_pareto_plot():
                pareto_section.clear()
                with pareto_section:
                    pareto_kernels = _find_pareto_optimal_kernels(kernels, rel_margin=rel_margin)
                    chart_options = _build_speedup_line_chart_options(
                        kernels, rel_margin=rel_margin, pareto_kernels=pareto_kernels
                    )
                    if chart_options is None:
                        ui.label("No runtime speedup data available to draw the report plot.").classes("text-gray-700")
                    else:
                        ui.echart(chart_options).classes("w-full").style("height: 460px;")
                        if pareto_kernels:
                            ui.label("Jump to kernel:").classes("text-caption text-gray-600 mt-2")
                            with ui.row().classes("flex-wrap gap-1 mt-1"):
                                for k in pareto_kernels:
                                    with ui.link(target=f"/kernel/{k.id}"):
                                        ui.button(f"kernel {k.id}", icon="open_in_new").props("flat dense size=sm")

            with pareto_section:
                ui.button("Compute pareto plot", on_click=_render_pareto_plot, icon="insights").props("outline")

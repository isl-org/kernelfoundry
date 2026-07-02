"""Page for showing detailed information about a kernel (language model, code, status, runtime stats, etc)"""

from nicegui import ui
from typing import Optional, Callable
from pathlib import Path

from kernelfoundry.gui.utils import get_id_by_uuid, get_kernel_by_id, get_task_by_id


def kernel_detail_page(
    kernel_id: int = 0,
    include_download: bool = False,
    get_tarball_fn: Optional[Callable] = None,
):
    """Display detailed info about the generated kernel (language model, code, status, runtime stats, etc)"""
    ui.traceviewer.patch_html()

    kernel = get_kernel_by_id(kernel_id)

    def download_task_files():
        if get_tarball_fn:
            tarball_bytes = get_tarball_fn(kernel)
            # Trigger download
            ui.download(tarball_bytes, f"kernel_{kernel_id}_task_files.tar.gz")

    with ui.row().classes("items-end mb-4"):
        ui.label(f"Kernel Detail for ID: {kernel_id}").classes("text-xl")
        ui.button("Roofline analysis", on_click=lambda: ui.navigate.to(f"/roofline/{kernel_id}")).props(
            "icon=stacked_bar_chart flat no-caps size=md"
        )
        if include_download and get_tarball_fn:
            ui.button("Download Task Files", on_click=download_task_files).props("icon=download")

    if kernel:
        with ui.row().classes("grid grid-cols-8 w-full"):
            with ui.column().classes("col-span-4 gap-0"):
                with ui.row().classes("h-10 w-full items-center"):
                    ui.label("Task Name").classes("font-bold ml-2").style("width: 100px;")
                    ui.label(kernel.task_name)

                with ui.row().classes("bg-gray-100 h-10 w-full items-center"):
                    ui.label("Job Name").classes("font-bold ml-2").style("width: 100px;")
                    ui.label(kernel.job_name)

                with ui.row().classes("h-10 w-full items-center"):
                    ui.label("Job ID").classes("font-bold ml-2").style("width: 100px;")
                    with ui.row().classes("gap-1 items-center"):
                        ui.label(str(kernel.job_id))
                        ui.label("-")
                        ui.link("View Logs", f"/job_logs/{kernel.job_id}")
                        ui.label("-")
                        ui.link("View Results", f"/graph?job_id={kernel.job_id}")

                with ui.row().classes("bg-gray-100 h-10 w-full items-center"):
                    ui.label("Iteration").classes("font-bold ml-2").style("width: 100px;")
                    ui.label(
                        f"{kernel.trial} (branch {kernel.version})" if kernel.trial is not None else "N/A (validation)"
                    )

                with ui.row().classes("h-10 w-full items-center"):
                    ui.label("Status").classes("font-bold ml-2").style("width: 100px;")
                    ui.label(kernel.status)

                with ui.row().classes("bg-gray-100 h-10 w-full items-center"):
                    ui.label("Timestamp").classes("font-bold ml-2").style("width: 100px;")
                    ui.label(kernel.timestamp.strftime("%Y-%m-%d %H:%M:%S") if kernel.timestamp else "N/A")

                with ui.row().classes("h-10 w-full items-center"):
                    ui.label("Parent").classes("font-bold ml-2").style("width: 100px;")
                    if kernel.parent_uuid:
                        parent_id = get_id_by_uuid(kernel.parent_uuid)
                        if parent_id:
                            ui.link(kernel.parent_uuid, f"/kernel/{parent_id}")
                    else:
                        ui.label("N/A")

                with ui.row().classes("bg-gray-100 h-10 w-full items-center"):
                    ui.label("Model").classes("font-bold ml-2").style("width: 100px;")
                    ui.label(kernel.language_model if kernel.language_model else "N/A")

                with ui.row().classes("h-10 w-full items-center"):
                    ui.label("GPU Arch").classes("font-bold ml-2").style("width: 100px;")
                    ui.label(kernel.gpu_arch if kernel.gpu_arch else "N/A")

            with ui.column().classes("col-span-4"):
                with ui.card():
                    runtime_stats = kernel.runtime_stats or {}
                    if len(runtime_stats) == 0 or "mean" in runtime_stats:
                        ui.label("Runtime stats (in ms)").classes("font-bold txt-lg mb-2")
                        # Old format: flat dict with mean, std, etc.
                        for key, value in runtime_stats.items():
                            if key == "num_trials":
                                continue
                            with ui.row().classes():
                                ui.label(key).classes("font-bold").style("width: 100px;")
                                ui.label(str(value))
                        with ui.row().classes():
                            ui.label("Speedup over reference").classes("font-bold").style("width: 100px;")
                            ui.label(
                                str(kernel.improve_over_native)
                                if kernel.improve_over_native is not None and kernel.improve_over_native > 0
                                else "N/A"
                            )
                        if kernel.improve_over_compile is not None and kernel.improve_over_compile > 0:
                            with ui.row().classes():
                                ui.label("Speedup over torch.compile").classes("font-bold").style("width: 100px;")
                                ui.label(str(kernel.improve_over_compile))
                    else:
                        # New format: {gpu: {benchmark: {mean, std, speedup, ...}}}
                        columns = [
                            {"name": "gpu", "label": "GPU", "field": "gpu", "align": "left"},
                            {"name": "benchmark", "label": "Benchmark", "field": "benchmark", "align": "left"},
                            {"name": "mean", "label": "Runtime (ms)", "field": "mean", "align": "right"},
                            {"name": "ref_speed", "label": "Ref. time (ms)", "field": "ref_speed", "align": "right"},
                            {"name": "speedup", "label": "Speedup", "field": "speedup", "align": "right"},
                        ]
                        rows = []
                        gpu_archs = kernel.gpu_arch.split(",")
                        if list(runtime_stats.keys())[0] not in gpu_archs:
                            assert len(gpu_archs) == 1, "This issue should only occur for single gpu jobs"
                            runtime_stats = {kernel.gpu_arch: runtime_stats}
                        for gpu_name, benchmarks in runtime_stats.items():
                            for bench_name, stats in benchmarks.items():
                                # Truncate benchmark name to last 30 chars for display
                                if "::" in bench_name:
                                    short_bench = bench_name.split("::")[-1]
                                    short_bench = short_bench[-30:] if len(short_bench) > 30 else short_bench
                                else:
                                    short_bench = bench_name[-30:] if len(bench_name) > 30 else bench_name

                                # Round speedup to 4 decimals
                                speedup_val = stats.get("speedup", "N/A")
                                if isinstance(speedup_val, (int, float)):
                                    speedup_val = round(speedup_val, 4)

                                # Round ref speed to 3 decimals
                                ref_speed_val = stats.get("ref_speed", "N/A")
                                if isinstance(ref_speed_val, (int, float)):
                                    ref_speed_val = round(ref_speed_val, 3)

                                # Format mean ± std
                                mean_val = stats.get("mean", "N/A")
                                std_val = stats.get("std", "N/A")
                                if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
                                    mean_str = f"{mean_val:.3f} ± {std_val:.2f}"
                                else:
                                    mean_str = "N/A"

                                rows.append(
                                    {
                                        "gpu": gpu_name,
                                        "benchmark": short_bench,
                                        "benchmark_full": bench_name,
                                        "mean": mean_str,
                                        "ref_speed": ref_speed_val,
                                        "speedup": speedup_val,
                                    }
                                )
                        rows.sort(key=lambda x: (x["gpu"], x["benchmark"]))

                        if len(rows) > 0:
                            rows.append(
                                {
                                    "gpu": "",
                                    "benchmark": "(Geometric) Mean:",
                                    "benchmark_full": "",
                                    "mean": round(kernel.runtime, 3) if kernel.runtime is not None else "N/A",
                                    "ref_speed": "",
                                    "speedup": (
                                        round(kernel.improve_over_native, 4)
                                        if kernel.improve_over_native is not None
                                        else "N/A"
                                    ),
                                }
                            )
                        # Display mean runtime and speedup above the table
                        with ui.row().classes("mb-2 gap-4"):
                            mean_runtime_str = (
                                f"{round(kernel.runtime, 2)}"
                                if kernel.runtime is not None and kernel.runtime >= 0
                                else "N/A"
                            )
                            ui.label(f"Mean runtime: {mean_runtime_str} ms").classes("font-semibold")

                            speedup_str = (
                                f"{round(kernel.improve_over_native, 4)}"
                                if kernel.improve_over_native is not None and kernel.improve_over_native >= 0
                                else "N/A"
                            )
                            ui.label(f"Speedup (geometric mean): {speedup_str}").classes("font-semibold")
                            ui.label("Runtime stats by input and hardware:").classes("font-bold txt-lg mb-2")

                        table = (
                            ui.table(columns=columns, rows=rows)
                            .classes("w-full")
                            .props("dense virtual-scroll")
                            .style("max-height: 250px; overflow-y: scroll; border: 1px solid #ddd;")
                        )
                        # Add tooltip to benchmark column showing full name on hover
                        table.add_slot(
                            "body-cell-benchmark",
                            '<q-td :props="props" :title="props.row.benchmark_full">{{ props.row.benchmark }}</q-td>',
                        )

        # prompt and answer
        with ui.row().classes("grid grid-cols-6 w-full"):
            with ui.column().classes("col-span-3"):
                ui.label("Prompt:").classes("font-bold ml-2").style("width: 80px;")
                ui.codemirror(language="markdown", value=kernel.prompt).style("height: 400px;")

            with ui.column().classes("col-span-3"):
                ui.label("Answer:").classes("font-bold ml-2").style("width: 80px;")
                ui.codemirror(language="markdown", value=kernel.answer).style("height: 400px;")

        # input and output code
        with ui.row().classes("grid grid-cols-6 w-full"):
            with ui.column().classes("col-span-3"):
                ui.label("Input Code:").classes("font-bold ml-2").style("width: 400px;")
                ui.codemirror(language="python", value=kernel.input_code).style("height: 100%;")

                ui.label("Evaluation log:").classes("font-bold ml-2 mt-4").style("width: 400px;")
                ui.codemirror(language="python", value=kernel.eval_log).style("height: 100%;")

            with ui.column().classes("col-span-3"):
                ui.label("Output Code:").classes("font-bold ml-2").style("width: 400px;")
                ui.codemirror(language="python", value=kernel.output_code).style("height: 100%;")

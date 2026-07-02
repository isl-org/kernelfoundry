"""Page for showing roofline charts and ALU breakdown charts for kernel performance analysis."""

from nicegui import ui
import os, json
from pathlib import Path
import re
import pandas as pd
import io
import warnings
import numpy as np
import itertools
from collections import OrderedDict
from kernelfoundry.eval_pipeline.utils.hardware_info import hardware_info, HardwareRoofs
from kernelfoundry.eval_pipeline.utils.performance_analysis import (
    get_roofs,
    compute_arithmic_intensity,
    get_roofline_points,
    get_median_row,
    get_row_with_closest_metric,
)
from kernelfoundry.gui.utils import get_kernel_by_id, get_profiler_data_reference_by_job_id
from kernelfoundry.eval_pipeline.profiler_feedback import UnitraceProfilerFeedback, VTuneProfilerFeedback


def _load_svg_templates(template_dir: Path | None = None):
    """Load and preprocess SVG templates for roofline chart rendering."""
    templates = {}
    if template_dir is None:
        template_dir = Path(__file__).parent / "figure_templates"

    def repl_fn(match):
        try:
            if len(match.groups()) == 1:
                return match.group(1)
        except:
            return match.group(0)
        return ""

    for filename in template_dir.iterdir():
        if filename.suffix == ".svg":
            t = filename.read_text()
            # remove everything before "<svg" and add the svg_content-scale class to make it responsive in the UI
            t.index("<svg")
            t = (
                '<svg class="svg-content-scale" preserveAspectRatio="xMidYMid meet" '
                + t[t.index("<svg ") + len("<svg ") :]
            )
            # Remove all data-cell-id from the drawio svgdata plugin and some id attributes to save space
            t = re.sub(r'\s*data-cell-id="[^"]*"|(g)\sid="[^"]*"', repl_fn, t)
            templates[filename.stem] = t
    return templates


def format_bytes(value) -> str:
    """Format byte values into human-readable format (B, KB, MB, GB, TB)."""
    try:
        num = float(value)
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        unit_idx = 0
        while num >= 1024 and unit_idx < len(units) - 1:
            num /= 1024
            unit_idx += 1
        return f"{num:.2f} {units[unit_idx]}"
    except (TypeError, ValueError):
        return str(value)


def fill_figure_template(template_str, series: list[pd.Series]) -> str:
    """Fill SVG template placeholders with data from series."""
    for idx, s in enumerate(series):
        data_dict = {k: v.item() if hasattr(v, "item") else v for k, v in s.items()}
        filled_template = template_str
        for key, value in data_dict.items():
            placeholder = f"{{{key}}}"
            filled_template = filled_template.replace(placeholder, str(value))
            # print(key, value)

    # Replace eval[...] patterns in the template
    eval_pattern = r"eval\[([^\]]+)\]"

    def eval_replacement(match):
        try:
            expr = match.group(1)
            result = eval(expr, dict(fmtb=format_bytes))
            return str(result)
        except ZeroDivisionError:
            return "nan"
        except:
            return match.group(0)

    filled_template = re.sub(eval_pattern, eval_replacement, filled_template)
    return filled_template


def _fill_figure_template_alu_bars(template_str: str, bars: dict[str, float]) -> str:
    """Fill SVG template with ALU bar chart data."""
    xywh_re = re.compile(r'x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)"')
    for name, value in bars.items():
        pos = template_str.find(f'data-XVE="{name}"')
        if pos >= 0:

            def bar_replacement(match):
                try:
                    x, y, width, height = map(int, match.groups())
                    new_height = int(height * value)
                    new_y = y + (height - new_height)
                    return f'x="{x}" y="{new_y}" width="{width}" height="{new_height}"'
                except Exception as e:
                    print(f"Error processing bar {name}: {e}")
                    return match.group(0)

            template_str1 = template_str[:pos]
            template_str2 = re.sub(
                r'x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)"', bar_replacement, template_str[pos:], count=1
            )
            template_str = template_str1 + template_str2
    return template_str


class RooflineUtils:
    MIN_VALUE = 0.001
    DISPLAY_NAME = {
        "GPU_MEMORY_BYTE_READ[bytes]": "GMEM",
        "GPU_MEMORY_BYTE_WRITE[bytes]": "GMEM",
        "SLM_BYTE_READ[bytes]": "SLM",
        "SLM_BYTE_WRITE[bytes]": "SLM",
        "XVE_INST_EXECUTED_FP16[events]": "FP16",
        "XVE_INST_EXECUTED_FP32[events]": "FP32",
        "XVE_INST_EXECUTED_FP64[events]": "FP64",
        "XVE_INST_EXECUTED_INT16[events]": "INT16",
        "XVE_INST_EXECUTED_INT32[events]": "INT32",
        "XVE_INST_EXECUTED_INT64[events]": "INT64",
        "XVE_INST_EXECUTED_MATH[events]": "EMATH",
        "XVE_INST_EXECUTED_XMX_BF16[events]": "XMX BF16",
        "XVE_INST_EXECUTED_XMX_FP16[events]": "XMX FP16",
        "XVE_INST_EXECUTED_XMX_INT2[events]": "XMX INT2",
        "XVE_INST_EXECUTED_XMX_INT4[events]": "XMX INT4",
        "XVE_INST_EXECUTED_XMX_INT8[events]": "XMX INT8",
        "XVE_INST_EXECUTED_ALU0_ALL[events]": "ALU0",
        "XVE_INST_EXECUTED_ALU1_ALL[events]": "ALU1",
        "XVE_INST_EXECUTED_ALU2_ALL[events]": "ALU2",
    }

    MEM_PAIRS = [
        ("GPU_MEMORY_BYTE_READ[bytes]", "GPU_MEMORY_BYTE_WRITE[bytes]"),
        ("SLM_BYTE_READ[bytes]", "SLM_BYTE_WRITE[bytes]"),
    ]

    COMPUTE_METRICS = [
        "XVE_INST_EXECUTED_FP16[events]",
        "XVE_INST_EXECUTED_FP32[events]",
        "XVE_INST_EXECUTED_FP64[events]",
        "XVE_INST_EXECUTED_INT16[events]",
        "XVE_INST_EXECUTED_INT32[events]",
        "XVE_INST_EXECUTED_INT64[events]",
        "XVE_INST_EXECUTED_MATH[events]",
        "XVE_INST_EXECUTED_XMX_BF16[events]",
        "XVE_INST_EXECUTED_XMX_FP16[events]",
        "XVE_INST_EXECUTED_XMX_INT2[events]",
        "XVE_INST_EXECUTED_XMX_INT4[events]",
        "XVE_INST_EXECUTED_XMX_INT8[events]",
        "XVE_INST_EXECUTED_ALU0_ALL[events]",
        "XVE_INST_EXECUTED_ALU1_ALL[events]",
        "XVE_INST_EXECUTED_ALU2_ALL[events]",
    ]

    @staticmethod
    def shorten_fn_name(fn_name: str) -> str:
        """Shorten function names by removing common namespace prefixes."""
        repl = [
            ("at::native::templates::", ""),
            ("at::native::", ""),
            ("at::detail::", ""),
            ("at::", ""),
            ("unsigned int", "uint"),
        ]
        for old, new in repl:
            fn_name = fn_name.replace(old, new)
        return fn_name

    @staticmethod
    def read_metrics(profiler_data: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """Reads the metrics from the profiler data dictionary.
        Args:
            profiler_data (dict): The profiler data dictionary.
        Returns:
            tuple: A tuple containing three DataFrames (or None) for different metric groups.
                (ComputeBasic, MemoryProfile, VectorEngineProfile)
        """
        compute_basic = next((v for k, v in profiler_data.items() if "ComputeBasic.metrics." in k), None)
        memory_profile = next((v for k, v in profiler_data.items() if "MemoryProfile.metrics." in k), None)
        vector_engine_profile = next((v for k, v in profiler_data.items() if "VectorEngineProfile.metrics." in k), None)
        if compute_basic is None:
            compute_basic = next((v for k, v in profiler_data.items() if ".metrics." in k), None)

        result = []
        for metric in [compute_basic, memory_profile, vector_engine_profile]:
            if metric is not None:
                lines = metric.splitlines()
                lines = [line for line in lines if line and not line.startswith("==")]
                result.append(pd.read_csv(io.StringIO("\n".join(lines))))
            else:
                result.append(None)
        return tuple(result)

    @staticmethod
    def vtune_counters_to_dataframe(vtune_raw_data: dict) -> pd.DataFrame | None:
        """Convert VTune raw counter data to a unitrace-compatible DataFrame for roofline plotting.

        VTune provides per-kernel aggregated counters with bandwidth in GB/s and instruction
        counts. This converts them into the column format expected by the roofline chart
        infrastructure (bytes transferred and instruction event counts).

        Accepts two formats:
        - DB format: {'counters': {kernel_name: {col: val, ...}}}
        - File format: {'counters.json': '<json string>'}

        Args:
            vtune_raw_data: The raw VTune data dict.
        Returns:
            A DataFrame with unitrace-compatible column names, or None if data is unavailable.
        """
        counters = vtune_raw_data["counters"]

        # Direct float mappings: VTune column name → output column name (default 0.0)
        DIRECT_MAPPINGS = {
            "GPU Memory Bandwidth, GB/sec:Read": "GPU_MEMORY_BYTE_READ_RATE[GBpS]",
            "GPU Memory Bandwidth, GB/sec:Write": "GPU_MEMORY_BYTE_WRITE_RATE[GBpS]",
            "XVE Instructions:ALU0 Instructions": "XVE_INST_EXECUTED_ALU0_ALL[events]",
            "XVE Instructions:ALU1 Instructions": "XVE_INST_EXECUTED_ALU1_ALL[events]",
            "XVE Instructions:XMX instructions": "XVE_INST_EXECUTED_ALU2_ALL[events]",
            "GPU L3:Busy(%)": "L3_BUSY[%]",
            "GPU L3:Average Bandwidth, GB/s:Read": "L3_READ[GBpS]",
            "GPU L3:Average Bandwidth, GB/s:Write": "L3_WRITE[GBpS]",
        }

        rows = []
        for kernel_name, cols in counters.items():

            def _f(key, _cols=cols):
                v = _cols.get(key)
                if v is None:
                    return None
                try:
                    return float(str(v).rstrip("%").replace(",", "").strip())
                except (ValueError, TypeError):
                    return None

            avg_time_s = _f("Computing Task:Average Time")
            instance_count = _f("Computing Task:Instance Count") or 1.0
            # VTune reports "Average Time" in seconds. Use total runtime
            # (avg × instances) so that instruction counts (which VTune
            # reports as totals over all instances) are correctly normalised.
            total_runtime_s = avg_time_s * instance_count

            row = {"Kernel": kernel_name, "GpuTime[ns]": total_runtime_s * 1e9}
            for vtune_col, out_col in DIRECT_MAPPINGS.items():
                row[out_col] = _f(vtune_col) or 0.0

            # Derived: bytes = bandwidth × time (approximate — VTune timing is
            # event-based, but these are needed by the roofline chart for
            # arithmetic intensity. The SVG labels are pre-substituted to show
            # GB/s directly, so the inaccuracy doesn't surface there.)
            row["GPU_MEMORY_BYTE_READ[bytes]"] = row["GPU_MEMORY_BYTE_READ_RATE[GBpS]"] * 1e9 * total_runtime_s
            row["GPU_MEMORY_BYTE_WRITE[bytes]"] = row["GPU_MEMORY_BYTE_WRITE_RATE[GBpS]"] * 1e9 * total_runtime_s
            # SVG template computes hit rate as 100*L3_HIT/(L3_HIT+L3_MISS);
            # setting hit=100-mr and miss=mr gives the correct percentage.
            l3_miss_ratio = _f("GPU L3:Miss Ratio(%)") or 0.0
            row["L3_HIT[events]"] = 100.0 - l3_miss_ratio
            row["L3_MISS[events]"] = l3_miss_ratio

            rows.append(row)

        if not rows:
            return None
        return pd.DataFrame(rows)

    @staticmethod
    def get_median_row_for_each_kernel(df: pd.DataFrame, column: str = "GpuTime[ns]") -> list[pd.Series]:
        """Returns a DataFrame with the median row for each kernel."""
        result_rows = []
        for kernel_name, group in df.groupby("Kernel"):
            median_row = get_median_row(group, column=column)
            result_rows.append(median_row)
        return result_rows

    @staticmethod
    def compute_plot_range(
        mem_bandwidths: list[float],
        compute_limits: list[float],
        measured_arithmic_intensities: list[float],
        measured_compute: list[float],
    ):
        """Computes the x and y ranges for the roofline plot.
        Args:
            mem_bandwidths (list[float]): List of memory bandwidths.
            compute_limits (list[float]): List of compute limits.
            measured_arithmic_intensities (list[float]): List of measured arithmetic intensities.
            measured_compute (list[float]): List of measured compute values.
        Returns:
            tuple: A tuple containing x_range and y_range.
        """
        custom_min = [RooflineUtils.MIN_VALUE]

        min_compute = min(measured_compute + custom_min)
        max_compute = max(measured_compute + compute_limits)

        y_range = (10 ** (np.floor(np.log10(min_compute))), 10 ** (np.ceil(np.log10(max_compute))))
        # print(
        #     f"Computed y_range: {y_range}, min_compute: {min_compute}, max_compute: {max_compute}, measured_compute: {measured_compute}, compute_limits: {compute_limits}"
        # )

        min_bandwidth = min(mem_bandwidths)
        min_ai = min(measured_arithmic_intensities + custom_min)
        max_ai = max(measured_arithmic_intensities + [max_compute / min_bandwidth])
        x_range = (10 ** (np.floor(np.log10(min_ai))), 10 ** (np.ceil(np.log10(max_ai) + 1)))
        return x_range, y_range

    @staticmethod
    def get_chart_options(
        title: str,
        data: pd.Series,
        mem_compute_pairs: list[tuple[tuple[str, str], str]],
        roofs: HardwareRoofs,
    ):
        """Generates the chart options for the roofline plot.
        Args:
            title (str): The title of the chart.
            data (pd.Series): The data series containing the metrics.
            mem_compute_pairs (list[tuple[tuple[str,str], str]]): List of memory-compute metric column name pairs.
            roofs (HardwareRoofs): Dictionary of roof names to their compute limits.
        Returns:
            dict: The chart options for the roofline plot.
        """

        def format_name(metric: str) -> str:
            return RooflineUtils.DISPLAY_NAME.get(metric, metric)

        mem_keys = [pair[0] for pair in mem_compute_pairs]
        compute_keys = [pair[1] for pair in mem_compute_pairs]
        mem_bandwidths = [roofs[mem] for mem in itertools.chain.from_iterable(mem_keys) if mem in roofs]
        compute_limits = [roofs[comp] for comp in compute_keys if comp in roofs]
        measured_arithmic_intensities = []
        measured_compute = []
        for mem_metrics, compute_metric in mem_compute_pairs:
            ai = compute_arithmic_intensity(data, compute_metric=compute_metric, mem_metrics=mem_metrics)
            if ai is not None:
                measured_arithmic_intensities.append(ai)
            try:
                compute_val = float(data[compute_metric] / data["GpuTime[ns]"])
                measured_compute.append(compute_val)
            except:
                pass
        x_range, y_range = RooflineUtils.compute_plot_range(
            mem_bandwidths, compute_limits, measured_arithmic_intensities, measured_compute
        )
        # print(f"Plot ranges: x={x_range}, y={y_range}")
        series = []
        for mem_metrics, compute_metric in mem_compute_pairs:
            mem_bw = sum(float(data[metric]) for metric in mem_metrics) / data["GpuTime[ns]"]
            s = dict(
                name=f"{format_name(mem_metrics[0])}-{format_name(compute_metric)}",
                type="scatter",
                tooltip={
                    ":formatter": f"function (params) {{ return '<b>I/byte: ' + params.value[0].toFixed(2) + '<br>GIPS: ' + params.value[1].toFixed(2) + '<br>BW GB/s: {mem_bw:.2f}</b>'; }}"
                },
                # encode={"tooltip": [0, 1]},
            )
            ai = compute_arithmic_intensity(data, compute_metric=compute_metric, mem_metrics=mem_metrics)
            # print(f"AI for {mem_metrics}-{compute_metric}: {ai}")
            if ai is not None:
                try:
                    compute_val = float(data[compute_metric] / data["GpuTime[ns]"])
                    # print(
                    #     f"Compute val for {mem_metrics}-{compute_metric}: {compute_val}, {data[compute_metric]}, {data['GpuTime[ns]']}"
                    # )
                    s["data"] = [[ai, compute_val]]
                    # print(f"series data for {mem_metrics}-{compute_metric}: {s['data']}")
                    s["markLine"] = dict(
                        symbol=["none", "none"], data=[], label=dict(distance=[len(series) * 200 + 10, 0])
                    )
                    comp_limit = roofs[compute_metric]
                    mem_limit = roofs[mem_metrics[0]]
                    start, end = get_roofline_points(mem_limit, comp_limit, min_value=x_range[0])
                    s["markLine"]["data"].append(
                        {
                            "name": f"{format_name(compute_metric)}: {comp_limit:.2f} GIPS",
                            "yAxis": end[1],
                            "label": {"formatter": "{b}", "position": "insideStartTop"},
                        }
                    )
                    s["markLine"]["data"].append(
                        [
                            {
                                "name": f"{format_name(mem_metrics[0])} BW {mem_limit:.2f} GB/s",
                                "coord": [start[0], start[1]],
                                "label": {"formatter": "{b}", "position": "insideMiddleTop"},
                            },
                            {
                                "coord": [end[0], end[1]],
                            },
                        ]
                    )
                    series.append(s)
                except Exception as e:
                    print(f"Error processing series for {mem_metrics}-{compute_metric}: {e}")
                    pass

        # print(title, len(series))
        options = {
            "title": {"text": title, "textStyle": {"fontSize": 12}},
            "xAxis": {
                "name": "instructions/byte",
                "type": "log",
                "min": x_range[0],
                "max": x_range[1],
            },
            "yAxis": {
                "name": "GIPS",
                "type": "log",
                "min": y_range[0],
                "max": y_range[1],
            },
            "tooltip": {},
            "legend": {},
            "series": series,
        }
        errors = []
        if (
            len(series) == 0
            and sum(data.get(k, 0) for k in itertools.chain.from_iterable(RooflineUtils.MEM_PAIRS)) == 0
        ):
            errors.append("Trace reports zero memory read/write bytes.")
        if errors:
            options["graphic"] = [
                {
                    "type": "text",
                    "left": "center",
                    "top": "middle",
                    "style": {
                        "text": "\n".join(errors),
                        "fontSize": 14,
                        "fill": "#f88",
                    },
                }
            ]
        return options

    def get_alu_breakdown_chart_options(metrics: pd.Series, roofs: HardwareRoofs) -> dict:
        """Generates chart options for the ALU breakdown bar chart.
        Args:
            metrics (pd.Series): The metrics series containing the ALU instruction counts.
            roofs (HardwareRoofs): The hardware roofs containing the ALU compute limits.
        Returns:
            dict: The chart options for the ALU breakdown bar chart.
        """
        dt = metrics["GpuTime[ns]"] / 100  # convert to percentage
        fp16 = metrics["XVE_INST_EXECUTED_FP16[events]"] / dt / roofs["XVE_INST_EXECUTED_FP16[events]"]
        fp32 = metrics["XVE_INST_EXECUTED_FP32[events]"] / dt / roofs["XVE_INST_EXECUTED_FP32[events]"]
        fp64 = metrics["XVE_INST_EXECUTED_FP64[events]"] / dt / roofs["XVE_INST_EXECUTED_FP64[events]"]
        int16 = metrics["XVE_INST_EXECUTED_INT16[events]"] / dt / roofs["XVE_INST_EXECUTED_INT16[events]"]
        int32 = metrics["XVE_INST_EXECUTED_INT32[events]"] / dt / roofs["XVE_INST_EXECUTED_INT32[events]"]
        int64 = metrics["XVE_INST_EXECUTED_INT64[events]"] / dt / roofs["XVE_INST_EXECUTED_INT64[events]"]
        em = metrics["XVE_INST_EXECUTED_MATH[events]"] / dt / roofs["XVE_INST_EXECUTED_MATH[events]"]
        xmx_bf16 = metrics["XVE_INST_EXECUTED_XMX_BF16[events]"] / dt / roofs["XVE_INST_EXECUTED_XMX_BF16[events]"]
        xmx_fp16 = metrics["XVE_INST_EXECUTED_XMX_FP16[events]"] / dt / roofs["XVE_INST_EXECUTED_XMX_FP16[events]"]
        xmx_int2 = metrics["XVE_INST_EXECUTED_XMX_INT2[events]"] / dt / roofs["XVE_INST_EXECUTED_XMX_INT2[events]"]
        xmx_int4 = metrics["XVE_INST_EXECUTED_XMX_INT4[events]"] / dt / roofs["XVE_INST_EXECUTED_XMX_INT4[events]"]
        xmx_int8 = metrics["XVE_INST_EXECUTED_XMX_INT8[events]"] / dt / roofs["XVE_INST_EXECUTED_XMX_INT8[events]"]

        bar_width = "95%"
        options = {
            "title": {"text": "ALU instruction breakdown", "textStyle": {"fontSize": 12}},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"bottom": "35%"},
            "legend": {
                "data": [
                    "FP16",
                    "FP32",
                    "FP64",
                    "INT16",
                    "INT32",
                    "INT64",
                    "EM",
                    "XMX BF16",
                    "XMX FP16",
                    "XMX INT2",
                    "XMX INT4",
                    "XMX INT8",
                ]
            },
            "xAxis": {"type": "category", "data": ["ALU0", "ALU1", "ALU2"]},
            "yAxis": {"type": "value", "min": 0, "max": 100, "axisLabel": {"formatter": "{value}%"}},
            "series": [
                {
                    "name": "FP16",
                    "type": "bar",
                    "stack": "total",
                    "data": [fp16, 0, 0],
                    "itemStyle": {"color": "#a0ebff"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "FP32",
                    "type": "bar",
                    "stack": "total",
                    "data": [fp32, 0, 0],
                    "itemStyle": {"color": "#00c7fd"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "FP64",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, fp64, 0],
                    "itemStyle": {"color": "#0095ca"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "INT16",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, int16, 0],
                    "itemStyle": {"color": "#d8f3a2"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "INT32",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, int32, 0],
                    "itemStyle": {"color": "#8bae46"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "INT64",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, int64, 0],
                    "itemStyle": {"color": "#515a3d"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "EM",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, em, 0],
                    "itemStyle": {"color": "#fee17a"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "XMX BF16",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, 0, xmx_bf16],
                    "itemStyle": {"color": "#98a1ff"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "XMX FP16",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, 0, xmx_fp16],
                    "itemStyle": {"color": "#5a69ff"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "XMX INT2",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, 0, xmx_int2],
                    "itemStyle": {"color": "#1f2db8"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "XMX INT4",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, 0, xmx_int4],
                    "itemStyle": {"color": "#030f8a"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
                {
                    "name": "XMX INT8",
                    "type": "bar",
                    "stack": "total",
                    "data": [0, 0, xmx_int8],
                    "itemStyle": {"color": "#030f8a"},
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "inside",
                        ":formatter": "function(params) { return params.value > 0 ? params.seriesName : ''; }",
                    },
                },
            ],
        }
        return options

    @staticmethod
    def get_vtune_alu_bar_chart_options(metrics: pd.Series, roofs: HardwareRoofs) -> dict:
        """Generates a simplified ALU utilization bar chart from VTune instruction counts.

        VTune provides ALU0/ALU1/XMX totals rather than per-instruction-type breakdowns,
        so this produces a three-bar chart showing utilisation of each pipeline as a
        percentage of the theoretical peak.

        Args:
            metrics (pd.Series): A row from the vtune_counters_to_dataframe() result.
            roofs (HardwareRoofs): The hardware roofs containing peak ALU compute limits.
        Returns:
            dict: ECharts options for the simplified ALU utilization bar chart.
        """
        gpu_time = metrics["GpuTime[ns]"]
        if gpu_time <= 0:
            return {}

        def _util(col, roof_key):
            count = metrics.get(col, 0) or 0
            roof = roofs[roof_key]
            if roof is None or roof == 0:
                return 0.0
            return min(100.0 * count / (gpu_time * roof), 100.0)

        alu0_util = _util("XVE_INST_EXECUTED_ALU0_ALL[events]", "XVE_INST_EXECUTED_ALU0_ALL[events]")
        alu1_util = _util("XVE_INST_EXECUTED_ALU1_ALL[events]", "XVE_INST_EXECUTED_ALU1_ALL[events]")
        xmx_util = _util("XVE_INST_EXECUTED_ALU2_ALL[events]", "XVE_INST_EXECUTED_ALU2_ALL[events]")

        bar_width = "60%"
        options = {
            "title": {"text": "ALU pipeline utilization (VTune)", "textStyle": {"fontSize": 12}},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "formatter": "{b}: {c:.1f}%"},
            "grid": {"bottom": "15%"},
            "xAxis": {"type": "category", "data": ["ALU0\n(FP16/FP32)", "ALU1\n(FP64/INT/Math)", "XMX\n(Matrix/DPAS)"]},
            "yAxis": {
                "type": "value",
                "min": 0,
                "max": 100,
                "axisLabel": {"formatter": "{value}%"},
                "name": "% of peak",
            },
            "series": [
                {
                    "name": "Utilization",
                    "type": "bar",
                    "data": [
                        {"value": alu0_util, "itemStyle": {"color": "#00c7fd"}},
                        {"value": alu1_util, "itemStyle": {"color": "#8bae46"}},
                        {"value": xmx_util, "itemStyle": {"color": "#5a69ff"}},
                    ],
                    "barWidth": bar_width,
                    "label": {
                        "show": True,
                        "position": "top",
                        ":formatter": "function(params) { return params.value.toFixed(1) + '%'; }",
                    },
                },
            ],
        }
        return options

    @staticmethod
    def get_chart_options_for_kernel(
        profiler_data: dict,
        worker_info: dict,
        profiler_key: str,
        shorten_fn_signatures: bool = True,
        figure_templates: dict | None = None,
    ) -> OrderedDict[str, dict]:
        """Generates chart options for all roofline plots for a given kernel entry
        Args:
            profiler_data (dict): The profiler data dictionary.
            worker_info (dict): The worker info dictionary.
            profiler_key (str): The key in the profiler data to use (e.g. 'unitrace', 'vtune').
            shorten_fn_signatures (bool): Whether to shorten function signatures in titles.
        Returns:
            OrderedDict[str, dict]: An ordered dictionary mapping kernel names to their chart data.
                Each value contains: 'roofline_options', 'mem_figure', 'alu_bar_chart_options'
        """
        result = OrderedDict()
        roofs_info = get_roofs(worker_info)
        if roofs_info is None:
            return result
        raw_data = profiler_data.get(profiler_key, {})
        # Detect vtune
        is_vtune = "counters" in raw_data
        if is_vtune:
            df_compute_basic = RooflineUtils.vtune_counters_to_dataframe(raw_data)
            df_memory_profile = None
            df_vector_engine_profile = None
            vtune_counters = raw_data.get("counters") or {}
            df_vtune_raw = (
                pd.DataFrame([{"Kernel": k, **v} for k, v in vtune_counters.items()]) if vtune_counters else None
            )
        else:
            df_compute_basic, df_memory_profile, df_vector_engine_profile = RooflineUtils.read_metrics(raw_data)
            vtune_counters = {}
            df_vtune_raw = None
        if df_compute_basic is None:
            return result
        df = df_compute_basic
        metrics_per_kernel = RooflineUtils.get_median_row_for_each_kernel(df_compute_basic)

        def _get_closest_per_kernel(df_other):
            result = []
            for m in metrics_per_kernel:
                group = df_other[df_other["Kernel"] == m["Kernel"]]
                if len(group) > 0:
                    result.append(get_row_with_closest_metric(group, m["GpuTime[ns]"]))
            return result if result else None

        mem_metrics_per_kernel = _get_closest_per_kernel(df_memory_profile) if df_memory_profile is not None else None
        ve_metrics_per_kernel = (
            _get_closest_per_kernel(df_vector_engine_profile) if df_vector_engine_profile is not None else None
        )
        for metrics in metrics_per_kernel:
            name = f"{metrics['Kernel']}"
            if shorten_fn_signatures:
                name = RooflineUtils.shorten_fn_name(name)
            mem_compute_pairs = []
            for mem_pair, compute_metric in itertools.product(RooflineUtils.MEM_PAIRS, RooflineUtils.COMPUTE_METRICS):
                if (
                    mem_pair[0] in metrics
                    and mem_pair[1] in metrics
                    and compute_metric in metrics
                    and metrics[compute_metric]  # not None and > 0
                ):
                    mem_compute_pairs.append((mem_pair, compute_metric))

            if len(mem_compute_pairs) == 0:
                # this can happen for kernels that are pure memory transfer operations (e.g. for OCL clEnqueueReadBuffer)
                warnings.warn(f"No valid memory-compute pairs found for kernel {name}, skipping roofline chart.")
                continue

            options = RooflineUtils.get_chart_options(
                title=name,
                data=metrics,
                mem_compute_pairs=mem_compute_pairs,
                roofs=roofs_info,
            )
            figure = None
            alu_bar_chart_options = None
            freqs_runtime_info = OrderedDict()

            if is_vtune:
                kernel_cols = vtune_counters.get(metrics["Kernel"], {})

                def _fv(key, _cols=kernel_cols):
                    v = _cols.get(key)
                    if v is None:
                        return None
                    try:
                        return float(str(v).rstrip("%").replace(",", "").strip())
                    except:
                        return str(v)

                # Sidebar info: (display_label, value) — pulled from transformed metrics or raw kernel_cols
                freqs_runtime_info["VTune"] = {
                    k: v
                    for k, v in {
                        "GpuTime[ns]": metrics.get("GpuTime[ns]"),
                        "SIMD Width": _fv("Computing Task:SIMD Width"),
                        "Instance Count": _fv("Computing Task:Instance Count"),
                        "Average Time (s)": _fv("Computing Task:Average Time"),
                        "XVE Active (%)": _fv("XVE Array:Active(%)"),
                        "XVE Stalled (%)": _fv("XVE Array:Stalled(%)"),
                        "XVE Idle (%)": _fv("XVE Array:Idle(%)"),
                        "Thread Occupancy (%)": _fv("XVE Threads Occupancy(%)"),
                        "GPU Mem BW Read (GB/s)": metrics.get("GPU_MEMORY_BYTE_READ_RATE[GBpS]"),
                        "GPU Mem BW Write (GB/s)": metrics.get("GPU_MEMORY_BYTE_WRITE_RATE[GBpS]"),
                        "L3 BW Read (GB/s)": metrics.get("L3_READ[GBpS]"),
                        "L3 BW Write (GB/s)": metrics.get("L3_WRITE[GBpS]"),
                        "L3 Busy (%)": _fv("GPU L3:Busy(%)"),
                        "L3 Miss Ratio (%)": _fv("GPU L3:Miss Ratio(%)"),
                        "L3 Stalled (%)": _fv("GPU L3:Stalled(%)"),
                        "TLB Misses": _fv("TLB Misses"),
                    }.items()
                    if v is not None
                }

                if figure_templates and "intel_mem" in figure_templates:
                    # Pad columns absent from VTune data so SVG eval expressions resolve to 0
                    _padded = metrics.copy()
                    for _col in [
                        "SLM_BANK_CONFLICT_COUNT[events]",
                        "SLM_ACCESS_COUNT[events]",
                        "SLM_BYTE_READ[bytes]",
                        "SLM_BYTE_WRITE[bytes]",
                        "LOAD_STORE_CACHE_ACCESS[events]",
                        "LOAD_STORE_CACHE_BYTE_READ[bytes]",
                        "LOAD_STORE_CACHE_BYTE_WRITE[bytes]",
                        "LOAD_STORE_CACHE_HIT[events]",
                        "GPU_MEMORY_BYTE_READ[bytes]",
                        "GPU_MEMORY_BYTE_WRITE[bytes]",
                    ]:
                        if _col not in _padded.index:
                            _padded[_col] = 0
                    _tmpl = figure_templates["intel_mem"]
                    for _old, _new in [
                        (
                            "eval[fmtb({GPU_MEMORY_BYTE_READ[bytes]})]",
                            f"{(_padded.get('GPU_MEMORY_BYTE_READ_RATE[GBpS]')  or 0):.2f} GB/s",
                        ),
                        (
                            "eval[fmtb({GPU_MEMORY_BYTE_WRITE[bytes]})]",
                            f"{(_padded.get('GPU_MEMORY_BYTE_WRITE_RATE[GBpS]') or 0):.2f} GB/s",
                        ),
                        ("eval[fmtb(64*{L3_READ[events]})]", f"{(_padded.get('L3_READ[GBpS]')  or 0):.2f} GB/s"),
                        ("eval[fmtb(64*{L3_WRITE[events]})]", f"{(_padded.get('L3_WRITE[GBpS]') or 0):.2f} GB/s"),
                    ]:
                        _tmpl = _tmpl.replace(_old, _new)
                    figure = fill_figure_template(_tmpl, [_padded])
                    _gpu_time = metrics["GpuTime[ns]"]
                    _bars = {
                        name: (
                            min(metrics.get(col, 0) / (_gpu_time * roofs_info[col]), 1.0)
                            if (_gpu_time and roofs_info[col])
                            else 0.0
                        )
                        for col, name in [
                            ("XVE_INST_EXECUTED_ALU0_ALL[events]", "ALU0"),
                            ("XVE_INST_EXECUTED_ALU1_ALL[events]", "ALU1"),
                            ("XVE_INST_EXECUTED_ALU2_ALL[events]", "ALU2"),
                        ]
                    }
                    figure = _fill_figure_template_alu_bars(figure, _bars)

                alu_bar_chart_options = RooflineUtils.get_vtune_alu_bar_chart_options(metrics, roofs_info)
            else:
                freqs_runtime_info["ComputeBasic"] = {
                    "GpuTime[ns]": metrics.get("GpuTime[ns]", "N/A"),
                    "AvgGpuCoreFrequency[MHz]": metrics.get("AvgGpuCoreFrequencyMHz[MHz]", "N/A"),
                    "CoreFrequency[MHz]": metrics.get("CoreFrequencyMHz[MHz]", "N/A"),
                    "CoreFrequencyChanged": metrics.get("CoreFrequencyChanged", "N/A"),
                }

                if mem_metrics_per_kernel is not None:
                    mem_metrics = next((m for m in mem_metrics_per_kernel if m["Kernel"] == metrics["Kernel"]), None)
                    figure = (
                        fill_figure_template(figure_templates["intel_mem"], [mem_metrics]) if figure_templates else None
                    )
                    freqs_runtime_info["MemoryProfile"] = {
                        "GpuTime[ns]": mem_metrics.get("GpuTime[ns]", "N/A"),
                        "AvgGpuCoreFrequency[MHz]": mem_metrics.get("AvgGpuCoreFrequencyMHz[MHz]", "N/A"),
                        "CoreFrequency[MHz]": mem_metrics.get("CoreFrequencyMHz[MHz]", "N/A"),
                        "CoreFrequencyChanged": mem_metrics.get("CoreFrequencyChanged", "N/A"),
                    }
                if figure and ve_metrics_per_kernel is not None:
                    ve_metrics = next((m for m in ve_metrics_per_kernel if m["Kernel"] == metrics["Kernel"]), None)
                    freqs_runtime_info["VectorEngineProfile"] = {
                        "GpuTime[ns]": ve_metrics.get("GpuTime[ns]", "N/A"),
                        "AvgGpuCoreFrequency[MHz]": ve_metrics.get("AvgGpuCoreFrequencyMHz[MHz]", "N/A"),
                        "CoreFrequency[MHz]": ve_metrics.get("CoreFrequencyMHz[MHz]", "N/A"),
                        "CoreFrequencyChanged": ve_metrics.get("CoreFrequencyChanged", "N/A"),
                        "XVE_THREADS_OCCUPANCY_ALL[%]": ve_metrics.get("XVE_THREADS_OCCUPANCY_ALL[%]", "N/A"),
                    }
                    bars = {}
                    gpu_time = metrics["GpuTime[ns]"]
                    bars["ALU0"] = (
                        ve_metrics.get("XVE_INST_EXECUTED_ALU0_ALL[events]", 0)
                        / gpu_time
                        / roofs_info["XVE_INST_EXECUTED_ALU0_ALL[events]"]
                    )
                    bars["ALU1"] = (
                        ve_metrics.get("XVE_INST_EXECUTED_ALU1_ALL[events]", 0)
                        / gpu_time
                        / roofs_info["XVE_INST_EXECUTED_ALU1_ALL[events]"]
                    )
                    bars["ALU2"] = (
                        ve_metrics.get("XVE_INST_EXECUTED_ALU2_ALL[events]", 0)
                        / gpu_time
                        / roofs_info["XVE_INST_EXECUTED_ALU2_ALL[events]"]
                    )
                    figure = _fill_figure_template_alu_bars(figure, bars)
                if ve_metrics_per_kernel is not None:
                    ve_metrics = next((m for m in ve_metrics_per_kernel if m["Kernel"] == metrics["Kernel"]), None)
                    alu_bar_chart_options = RooflineUtils.get_alu_breakdown_chart_options(ve_metrics, roofs_info)

            result[name] = {
                "roofline_options": options,
                "mem_figure": figure,
                "alu_bar_chart_options": alu_bar_chart_options,
                "df": df_vtune_raw if is_vtune else df_compute_basic,
                "df_title": "VTune Counters" if is_vtune else "ComputeBasic",
                "dfmem": df_memory_profile,
                "dfve": df_vector_engine_profile,
                "freqs_runtime_info": freqs_runtime_info,
            }

        return result


def render_kernel_charts_section(
    label: str,
    profiler_data: dict,
    worker_info: dict,
    profiler_key: str,
    profiler_feedback: str | None = None,
    figure_templates: dict | None = None,
):
    """Renders the profiler charts section for a kernel (custom or reference).

    Args:
        label: Section header label (e.g. "custom" or "reference").
        profiler_data: The profiler data dictionary for the kernel.
        worker_info: The worker info dictionary for the kernel.
        profiler_key: The key in profiler_data to use (e.g. "unitrace").
        profiler_feedback: Optional profiler feedback text to display after the charts.
    """
    ui.label(label).classes("text-center w-full text-xl")
    for title, chart_data in RooflineUtils.get_chart_options_for_kernel(
        profiler_data, worker_info, profiler_key, figure_templates=figure_templates
    ).items():
        with ui.row().classes("w-full"):
            ui.echart(options=chart_data["roofline_options"]).classes("w-full h-96")
            if chart_data.get("mem_figure") is not None:
                ui.html(chart_data["mem_figure"], sanitize=False).classes("w-1/3 h-96 mt-6")
            if chart_data.get("alu_bar_chart_options"):
                ui.echart(options=chart_data["alu_bar_chart_options"]).classes("w-1/3 h-96")
            if chart_data.get("freqs_runtime_info") is not None:
                info = chart_data["freqs_runtime_info"]
                with ui.column().classes("w-3/11 h-96 overflow-y-auto p-2 text-xs"):
                    with ui.grid(columns="auto auto").classes("w-full gap-x-1"):
                        for section, metrics in info.items():
                            ui.label(section).classes("font-bold mt-4 break-all")
                            ui.label("")
                            for key, value in metrics.items():
                                ui.label(key).classes("break-all")
                                if "[%]" in key and isinstance(value, (int, float)):
                                    value = f"{value:.2f}%"
                                ui.label(str(value))
            if chart_data.get("df") is not None:
                with ui.expansion("Raw Data").classes("w-full mt-4"):
                    column_defaults = {
                        "style": "text-wrap: wrap; max-width=32px;",
                    }
                    df = chart_data["df"]
                    if df is not None:
                        ui.table.from_pandas(
                            df,
                            title=chart_data.get("df_title", "ComputeBasic"),
                            pagination=10,
                            column_defaults=column_defaults,
                        ).classes("w-full")
                    df = chart_data["dfmem"]
                    if df is not None:
                        ui.table.from_pandas(
                            df, title="MemoryProfile", pagination=10, column_defaults=column_defaults
                        ).classes("w-full")
                    df = chart_data["dfve"]
                    if df is not None:
                        ui.table.from_pandas(
                            df, title="VectorEngineProfile", pagination=10, column_defaults=column_defaults
                        ).classes("w-full")
    if profiler_feedback:
        with ui.expansion("Profiler Feedback").classes("w-full mt-4"):
            ui.code(profiler_feedback, language="markdown").classes("w-full whitespace-pre-wrap text-sm")


def write_trace_and_create_traceviewer_button(kernel, kernel_id):
    """Write profiler trace to file and create a viewer button for timeline analysis."""
    trace_file_name = f"profiler_data_{kernel_id}.json"
    profiler_data_dir = ui.traceviewer.profiler_data_dir
    temp_trace_path = os.path.join(profiler_data_dir, trace_file_name)

    def get_timeline_data_from_kernel(kernel):
        try:
            return {"traceEvents": kernel.profiler_data_detail["unitrace"]["timeline"]["traceEvents"]}
        except:
            pass
        try:
            return {"traceEvents": kernel.profiler_data["unitrace"]["custom"]["timeline"]["traceEvents"]}
        except:
            pass
        try:
            return {"traceEvents": kernel.profiler_data["unitrace"]["timeline"]["traceEvents"]}
        except:
            pass
        return {}

    data = get_timeline_data_from_kernel(kernel)

    if data:
        with open(temp_trace_path, "w") as f:
            json_data = json.dumps(data, indent=4)
            f.write(json_data)

    trace_button = ui.traceviewer("Trace View", f"/profiler_data/{trace_file_name}", trace_file_name).props(
        "icon=launch flat no-caps size=md"
    )
    if not data:
        trace_button.disable()
    return trace_button


def roofline_page(kernel_id: int = 0, template_dir: Path | None = None):
    """Render the roofline analysis page for a kernel with profiler data visualization."""
    ui.traceviewer.patch_html()
    ui.add_css("""
.svg-content-scale { width: 100%; height: auto; max-height: 100%;}
div.nicegui-code * > pre {
    white-space: pre-wrap;
    text-wrap: wrap;
    word-break: break-all;
}
    """)

    FIGURE_TEMPLATES = _load_svg_templates(template_dir)

    kernel = get_kernel_by_id(kernel_id)
    if kernel.profiler_data_reference is None:
        # Check kernels from the same job for the profiler data reference
        profiler_data_reference, eval_worker_info_reference = get_profiler_data_reference_by_job_id(kernel.job_id)
        if eval_worker_info_reference is not None and "gpu_name" in eval_worker_info_reference:
            eval_worker_info_reference = eval_worker_info_reference
        elif eval_worker_info_reference is not None and len(eval_worker_info_reference) == 1:
            eval_worker_info_reference = next(iter(eval_worker_info_reference.values()))
    else:
        # Reference profiler data was captured together with this kernel
        profiler_data_reference = kernel.profiler_data_reference
        if kernel.eval_worker_info is not None and "gpu_name" in kernel.eval_worker_info:
            eval_worker_info_reference = kernel.eval_worker_info
        elif kernel.eval_worker_info is not None and len(kernel.eval_worker_info) == 1:
            eval_worker_info_reference = next(iter(kernel.eval_worker_info.values()))

    with ui.row().classes("items-end mb-4"):
        ui.label(f"Kernel Roofline for ID: {kernel_id}").classes("text-xl")
        # ui.link("Kernel Detail", f"/kernel/{kernel_id}")
        ui.button("Kernel Detail", on_click=lambda: ui.navigate.to(f"/kernel/{kernel_id}")).props(
            "icon=info flat no-caps size=md"
        )
        write_trace_and_create_traceviewer_button(kernel, kernel_id).classes("ml-4")

    if kernel is None:
        ui.label("Kernel not found").classes("text-red-500")
        return
    with ui.expansion("Worker info").classes("w-full mb-2"):
        with ui.row().classes("grid grid-cols-8 min-w-full"):
            with ui.column().classes("col-span-4 gap-0"):
                if kernel.eval_worker_info is None:
                    ui.label("No worker info available")
                else:
                    eval_worker_info = kernel.eval_worker_info
                    if "gpu_name" not in eval_worker_info and len(eval_worker_info) == 1:
                        eval_worker_info = next(iter(kernel.eval_worker_info.values()))

                    for key, value in sorted(eval_worker_info.items()):
                        with ui.row().classes():
                            ui.label(key).classes("font-bold").style("width: 240px;")
                            ui.label(str(value))
            with ui.column().classes("col-span-4 gap-0"):
                if eval_worker_info_reference is None:
                    ui.label("No worker info for reference available")
                else:
                    for key, value in sorted(eval_worker_info_reference.items()):
                        with ui.row().classes():
                            ui.label(key).classes("font-bold").style("width: 240px;")
                            classes = "text-red-600" if eval_worker_info and eval_worker_info.get(key) != value else ""
                            ui.label(str(value)).classes(classes)

    with ui.row().classes("grid grid-cols-8 min-w-full"):
        with ui.column().classes("col-span-4 gap-0"):
            # Custom kernel info and charts — detect available profiler (vtune takes priority over unitrace)
            custom_profiler_key = "vtune" if kernel.profiler_data and "vtune" in kernel.profiler_data else "unitrace"
            if custom_profiler_key == "vtune":
                _pf = VTuneProfilerFeedback()
                feedback = _pf.create_feedback(kernel.profiler_data.get("vtune", {}), eval_worker_info or {})
            else:
                feedback = UnitraceProfilerFeedback().create_feedback(
                    kernel.profiler_data.get("unitrace", {}) if kernel.profiler_data else {},
                    eval_worker_info or {},
                )
            content_custom = render_kernel_charts_section(
                "custom",
                kernel.profiler_data or {},
                eval_worker_info or {},
                custom_profiler_key,
                profiler_feedback=feedback or None,
                figure_templates=FIGURE_TEMPLATES,
            )

        if profiler_data_reference is not None:
            with ui.column().classes("col-span-4 gap-0"):
                # Reference profiler data and charts
                ref_profiler_key = "vtune" if "vtune" in profiler_data_reference else "unitrace"
                content_reference = render_kernel_charts_section(
                    "reference",
                    profiler_data_reference or {},
                    eval_worker_info_reference or {},
                    ref_profiler_key,
                    figure_templates=FIGURE_TEMPLATES,
                )

        else:  # old format code path. TODO remove this eventually
            with ui.column().classes("col-span-4 gap-0"):
                content_reference = render_kernel_charts_section(
                    "reference",
                    kernel.profiler_data or {},
                    kernel.eval_worker_info or {},
                    "unitrace_ref",
                    figure_templates=FIGURE_TEMPLATES,
                )

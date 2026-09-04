"""Profiler feedback generators for different profilers (Unitrace, NCU, VTune)"""

import pandas as pd
import re
import io
import csv
import json
from collections import OrderedDict
from kernelfoundry.eval_pipeline.utils.performance_analysis import *
from kernelfoundry.eval_pipeline.utils.hardware_info import HardwareRoofs
from io import StringIO
from abc import ABC, abstractmethod
import logging
from jinja2 import Environment, PackageLoader, select_autoescape

# GPU architectures that are supported - can be extended
VALID_CUDA_ARCHS = [
    "Maxwell",
    "Pascal",
    "Volta",
    "Turing",
    "Ampere",
    "Hopper",
    "Ada",
    "native",
    "A100",
    "A6000",
    "L40S",
    "L4",
]

PROFILER_ANALYSIS_PROMPT = """
Your kernel has been analyzed with a profiler. Here is a summary of the results:
{profiler_summary}
"""

PROFILER_ANALYSIS_INTRO = "Your code has been analyzed with a profiler. Here is a summary of the results:"


class ProfilerFeedback:
    """Abstract base class for profiler feedback generators.

    Subclasses implement profiler-specific logic to collate raw output data
    from multiple profiling passes and produce human-readable feedback strings
    that can be injected into LLM prompts.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the profiler"""
        pass

    @abstractmethod
    def collate_data(self, outputs: dict[str, dict]) -> dict:
        """Collate the output data from multiple profiling runs into a single dictionary for analysis.

        Args:
            outputs: A dictionary of dictionaries containing the output data from multiple profiling runs.
                Example: {'unitrace.001': {'timeline': '...', 'metrics': '...'}, 'unitrace.002': {...}, ...}

        Returns:
            A single dictionary that collates the relevant data from the multiple profiling runs for analysis.
            Example::

                {'timeline': '...', 'ComputeBasic.metrics.pid': '...', 'MemoryBasic.metrics.pid': '...'}

            Note that the data keys in outputs and the collated dictionary are specific to the profiler.
        """
        pass

    @abstractmethod
    def create_feedback(self, data: dict, worker_info: dict) -> str:
        """Create feedback based on the collated data and worker information.

        Args:
            data: A dictionary containing the collated data from the profiling runs, as returned by collate_data().
            worker_info: A dictionary containing information about the worker that executed the profiling runs,
                such as GPU architecture, device ID, etc.

        Returns:
            A string containing the generated feedback based on the collated data and worker information.
        """
        pass

    def collate_and_create_feedback(self, outputs: dict[str, dict], worker_info: dict) -> tuple[dict, str]:
        """Collate the output data from multiple profiling runs and create feedback.

        Args:
            outputs: A dictionary of dictionaries containing the output data from multiple profiling runs.
                Example: {'unitrace.001': {'timeline': '...', 'trace.metrics.pid': '...'}, 'unitrace.002': {...}, ...}
            worker_info: A dictionary containing information about the worker that executed the profiling runs,
                such as GPU architecture, device ID, etc.

        Returns:
            A tuple containing the collated data dictionary and the generated feedback string.
            Example::

                (
                    {'timeline': '...', 'ComputeBasic.metrics.pid': '...', 'MemoryBasic.metrics.pid': '...'},
                    "The kernel is memory bound ..."
                )
        """
        data = self.collate_data(outputs)
        feedback = self.create_feedback(data, worker_info)
        return data, feedback


class UnitraceProfilerFeedback(ProfilerFeedback):
    """Profiler feedback for SYCL kernels using Intel unitrace (metric-query mode).

    Supports multi-pass profiling with ComputeBasic, MemoryProfile, and
    VectorEngineProfile metric groups.
    """

    MODEL_RUN_LOOP_EVENT_PREFIX = "ittapi::model run loop"

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "unitrace"

    @staticmethod
    def _short_kernel_name_and_grid(full_name: str) -> tuple[str, str]:
        """Returns a shortened version of the kernel name and its grid configuration."""
        grid = re.search(r"\[.+\]$", full_name)
        name = full_name.replace("(anonymous namespace)::", "")
        if grid:
            grid = grid.group(0)
            name = name.replace(grid, "")
        return name, grid

    @staticmethod
    def _read_metrics(profiler_data: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """Reads the metrics from the profiler data dictionary.
        Args:
            profiler_data (dict): The profiler data dictionary.
        Returns:
            tuple: A tuple containing three DataFrames (or None) for different metric groups.
                (ComputeBasic, MemoryProfile, VectorEngineProfile)
        """
        compute_basic = next(((k, v) for k, v in profiler_data.items() if "ComputeBasic.metrics." in k), None)
        memory_profile = next(((k, v) for k, v in profiler_data.items() if "MemoryProfile.metrics." in k), None)
        vector_engine_profile = next(
            ((k, v) for k, v in profiler_data.items() if "VectorEngineProfile.metrics." in k), None
        )
        if compute_basic is None:
            compute_basic = next(((k, v) for k, v in profiler_data.items() if ".metrics." in k), None)

        result = []
        for metric in [compute_basic, memory_profile, vector_engine_profile]:
            if metric is not None:
                name, data = metric
                pid = name.split(".metrics.")[-1]
                lines = data.splitlines()
                lines = [line for line in lines if line and not line.startswith("==")]
                df = pd.read_csv(io.StringIO("\n".join(lines)))
                df.attrs["process_id"] = pid
                result.append(df)
            else:
                result.append(None)
        return tuple(result)

    def _get_model_run_segments_from_timeline(self, timeline: dict) -> list[dict]:
        """Extract model run loop segments from the timeline data.

        Args:
            timeline (dict): The timeline data from the profiler output.

        Returns:
            list[dict]: Segment metadata with labels and time ranges.
        """
        model_run_loop_entries = []
        for event in timeline.get("traceEvents", []):
            event_name = event.get("name", "")
            if event_name == self.MODEL_RUN_LOOP_EVENT_PREFIX:
                label = "default"
            elif event_name.startswith(f"{self.MODEL_RUN_LOOP_EVENT_PREFIX}::"):
                label = event_name.split("::", 2)[-1]
            else:
                continue

            start_time = event.get("ts")
            duration = event.get("dur")
            if start_time is None or duration is None:
                continue

            model_run_loop_entries.append(
                {
                    "label": label,
                    "trace_event_name": event_name,
                    "time_range": (start_time, start_time + duration),
                }
            )

        if not model_run_loop_entries:
            logging.warning(
                f"[Profiling] Entry '{self.MODEL_RUN_LOOP_EVENT_PREFIX}' not found in timeline (ignore if user-defined benchmarking)."
            )
            return [{"label": "default", "trace_event_name": None, "time_range": (0, float("inf"))}]

        return model_run_loop_entries

    def _get_global_ids_from_timeline(self, timeline: dict, time_range: tuple) -> list:
        """Extract global IDs of events from the timeline data within a specified time range.

        Args:
            timeline (dict): The timeline data from the profiler output.
            time_range (tuple): A tuple containing the start and end times to filter events.

        Returns:
            list: A list of global IDs within the time range.
        """
        start_time, end_time = time_range
        global_ids = []
        for event in timeline.get("traceEvents", []):
            event_start = event.get("ts", 0)
            event_end = event_start + event.get("dur", 0)

            if event_start >= start_time and event_end <= end_time:
                gid = event.get("args", {}).get("id")
                if gid is not None:
                    global_ids.append(int(gid))

        return global_ids

    def analyze_kernel(
        self,
        compute_basic: pd.Series,
        memory_profile: pd.Series,
        vector_engine_profile: pd.Series,
        roofs: HardwareRoofs,
    ) -> tuple[str, dict]:
        """Analyze the kernel performance based on profiling data.
        Args:
            df (pd.Series): A pandas Series containing profiling metrics for a kernel.

        Returns:
            list: A list of tuples containing importance scores and corresponding analysis messages.
        """
        kernel_name = compute_basic["Kernel"]
        kernel_name, launch_config = self._short_kernel_name_and_grid(kernel_name)
        results = dict()

        runtime_ns = compute_basic["GpuTime[ns]"]
        results["runtime"] = runtime_ns / 1e6  # convert to ms
        results["launch_config"] = launch_config
        results["thread_occupancy"] = vector_engine_profile.get("XVE_THREADS_OCCUPANCY_ALL[%]", None)
        gmem_rw_rate = (
            compute_basic["GPU_MEMORY_BYTE_READ_RATE[GBpS]"] + compute_basic["GPU_MEMORY_BYTE_WRITE_RATE[GBpS]"]
        )
        results["gmem_bandwidth"] = gmem_rw_rate
        results["gmem_bandwidth_pct"] = 100 * gmem_rw_rate / roofs.GPU_MEMORY_BYTE_READ
        slm_rw_rate = compute_basic["SLM_BYTE_READ[bytes]"] + compute_basic["SLM_BYTE_WRITE[bytes]"]
        slm_rw_rate /= runtime_ns  # convert to bytes per ns
        results["slm_bandwidth_pct"] = 100 * slm_rw_rate / (roofs.SLM_BYTE_READ + roofs.SLM_BYTE_WRITE)
        results["ve_stall_pct"] = compute_basic["XVE_STALL[%]"]
        if memory_profile.get("SLM_ACCESS_COUNT[events]", 0) > 0:
            results["slm_conflicts_pct"] = (
                100 * memory_profile["SLM_BANK_CONFLICT_COUNT[events]"] / memory_profile["SLM_ACCESS_COUNT[events]"]
            )
        else:
            results["slm_conflicts_pct"] = 0
        bounds = []
        alu_util = []
        for i in range(3):
            compute_metric = f"XVE_INST_EXECUTED_ALU{i}_ALL[events]"
            compute = compute_basic[compute_metric] / runtime_ns  # convert to instructions per ns
            ai = compute_arithmic_intensity(compute_basic, compute_metric=f"XVE_INST_EXECUTED_ALU{i}_ALL[events]")
            roofline_point_ai_low = get_roofline_points(
                max_mem_bw=roofs.GPU_MEMORY_BYTE_READ, max_compute=0.9 * roofs.get(compute_metric)
            )[1][0]
            roofline_point_ai_high = get_roofline_points(
                max_mem_bw=0.9 * roofs.GPU_MEMORY_BYTE_READ, max_compute=roofs.get(compute_metric)
            )[1][0]
            if ai is None:
                bounds.append((compute, "unknown (missing arithmetic intensity)"))
            elif ai < roofline_point_ai_low:
                bounds.append((compute, "memory"))
            elif ai > roofline_point_ai_high:
                bounds.append((compute, "compute"))
            else:
                bounds.append((compute, "balanced"))

            # ALU breakdown
            alu_dict = {"name": f"ALU{i}", "util_pct": 100 * compute / roofs.get(compute_metric)}
            alu_to_instructions = {
                "ALU0": ["XVE_INST_EXECUTED_FP16[events]", "XVE_INST_EXECUTED_FP32[events]"],
                "ALU1": [
                    "XVE_INST_EXECUTED_FP64[events]",
                    "XVE_INST_EXECUTED_INT16[events]",
                    "XVE_INST_EXECUTED_INT32[events]",
                    "XVE_INST_EXECUTED_INT64[events]",
                    "XVE_INST_EXECUTED_MATH[events]",
                ],
                "ALU2": [
                    "XVE_INST_EXECUTED_XMX_BF16[events]",
                    "XVE_INST_EXECUTED_XMX_FP16[events]",
                    "XVE_INST_EXECUTED_XMX_INT2[events]",
                    "XVE_INST_EXECUTED_XMX_INT4[events]",
                    "XVE_INST_EXECUTED_XMX_INT8[events]",
                ],
            }
            instruction_to_short_name = {
                "XVE_INST_EXECUTED_FP16[events]": "FP16",
                "XVE_INST_EXECUTED_FP32[events]": "FP32",
                "XVE_INST_EXECUTED_FP64[events]": "FP64",
                "XVE_INST_EXECUTED_INT16[events]": "INT16",
                "XVE_INST_EXECUTED_INT32[events]": "INT32",
                "XVE_INST_EXECUTED_INT64[events]": "INT64",
                "XVE_INST_EXECUTED_MATH[events]": "MATH",
                "XVE_INST_EXECUTED_XMX_BF16[events]": "XMX_BF16",
                "XVE_INST_EXECUTED_XMX_FP16[events]": "XMX_FP16",
                "XVE_INST_EXECUTED_XMX_INT2[events]": "XMX_INT2",
                "XVE_INST_EXECUTED_XMX_INT4[events]": "XMX_INT4",
                "XVE_INST_EXECUTED_XMX_INT8[events]": "XMX_INT8",
            }
            instr_mix = []
            for instr_metric in alu_to_instructions[alu_dict["name"]]:
                instr_pct = 0
                if vector_engine_profile.get(compute_metric, 0) > 0:
                    instr_pct = 100 * vector_engine_profile[instr_metric] / vector_engine_profile[compute_metric]
                instr_mix.append({"name": instruction_to_short_name[instr_metric], "pct": instr_pct})
            alu_dict["instr_mix"] = instr_mix

            alu_util.append(alu_dict)
        alu_util = sorted(alu_util, key=lambda x: x["util_pct"], reverse=True)
        results["alu_util"] = alu_util

        if bounds:
            bound = sorted(bounds, key=lambda x: x[0], reverse=True)[0][1]
        else:
            bound = None
        results["bound"] = bound

        return kernel_name, results

    def _create_feedback_for_single_run(self, data: dict, worker_info: dict) -> str:
        df_compute_basic, df_memory_profile, df_vector_engine_profile = self._read_metrics(data)
        if df_compute_basic is None:
            logging.warning("[Profiling] ComputeBasic metrics missing — cannot generate feedback.")
            return ""
        if df_memory_profile is None:
            logging.warning("[Profiling] MemoryProfile metrics missing — SLM conflict analysis will be skipped.")
        if df_vector_engine_profile is None:
            logging.warning(
                "[Profiling] VectorEngineProfile metrics missing — thread occupancy and instruction mix analysis will be skipped."
            )

        hardware_roofs = get_roofs(worker_info)
        if hardware_roofs is None:
            return ""

        analysis_results = OrderedDict()
        for kernel_name, group in df_compute_basic.groupby("Kernel"):
            compute_basic = get_median_row(group)
            gpu_time = compute_basic["GpuTime[ns]"]
            memory_profile = (
                get_row_with_closest_metric(df_memory_profile[df_memory_profile["Kernel"] == kernel_name], gpu_time)
                if df_memory_profile is not None
                else pd.Series(dtype=float)
            )
            vector_engine_profile = (
                get_row_with_closest_metric(
                    df_vector_engine_profile[df_vector_engine_profile["Kernel"] == kernel_name], gpu_time
                )
                if df_vector_engine_profile is not None
                else pd.Series(dtype=float)
            )
            name, analysis = self.analyze_kernel(compute_basic, memory_profile, vector_engine_profile, hardware_roofs)
            analysis_results[name] = analysis

        env = Environment(
            loader=PackageLoader("kernelfoundry.eval_pipeline.utils", package_path=""),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("intel_profiler_kernel_feedback.j2")
        return template.render(kernels=analysis_results)

    @staticmethod
    def _format_segment_heading(index: int, label: str) -> str:
        return f"# Benchmark {index}: pytest test {label}"

    def _get_segment_runtime_ns(self, data: dict) -> float:
        df_compute_basic, _, _ = self._read_metrics(data)
        if df_compute_basic is None:
            return float("-inf")

        runtime_ns = 0.0
        for _, group in df_compute_basic.groupby("Kernel"):
            runtime_ns += float(get_median_row(group)["GpuTime[ns]"])
        return runtime_ns

    def _select_segments_for_feedback(self, segments: OrderedDict) -> list[tuple[str, dict]]:
        segment_items = list(segments.items())
        if len(segment_items) <= 3:
            return segment_items

        sorted_segments = sorted(segment_items, key=lambda item: self._get_segment_runtime_ns(item[1]))
        median_segment = sorted_segments[(len(sorted_segments) - 1) // 2]
        slowest_segment = sorted_segments[-1]

        selected_labels = {median_segment[0], slowest_segment[0]}
        return [item for item in segment_items if item[0] in selected_labels]

    def create_feedback(self, data: dict, worker_info: dict) -> str:
        try:
            segments = data.get("segments", {})
            if segments:
                sections = []
                selected_segments = self._select_segments_for_feedback(segments)
                for index, (label, segment_data) in enumerate(selected_segments, start=1):
                    segment_feedback = self._create_feedback_for_single_run(segment_data, worker_info)
                    if segment_feedback:
                        sections.append(f"{self._format_segment_heading(index, label)}\n\n{segment_feedback}")
                feedback_body = "\n\n".join(sections)
            else:
                feedback_body = self._create_feedback_for_single_run(data, worker_info)
            feedback = f"{PROFILER_ANALYSIS_INTRO}\n\n{feedback_body}" if feedback_body else ""
        except Exception as e:
            logging.exception(f"Error creating profiler feedback: {e}")
            feedback = ""
        return feedback

    def collate_data(self, outputs: dict[str, dict]) -> dict:
        result = {"segments": OrderedDict()}
        passes = {}
        aggregate_metrics = {"ComputeBasic": [], "MemoryProfile": [], "VectorEngineProfile": []}

        identify_metric_group_using_column_headers = {
            ("SLM_BANK_CONFLICT_COUNT[events]", "XVE_INST_EXECUTED_ALU0_ALL[events]"): "ComputeBasic",
            ("XVE_INST_EXECUTED_XMX_INT8[events]",): "VectorEngineProfile",
            ("SLM_ACCESS_COUNT[events]",): "MemoryProfile",
        }

        for k, v in outputs.items():
            if k.startswith("unitrace"):
                passes[k] = v

        for pass_name, pass_data in passes.items():
            metrics_df = next((v for v in self._read_metrics(pass_data) if v is not None), None)
            if metrics_df is None:
                continue
            metrics_df.columns = [col.strip() for col in metrics_df.columns]
            process_id = metrics_df.attrs.get("process_id", "unknown")
            timeline = pass_data.get("timeline", {})
            segment_entries = self._get_model_run_segments_from_timeline(timeline)

            metric_group = "unknown"
            for column_set, candidate_group in identify_metric_group_using_column_headers.items():
                if set(column_set).issubset(metrics_df.columns):
                    metric_group = candidate_group
                    break

            if metric_group == "unknown":
                logging.warning(f"Could not determine metric group for unitrace pass '{pass_name}'.")

            if metric_group == "ComputeBasic" or "timeline" not in result:
                result["timeline"] = timeline

            matched_any_segment = False
            for segment_entry in segment_entries:
                time_range = segment_entry["time_range"]
                global_ids = self._get_global_ids_from_timeline(timeline, time_range)
                filtered_df = metrics_df[metrics_df["GlobalInstanceId"].isin(global_ids)]
                if len(filtered_df) == 0:
                    continue

                matched_any_segment = True
                if metric_group != "unknown":
                    aggregate_metrics[metric_group].append(filtered_df)

                segment_label = segment_entry["label"]
                segment_data = result["segments"].setdefault(
                    segment_label,
                    {
                        "time_range": [x if x != float("inf") else None for x in time_range],
                        "trace_event_name": segment_entry["trace_event_name"],
                    },
                )
                segment_data[f"{metric_group}.metrics.{process_id}"] = filtered_df.to_csv(index=False)

            if matched_any_segment:
                continue

            logging.warning("No metrics found within any model execution time range, using full metrics dataframe.")
            if metric_group != "unknown":
                aggregate_metrics[metric_group].append(metrics_df)
            fallback_segment = result["segments"].setdefault(
                "default",
                {"time_range": [0, None], "trace_event_name": None},
            )
            fallback_segment[f"{metric_group}.metrics.{process_id}"] = metrics_df.to_csv(index=False)

        for metric_group, frames in aggregate_metrics.items():
            if frames:
                combined_df = pd.concat(frames, ignore_index=True)
                result[f"{metric_group}.metrics.segmented"] = combined_df.to_csv(index=False)

        if not result["segments"]:
            result.pop("segments")

        return result


class OCLUnitraceProfilerFeedback(UnitraceProfilerFeedback):
    """Profiler feedback for OpenCL using unitrace metric-sampling.

    Metric-sampling produces independent samples from separate program runs for each
    metric group (ComputeBasic, MemoryProfile, VectorEngineProfile). Unlike metric-query,
    the samples are not aligned by GlobalInstanceId or timestamp across groups, and
    GpuTime[ns] is just the fixed sampling interval duration. Segments cannot be
    reliably filtered from sampling data, so this class averages all samples per kernel
    and matches the three groups by kernel name only.
    """

    @property
    def name(self) -> str:
        return "unitrace"

    def _create_feedback_for_single_run(self, data: dict, worker_info: dict) -> str:
        df_compute_basic, df_memory_profile, df_vector_engine_profile = self._read_metrics(data)
        if df_compute_basic is None:
            return ""

        hardware_roofs = get_roofs(worker_info)
        if hardware_roofs is None:
            return ""

        cb_agg = df_compute_basic.groupby("Kernel").mean(numeric_only=True)
        mp_agg = df_memory_profile.groupby("Kernel").mean(numeric_only=True) if df_memory_profile is not None else None
        ve_agg = (
            df_vector_engine_profile.groupby("Kernel").mean(numeric_only=True)
            if df_vector_engine_profile is not None
            else None
        )

        analysis_results = OrderedDict()
        for kernel_name in cb_agg.index:
            compute_basic = cb_agg.loc[kernel_name].copy()
            compute_basic["Kernel"] = kernel_name
            memory_profile = (
                mp_agg.loc[kernel_name]
                if mp_agg is not None and kernel_name in mp_agg.index
                else pd.Series(dtype=float)
            )
            vector_engine_profile = (
                ve_agg.loc[kernel_name]
                if ve_agg is not None and kernel_name in ve_agg.index
                else pd.Series(dtype=float)
            )
            name, analysis = self.analyze_kernel(compute_basic, memory_profile, vector_engine_profile, hardware_roofs)
            analysis_results[name] = analysis

        env = Environment(
            loader=PackageLoader("kernelfoundry.eval_pipeline.utils", package_path=""),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("intel_profiler_kernel_feedback.j2")
        return template.render(kernels=analysis_results)

    def collate_data(self, outputs: dict[str, dict]) -> dict:
        result = {}
        aggregate_metrics = {"ComputeBasic": [], "MemoryProfile": [], "VectorEngineProfile": []}

        identify_metric_group_using_column_headers = {
            ("SLM_BANK_CONFLICT_COUNT[events]", "XVE_INST_EXECUTED_ALU0_ALL[events]"): "ComputeBasic",
            ("XVE_INST_EXECUTED_XMX_INT8[events]",): "VectorEngineProfile",
            ("SLM_ACCESS_COUNT[events]",): "MemoryProfile",
        }

        for k, v in outputs.items():
            if not k.startswith("unitrace"):
                continue

            metrics_df = next((df for df in self._read_metrics(v) if df is not None), None)
            if metrics_df is None:
                continue
            metrics_df.columns = [col.strip() for col in metrics_df.columns]
            process_id = metrics_df.attrs.get("process_id", "unknown")

            if "timeline" not in result:
                timeline = v.get("timeline", {})
                if timeline:
                    result["timeline"] = timeline

            metric_group = "unknown"
            for column_set, candidate_group in identify_metric_group_using_column_headers.items():
                if set(column_set).issubset(metrics_df.columns):
                    metric_group = candidate_group
                    break

            if metric_group == "unknown":
                logging.warning(f"Could not determine metric group for unitrace pass '{k}'.")
                continue

            aggregate_metrics[metric_group].append(metrics_df)
            result[f"{metric_group}.metrics.{process_id}"] = metrics_df.to_csv(index=False)

        for metric_group, frames in aggregate_metrics.items():
            if frames:
                combined_df = pd.concat(frames, ignore_index=True)
                result[f"{metric_group}.metrics.segmented"] = combined_df.to_csv(index=False)

        return result


class NCUProfilerFeedback(ProfilerFeedback):
    """Profiler feedback for CUDA kernels using NVIDIA Nsight Compute (NCU).

    Parses NCU CSV reports to extract throughput metrics, roofline analysis,
    and optimization hints for the slowest kernel in the profile.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "ncu"

    def _read_output_csv(self, trace_data: dict) -> pd.DataFrame | None:
        """Read and parse NCU output CSV data into a pandas DataFrame.

        Creates pandas DataFrames from NCU output trace files for further processing.

        Args:
            trace_data: Dictionary with the data of the NCU output files
        Returns:
            pd.DataFrame: A pandas DataFrame containing information for all devices.
        """
        data = trace_data.get("ncu_report.csv")
        if not data:
            return None

        # remove lines that start with '=='
        lines = [line for line in data.splitlines() if not line.startswith("==")]

        if not lines:
            return None

        # Parse CSV data
        csv_data = StringIO("\n".join(lines))
        df = pd.read_csv(csv_data)

        # Remove the thousand commas from the Metric Value column
        df["Metric Value"] = df["Metric Value"].astype(str).str.replace(",", "")
        # make sure the Metric Value column is numeric
        df["Metric Value"] = pd.to_numeric(df["Metric Value"], errors="coerce")

        # Convert numeric columns
        new_series = {}
        for series_name, series in df.items():
            try:
                new_series[series_name] = pd.to_numeric(series)
            except (ValueError, TypeError):
                new_series[series_name] = series

        return pd.DataFrame(new_series)

    def kernel_id_runtime(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        """Get a list of kernel IDs and their runtimes from the DataFrame."""
        result = []

        # Filter for GPU time metrics
        time_df = df[df["Metric Name"].str.contains("duration", case=False, na=False)]

        for kernel_id in time_df["ID"].unique():
            kernel_time_df = time_df[time_df["ID"] == kernel_id]

            if len(kernel_time_df) == 0:
                continue

            # Get the metric value and unit
            metric_value = kernel_time_df.iloc[0]["Metric Value"]
            metric_unit = kernel_time_df.iloc[0]["Metric Unit"]

            # Convert to nanoseconds based on unit
            runtime_ns = metric_value
            if metric_unit == "us":
                runtime_ns = metric_value * 1000
            elif metric_unit == "ms":
                runtime_ns = metric_value * 1000000
            elif metric_unit == "s":
                runtime_ns = metric_value * 1000000000

            result.append((int(kernel_id), float(runtime_ns)))

        return result

    def kernel_feedback(self, df: pd.DataFrame, kernel_id: int) -> str:
        """Generate feedback for a specific kernel based on its metrics."""
        section_name = "GPU Speed Of Light Throughput"
        thrughput_metrics = [
            "Compute (SM) Throughput",
            "L1/TEX Cache Throughput",
            "L2 Cache Throughput",
            "DRAM Throughput",
        ]

        section_rule_pair = [
            ("SpeedOfLight_RooflineChart", "SOLFPRoofline"),
            ("SpeedOfLight", "SOLBottleneck"),
        ]
        feedback = []

        # Filter dataframe for this kernel
        kernel_df = df[df["ID"] == kernel_id]

        # Get throughput metrics overview
        throughput_df = kernel_df[
            (kernel_df["Section Name"] == section_name) & (kernel_df["Metric Name"].isin(thrughput_metrics))
        ]

        if not throughput_df.empty:
            feedback.append("Throughput Metrics:")
            for _, row in throughput_df.iterrows():
                metric_name = row["Metric Name"]
                metric_value = row["Metric Value"]
                metric_unit = row["Metric Unit"]
                feedback.append(f"  - {metric_name}: {metric_value:.1f}{metric_unit}")

        # Get roofline and bottleneck information
        for section, rule_name in section_rule_pair:
            rule_df = kernel_df[(kernel_df["Section Name"] == section) & (kernel_df["Rule Name"] == rule_name)]

            if not rule_df.empty:
                rule_desc = rule_df.iloc[0]["Rule Description"]
                # Remove sentences that refer to external documentation of the form "See ... (url) for more ..."
                rule_desc = re.sub(r'See\s+(?:the\s+)?[^(]+\s+\([^)]+\)\s+for\s+more\s+[^."]+\.', "", rule_desc).strip()

                if rule_name == "SOLFPRoofline":
                    feedback.append(f"\nRoofline Analysis:\n{rule_desc}")
                elif rule_name == "SOLBottleneck":
                    feedback.append(f"\nOptimization Hint:\n{rule_desc}")

        return "\n".join(feedback)

    def __call__(self, trace_data) -> str:
        # process the trace data and extract relevant profiling information
        # for now, just return the raw trace data
        metrics_df = self._read_output_csv(trace_data)
        if metrics_df is None:
            return ""

        # Get kernel runtimes
        kernel_runtimes = self.kernel_id_runtime(metrics_df)

        if not kernel_runtimes:
            return ""

        # Find kernel with longest runtime
        longest_kernel_id, _ = max(kernel_runtimes, key=lambda x: x[1])

        # Get feedback for the longest running kernel
        feedback = self.kernel_feedback(metrics_df, longest_kernel_id)

        if not feedback:
            return ""

        return PROFILER_ANALYSIS_PROMPT.format(profiler_summary=feedback)

    def collate_data(self, outputs: dict[str, dict]) -> dict:
        # No multipass supported for NCU, just return the data of the first output
        for k, v in outputs.items():
            if k.startswith("ncu"):
                return v
        # Return empty dict if no NCU output is found
        return {}

    def create_feedback(self, data: dict, worker_info: dict) -> str:
        if not data:
            # Reached when profiling failed upstream
            logging.warning("No NCU profile data for this run, so no profiler feedback is included in the next prompt.")
            return "No NCU profile data available."
        try:
            feedback = self.__call__(data)
        except Exception as e:
            logging.exception(f"Error creating profiler feedback: {e}")
            feedback = ""
        return feedback


_VTUNE_OVERHEAD_KERNEL_PATTERNS = [
    re.compile(r"VectorizedElementwiseKernel"),
    re.compile(r"UnrolledElementwiseKernel"),
    re.compile(r"zeCommandListAppendMemoryCopy"),
    re.compile(r"ReduceKernelEmptyFunctor"),
    re.compile(r"\[Outside any task\]"),
]


class VTuneProfilerFeedback(ProfilerFeedback):
    """VTune profiler feedback generator for Intel GPUs.

    Works with the counters dict produced by VTune._extract_counters,
    serialised as JSON under the key 'vtune_counters.json'.
    """

    @property
    def name(self) -> str:
        return "vtune"

    @staticmethod
    def _is_overhead_kernel(name: str) -> bool:

        return any(pat.search(name) for pat in _VTUNE_OVERHEAD_KERNEL_PATTERNS)

    @staticmethod
    def _identify_primary_kernel(counters: dict) -> str | None:
        """Pick the user compute kernel with the highest total time.

        Falls back to the overall hottest kernel if no user kernels were
        captured — better to return something with a warning than fail silently.
        """
        best_user: tuple[float, str] | None = None
        best_any: tuple[float, str] | None = None
        for name, cols in counters.items():
            try:
                t = float(str(cols.get("Computing Task:Total Time", 0)).replace(",", ""))
            except (ValueError, TypeError):
                continue
            if best_any is None or t > best_any[0]:
                best_any = (t, name)
            if not VTuneProfilerFeedback._is_overhead_kernel(name):
                if best_user is None or t > best_user[0]:
                    best_user = (t, name)
        if best_user is not None:
            return best_user[1]
        if best_any is not None:
            logging.warning("Only overhead kernels captured by VTune; using %s", best_any[1])
            return best_any[1]
        return None

    @staticmethod
    def _build_metrics(cols: dict) -> dict:
        """Parse raw VTune counter strings into typed metric values."""

        def _f(key: str) -> float | None:
            v = cols.get(key)
            if v is None:
                return None
            try:
                return float(str(v).rstrip("%").replace(",", "").strip())
            except (ValueError, TypeError):
                return None

        def _pct(key: str) -> float | None:
            return _f(f"{key}(%)") if f"{key}(%)" in cols else _f(key)

        # Instruction counts for mix analysis
        alu0_count = _f("XVE Instructions:ALU0 Instructions")
        alu1_count = _f("XVE Instructions:ALU1 Instructions")
        xmx_count = _f("XVE Instructions:XMX instructions")
        total_instr = (alu0_count or 0) + (alu1_count or 0) + (xmx_count or 0)

        return {
            # Runtime info
            "simd_width": _f("Computing Task:SIMD Width"),
            "instance_count": _f("Computing Task:Instance Count"),
            "avg_runtime_ms": _f("Computing Task:Average Time"),
            "total_time_s": _f("Computing Task:Total Time"),
            # XVE utilization
            "xve_active_pct": _pct("XVE Array:Active"),
            "xve_stalled_pct": _pct("XVE Array:Stalled"),
            "xve_idle_pct": _pct("XVE Array:Idle"),
            # Occupancy
            "avg_thread_occupancy_pct": _pct("XVE Threads Occupancy"),
            "peak_occupancy_pct": _pct("Peak XVE Threads Occupancy"),
            # Stall reasons (% of total execution time; values can sum > stall% due to thread-level averaging)
            "stall_send_pct": _pct("XVE Stall Reasons:Send"),
            "stall_sbid_pct": _pct("XVE Stall Reasons:SBID"),
            "stall_dist_acc_pct": _pct("XVE Stall Reasons:Dist or Acc"),
            "stall_instr_fetch_pct": _pct("XVE Stall Reasons:Instruction Fetch"),
            "stall_barrier_pct": _pct("XVE Stall Reasons:Barrier"),
            # Pipeline activity (% of time each pipe is busy executing)
            "alu0_active_pct": _pct("XVE Pipelines:ALU0 active"),
            "alu1_active_pct": _pct("XVE Pipelines:ALU1 active"),
            "xmx_active_pct": _pct("XVE Pipelines:XMX active"),
            # Instruction mix (derived from raw counts)
            "alu0_instr_count": alu0_count,
            "alu1_instr_count": alu1_count,
            "xmx_instr_count": xmx_count,
            "alu0_instr_pct": 100 * (alu0_count or 0) / total_instr if total_instr > 0 else None,
            "alu1_instr_pct": 100 * (alu1_count or 0) / total_instr if total_instr > 0 else None,
            "xmx_instr_pct": 100 * (xmx_count or 0) / total_instr if total_instr > 0 else None,
            # Memory
            "l3_miss_pct": _pct("GPU L3:Miss Ratio"),
            "l3_bw_read_gbps": _f("GPU L3:Average Bandwidth, GB/s:Read"),
            "l3_bw_write_gbps": _f("GPU L3:Average Bandwidth, GB/s:Write"),
            "gpu_memory_bw_read_gbps": _f("GPU Memory Bandwidth, GB/sec:Read"),
            "gpu_memory_bw_write_gbps": _f("GPU Memory Bandwidth, GB/sec:Write"),
        }

    @staticmethod
    def _determine_bound(m: dict) -> str:
        """Classify the kernel as memory-bound, compute-bound, or balanced.

        Uses stall reasons and pipeline activity as proxies for the roofline
        position.  Returns one of: 'memory_latency', 'memory_bandwidth',
        'compute_xmx', 'compute_alu0', 'compute_alu1', 'balanced'.
        """
        stall_send = m.get("stall_send_pct") or 0
        stall_sbid = m.get("stall_sbid_pct") or 0
        stall_dist = m.get("stall_dist_acc_pct") or 0
        stall_total = m.get("xve_stalled_pct") or 0
        xmx_active = m.get("xmx_active_pct") or 0
        alu0_active = m.get("alu0_active_pct") or 0
        alu1_active = m.get("alu1_active_pct") or 0

        # Send stalls = waiting for memory load/store to complete → memory-latency bound
        if stall_send > 5 and stall_total > 0 and stall_send >= max(stall_sbid, stall_dist):
            return "memory_latency"

        # XMX pipeline saturated → matrix compute bound
        if xmx_active > 20:
            return "compute_xmx"

        # High ALU activity with relatively low stalls → compute bound
        if alu0_active > 50:
            return "compute_alu0"
        if alu1_active > 50:
            return "compute_alu1"

        return "balanced"

    @staticmethod
    def _generate_recommendations(m: dict, bound: str) -> list[str]:
        """Generate optimization hints based on metrics and bound classification."""
        recs = []

        stall_send = m.get("stall_send_pct") or 0
        stall_sbid = m.get("stall_sbid_pct") or 0
        stall_dist = m.get("stall_dist_acc_pct") or 0
        stall_barrier = m.get("stall_barrier_pct") or 0
        avg_occ = m.get("avg_thread_occupancy_pct")
        peak_occ = m.get("peak_occupancy_pct")
        l3_miss = m.get("l3_miss_pct")
        xmx_active = m.get("xmx_active_pct") or 0
        alu0_active = m.get("alu0_active_pct") or 0

        if bound == "memory_latency":
            recs.append(
                f"[memory_latency] Send stalls ({stall_send:.1f}%) dominate — kernel is stalled waiting for "
                "memory loads/stores. Use tensor descriptors, prefetching, or bf16 inputs to reduce latency."
            )
        elif bound == "compute_xmx":
            recs.append(
                f"[compute_xmx] XMX pipeline is the bottleneck ({xmx_active:.1f}% active). "
                "The kernel is XMX/matrix compute bound — this is a good sign. Increase tile sizes "
                "or use DPAS chaining to sustain throughput."
            )
        elif bound in ("compute_alu0", "compute_alu1"):
            alu_name = "ALU0 (FP16/FP32)" if bound == "compute_alu0" else "ALU1 (FP64/INT/Math)"
            alu_pct = alu0_active if bound == "compute_alu0" else m.get("alu1_active_pct") or 0
            recs.append(
                f"[compute_alu] {alu_name} pipeline is active {alu_pct:.1f}% of the time — kernel is "
                "ALU compute bound. Consider using XMX/DPAS instructions for matrix operations "
                "or vectorizing scalar loops to improve throughput."
            )

        if avg_occ is not None and avg_occ < 50:
            recs.append(
                f"[low_occupancy] Average thread occupancy {avg_occ:.0f}% is low. "
                "Try larger work-groups, reduce register pressure, or use a persistent kernel pattern."
            )
        elif peak_occ is not None and peak_occ < 50:
            recs.append(
                f"[low_peak_occupancy] Peak occupancy {peak_occ:.0f}% — try larger tiles, "
                "fewer registers, or a persistent kernel."
            )

        if stall_sbid > 5 or stall_dist > 5:
            recs.append(
                f"[dependency_stalls] SBID ({stall_sbid:.1f}%) and Dist/Acc ({stall_dist:.1f}%) stalls indicate "
                "in-flight instruction dependency bottlenecks. Reorganize instructions to hide latency "
                "or reduce register dependency chains."
            )

        if stall_barrier > 5:
            recs.append(
                f"[barrier_stalls] Barrier stalls at {stall_barrier:.1f}% — work-group synchronization "
                "is a bottleneck. Reduce barrier frequency or restructure the algorithm."
            )

        if l3_miss is not None and l3_miss > 50:
            recs.append(
                f"[l3_thrashing] L3 miss ratio {l3_miss:.0f}% — significant DRAM traffic. "
                "Reduce tile sizes or improve data reuse to fit working set into L3 cache."
            )

        if xmx_active < 5 and alu0_active > 40:
            recs.append(
                "[no_xmx] XMX pipeline is unused. If the kernel performs matrix multiplications, "
                "consider using DPAS/XMX instructions (e.g., joint_matrix in SYCL) for up to "
                "~8x throughput gain on matrix ops."
            )

        return recs

    @staticmethod
    def _fmt_count(n: float | None) -> str:
        """Format a raw instruction count as a human-readable abbreviated string."""
        if n is None:
            return "N/A"
        if n >= 1e12:
            return f"{n/1e12:.1f}T"
        if n >= 1e9:
            return f"{n/1e9:.1f}B"
        if n >= 1e6:
            return f"{n/1e6:.1f}M"
        return f"{n:.0f}"

    @staticmethod
    def _format_feedback(kernel_name: str, m: dict, recs: list[str], bound: str) -> str:
        """Format metrics and recommendations as a structured human-readable string."""
        _BOUND_DESC = {
            "memory_latency": "memory-latency bound (high Send stalls)",
            "memory_bandwidth": "memory-bandwidth bound",
            "compute_xmx": "XMX/matrix compute bound",
            "compute_alu0": "ALU0-compute bound (FP16/FP32)",
            "compute_alu1": "ALU1-compute bound (FP64/INT/Math)",
            "balanced": "balanced (not strongly bound by a single resource)",
        }
        parts = [f"## VTune Analysis: {kernel_name}"]

        # --- Runtime and Occupancy ---
        parts.append("\n### Runtime and Occupancy:")
        runtime_parts = []
        if m.get("simd_width") is not None:
            runtime_parts.append(f"SIMD width: {int(m['simd_width'])}")
        if m.get("instance_count") is not None:
            runtime_parts.append(f"{int(m['instance_count'])} instances")
        if m.get("avg_runtime_ms") is not None:
            runtime_parts.append(f"average runtime: {m['avg_runtime_ms']:.3f} ms")
        if m.get("total_time_s") is not None:
            runtime_parts.append(f"total time: {m['total_time_s']:.3f} s")
        if runtime_parts:
            parts.append(", ".join(runtime_parts) + ".")
        if m.get("avg_thread_occupancy_pct") is not None:
            occ_line = f"Average thread occupancy: {m['avg_thread_occupancy_pct']:.1f}%"
            if m.get("peak_occupancy_pct") is not None:
                occ_line += f", peak: {m['peak_occupancy_pct']:.1f}%"
            parts.append(occ_line + ".")

        # --- XVE Utilization ---
        parts.append("\n### XVE Utilization:")
        util_parts = []
        if m.get("xve_active_pct") is not None:
            util_parts.append(f"Active: {m['xve_active_pct']:.1f}%")
        if m.get("xve_stalled_pct") is not None:
            util_parts.append(f"Stalled: {m['xve_stalled_pct']:.1f}%")
        if m.get("xve_idle_pct") is not None:
            util_parts.append(f"Idle: {m['xve_idle_pct']:.1f}%")
        if util_parts:
            parts.append(" | ".join(util_parts))

        # --- Stall Breakdown ---
        stall_items = [
            ("Send (memory)", "stall_send_pct"),
            ("SBID", "stall_sbid_pct"),
            ("Dist/Acc", "stall_dist_acc_pct"),
            ("Instruction Fetch", "stall_instr_fetch_pct"),
            ("Barrier", "stall_barrier_pct"),
        ]
        stall_data = [(label, m[key]) for label, key in stall_items if m.get(key) is not None]
        if stall_data:
            # Sort by descending value for readability
            stall_data.sort(key=lambda x: x[1], reverse=True)
            parts.append("\n### Stall Reasons:")
            parts.append(", ".join(f"{label}: {val:.1f}%" for label, val in stall_data))

        # --- Pipeline Utilization ---
        pipeline_rows = [
            ("ALU0 (FP16/FP32)", "alu0_active_pct", "alu0_instr_count", "alu0_instr_pct"),
            ("ALU1 (FP64/INT/Math)", "alu1_active_pct", "alu1_instr_count", "alu1_instr_pct"),
            ("XMX (Matrix/DPAS)", "xmx_active_pct", "xmx_instr_count", "xmx_instr_pct"),
        ]
        pipeline_lines = []
        for label, active_key, count_key, pct_key in pipeline_rows:
            active = m.get(active_key)
            count = m.get(count_key)
            pct = m.get(pct_key)
            if active is not None:
                line = f"  {label}: {active:.1f}% active"
                if count is not None and pct is not None:
                    line += f" — {VTuneProfilerFeedback._fmt_count(count)} instructions ({pct:.1f}%)"
                pipeline_lines.append(line)
        if pipeline_lines:
            parts.append("\n### Pipeline Utilization:")
            parts.extend(pipeline_lines)

        # --- Roofline Analysis ---
        parts.append("\n### Roofline Analysis:")
        parts.append(f"The kernel is {_BOUND_DESC.get(bound, bound)}.")
        # Dominant instruction type
        if m.get("alu0_instr_pct") is not None:
            instr_summary = []
            if m["alu0_instr_pct"] > 0:
                instr_summary.append(f"ALU0 {m['alu0_instr_pct']:.1f}%")
            if m.get("alu1_instr_pct", 0) > 0:
                instr_summary.append(f"ALU1 {m['alu1_instr_pct']:.1f}%")
            if m.get("xmx_instr_pct", 0) > 0:
                instr_summary.append(f"XMX {m['xmx_instr_pct']:.1f}%")
            if instr_summary:
                parts.append("Instruction mix: " + ", ".join(instr_summary) + ".")

        # --- Memory ---
        parts.append("\n### Memory:")
        if m.get("gpu_memory_bw_read_gbps") is not None or m.get("gpu_memory_bw_write_gbps") is not None:
            bw_parts = []
            if m.get("gpu_memory_bw_read_gbps") is not None:
                bw_parts.append(f"Read: {m['gpu_memory_bw_read_gbps']:.1f} GB/s")
            if m.get("gpu_memory_bw_write_gbps") is not None:
                bw_parts.append(f"Write: {m['gpu_memory_bw_write_gbps']:.1f} GB/s")
            parts.append("GPU (DRAM) Memory Bandwidth (VTune-sampled, approximate) — " + ", ".join(bw_parts) + ".")
        if m.get("l3_bw_read_gbps") is not None or m.get("l3_bw_write_gbps") is not None:
            l3_bw = []
            if m.get("l3_bw_read_gbps") is not None:
                l3_bw.append(f"Read: {m['l3_bw_read_gbps']:.1f} GB/s")
            if m.get("l3_bw_write_gbps") is not None:
                l3_bw.append(f"Write: {m['l3_bw_write_gbps']:.1f} GB/s")
            parts.append("L3 Cache Bandwidth — " + ", ".join(l3_bw) + ".")
        if m.get("l3_miss_pct") is not None:
            parts.append(f"L3 miss ratio: {m['l3_miss_pct']:.1f}% (fraction of L3 requests served from DRAM).")

        # --- Recommendations ---
        if recs:
            parts.append("\n### Recommendations:")
            for rec in recs:
                parts.append(f"  {rec}")

        return "\n".join(parts)

    def collate_data(self, outputs: dict[str, dict]) -> dict:
        """Collate VTune output: deserialise the first vtune_counters.json found."""
        for k, v in outputs.items():
            if k.startswith("vtune"):
                counters_json = v.get("vtune_counters.json")
                if counters_json:
                    return {"counters": json.loads(counters_json)}
        return {}

    def create_feedback(self, data: dict, worker_info: dict) -> str:
        """Create feedback string from VTune profiling data."""
        counters = data.get("counters")
        if not counters:
            logging.warning("No VTune counter data found")
            return ""

        try:
            sections = []
            for name, cols in counters.items():
                if self._is_overhead_kernel(name):
                    continue
                m = self._build_metrics(cols)
                bound = self._determine_bound(m)
                recs = self._generate_recommendations(m, bound)
                sections.append(self._format_feedback(name, m, recs, bound))

            if not sections:
                # Fall back to showing the primary kernel even if it's overhead
                primary = self._identify_primary_kernel(counters)
                if primary:
                    m = self._build_metrics(counters[primary])
                    bound = self._determine_bound(m)
                    recs = self._generate_recommendations(m, bound)
                    sections.append(self._format_feedback(primary, m, recs, bound))

            feedback_body = "\n\n".join(sections)
            return f"{PROFILER_ANALYSIS_INTRO}\n\n{feedback_body}" if feedback_body else ""
        except Exception as e:
            logging.exception("Error creating VTune profiler feedback: %s", e)
            return ""


def get_reference_language_for_profiling(config) -> str:
    """Get the reference language for profiler feedback based on the current language."""
    ref_language = config.get("prompt", {}).get("reference_language", config.get("language"))
    if ref_language == "Pytorch":
        return config.get("language")  # use same language as custom kernel for profiling
    return ref_language


# Mapping of (language, profiler_type) -> profiler feedback class
_PROFILER_FEEDBACK_MAPPING = {
    ("sycl", "unitrace"): UnitraceProfilerFeedback,
    ("sycl", "vtune"): VTuneProfilerFeedback,
    ("ocl", "unitrace"): OCLUnitraceProfilerFeedback,
    ("ocl", "vtune"): VTuneProfilerFeedback,
    ("cuda", "ncu"): NCUProfilerFeedback,
    ("pytorch", "ncu"): NCUProfilerFeedback,
    ("pytorch", "unitrace"): UnitraceProfilerFeedback,
    ("triton", "ncu"): NCUProfilerFeedback,
    ("triton", "unitrace"): UnitraceProfilerFeedback,
}

# Default profiler feedback for each language
_DEFAULT_PROFILER_FEEDBACK = {
    "sycl": "unitrace",
    "ocl": "vtune",
    "cuda": "ncu",
}


def get_profiler_feedback_class(
    language: str, gpu_arch: str, profiler_type: str | None = None
) -> type[ProfilerFeedback]:
    """Get the appropriate profiler feedback class based on user-specified profiler type or language & GPU architecture."""
    language = language.lower()

    if profiler_type is None:
        if language in ["pytorch", "triton"]:
            gpu_arch_is_cuda = gpu_arch in VALID_CUDA_ARCHS
            return NCUProfilerFeedback if gpu_arch_is_cuda else UnitraceProfilerFeedback
        profiler_type = _DEFAULT_PROFILER_FEEDBACK.get(language)

    key = (language, profiler_type.lower() if profiler_type else None)
    if key in _PROFILER_FEEDBACK_MAPPING:
        return _PROFILER_FEEDBACK_MAPPING[key]

    raise ValueError(
        f"Unknown profiler '{profiler_type}' for language '{language}'. "
        f"Available profilers for '{language}': {[p for (l, p) in _PROFILER_FEEDBACK_MAPPING.keys() if l == language]}"
    )


if __name__ == "__main__":
    with open("profiling/unitrace/2025-08-18-trace_data.json", "r") as f:
        trace_data = json.load(f)
    if "device_id" not in trace_data:
        trace_data["device_id"] = "0xe20b"
    pf = UnitraceProfilerFeedback({})
    print(pf(trace_data))

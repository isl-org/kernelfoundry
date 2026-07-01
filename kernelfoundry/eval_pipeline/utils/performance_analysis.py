"""Helper functions for analyzing performance data from profilers like Unitrace"""

import pandas as pd
from kernelfoundry.eval_pipeline.utils.hardware_info import hardware_info, HardwareRoofs

__all__ = [
    "get_roofs",
    "compute_arithmic_intensity",
    "get_roofline_points",
    "get_median_row",
    "get_row_with_closest_metric",
]


def get_roofs(worker_info: dict) -> HardwareRoofs | None:
    """Uses the cpu and gpu names of the eval worker to lookup the hardware roofs
    Args:
        worker_info: The worker info dictionary. With information about the cpu and gpu of the worker.
    Returns:
        HardwareRoofs | None: The hardware roofs or None if not found.
    """

    try:
        cpu_name = worker_info["cpu_info"]
        hw_info = hardware_info.get(cpu_name, None)
        if hw_info is None:
            hw_info = hardware_info.get(worker_info["gpu_name"], None)
        return hw_info.roofs
    except:
        return None


def compute_arithmic_intensity(
    series: pd.Series, compute_metric="XVE_INST_EXECUTED_FP32[events]", mem_metrics: tuple[str, str] = None
):
    """Calculates the arithmetic intensity for a given series (row) of metrics.
    Args:
        series (pd.Series): A row from the metrics DataFrame.
        compute_metric (str): The compute metric to use.
    Returns:
        float: The arithmetic intensity (compute / memory).
    """
    result = None
    if mem_metrics is None:
        mem_metrics = ("GPU_MEMORY_BYTE_READ[bytes]", "GPU_MEMORY_BYTE_WRITE[bytes]")
    try:
        mem_moves = sum(float(series[metric]) for metric in mem_metrics)
        compute_instructions = float(series[compute_metric])
        result = compute_instructions / mem_moves
    except:
        pass
    return result


def get_roofline_points(
    max_mem_bw: float, max_compute: float, min_value: float = 0.001
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calculates the roofline points given max memory bandwidth and max compute.
    Args:
        max_mem_bw (float): Maximum memory bandwidth.
        max_compute (float): Maximum compute.
        min_value (float): Minimum value for the x-axis. Default is 0.001 to avoid log(0) issues.
    Returns:
        tuple: Two tuples representing the start and end points of the diagonal part of the roofline.
    """
    x1, y1 = min_value, max_mem_bw * min_value
    x2, y2 = max_compute / max_mem_bw, max_compute
    return (x1, y1), (x2, y2)


def get_median_row(df: pd.DataFrame, column: str = "GpuTime[ns]") -> pd.Series:
    """Returns the median row based on the specified column."""
    return df.sort_values(by=[column]).iloc[len(df) // 2]


def get_row_with_closest_metric(df: pd.DataFrame, target_value: float, column: str = "GpuTime[ns]") -> pd.Series:
    """Returns the row with the closest value to the target_value in the specified column."""
    if len(df) == 0:
        return pd.Series()  # Return an empty Series if the DataFrame is empty
    return df.iloc[(df[column] - target_value).abs().argsort()[:1]].iloc[0]

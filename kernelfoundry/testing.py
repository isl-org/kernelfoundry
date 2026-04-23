"""Testing utilities for kernel output validation."""

from typing import Callable
import numpy as np


def _name_is_tensor(obj):
    """Check if the object is a tensor.

    This function just checks the type name.
    """
    return "tensor" in str(type(obj)).lower()


def _convert_to_numpy(obj):
    """Convert an object to a NumPy array if applicable.
    Args:
        obj: The object to convert.
    Returns:
        A NumPy array if the object can be converted or is already a NumPy array, otherwise
        returns None.
    """
    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return obj.detach().cpu().numpy()
    elif hasattr(obj, "numpy"):
        return obj.numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return None


def all_close_with_slack(
    output_reference: "torch.Tensor",
    output_kernel: "torch.Tensor",
    epsilon: float = 1e-7,
    max_rel_err: float = 0.01,
    ratio_below_max_err: float = 0.99,
) -> bool:
    """Check the accuracy of the kernel output compared to the reference output.

    This function computes the absolute relative error between the new and original outputs,
    and determines if the proportion of elements within a specified maximum relative error
    is above a given ratio.

    Args:
        output_reference (torch.Tensor): The reference output tensor.
        output_kernel (torch.Tensor): The kernel output tensor to compare.
        epsilon (float, optional): A small constant to avoid division by zero. Default is 1e-7.
        max_rel_err (float, optional): The maximum relative error allowed. Default is 0.01.
        ratio_below_max_err (float, optional): The minimum required ratio of elements
            with error below the maximum relative error. Default is 0.99.

    Returns:
        bool: True if the ratio of elements with a relative error below `max_rel_err`
            is greater than `ratio_below_max_err`, False otherwise.
    """
    import torch

    abs_rel_error = torch.abs((output_kernel - output_reference) / (torch.abs(output_reference) + epsilon))
    is_correct = (torch.sum(abs_rel_error < max_rel_err) / torch.numel(abs_rel_error)) > ratio_below_max_err
    return is_correct


def cosine_similarity(output_reference: "torch.Tensor", output_kernel: "torch.Tensor", min_sim: float = 0.99985):
    """Compute cosine similarity of flattened output tensors.

    Returns:
        bool: True if similarity meets the threshold, False otherwise.
    """
    import torch

    sim_computer = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
    sim = sim_computer(output_reference.flatten(), output_kernel.flatten())
    return sim >= min_sim


def assert_allclose(
    actual,
    expected,
    *,
    epsilon: float = 1e-7,
    rtol: float = 0.01,
    ratio_below_max_err: float = 0.99,
    msg: str | Callable[[str], str] | None = None,
    err_stats: bool = True,
) -> None:
    """Asserts that two arrays are close within a given relative tolerance.

    This function computes the absolute relative error between the new and original outputs,
    and determines if the proportion of elements within a specified maximum relative error
    is above a given ratio.

    This function behaves like all_close_with_slack, but raises an AssertionError with a detailed message

    Args:
        actual (np.ndarray|torch.Tensor): The output tensor to validate.
        expected (np.ndarray|torch.Tensor): The reference output tensor.
        epsilon (float, optional): A small constant to avoid division by zero. Default is 1e-7.
        rtol (float, optional): The maximum relative error allowed. Default is 0.01.
        ratio_below_max_err (float, optional): The minimum required ratio of elements
            with error below the maximum relative error. Default is 0.99.
        msg (str | Callable[[str], str] | None, optional): Optional custom error message.
        err_stats (bool, optional): Whether to include error statistics. Default is True.

    Raises:
        AssertionError: If the arrays are not close enough.
    """
    __tracebackhide__ = True  # Hide traceback for cleaner test output

    actual = _convert_to_numpy(actual)
    expected = _convert_to_numpy(expected)
    if _name_is_tensor(actual) or _name_is_tensor(expected):
        arrays_name_str = "Tensors"
    else:
        arrays_name_str = "Arrays"

    abs_rel_error = np.abs((actual - expected) / (np.abs(expected) + epsilon))
    num_correct = np.sum(abs_rel_error < rtol)
    total_elements = abs_rel_error.size
    is_correct = (num_correct / total_elements) > ratio_below_max_err

    if not is_correct:
        standard_msg = f"{arrays_name_str} are not equal within tolerance.\n"
        if err_stats:
            stat_str = (
                f"Mismatched elements: {total_elements - num_correct} / {total_elements} "
                f"({((total_elements - num_correct) / total_elements) * 100:.1f}%)\n"
                f"Greatest absolute difference: {np.max(np.abs(actual - expected)):.15e}\n"
                f"Greatest relative difference: {np.max(abs_rel_error):.15e}"
            )
            standard_msg += stat_str

        if msg is None:
            full_msg = standard_msg
        elif callable(msg):
            full_msg = msg(standard_msg)
        else:
            full_msg = msg + "\n" + standard_msg
        raise AssertionError(full_msg)

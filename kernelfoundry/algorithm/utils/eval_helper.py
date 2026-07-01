from typing import Iterable
import re
import numpy as np
from kernelfoundry.eval_pipeline.task import ProcessResult
import logging


def truncate_message(msg: str, max_length: int = 5000) -> str:
    truncation_notice = "...[truncated]"
    if len(msg) > max_length:
        return msg[: max_length - len(truncation_notice)] + truncation_notice
    return msg


def _shorten_pytest_error_tensor_output(text, error_type="TypeError"):
    """
    Shorten the tensor output for specific error types in pytest output

    Args:
        text: pytest output
        error_type: The error type

    Returns:
        List of strings containing the shortened output
    """
    # Add empty line at end to avoid special case handling
    lines = text.splitlines()
    lines = lines + [""]

    result = []
    error_lines = []
    in_error_section = False

    for line in lines:
        # Check if we've found the error type
        if f"{error_type}: " in line:
            in_error_section = True
            error_lines.append(line)
        # If we're in the error section, collect lines starting with "E"
        elif in_error_section:
            if line.strip().startswith("E"):
                error_lines.append(line)
            # Stop when we hit a non-error line (e.g., blank line or new section)
            elif line.strip() and not line.strip().startswith("E"):
                in_error_section = False
        elif len(error_lines):
            # Shorten tensor outputs in the collected error lines
            # Join all error lines to handle multi-line tensors
            error_text = "\n".join(error_lines)

            # Replace multi-line tensor contents with ...
            # This handles tensor( ... ) that may span multiple lines
            shortened_text = re.sub(r"tensor\([^)]*\)", "tensor(...)", error_text, flags=re.DOTALL)

            error_lines = []
            result.append(shortened_text)
            result.append(line)
        else:
            result.append(line)

    return "\n".join(result[:-1])  # remove added empty line


def _remove_skipped_dependency_tests(output: str) -> str:
    """Remove skipped tests due to missing dependencies from pytest output.

    Args:
        output: pytest output string

    Returns:
        Output with skipped dependency tests removed
    """
    # remove the skipped tests due to missing dependencies
    # They look like this
    # task.py::TestKernelBench::test_all_close[4]
    # -------------------------------- live log setup --------------------------------
    # INFO     pytest_dependency:pytest_dependency.py:100 skip test_all_close[4] because it depends on TestKernelBench::test_output_shapes_match
    # SKIPPED (test_all_close[4] depends on TestKernelBench::test_output_s...) [ 54%]
    pattern = r"""(?x)  # verbose mode
^task\.py::\w+::\w+\[\d+\]\s*$\n
^-+\s+live\s+log\s+setup\s+-+$\n
^INFO\s+pytest_dependency:pytest_dependency\.py:\d+\s+skip\s+\w+\[\d+\]\s+because\s+it\s+depends\s+on\s+\w+::\w+$\n
^SKIPPED\s+\(\w+\[\d+\]\s+depends\s+on\s+\w+::\w+.*?\)\s+\[\s*\d+%\]$\n
    """
    return re.sub(pattern, "", output, flags=re.MULTILINE)


def postprocess_pytest_output(result: ProcessResult) -> str:
    """Postprocesses pytest output to remove unnecessary lines before the first test case.

    Args:
        result (ProcessResult): Process result with pytest stdout/stderr.
    """

    output = result.stdout + "\n" + result.stderr
    output = output.split("\n")

    # search for the first line that starts with "task.py::"
    for i, line in enumerate(output):
        if line.startswith("task.py::"):
            output = output[i:]
            break
    # remove empty lines at the end
    while output and output[-1].strip() == "":
        output.pop()
    output = ["==== test session starts", ""] + output
    output = "\n".join(output)

    output = _remove_skipped_dependency_tests(output)

    output = _shorten_pytest_error_tensor_output(output, error_type="TypeError")

    return output


# used for parsing clang output
class _parse_clang_output:
    # the values with the prefix "src_" indicate messages related to source files of the task
    msg_types: dict[str, int] = {
        "unknown": 0,
        "warning": 1,
        "error": 2,
        "fatal error": 3,
        "note": 4,
        "src_warning": 5,
        "src_error": 6,
        "src_fatal error": 7,
        "src_note": 8,
    }
    msg_re = re.compile(r"^(.*?):(\d+):(\d+):\s+(warning|error|fatal error|note):\s+(.+?)$")
    codeloc_re = re.compile(r"^\s+(\d+)?\s\|\s+(.*)$")
    in_file_included_re = re.compile(r"\s*(In file included)?\s+from (.*)(:\d+)+,\s*")
    required_from_re = re.compile(r"\s*(.*)(:\d+)+:\s*required from\s+.*")


def _classify_clang_lines(lines: list[str], src_files: Iterable[str] | None = None) -> np.ndarray:
    """Classify each line of clang output into message types."""
    msg_types = _parse_clang_output.msg_types
    msg_re = _parse_clang_output.msg_re
    codeloc_re = _parse_clang_output.codeloc_re
    unknown = msg_types["unknown"]
    linetypes = np.full(len(lines), unknown, dtype=np.int32)
    if src_files is not None:
        logging.debug(f"_classify_clang_lines: src files:\n {'\n    '.join(src_files)}")

    def is_src_file(fname: str) -> bool:
        if src_files is None:
            return True
        for src_file in src_files:
            if src_file in fname:
                return True
        return False

    i = 0
    while i < len(lines):
        line = lines[i]
        m = msg_re.match(line)
        if m:
            anchor_i = i
            src_file, src_line, src_col, msg_type, msg_content = m.groups()
            if is_src_file(src_file):
                value = msg_types.get("src_" + msg_type) or unknown
            else:
                value = msg_types.get(msg_type) or unknown
            linetypes[i] = value
            for j in range(1, 10):  # usually there are only 2 or 3 lines of code location info
                i += 1
                line = lines[i]
                if (
                    codeloc_re.match(line) or "MSVC" in line
                ):  # Sometimes there is a hint about MSVC between the message and the code location
                    linetypes[i] = value
                else:
                    break
            # mark the lines before line anchor_i with the same msg type if they belong to a stack of paths
            for k in range(anchor_i - 1, -1, -1):
                line = lines[k]
                if linetypes[k] == unknown and line.strip().startswith("/"):
                    linetypes[k] = value
                else:
                    break
        else:
            i += 1
    return linetypes


def _filter_clang_warnings(lines: list[str], src_file_paths: Iterable[str] | None = None) -> list[str]:
    """Filter out warning messages from clang compiler output that do not originate from the task source files

    Args:
        text: String containing compiler output

    Returns:
        Filtered output without warning messages
    """
    linetypes = _classify_clang_lines(lines, src_file_paths)

    unknown = _parse_clang_output.msg_types["unknown"]
    src_warns = _parse_clang_output.msg_types["src_warning"]
    src_error = _parse_clang_output.msg_types["src_error"]
    src_fatal = _parse_clang_output.msg_types["src_fatal error"]
    error = _parse_clang_output.msg_types["error"]
    fatal = _parse_clang_output.msg_types["fatal error"]
    keeptypes = (unknown, src_warns, src_error, src_fatal, error, fatal)
    keep = np.isin(linetypes, keeptypes)
    return list(np.asarray(lines)[keep])


def _shorten_stack(lines: list[str], regular_expression) -> list[str]:
    """Shorten the include stack in clang output by keeping only the first and last lines of the stack.

    Args:
        lines: List of lines from clang output
    """
    result = []
    i = 0
    # keep track of included lines to avoid long list of duplicated included lines
    included_lines_set = set()
    while i < len(lines):
        line = lines[i]
        m = regular_expression.match(line)
        if m:
            include_stack = [line]
            for j in range(i + 1, len(lines)):
                line = lines[j]
                m = regular_expression.match(line)
                if not m:
                    break
                include_stack.append(line)
            if tuple(include_stack) not in included_lines_set:
                if len(include_stack) > 2:
                    result.append(include_stack[0])
                    indent = len(include_stack[1]) - len(include_stack[1].lstrip())
                    result.append(" " * indent + "...")
                else:
                    result.extend(include_stack)
                included_lines_set.add(tuple(include_stack))
            i = j
        else:
            result.append(line)
            i += 1
    return result


def _skip_until_compiler(lines: list[str]) -> list[str]:
    """Skip lines until the first compiler call is found."""
    for i in range(len(lines)):
        line = lines[i]
        if re.search(r"\b(clang|clang\+\+|gcc|g\+\+|icx|icpx|dpcpp|nvcc)\b", line):
            return lines[i:]
    return lines


def _skip_after_ninja_build_stopped(lines: list[str]) -> list[str]:
    """Skip lines after 'ninja: build stopped' message."""
    for i in range(len(lines)):
        line = lines[i]
        if "ninja: build stopped" in line:
            return lines[: i + 1]
    return lines


def _shorten_compiler_command(lines: list[str]) -> list[str]:
    """Shorten long compiler command lines by truncating arguments after a certain length."""
    result = []
    compiler_re = re.compile(r"\b(clang|clang\+\+|gcc|g\+\+|icx|icpx|dpcpp|nvcc)\s")
    for line in lines:
        if compiler_re.search(line):
            # Keep compiler name, input files (.cu, .cpp, .sycl, etc.), and -o output
            parts = line.split()
            shortened = []
            i = 0
            while i < len(parts):
                part = parts[i]
                # Keep compiler command
                if compiler_re.search(part + " "):
                    shortened.extend(parts[0 : i + 1])
                    shortened.append(" [flags truncated] ")
                # Keep -o and its argument
                elif part == "-o" and i + 1 < len(parts):
                    shortened.append(part)
                    shortened.append(parts[i + 1])
                    i += 1
                # Keep source files (.cu, .cpp, .sycl, etc.)
                elif any(part.endswith(ext) for ext in [".cu", ".cpp", ".sycl", ".c", ".cc", ".cxx"]):
                    shortened.append(part)
                i += 1
            result.append(" ".join(shortened))
        else:
            result.append(line)
    return result


def _shorten_paths(lines: list[str]) -> list[str]:
    """Shorten long file paths in the lines by keeping only the last two components."""
    result = []
    path_re = re.compile(r"(?:[a-zA-Z]:)?(?:[/\\][\w\-. ]+)+[/\\]?")
    oneapi_include_re = re.compile(r"/opt/intel/oneapi/compiler/\d+\.\d+/bin/compiler/../../include/")
    site_packages_re = re.compile(r"(?:[a-zA-Z]:)?(?:[/\\][\w\-. ]+)*/site-packages/")
    for line in lines:
        paths = path_re.findall(line)
        for path in paths:
            # Replace everything up to and including '/task_data/' with '<project_dir>'
            idx = path.find("/task_data/")
            if idx != -1:
                shortened_path = "<project_dir>/" + path[idx + len("/task_data/") :]
                line = line.replace(path, shortened_path)
                continue
            # Shorten oneAPI include paths
            m = oneapi_include_re.search(path)
            if m:
                shortened_path = "<oneapi_include_path>/" + path[m.end() :]
                line = line.replace(path, shortened_path)
                continue
            # Shorten site-packages paths
            m = site_packages_re.search(path)
            if m:
                shortened_path = "<python_env_path>/" + path[m.end() :]
                line = line.replace(path, shortened_path)
                continue

        result.append(line)
    return result


def _remove_duplicates(lines: list[str]) -> list[str]:
    """Remove duplicate lines if they are consecutive."""
    new_lines = []
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1]:
            continue
        new_lines.append(lines[i])
    return new_lines


def _truncate_but_keep_errors(
    lines: list[str],
    src_file_paths: Iterable[str] | None = None,
    max_length_initial: int = 5000,
    max_length_errors: int = 2000,
) -> list[str]:
    """
    Truncate lines while keeping error messages.

    Args:
        lines: List of lines to truncate
        src_file_paths: Iterable of source file paths to identify messages related to the task source files
        max_length_initial: Maximum length of the initial truncated message
        max_length_errors: Maximum length of the error messages to keep
    """
    # make sure that errors are included for sure - classify lines to get error lines
    linetypes = _classify_clang_lines(lines, src_file_paths)
    error, src_error = _parse_clang_output.msg_types["error"], _parse_clang_output.msg_types["src_error"]
    keep_errors = np.isin(linetypes, (error, src_error))
    error_lines = list(np.asarray(lines)[keep_errors])

    # truncate message
    truncated = truncate_message("\n".join(lines), max_length=max_length_initial)

    # re-add error part with up to 2000 characters
    error_part = ""
    for error_line in error_lines:
        if error_line not in truncated:
            error_part += "\n" + error_line
    truncated_error_part = truncate_message(error_part, max_length=max_length_errors)
    return truncated + truncated_error_part


def postprocess_compiler_output(result: ProcessResult, src_file_paths: Iterable[str] | None = None) -> str:
    """Postprocesses compiler output to remove unnecessary lines and shorten include stacks.

    Args:
        result (ProcessResult): Process result with compiler stdout/stderr.
        src_file_paths (Iterable[str] | None): Iterable of source file paths to identify messages
            related to the task source files. If None, all files are considered related.
    Returns:
        str: Postprocessed compiler output.
    """
    output = result.stdout + "\n" + result.stderr
    lines = output.splitlines()
    lines = _skip_until_compiler(lines)
    lines = _skip_after_ninja_build_stopped(lines)
    lines = _shorten_compiler_command(lines)
    lines = _filter_clang_warnings(lines, src_file_paths)
    lines = _shorten_stack(lines, _parse_clang_output.in_file_included_re)
    lines = _shorten_stack(lines, _parse_clang_output.required_from_re)
    lines = _shorten_paths(lines)
    lines = _remove_duplicates(lines)

    truncated = _truncate_but_keep_errors(lines, src_file_paths, max_length_initial=5000, max_length_errors=2000)

    return truncated


def check_shape_test_result(result: ProcessResult) -> bool:
    """Check whether the shape test passed based on the pytest output in the ProcessResult.

    This is specific to KernelBench. The name of the shape test is expected to be
    "test_output_shapes_match".

    Args:
        result: ProcessResult containing pytest output

    Returns:
        True if the shape test passed, False otherwise
    """
    output = result.stdout + "\n" + result.stderr
    for line in output.splitlines():
        if re.search(r"::test_output_shapes_match\s+PASSED", line):
            return True
    return False

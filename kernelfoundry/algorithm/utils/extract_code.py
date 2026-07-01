import json
import re
import os
from typing import List, Tuple

from kernelfoundry.algorithm.prompts.languages import EXTRACT_CODE_LANGUAGES


def extract_python_code(text):
    """
    Extract python code from model output
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches) if matches else ""


def remove_code_block_header(code, code_language_type):
    """Assume input is code but just with like python, cpp, etc. at the top"""
    if code.startswith(code_language_type):
        code = code[len(code_language_type) :].strip()
    return code


def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None


def extract_code_flexible(
    output_string: str,
    tag: str = "cuda",
    use_first: bool = False,
    code_language_types: list = EXTRACT_CODE_LANGUAGES,
) -> str | None:
    """Extracts code either from codeblocks, from string, or from tags"""
    matches_list = list(re.finditer(rf"\"\"\"\n(.*?)\"\"\"", output_string, re.DOTALL))
    if matches_list:
        match = matches_list[0]
        code = match.group(1).strip()
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()
        return code
    else:
        code_from_tags = extract_code_from_tags(
            output_string, tag=tag, use_first=use_first, code_language_types=code_language_types
        )
        if code_from_tags is None:
            return extract_code_from_tags(
                output_string, tag="python", use_first=use_first, code_language_types=code_language_types
            )
        return code_from_tags


def extract_cpp_code_heuristic(output_string: str) -> str | None:
    """Extract C++ code from model output"""
    # fmt: off
    cpp_keywords = [ "int", "float", "double", "char", "void", "bool", "long", "short", "unsigned", "signed", "auto",
        "const", "static", "extern", "register", "volatile", "inline", "namespace", "using", "typedef", "typename",
        "class", "struct", "union", "enum", "public", "private", "protected", "virtual", "friend", "template", 
        "operator", "if", "else", "switch", "case", "default", "for", "while", "do", "break", "continue", "return", 
        "try", "catch", "throw", "new", "delete", "sizeof", "__global__", "__device__", "__host__", "half",
        "float2", "float3", "float4", "int2", "int3", "int4", "dim3", "short2", "short3", "short4", "uchar2", "uchar3", 
        "uchar4", "__restrict__"
    ]
    preprocessor_directives = ["#include", "#define", "#ifdef", "#ifndef", "#endif", "#pragma", "#if", "#else", 
        "#elif", "#undef"
    ]
    # fmt: on
    cpp_keywords_re = re.compile(
        r"\b("
        + "|".join(re.escape(kw) for kw in cpp_keywords)
        + r")\b|("
        + "|".join(re.escape(pd) for pd in preprocessor_directives)
        + r")"
    )
    # line_with_comments_re = re.compile(r"^\s*//|/\*|\*/")
    line_with_brackets_re = re.compile(r"[{}\[\]()]")
    line_with_semicolon_re = re.compile(r";\s*(//.*)?$")
    line_with_comma_re = re.compile(r",\s*(//.*)?$")
    line_with_stream_op_re = re.compile(r"<<|>>")
    line_ending_with_continuation_re = re.compile(r"(<<|>>)\s*(//.*)?$")
    line_starting_with_stream_op_re = re.compile(r"^\s*(<<|>>)")

    trimmed = output_string.strip()
    lines = trimmed.splitlines()
    line_score = [0] * len(lines)

    # First pass: remove comments while preserving line count
    trimmed_without_comments_lines = []
    in_multiline_comment = False
    for line in lines:
        modified_line = line

        # Handle multiline comments
        if in_multiline_comment:
            end_pos = modified_line.find("*/")
            if end_pos != -1:
                modified_line = " " * (end_pos + 2) + modified_line[end_pos + 2 :]
                in_multiline_comment = False
            else:
                modified_line = ""

        # Check for start of multiline comment
        start_pos = modified_line.find("/*")
        if start_pos != -1:
            end_pos = modified_line.find("*/", start_pos)
            if end_pos != -1:
                # Single line multiline comment
                modified_line = (
                    modified_line[:start_pos] + " " * (end_pos - start_pos + 2) + modified_line[end_pos + 2 :]
                )
            else:
                # Start of multiline comment
                modified_line = modified_line[:start_pos]
                in_multiline_comment = True

        # Remove single line comments
        single_line_pos = modified_line.find("//")
        if single_line_pos != -1:
            modified_line = modified_line[:single_line_pos]

        trimmed_without_comments_lines.append(modified_line)

    trimmed_without_comments = "\n".join(trimmed_without_comments_lines)

    lines = trimmed_without_comments.splitlines()
    for i, line in enumerate(lines):
        matches = cpp_keywords_re.findall(line)
        line_score[i] += len(matches)
        if line_with_brackets_re.search(line):
            line_score[i] += 1
        if line_with_semicolon_re.search(line):
            line_score[i] += 1
        if line_with_stream_op_re.search(line):
            line_score[i] += 1
        # assign empty lines the score of the previous line
        if line.strip() == "" and i > 0:
            line_score[i] = line_score[i - 1]
        elif line_with_comma_re.search(line) and i > 0:
            line_score[i] = line_score[i - 1]  # likely continuation of previous line
        elif line_ending_with_continuation_re.search(line) and i > 0:
            line_score[i] = line_score[i - 1]  # likely continuation of previous line
        elif line_starting_with_stream_op_re.search(line) and i > 0 and line_score[i - 1] > 0:
            line_score[i] = max(line_score[i], line_score[i - 1])

    for i, line in enumerate(lines):
        if line_score[i] == 0 and i > 0 and i < len(lines) - 1:
            if line_score[i - 1] > 0 and line_score[i + 1] > 0:
                line_score[i] = line_score[i - 1]

    # Find the longest contiguous block of lines with line_score > 0
    max_length = 0
    max_start = 0
    current_length = 0
    current_start = 0

    for i, score in enumerate(line_score):
        if score > 0:
            if current_length == 0:
                current_start = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
            current_length = 0

    # Check the last block
    if current_length > max_length:
        max_length = current_length
        max_start = current_start

    # for i, line in enumerate(lines):
    #     print(f"{line_score[i]:2} {line}")

    if max_length > 0:
        return "\n".join(trimmed.splitlines()[max_start : max_start + max_length])

    return None


def extract_code_from_tags(
    output_string: str,
    tag: str = "cuda",
    use_first: bool = False,
    code_language_types: list = EXTRACT_CODE_LANGUAGES,
) -> str | None:
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(rf"<{tag}>(.*?)</{tag}>", trimmed, re.DOTALL)

    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)

    if not matches_list:
        # check if the output was in code blocks instead, with code tag specified
        matches_list = list(re.finditer(rf"```{tag}(.*?)```", trimmed, re.DOTALL))

    if not matches_list:
        # check if the output was in code blocks instead, with no tag specified
        matches_list = list(re.finditer(rf"```(.*?)```", trimmed, re.DOTALL))

    if matches_list:
        if use_first:
            match = matches_list[0]
        else:
            # use longest
            match = max(matches_list, key=lambda m: len(m.group(1)))
        code = match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type) or code.startswith(code_type.upper()):
                code = code[len(code_type) :].strip()

        return code

    return None


def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)

    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None


def extract_code_blocks(text, code_language_types: list[str]) -> str:
    """
    Extract all code blocks from text, combine them to return as a single string
    """
    pattern = r"```.*?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type) :].strip()
        combined_code.append(code)

    return " \n ".join(combined_code) if combined_code else ""


########### Diff utils


def apply_diff(original_code: str, diff_text: str) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            replace_summary = f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def replace_function_calls(text):
    """For templated kernels, add template_args to pytorch functional"""
    if "template_args=[]" in text and "*template_args" in text:
        # template args is already included in reference architecture
        return text

    # first, replace function head:
    text = text.replace("fn=module_fn", "fn=module_fn, template_args=[]")

    # Regular expression pattern to match the occurrences of `return fn(...)`
    pattern = r"(return\s+\w+\s*\(\s*[^)]*\))"

    # Function to add `, *template_args` to the arguments inside the parenthesis
    def replacer(match):
        # Get the matched group
        function_call = match.group(1)
        # Find the position before the closing parenthesis
        position = function_call.rfind(")")
        # Check if there's a trailing comma before the closing parenthesis
        function_wo_paragraphs = function_call.replace(" ", "").replace("\n", "")
        if ",)" in function_wo_paragraphs:
            # Insert `*template_args` in place of the trailing comma
            return function_call[:position] + " *template_args" + function_call[position:]
        else:
            # Insert `, *template_args` if no trailing comma
            return function_call[:position] + ", *template_args" + function_call[position:]

    # Use re.sub to replace all occurrences
    modified_text = re.sub(pattern, replacer, text)

    return modified_text

import warnings
from pathlib import Path

import clang
import clang.cindex
from clang.cindex import CursorKind, Cursor
import re


def _bundled_libclang() -> str | None:
    """Return the libclang shared library bundled by the libclang-ng package."""
    native = Path(clang.__file__).parent / "native"
    for name in ("libclang.dll", "libclang.so", "libclang.dylib"):
        candidate = native / name
        if candidate.is_file():
            return str(candidate)
    return None


# Always use the libclang bundled by libclang-ng, never a system install: the 'clang' package's
# cindex.py bindings are only guaranteed to match the libclang-ng build they were pinned against
_libclang_file = _bundled_libclang()

if _libclang_file:
    clang.cindex.Config.set_library_file(_libclang_file)
else:
    warnings.warn(
        "Could not locate the libclang shared library bundled with the 'libclang-ng' package. "
        "clang.cindex will fail at Index.create(). Reinstall 'libclang-ng'.",
        RuntimeWarning,
        stacklevel=1,
    )
from collections import defaultdict
from typing import NamedTuple


class CodeEditCommand(NamedTuple):
    """Object describing a code edit operation."""

    start: int
    end: int
    replacement: str

    def __repr__(self) -> str:
        return f"CodeEditCommand(start={self.start}, end={self.end}, replacement='{self.replacement}')"

    def _apply(self, source_code: str) -> str:
        result = source_code[: self.start] + self.replacement + source_code[self.end :]
        return result

    @staticmethod
    def apply_all(commands: list["CodeEditCommand"], source_code: str) -> str:
        # Apply commands in reverse order to avoid messing up offsets
        for command in sorted(commands, key=lambda c: c.start, reverse=True):
            source_code = command._apply(source_code)
        return source_code


def _convert_byte_offset_to_string_offset(source_code: str, byte_offset: int) -> int:
    """Convert a byte offset to a string offset in the given source code.
    Args:
        source_code (str): The source code.
        byte_offset (int): The byte offset to convert.
    Returns:
        int: The corresponding string offset.
    """
    encoded = source_code.encode("utf-8")
    if byte_offset > len(encoded):
        raise ValueError("byte_offset is out of range")
    decoded = encoded[:byte_offset].decode("utf-8", errors="ignore")
    return len(decoded)


class CodeEditing:

    def __init__(self, source_code: str, parse_header_files: bool = True):
        """CodeEditing class for manipulating C++ source code using libclang
        Args:
            source_code (str): The C++ source code to manipulate.
            parse_header_files (bool): If True, header files will be parsed.
        """
        self._source_code = None
        self._parse_header_files = parse_header_files
        self.source_code = source_code

    @property
    def translation_unit(self) -> clang.cindex.TranslationUnit:
        if self._translation_unit is None:
            code = self._source_code
            if not self._parse_header_files:
                # We don't want to parse header files, remove include statements by
                # replacing them with whitespace
                lines = self._source_code.split("\n")
                modified_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("#include"):
                        # Replace each character with space to preserve positions
                        modified_lines.append(" " * len(line))
                    else:
                        modified_lines.append(line)
                code = "\n".join(modified_lines)
                assert len(code) == len(
                    self._source_code
                ), "Length of source code must remain the same after removing includes"
            self._index = clang.cindex.Index.create()
            self._translation_unit = self._index.parse(
                "tmp.cpp", args=["-std=c++17"], unsaved_files=[("tmp.cpp", code)]
            )

        self._node_start_map = defaultdict(list)
        for node in self._translation_unit.cursor.walk_preorder():
            if node.location.file and node.location.file.name == "tmp.cpp":
                self._node_start_map[node.extent.start.offset].append(node)

        return self._translation_unit

    @property
    def source_code(self) -> str:
        return self._source_code

    @source_code.setter
    def source_code(self, value: str) -> None:
        assert isinstance(value, str), "source_code must be a string"
        if self._source_code == value:
            return
        self._source_code = value
        # for convenience, also store the source code as bytes for printing individual nodes or the AST
        self._source_code_as_bytes = value.encode("utf-8")
        # we need to re-parse the translation unit
        self._translation_unit = None

    def print_nodes(self, nodes: list[Cursor]):
        if isinstance(nodes, Cursor):
            nodes = [nodes]
        for node in nodes:
            if node.location.file and node.location.file.name == "tmp.cpp":
                lp_hash = node.lexical_parent.hash if node.lexical_parent else None
                sp_hash = node.semantic_parent.hash if node.semantic_parent else None
                print(
                    f"""{node.kind} {node.spelling} {node.type.get_canonical().spelling} {node.location} h={node.hash} lp={lp_hash} sp={sp_hash}
>|{self._source_code_as_bytes[node.extent.start.offset:node.extent.end.offset].decode('utf-8')}|<
"""
                    + 80 * "="
                )

    def print_ast(self):
        nodes = []
        for node in self.translation_unit.cursor.walk_preorder():
            if node.location.file and node.location.file.name == "tmp.cpp":
                nodes.append(node)
        self.print_nodes(nodes)

    def get_node_str(self, node: Cursor) -> str:
        return self._source_code_as_bytes[node.extent.start.offset : node.extent.end.offset].decode("utf-8")

    def find_nodes_by_kind(self, parent: Cursor, kind: CursorKind, recursive: bool = True) -> list[Cursor]:
        if recursive:
            return list(filter(lambda n: n.kind == kind, parent.walk_preorder()))
        else:
            return list(filter(lambda n: n.kind == kind, parent.get_children()))

    def find_enclosing_node(self, node: Cursor, kind: CursorKind | None) -> Cursor | None:
        _ = self.translation_unit  # ensure translation unit is parsed
        for n in self._node_start_map[node.extent.start.offset]:
            if (n.kind == kind or kind is None) and n.extent.end.offset >= node.extent.end.offset:
                return n

    def find_sycl_queue(self) -> list[Cursor]:
        """Find SYCL queue variable declaration nodes in the AST.
        Returns:
            list[Cursor]: A list of Cursor nodes representing SYCL queue variable declarations.
        """
        result = []
        for node in self.translation_unit.cursor.walk_preorder():
            if node.location.file and node.location.file.name == "tmp.cpp":
                if node.kind == CursorKind.VAR_DECL:
                    if node.type.get_canonical().spelling in ("sycl::queue", "sycl::queue &"):
                        result.append(node)
        return result

    def find_sycl_wait(self, queue_node: Cursor) -> list[Cursor]:
        """Find SYCL queue wait call expressions in the AST for a given queue variable.
        Args:
            queue_node (Cursor): The Cursor node representing the SYCL queue variable.
        Returns:
            list[Cursor]: A list of Cursor nodes representing SYCL queue wait call expressions.
        """
        result = []
        for node in self.translation_unit.cursor.walk_preorder():
            if node.location.file and node.location.file.name == "tmp.cpp":
                if node.kind == CursorKind.CALL_EXPR:
                    if node.spelling == "wait":
                        decl_ref_expr = self.find_nodes_by_kind(node, CursorKind.DECL_REF_EXPR)
                        if decl_ref_expr and decl_ref_expr[0].spelling == queue_node.spelling:
                            result.append(node)
        return result

    @staticmethod
    def replace_node_with_str(
        node: Cursor, replacement_str: str, source_code: str, remove_semicolon: bool = False
    ) -> CodeEditCommand:
        """Helper function for manipulating the source code by replacing a node with a string.
        Args:
            node (Cursor): The node to replace.
            replacement_str (str): The string to replace the node with.
            source_code (str): The source code to manipulate.
            remove_semicolon (bool): If True, removes a trailing semicolon after the node's extent.
        Returns:
            str: The modified source code with the node replaced by the string.
        """
        start_offset = node.extent.start.offset
        end_offset = node.extent.end.offset
        start_offset = _convert_byte_offset_to_string_offset(source_code, start_offset)
        end_offset = _convert_byte_offset_to_string_offset(source_code, end_offset)
        if remove_semicolon:
            # Look for whitespace and semicolon after the node's end
            i = end_offset
            while i < len(source_code) and source_code[i].isspace():
                i += 1
            if i < len(source_code) and source_code[i] == ";":
                end_offset = i + 1

        return CodeEditCommand(start_offset, end_offset, replacement_str)

    def replace_queue_with_torch_queue_and_remove_wait(self):
        """This method finds the SYCL queue declaration in the source code and replaces it with a
        declaration that uses the torch XPU stream queue. It removes the last queue.wait() call
        and adds the necessary include for c10/xpu/XPUStream.h.
        """
        code = self.source_code
        code_edit_commands = []
        q_nodes = self.find_sycl_queue()
        if not q_nodes:
            return

        # add include for c10/xpu/XPUStream.h
        if "XPUStream.h" not in code:
            include_str = "#include <c10/xpu/XPUStream.h>\n"
            lines = self.source_code.split("\n")
            last_include_line = -1
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#include"):
                    last_include_line = i

            end_offset = 0
            if last_include_line >= 0:
                # Calculate end offset of the last include line
                end_offset = sum(len(lines[j]) + 1 for j in range(last_include_line + 1))  # +1 for newline
            code_edit_commands.append(CodeEditCommand(end_offset, end_offset, include_str))

        # change queue declaration to use torch XPU stream
        for q_node in q_nodes:
            if "getCurrentXPUStream" not in code:
                queue_var_name = q_node.spelling
                torch_queue = f"sycl::queue& {queue_var_name} = c10::xpu::getCurrentXPUStream().queue()"
                if "static" in self.get_node_str(q_node):
                    q_node = self.find_enclosing_node(q_node, CursorKind.DECL_STMT)
                    torch_queue = torch_queue + ";"
                if q_node is not None:
                    code_edit_commands.append(self.replace_node_with_str(q_node, torch_queue, code))

        # remove the queue.wait() calls
        w_nodes = self.find_sycl_wait(q_node)
        for w_node in w_nodes:
            code_edit_commands.append(self.replace_node_with_str(w_node, "", code, remove_semicolon=True))

        self.source_code = CodeEditCommand.apply_all(code_edit_commands, code)

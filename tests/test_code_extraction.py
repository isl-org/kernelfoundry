from kernelfoundry.algorithm.utils.extract_code import extract_cpp_code_heuristic


class TestProseIsNotCode:
    """The heuristic scores C++ keywords, and English is full of them."""

    def test_a_refusal_to_change_anything_is_rejected(self):
        assert extract_cpp_code_heuristic("The kernel is already optimal, so I left it alone.") is None

    def test_a_paragraph_of_reasoning_is_rejected(self):
        text = (
            "Nothing to change here. The current implementation returns the best result\n"
            "for this case and I would not recommend a different approach.\n"
        )
        assert extract_cpp_code_heuristic(text) is None


class TestCodeSurvives:
    def test_a_function(self):
        code = "void f(int n) {\n    for (int i = 0; i < n; ++i) g(i);\n}"
        assert extract_cpp_code_heuristic(code) == code

    def test_preprocessor_only(self):
        code = "#include <sycl/sycl.hpp>\n#define QK 32"
        assert extract_cpp_code_heuristic(code) == code

    def test_a_single_statement(self):
        code = "acc[a][b][e] = sycl::fma(x, y, acc[a][b][e]);"
        assert extract_cpp_code_heuristic(code) == code

    def test_code_is_found_inside_surrounding_prose(self):
        text = "Here is the improved kernel.\n\nint add(int a, int b) { return a + b; }\n\nIt is faster."
        assert "int add(int a, int b) { return a + b; }" in extract_cpp_code_heuristic(text)

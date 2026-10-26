import textwrap

import pytest

from kernelfoundry.algorithm.utils.skills import (
    Skill,
    load_skills_for_language,
    skills_as_prompt_sections,
)


def write_skill(root, name, language, body="Body text."):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        textwrap.dedent(f"""\
            ---
            name: {name}
            description: A skill for {language}.
            metadata:
              programming-languages: {language}
            ---

            {body}
            """),
        encoding="utf-8",
    )
    return d


class TestLoadSkillsForLanguage:
    def test_no_skills_configured_loads_nothing(self):
        assert load_skills_for_language({"language": "SYCL"}) == []

    def test_filters_by_language(self, tmp_path):
        write_skill(tmp_path, "sycl-gemm", "SYCL")
        write_skill(tmp_path, "cuda-gemm", "CUDA")
        config = {"skills": {"directories": [str(tmp_path)]}, "language": "SYCL"}
        assert [s.name for s in load_skills_for_language(config)] == ["sycl-gemm"]

    @pytest.mark.parametrize("declared", ["SYCL", "sycl", "SyCl"])
    @pytest.mark.parametrize("requested", ["SYCL", "sycl"])
    def test_language_matching_ignores_case(self, tmp_path, declared, requested):
        """A config says SYCL and a skill may say sycl; they mean the same language."""
        write_skill(tmp_path, "sycl-gemm", declared)
        config = {"skills": {"directories": [str(tmp_path)]}, "language": requested}
        assert [s.name for s in load_skills_for_language(config)] == ["sycl-gemm"]

    def test_opencl_matches_the_ocl_alias(self, tmp_path):
        write_skill(tmp_path, "ocl-gemm", "ocl")
        config = {"skills": {"directories": [str(tmp_path)]}, "language": "OpenCL"}
        assert [s.name for s in load_skills_for_language(config)] == ["ocl-gemm"]

    def test_a_missing_directory_is_skipped(self, tmp_path):
        config = {"skills": {"directories": [str(tmp_path / "nope")]}, "language": "SYCL"}
        assert load_skills_for_language(config) == []


class TestSkillsAsPromptSections:
    def make(self, name, body):
        return Skill.parse(f"---\nname: {name}\ndescription: d\n---\n\n{body}\n")

    def test_body_and_description_are_included(self):
        [section] = skills_as_prompt_sections([self.make("a", "The register layout is X.")])
        assert "The register layout is X." in section
        assert section.startswith("### a")

    def test_a_skill_over_budget_is_dropped_not_truncated(self, caplog):
        skills = [self.make("small", "x"), self.make("big", "y" * 5000)]
        sections = skills_as_prompt_sections(skills, max_chars=1000)
        assert len(sections) == 1
        assert "small" in sections[0]
        assert "big" not in sections[0]

    def test_budget_counts_across_skills(self):
        skills = [self.make(n, "z" * 400) for n in ("a", "b", "c")]
        assert len(skills_as_prompt_sections(skills, max_chars=900)) == 2

    def test_warns_when_skill_resources_are_not_inlined(self, tmp_path, caplog):
        skill_dir = write_skill(tmp_path, "with-resource", "SYCL", body="Body text.")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "layout.txt").write_text("layout", encoding="utf-8")

        skill = Skill.load(skill_dir)
        skills_as_prompt_sections([skill])

        assert "not inlined into classic prompts" in caplog.text


def render(sections):
    from kernelfoundry.algorithm.prompts.template_manager import TemplateManager

    manager = TemplateManager(
        template_example=None,
        language="SYCL",
        gpu_arch="bmg",
        ref_language="Pytorch",
        skill_prompt_list=sections,
    )
    return manager.construct_prompt(reference_code="def f(x): return x")


class TestTheRenderedPrompt:
    """Storing the sections is not the risk; the template silently dropping them is."""

    def test_skill_text_reaches_the_prompt(self):
        prompt = render(["### dpas\n\nHow to use DPAS.\n\nByte i of lane l holds A[i/2]."])
        assert "Byte i of lane l holds A[i/2]." in prompt
        assert "Reference material" in prompt

    def test_no_skills_adds_no_heading(self):
        assert "Reference material" not in render([])

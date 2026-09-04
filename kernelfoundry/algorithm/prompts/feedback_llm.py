"""Handler for LLM that provides feedback based on console output and documentation."""

import os
import json
import re
import hydra
import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape

from kernelfoundry.algorithm.problem_logger import ProblemLogger
from kernelfoundry.algorithm.utils.extract_code import extract_code_from_tags


class DocumentationHandler:
    """Loads and retrieves documentation snippets referenced by keyword tags."""

    def __init__(self, doc_path: str = "data/rag/docs/sycl_specification.json"):
        """Load json file with list of documentation sections"""
        assert os.path.exists(doc_path), f"SYCL specification not found at {doc_path}"

        with open(doc_path, "r", encoding="utf-8") as inf:
            self.documentation = json.load(inf)

    def _find_best_match(self, keyword: str):
        """Find most suitable section from the documentation with this keyword"""
        # remove "sycl from keyword because everything is about sycl"
        simplified_kw = keyword.lower().replace("sycl", "")
        # break
        candidate_sections = []
        for doc_section in self.documentation:
            if simplified_kw in doc_section.lower():
                candidate_sections.append(doc_section)
        if len(candidate_sections) == 0:
            return None
        if len(candidate_sections) == 1:
            return candidate_sections[0]
        else:
            # multiple matches
            in_header = [simplified_kw in doc_section.lower().split("\n")[0] for doc_section in candidate_sections]
            # Option 1: return one of the sections where the keyword is in the header
            if any(in_header):
                return candidate_sections[np.where(in_header)[0][0]]
            # Option 2: Return section with most occurences of the keyword
            num_matches = [doc_section.lower().count(simplified_kw) for doc_section in candidate_sections]
            return candidate_sections[np.argmax(num_matches)]

    def _extract_kw_and_find_docs(self, text_with_keywords: str, pattern: str = r"<kw>(.*?)</kw>"):
        """Find keywords in raw text and try to retrieve documentation for each of them"""
        matches = re.findall(pattern, text_with_keywords)

        for keyword in matches:
            # find best match
            docs_section = self._find_best_match(keyword)
            # if one of the keywords was found, stop -> we cannot include several sections
            if docs_section is not None:
                return docs_section
        return ""

    def replace_keywords_by_documentation(self, text_with_keywords: str):
        """Take feedback from LLM with keywords and replace keyword part by the documentation"""

        # first check if the last section of the feedback contains any keywords
        if "kw" not in text_with_keywords.split("###")[-1]:
            return text_with_keywords

        # remove the keyword part
        feedback_base = text_with_keywords.split("###")[:-1]

        # find the corresponding docs
        doc_to_insert = self._extract_kw_and_find_docs(text_with_keywords)

        # if we didn't find any, just return the original feedback
        if len(doc_to_insert) == 0:
            return "###".join(feedback_base)

        doc_to_insert = " Relevant excerpt from the SYCL specification:\n\n" + doc_to_insert
        feedback_and_docs = "###".join(feedback_base + [doc_to_insert])

        return feedback_and_docs


class FeedbackHelper:
    """Coordinates prompting an LLM for feedback based on the eval log."""

    def __init__(
        self,
        use_feedback_llm: bool,
        language: str,
        server_config,
        use_docs_via_keywords: bool = False,
    ):
        """Initialize feedback behavior and optional documentation enrichment."""
        self.use_feedback_llm = use_feedback_llm
        if use_feedback_llm:
            self.inference_server = hydra.utils.instantiate(server_config)
        else:
            self.inference_server = None
        self.language = language

        self.kw_prompt = self.language == "SYCL" and use_docs_via_keywords
        if self.kw_prompt:
            self.docs_handler = DocumentationHandler()

        env = Environment(loader=PackageLoader("kernelfoundry.algorithm.prompts"), autoescape=select_autoescape())
        self.prompt_template = env.get_template("feedback_llm_prompt.j2")

    def load_parent_and_get_feedback(self, problem_logger: ProblemLogger, parent_program=None):
        """Call feedback LLM that analyses the console output, or fall back to the console output itself."""
        if parent_program is not None:
            #### Task code path
            prior_gen_code = parent_program.code_as_str
            if parent_program.is_program0:  # The first program does not have console output
                console_output = None
            else:
                console_output = parent_program.kernel_exec_result.eval_log
            if self.use_feedback_llm and console_output is not None:
                feedback_prompt = self.construct_feedback_prompt(prior_gen_code, console_output)
                feedback = self.inference_server(messages=[{"role": "user", "content": feedback_prompt}])
                return prior_gen_code, feedback
            # return generated kernel and console output if feedback LLM is not used
            return prior_gen_code, [console_output]

        else:
            # load console output from one iteration earlier
            console_output = problem_logger.read_prior_stdout()
            prior_gen_code = problem_logger.read_prior_gen_code()
            if prior_gen_code is not None:
                prior_gen_code = extract_code_from_tags(prior_gen_code, self.language.lower())
            if self.use_feedback_llm and console_output is not None:
                feedback_prompt = self.construct_feedback_prompt(prior_gen_code, console_output)
                feedback = self.inference_server(messages=[{"role": "user", "content": feedback_prompt}])
            else:
                feedback = [console_output]
            # return generated kernel, console output, and feedback
            return prior_gen_code, console_output, feedback

    def get_feedback(self, generated_kernel: str, console_output: str):
        """Get feedback from the feedback LLM."""
        if not self.use_feedback_llm:
            return [console_output]
        feedback_prompt = self.construct_feedback_prompt(generated_kernel, console_output)
        feedback = self.inference_server(messages=[{"role": "user", "content": feedback_prompt}])
        # replace keywords with relevant documentation
        if self.kw_prompt:
            feedback = [self.docs_handler.replace_keywords_by_documentation(fb) for fb in feedback]
        return feedback

    def construct_feedback_prompt(self, generated_kernel, console_output):
        """
        Generate a prompt for the feedback LLM to analyze the generated kernel and console output.
        """
        status = "correct" if "compiles and is correct" in console_output else "error"

        prompt = self.prompt_template.render(
            generated_kernel=generated_kernel,
            status=status,
            console_output=console_output,
            use_keyword_prompt=self.kw_prompt,
        )
        return prompt

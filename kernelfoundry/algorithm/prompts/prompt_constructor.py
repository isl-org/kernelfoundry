"""Prompt construction based on RAG databases, templates, and examples."""

import os

# from urllib import response
import warnings
import random
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from jinja2 import Environment, PackageLoader, select_autoescape

from kernelfoundry.algorithm.prompts.template_manager import TemplateManager
from kernelfoundry.algorithm.schemas import Program
from kernelfoundry.algorithm.inference_server import InferenceServer

# path to examples
REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KERNEL_EXAMPLES = os.path.join(REPO_TOP_PATH, "kernelfoundry", "algorithm", "prompts", "kernel_examples")
TEMPLATES_PATH = os.path.join(REPO_TOP_PATH, "kernelfoundry", "algorithm", "prompts", "templates")

SYSTEM_PROMPT = "You are an expert CUDA engineer tasked with translating PyTorch code into performant CUDA kernel code."


# Keywords for kernel categorization
TOPIC_KEYWORDS = [
    # Matrix operations
    "gemm",  # covers matmul and most variants
    "batched_matmul",  # BMM / batched; distinct enough to keep
    "matrix_transpose",
    "dot_product",
    # Reductions & scans
    "reduction",  # sum, mean, min, max, argmin, argmax, product
    "scan",  # prefix_sum, cumsum, cumprod and variants
    "histogram",
    # Sorting
    "sort",  # bitonic, radix, merge, etc.
    # Data movement
    "gather_scatter",
    "stencil",
    "copy",
    # Convolution
    "convolution",  # standard 1D/2D/3D
    "conv_transposed",
    "conv_depthwise",  # covers depthwise-separable and pointwise too
    # Pooling
    "pooling",  # max, average, global avg
    # Activations — keep only the ones with genuinely distinct compute shapes
    "activation_function",  # umbrella / catch-all
    "relu",  # relu, leaky_relu, elu, selu, hardtanh
    "sigmoid",  # sigmoid, hard_sigmoid
    "tanh",
    "gelu",  # gelu, swish, mish, silu
    "softmax",  # softmax, log_softmax, logsumexp
    # Normalization
    "normalization",  # umbrella
    "batch_norm",
    "layer_norm",  # important for transformers
    "group_norm",
    # Attention
    "attention",  # self-attention, multi-head attention, flash attention
    # Element-wise arithmetic
    "elementwise_arithmetic",  # add, sub, mul, div, clamp, scale, bias_add, residual_add
    # Loss functions
    "loss_function",  # umbrella; mse, cross_entropy, kl_div, etc.
    # Math
    "math",  # sin, cos, exp, log, sqrt, pow, ...
    # Other ML ops
    "dropout",
    "embedding",  # embedding lookup / table
    # Lower-level / systems
    "sort",
    "fft",
    "sparse",  # sparse matmul, SpMM, etc.
    "half_precision",
]


def get_system_prompt(language: str) -> str:
    return SYSTEM_PROMPT.replace("CUDA", language)


class PromptConstructor:
    """Builds generation prompts with templates, examples, and RAG inputs."""

    def __init__(
        self,
        language: str,
        gpu_arch: str | list,
        prompt_config: DictConfig,
        reference_language: str = "Pytorch",
        mode: str = "functional",
        use_feedback_llm: bool = False,
    ):
        """Initialize prompt construction dependencies and retrieval backends."""

        self.language = language
        self.gpu_arch = gpu_arch
        self.reference_language = reference_language
        self.mode = mode
        self.diff_format = prompt_config.diff_format
        self.use_feedback_llm = use_feedback_llm
        self._cached_reference_src = None
        self._cached_reference_keywords = None

        # construct template manager kwargs
        template_kwargs = {
            "language": language,
            "gpu_arch": gpu_arch,
            "ref_language": self.reference_language,
            "n_tips": prompt_config.get("num_optimization_tips", 2),
            "include_inspirations": prompt_config.get("include_inspirations", True),
            "include_top": prompt_config.get("include_best_program", True),
            "use_hardware_prompt": prompt_config.get("include_hardware_specs", True),
            "allow_templated": prompt_config.get("allow_templated", False),
        }

        # load template example
        lang_lower = language.lower()
        fn_end = "cu" if lang_lower == "cuda" else "sycl"
        template_example_path = os.path.join(KERNEL_EXAMPLES, f"templated_{mode}_{lang_lower}.{fn_end}")
        if os.path.exists(template_example_path):
            with open(template_example_path, "r") as inf:
                self.template_example = inf.read()
        else:
            assert not template_kwargs.get("allow_templated", True), "Must provide example if allow_templated=True"
            self.template_example = None

        self.template_manager = TemplateManager(template_example=self.template_example, **template_kwargs)

        # INITIALIZE RAG DATABASES
        self.rag_databases = []
        # in the config, there is a list of rag databases to setup (e.g. standard SYCL, ESIMD, etc)
        for rag_config in prompt_config.rag:
            rag_init_args = OmegaConf.to_container(rag_config)
            rag_init_args["language"] = language  # pass language to RAG init
            rag_db = hydra.utils.instantiate(rag_init_args)
            self.rag_databases.append(rag_db)
            print("RAG: Initialized rag db:", rag_config)

    def __call__(
        self,
        reference_src: str,
        problem_name: str,
        last_program: Program | None = None,
        second_ref_code: str | None = None,
        inspirations: list[Program] | None = None,
        top_program: Program | None = None,
        evolvable_content: dict[str, str] | None = None,
        target_optimization_profile: dict[str, int] | None = None,
        ref_keywords: list[str] | None = None,
    ) -> str:
        """
        Generate a prompt for the given reference source code.

        Args:
            reference_src: The reference source code to translate
            problem_name: Name of the problem for RAG lookup
            last_program: Previous iteration's program (for feedback)
            second_ref_code: Optional secondary reference code
            inspirations: List of inspiration programs to include
            top_program: Best performing program so far
            evolvable_content: Optional evolved content for template regions
            target_optimization_profile: Optimization coordinates selected for this iteration
            ref_keywords: List of keywords for reference, usually computed in first iteration
        """
        if inspirations is None:
            inspirations = []

        # get example
        is_first_iter = last_program is None and top_program is None

        if (
            ref_keywords is None
            and (self._cached_reference_src != reference_src or self._cached_reference_keywords is None)
            and len(self.rag_databases) > 0  # only necessary to categorize if we want to use RAG examples
        ):
            self._cached_reference_src = reference_src
            logging.info("Categorizing reference code for RAG retrieval...")
            self._cached_reference_keywords = self.categorize_code(reference_src, input_type="PyTorch code")
        elif ref_keywords is not None:
            self._cached_reference_keywords = ref_keywords

        reference_keywords = self._cached_reference_keywords

        rag_input = self.get_examples(
            reference_src,
            problem_name,
            is_first_iter,
            last_program,
            top_program,
            reference_keywords=reference_keywords,
            target_optimization_profile=target_optimization_profile,
        )

        # check whether forward or backward
        is_backward = "backward" in problem_name

        return self.template_manager.construct_prompt(
            reference_src,
            last_program=last_program,
            prior_versions=inspirations,
            top_program=top_program,
            rag_input=rag_input,
            second_ref_code=second_ref_code,
            is_feedback=self.use_feedback_llm,
            evolvable_content=evolvable_content,
            is_backward=is_backward,
        )

    ################## Load examples (from simple examples or RAG-based) ##################

    def get_examples(
        self,
        reference_src: str,
        problem_name: str,
        is_first_iter: bool,
        last_program=None,
        top_program=None,
        reference_keywords: list[str] | None = None,
        target_optimization_profile: dict[str, int] | None = None,
    ) -> list:
        """
        Generate an initial prompt for the given reference source code.
        """
        rag_input_list = []

        if is_first_iter and self.reference_language == "Pytorch":
            # Use single example (vector addition)
            simple_init_example = self.load_vector_add_example()
            rag_input_list.append(simple_init_example)

        # iterate through rag databases
        for rag_db in self.rag_databases:
            rag_db_input = rag_db.get_rag_examples(
                problem_name=problem_name,
                is_first_iter=is_first_iter,
                reference_keywords=reference_keywords,
                target_optimization_profile=target_optimization_profile,
            )
            rag_input_list.append(rag_db_input)
        return "\n\n".join(rag_input_list)

    def load_vector_add_example(self):
        """Load the canonical vector-add translation example for the active language."""
        functional_form = self.mode == "functional"
        # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
        in_path_dict = {True: "pytorch_functional_ex_add.py", False: "model_ex_add.py"}
        out_path_dict_cuda = {True: "cuda_example_add_raw.cu", False: "model_new_ex_add.py"}
        out_path_dict_sycl = {True: "sycl_example_add_raw.sycl", False: "model_new_ex_add_sycl.py"}
        out_path_dict_triton = {True: "triton_functional.py", False: "triton_class.py"}

        in_example = os.path.join(KERNEL_EXAMPLES, in_path_dict[functional_form])
        if self.language == "CUDA":
            out_example = os.path.join(KERNEL_EXAMPLES, out_path_dict_cuda[functional_form])
        elif self.language == "SYCL":
            out_example = os.path.join(KERNEL_EXAMPLES, out_path_dict_sycl[functional_form])
        elif self.language == "triton":
            out_example = os.path.join(KERNEL_EXAMPLES, out_path_dict_triton[functional_form])
        else:
            raise NotImplementedError("No other language than CUDA, SYCL, triton are supported")

        example_arch_path = os.path.join(REPO_TOP_PATH, in_example)
        example_new_arch_path = os.path.join(REPO_TOP_PATH, out_example)

        if not os.path.exists(example_arch_path):
            raise FileNotFoundError(f"Example architecture file not found: {example_arch_path}")
        if not os.path.exists(example_new_arch_path):
            raise FileNotFoundError(f"Example new architecture file not found: {example_new_arch_path}")

        with open(example_arch_path, "r") as f:
            example_arch = f.read()
        with open(example_new_arch_path, "r") as f:
            example_new_arch = f.read()

        example = f"""
        ### Example:

        { self.reference_language} reference:
        ```
        {example_arch}
        ```
        Correct {self.language} kernel:
        ```
        {example_new_arch}
        ```
        """
        return example

    @staticmethod
    def categorize_code(
        code: str,
        input_type: str = "PyTorch code",
        allowed_keywords: list[str] | None = None,
        server_type: str = "intel_gnai",
        model_name: str = "claude-4-5-sonnet",
    ) -> list[str]:
        """Categorize code using LLM to extract relevant topic keywords.

        Args:
            code: Source code to categorize.
            input_type: Type of input code (e.g., "PyTorch code", "CUDA kernel").
            allowed_keywords: List of keywords to use for categorization.
            server_type: Inference server type (e.g., "intel_gnai").
            model_name: Model to use for categorization.

        Returns:
            List of extracted keywords.
        """
        if allowed_keywords is None:
            allowed_keywords = TOPIC_KEYWORDS

        # Setup LLM inference
        inf_server = InferenceServer(
            server_type=server_type,
            model_name=model_name,
            max_tokens=500,
            temperature=0.0,
            num_completions=1,
        )

        # Render categorization prompt
        env = Environment(loader=PackageLoader("kernelfoundry.algorithm.prompts"), autoescape=select_autoescape())
        prompt_template = env.get_template("categorize_code.j2")
        prompt = prompt_template.render(
            kernel=code,
            input_type=input_type,
            allowed_keywords=", ".join(allowed_keywords),
        )
        # Call LLM to categorize
        messages = [
            {"role": "system", "content": "You are a helpful code categorization assistant."},
            {"role": "user", "content": prompt},
        ]
        response = inf_server(messages)

        # Extract keywords from response
        response_text = response[0] if isinstance(response, list) else response
        keywords = []
        for line in response_text.split("\n"):
            if "TOPICS:" in line:
                topics_str = line.split("TOPICS:")[-1].strip()
                keywords = [k.strip() for k in topics_str.split(",")]
                break
            if "TOPIC:" in line:
                topics_str = line.split("TOPIC:")[-1].strip()
                keywords = [k.strip() for k in topics_str.split(",")]
                break

        return keywords

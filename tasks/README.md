# Tasks

Example tasks and an interface to commonly used benchmarks.

| Directory | Contents |
| --- | --- |
| `example_custom/` | A complete SYCL matrix-multiply task. The best starting point for your own. |
| `kernelbench/` | Template for converting a KernelBench task into KernelFoundry format. |
| `pytorch_functionals.csv` | Reference PyTorch operations used by the KernelBench tasks. |

**The task format is documented in the user guide, not here:**

- [Anatomy of a task package](https://isl-org.github.io/kernelfoundry/docs/guide/task-package.html)
  covers files, the `[EVOLVE_START]` / `[REFERENCE_START]` / `[USER_INSTRUCTIONS_START]` blocks, and
  how to create a task
- [Writing tests](https://isl-org.github.io/kernelfoundry/docs/guide/writing-tests.html)
  covers correctness tests and the benchmark
- [Config parameters](https://isl-org.github.io/kernelfoundry/docs/guide/config-parameters.html)
  documents every `config.yaml` key
- [Quickstart](https://isl-org.github.io/kernelfoundry/docs/guide/quickstart.html)
  runs `example_custom/` end to end without an LLM API key

## KernelBench

### KernelBench Foundation

The core task collection is based on [KernelBench](https://github.com/KernelBench/KernelBench), a benchmark suite for evaluating GPU kernel generation capabilities. KernelBench provides a comprehensive set of kernel optimization problems with different difficulty levels.

### SakanaAI Functional Implementation

We leverage the functional implementations of KernelBench tasks provided by SakanaAI in the [AI-CUDA-Engineer-Archive](https://huggingface.co/datasets/SakanaAI/AI-CUDA-Engineer-Archive) on HuggingFace (CC-BY 4.0 License). This implementation is algorithmically equivalent to the original KernelBench tasks, but isolates the computations into standalone functions rather than as part of nn.Module. 

## Usage

A placeholder template for converting any KernelBench task into the KernelFoundry format can be found in the `kernelbench/` subdirectory. The template includes
* A `config.yaml` file where task_name, language and other metadata are inserted
* A `task.py` file with tests and the reference implementation, which is inserted from the respective KernelBench task.
* A file `conftest.py` that provides an interface to the KernelFoundry test fixtures (not modified)

## Citation

If you use these tasks in your research, please cite the original KernelBench paper or the work by SakanaAI that provides the functional version:

```bibtex
@inproceedings{ouyang2025kernelbench,
  title={KernelBench: Can LLMs Write Efficient GPU Kernels?},
  author={Ouyang, Anne and Guo, Simon and Arora, Simran and Zhang, Alex L and Hu, William and Re, Christopher and Mirhoseini, Azalia},
  booktitle={International Conference on Machine Learning},
  pages={47356--47415},
  year={2025},
  organization={PMLR}
}
@article{lange2025towards,
  title={Towards robust agentic cuda kernel benchmarking, verification, and optimization},
  author={Lange, Robert Tjarko and Sun, Qi and Prasad, Aaditya and Faldor, Maxence and Tang, Yujin and Ha, David},
  journal={arXiv preprint arXiv:2509.14279},
  year={2025}
}
```

## License

The functional implementations provided by SakanaAI are licensed under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). The original KernelBench benchmark is published under MIT license.

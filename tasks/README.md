# Tasks

KernelFoundry provides a standardized format for GPU kernel optimization tasks. This directory hosts example tasks and an interface to commonly used benchmarks.

## Task definition

### File structure

A custom task broadly consists of the following components:

1.  Config file (usually config.yaml): Defines parameters such as a task ID / name, language (e.g. SYCL), number of iterations to run, etc
2.  Task file (usually task.py): Main file that defines the kernel generation task (usually via a reference implementation, e.g., a PyTorch operation) and the tests that the kernel has to pass. 
3.  Kernel file (usually kernel.cpp): Kernel file that the LLM will modify. Can also be almost empty in the beginning, except for the evolve-start and end markers (see below)
4. Reference file (e.g. reference.cpp): A baseline implementation that the new kernel will be compared to. Usually provided by the user.

The file `conftest.py` , which you can see in all the template tasks, must be included unchanged in every custom task. Its only function is to configure pytest to work with our testing routine.

### Block structure

There are three predefined markers that you can use to mark special parts of your code:

1.  Evolve (\[EVOLVE\_START\] - \[EVOLVE\_END\]): 
    1.  Defines the part that should be modified when optimizing the kernel. The KernelFoundry framework or the agent responsible for that will modify only the block between the EVOLVE tags, and leave all other code untouched.
    2.  This part between the evolve-tags can be empty! If you want to generate a kernel from scratch, without any function header/imports / bindings given, you can simply leave this part empty.
    3.  Having an Evolve-part somewhere in the code is compulsory!
2.  Reference (\[REFERENCE\_START\] - \[REFERENCE\_END\]):  
    1.  The reference is an implementation of the operation for which the kernel should be generated. For example, the reference could be Pytorch code (e.g. torch.matmul(a, b)) and the task is to write a custom kernel that is faster than the reference implementation.
    2.  Oftentimes in our examples, the reference is used to check the correctness of the kernel implementation (comparing kernel output tensors to the reference output tensors).
    3.  Providing a reference is optional. It is possible to guide the LLM solely via user instructions and to test the kernel in a way other than by comparison with the reference.
3.  User instructions (\[USER\_INSTRUCTIONS\_START\] - \[USER\_INSTRUCTIONS\_END\]):
    1.  Optionally, you can provide further instructions, based on any guidance the user has provided. 
    2.  Examples include specific optimization strategies to try out, info about what will be tested, etc.

Notes:

-   By default, these three blocks will be included in the prompt (if provided). Thus, the LLM considers reference implementation, the code to evolve, and user instructions when proposing a new kernel.
-   It is worth noting that these blocks can be located anywhere. KernelFoundry will search all the files you provide to find these blocks. However, it makes the most sense to have a dedicated  or kernel.sycl file with the Evolve-block.
-   Of course, including the tags in a .py or .cpp file would lead to Syntax Errors. They need to be commented out.


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

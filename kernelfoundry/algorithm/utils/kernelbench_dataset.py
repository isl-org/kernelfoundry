################################################################################
# Helpers for KernelBench Tasks
################################################################################

import os
import re
import hashlib
import shutil
import tempfile
import pandas as pd
from omegaconf import DictConfig

from kernelfoundry import PACKAGE_ROOT
import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.eval_pipeline.task import Task
from kernelfoundry.algorithm.utils.extract_code import replace_function_calls

# The KernelBench task templates and pytorch_functionals.csv are data, not code, and are
# deliberately not shipped in the wheel. Using task_origin=KernelBench or robust_kbench requires a
# git checkout. Anchored on PACKAGE_ROOT.
_TASKS_DIR = os.path.join(PACKAGE_ROOT.parent, "tasks")
KERNELBENCH_TASK_TEMPLATE = os.path.join(_TASKS_DIR, "kernelbench")
BACKWARD_TASK_TEMPLATE = os.path.join(_TASKS_DIR, "robust_kbench_backward")
FN_ENDING_DICT = {"SYCL": "sycl", "triton": "py", "CUDA": "cu"}
COMMENT_DICT = {"SYCL": "//", "triton": "#", "CUDA": "//"}

PYTORCH_FUNCTIONALS_PATH = os.path.join(_TASKS_DIR, "pytorch_functionals.csv")


PYTORCH_FUNCTIONALS = None

# Tasks that we filtered out because they do not allow proper correctness testing (low output range / std, low sensitivity to input tensors)
FILTERED_OUT = [
    "80_Gemm_Max_Subtract_GELU",
    "84_Gemm_BatchNorm_Scaling_Softmax",
    "23_Softmax",
    "37_FrobeniusNorm_",
    "38_L1Norm_",
    "50_Product_reduction_over_a_dimension",
    "94_MSELoss",
    "95_CrossEntropyLoss",
    "96_HuberLoss",
    "97_CosineSimilarityLoss",
    "98_KLDivLoss",
    "9_Matmul_Subtract_Multiply_ReLU",
    "13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling",
    "23_Conv3d_GroupNorm_Mean",
    "41_Gemm_BatchNorm_GELU_GroupNorm_Mean_ReLU",
    "44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean",
    "48_Conv3d_Scaling_Tanh_Multiply_Sigmoid",
    "66_Matmul_Dropout_Mean_Softmax",
    "27_Conv3d_HardSwish_ReLU_Softmax_Mean",
    "75_Conv3d_GroupNorm_Meanbfloat16",
]


def init_pytorch_functionals():
    """Load functionals csv if not yet in memory"""
    global PYTORCH_FUNCTIONALS
    if PYTORCH_FUNCTIONALS is None:
        assert os.path.isfile(
            PYTORCH_FUNCTIONALS_PATH
        ), "KernelBench and robust_kbench tasks need the tasks/ directory, please clone KF and run from its root rather than pip installing."
        PYTORCH_FUNCTIONALS = pd.read_csv(PYTORCH_FUNCTIONALS_PATH, index_col="task_name")


def load_kernelbench_task_from_csv(task_name: str, config: DictConfig) -> str:
    """
    Load KernelBench task from the pytorch functionals csv
    Arguments:
        task_name (str): The name of the operation to get from the DB
        config: config parameters
    Returns:
        str: reference code
    """
    assert (
        task_name not in FILTERED_OUT
    ), f"Trying to get task {task_name} from csv, but this task was removed due to flaws!"
    # Load csv if not yet in memory
    init_pytorch_functionals()

    # get kernelbench task for this task_name
    if config.mode == "functional":
        code = PYTORCH_FUNCTIONALS.loc[task_name, "PyTorch_Code_Functional"]
    elif config.mode == "class":
        code = PYTORCH_FUNCTIONALS.loc[task_name, "PyTorch_Code_Module"]
    else:
        raise NotImplementedError("only class or functional modes are supported")

    level = PYTORCH_FUNCTIONALS.loc[task_name, "Level_ID"]
    return code, level


def load_kernelbench_task_from_db(task_name: str, config: DictConfig, origin: str = "KernelBench") -> str:
    """
    Load KernelBench task from the Tasks table in the database
    Arguments:
        task_name (str): The name of the operation to get from the DB
        config: config to initialize the database
        origin (str): task_origin (supported: {KernelBench, robust_kbench})
    Returns:
        str: reference code
    """
    assert (
        task_name not in FILTERED_OUT
    ), f"Trying to get task {task_name} from database, but this task was removed due to flaws!"
    # initialize database
    if not db.engine_readonly:
        db.init(config)
    # Read task from database
    eng = db.engine_readonly
    task = pd.read_sql(f"""SELECT * FROM tasks WHERE "task_name"='{task_name}' AND task_origin='{origin}'; """, eng)
    assert len(task) == 1, f"The database contains more than one task with name {task_name}"
    task_infos = task.iloc[0]
    reference = task_infos["reference_block"]
    level = task_infos["Level_ID"]
    return reference, level


def create_custom_task_files(tmpdir: str, config: DictConfig, task_name: str, reference: str, level: int):
    """
    Fill the KernelBench template with the given reference and config and other info, copy to tmpdir

    Args:
        tmpdir (str):  path to temporary directory to which the files will be copied
        config (DictConfig): configuration parameters
        task_name (str): name of the operation
        reference (str): reference code
        level (int): level of the operation
    """
    if "backward" in task_name.lower():
        task_template = BACKWARD_TASK_TEMPLATE
    else:
        task_template = KERNELBENCH_TASK_TEMPLATE

    def read_template(fn):
        with open(os.path.join(task_template, fn), "r") as inf:
            file_content = inf.read()
        return file_content

    def write_to_tmpdir(fn, content):
        with open(os.path.join(tmpdir, fn), "w") as outf:
            outf.write(content)

    # load config template and fill in information
    task_cfg = read_template("config.yaml")
    task_cfg = task_cfg.replace("job_name_PLACEHOLDER", config.job_name)
    task_cfg = task_cfg.replace("Op_name_PLACEHOLDER", task_name)
    task_cfg = task_cfg.replace("Op_ID_PLACEHOLDER", task_name.split("_")[0])
    gpu_arch = config.gpu_arch if isinstance(config.gpu_arch, str) else config.gpu_arch[0]
    task_cfg = task_cfg.replace("gpu_arch_PLACEHOLDER", gpu_arch)
    task_cfg = task_cfg.replace("language_PLACEHOLDER", config.language)
    level = 99 if level is None else level
    task_cfg = task_cfg.replace("level_PLACEHOLDER", str(level))
    write_to_tmpdir("config.yaml", task_cfg)

    # load task and insert reference
    task = read_template("task.py")
    fn_ending = FN_ENDING_DICT[config.language]
    task = task.replace("REFERENCE_PLACEHOLDER", reference).replace("PLACEHOLDER_FN_END", fn_ending)
    if config.language == "triton":
        # remove build function from task
        pattern = r"### remove for triton ###.*?### end remove ###"
        task = re.sub(pattern, "", task, flags=re.DOTALL)
    write_to_tmpdir("task.py", task)

    # copy file
    shutil.copyfile(os.path.join(task_template, "conftest.py"), os.path.join(tmpdir, "conftest.py"))

    # create new file with evolve block and save to pytorch_operation.sycl or pytorch_operation.cu
    with open(os.path.join(tmpdir, f"pytorch_operation.{fn_ending}"), "w") as outf:
        cm = COMMENT_DICT[config.language]
        outf.write(f"{cm} [EVOLVE_START]\n{cm} [EVOLVE_END]\n")

    if "backward" in task_name.lower():
        shutil.copyfile(os.path.join(task_template, "backward_eval.py"), os.path.join(tmpdir, "backward_eval.py"))


def kernelbench_reference_to_custom_task(task_name: str, config: DictConfig, reference: str, level: int) -> Task:
    """Convert KernelBench task for Task object"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy all files into temp directory and fill in reference and config
        create_custom_task_files(tmpdir, config, task_name, reference, level)
        # Create custom task
        ct, _ = Task.create(tmpdir)

    return ct


def load_kernelbench_task(
    config: DictConfig, task_name: str, from_db: bool = False, as_custom_task: bool = False, origin: str = "KernelBench"
):
    """
    Load KernelBench task either from database or csv file, return as string or as customTask
    Args:
        config: config with job parameters
        task_name (str): name of the operation
        from_db (bool): if True, load from db (otherwise load from csv)
        as_custom_task (bool): whether to return
        origin (str): task_origin if fetching from database (supported: {KernelBench, robust_kbench})
    Returns:
        task as str or Task
    """
    # load problem reference
    if from_db:
        reference, level = load_kernelbench_task_from_db(task_name, config, origin=origin)
    else:
        assert (
            origin == "KernelBench"
        ), f"Tasks from origin {origin} are not part of the pytorch_functionals csv. Set from_db=True to load from db."
        reference, level = load_kernelbench_task_from_csv(task_name, config)

    # add template_args:
    if config.mode == "functional":
        reference = replace_function_calls(reference)

    # convert to custom task
    if as_custom_task:
        return kernelbench_reference_to_custom_task(task_name, config, reference, level)
    else:
        return reference


def get_kernelbench_task_id(task_name: str):
    """Get ID of KernelBench task from database"""
    task_name = task_name.split(".")[0]
    # # if tasks are stored in database, return db ID
    eng = db.engine_readonly
    # tasks = pd.read_sql(f"""SELECT id FROM tasks WHERE "task_name"='{task_name}' """, eng)
    # assert len(tasks) == 1, f"Op name not found in tasks database or multiple entries (counted {len(tasks)})"
    # return tasks.iloc[0]["id"]
    return task_name


REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def assign_problem_hash(problem_path: str) -> list[int]:
    """
    Assign a unique hash to a problem in the dataset
    """
    with open(problem_path, "r") as f:
        problem_src = f.read()
    return get_code_hash(problem_src)


def get_id_to_name_mapping(level: int):
    init_pytorch_functionals()
    list_of_problems = PYTORCH_FUNCTIONALS[PYTORCH_FUNCTIONALS["Level_ID"] == level].index
    return {int(p.split("_")[0]): p + ".py" for p in list_of_problems}


def get_problem_list_by_level(level: int):
    init_pytorch_functionals()
    list_of_problems = PYTORCH_FUNCTIONALS[PYTORCH_FUNCTIONALS["Level_ID"] == level].index
    # filter out the tasks that were excluded due to flaws
    list_of_problems = [op for op in list_of_problems if op not in FILTERED_OUT]
    return [p + ".py" for p in list_of_problems]


def get_code_hash(problem_src: str) -> str:
    """
    Assign a unique hash to some piece of code
    Important to strip out the comments and whitespace as they are not functionally part of the code
    """
    # Remove multi-line comments first
    problem_src = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", problem_src)
    # Remove inline comments and all whitespace
    cleaned_problem_src = re.sub(r"#.*$|\s+", "", problem_src, flags=re.MULTILINE)
    # hash only on code
    return hashlib.md5(cleaned_problem_src.encode()).hexdigest()


def get_kernelbench_subset(data_config):
    """Construct task_set from configuration"""
    list_of_problems = get_problem_list_by_level(data_config.level)

    if data_config.use_representative:
        list_of_problems = level_to_representative[data_config.level]

    task_set = {int(p.split("_")[0]): p for p in list_of_problems}
    if data_config.subset[0] is not None and data_config.subset[1] is not None:
        start_id, end_id = data_config.subset
        task_set = {k: v for k, v in task_set.items() if start_id <= k <= end_id}

    # restrict to finetuning
    if data_config.use_finetune_test_set:
        test_set_for_level = level_to_finetune_test_set[data_config.level]
        task_set = {k: v for k, v in task_set.items() if k in test_set_for_level}

    return task_set


def get_robust_kbench_task_list(config: DictConfig) -> dict:
    """
    Get dictionary of available tasks for robust-kbench.
    Arguments:
        config: general configuration (for db init)
    Returns:
        dict: Dictionary of {Op_ID: task_name} with all tasks with origin robust_kbench
    """
    # initialize database
    if not db.engine_readonly:
        db.init(config)
    # Read task from database
    eng = db.engine_readonly
    tasks = pd.read_sql(f"""SELECT "Op_ID", "task_name" FROM tasks WHERE task_origin='robust_kbench' """, eng)
    tasks["Op_ID"] = tasks["Op_ID"].astype(int)
    # convert to dict
    task_set = tasks.set_index("Op_ID")["task_name"].to_dict()
    # filter subset
    if config.task_set.subset[0] is not None and config.task_set.subset[1] is not None:
        start_id, end_id = config.task_set.subset
        task_set = {k: v for k, v in task_set.items() if start_id <= k <= end_id}
    # filter representative
    if config.task_set.use_representative:
        representative_task_set = level_to_representative[config.task_set.level]
        task_set = {k: v for k, v in task_set.items() if v in representative_task_set}
    return task_set


################################################################################
# Representative subsets of KernelBench
# use this if you want to iterate on methods without the hassle of running the full dataset
# problem_ids are 1-indexed (logical index)
################################################################################

level1_representative_subset = [
    "4_Matrix_vector_multiplication_.py",
    "5_Matrix_scalar_multiplication.py",
    "7_Matmul_with_small_K_dimension_.py",
    "20_LeakyReLU.py",
    "21_Sigmoid.py",
    "25_Swish.py",
    "30_Softsign.py",
    "33_BatchNorm.py",
    "44_Average_Pooling_1D.py",
    "48_Mean_reduction_over_a_dimension.py",
    "53_Min_reduction_over_a_dimension.py",
    "64_conv_transposed_1D.py",
    "67_conv_standard_1D.py",
    "72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool.py",
    "76_conv_standard_1D_dilated_strided__.py",
    "82_conv_depthwise_2D_square_input_square_kernel.py",
    "86_conv_depthwise_separable_2D.py",
    "87_conv_pointwise_2D.py",
    "89_cumsum.py",
    "99_TripletMarginLoss.py",
]


level2_representative_subset = [
    "1_Conv2D_ReLU_BiasAdd.py",
    "5_ConvTranspose2d_Subtract_Tanh.py",
    "16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling.py",
    "17_Conv2d_InstanceNorm_Divide.py",
    "21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py",
    "24_Conv3d_Min_Softmax.py",
    "32_Conv2d_Scaling_Min.py",
    "35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py",
    "37_Matmul_Swish_Sum_GroupNorm.py",
    "46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py",
    "47_Conv3d_Mish_Tanh.py",
    "50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling.py",
    "59_Matmul_Swish_Scaling.py",
    "67_Conv2d_GELU_GlobalAvgPool.py",
    "70_Gemm_Sigmoid_Scaling_ResidualAdd.py",
    "73_Conv2d_BatchNorm_Scaling.py",
    "82_Conv2d_Tanh_Scaling_BiasAdd_Max.py",
    "85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py",
    "97_Matmul_BatchNorm_BiasAdd_Divide_Swish.py",
    "99_Matmul_GELU_Softmax.py",
]

level3_representative_subset = [
    "1_MLP.py",
    "5_AlexNet.py",
    "8_ResNetBasicBlock.py",
    "11_VGG16.py",
    "20_MobileNetV2.py",
    "21_EfficientNetMBConv.py",
    "33_VanillaRNN.py",
    "38_LTSMBidirectional.py",
    "43_MinGPTCausalAttention.py",
]


# subset of robust-kbench (the problems for which there are kernels published)
level0_representative_subset = [
    "layernorm_forward",
    "llama_ffw",
    "llama_rmsnorm_forward",
    "mnist_conv_relu_pool_forward",
    "mnist_cross_entropy_backward",
    "mnist_cross_entropy_forward",
    "mnist_linear_backward",
    "mnist_linear_forward",
    "mnist_linear_relu_backward",
    "mnist_linear_relu_forward",
    "mnist_pool_backward",
    "resnet_block",
]


level_to_representative = {
    0: level0_representative_subset,
    1: level1_representative_subset,
    2: level2_representative_subset,
    3: level3_representative_subset,
}

TEST_SET_LEVEL_1 = [9, 11, 16, 19, 28, 35, 37, 45, 46, 51, 53, 56, 63, 67, 68, 70, 72, 74, 80, 95]

level_to_finetune_test_set = {1: TEST_SET_LEVEL_1}

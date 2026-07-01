import autoroot
import os
import json
from collections import defaultdict
import pytest
import shutil
from hydra import initialize, compose
from omegaconf import open_dict

from kernelfoundry.algorithm.controller import Controller
from kernelfoundry.algorithm.utils.kernelbench_dataset import load_kernelbench_task
import kernelfoundry.eval_pipeline.database as db


# Function to set up the configuration for the test
def setup_cfg(cfg):
    # clean up existing test directory if it exists
    shutil.rmtree("runs/test_controller", ignore_errors=True)
    os.environ["OPENAI_API_KEY"] = "placeholder"  # no api key needed for test, but must be set

    # test on relu
    test_problem = 19

    # set run name and inference
    cfg.task_name = "19_ReLU"
    cfg.job_name = "test_controller"
    cfg.task_set.use_representative = False
    with open_dict(cfg):
        cfg.level = cfg.task_set.level
    cfg.max_iters = 1
    cfg.debug = True
    cfg.kernels_iter_0_path = os.path.join("tests", "relu_correct.sycl")

    cfg.language = "SYCL"
    cfg.gpu_arch = "dg2"
    desired_runtime = 15  # seconds

    cfg.eval_config.use_queue = True
    with open_dict(cfg):
        cfg.eval_config.language = cfg.language
    cfg.task_set.subset = [test_problem, test_problem]
    db.init(cfg)
    return desired_runtime


# Fixture to initialize and compose configuration
@pytest.fixture
def config():
    with initialize(version_base="1.3", config_path="../configs", job_name="test"):
        _cfg = compose(config_name="run.yaml")
    desired_runtime = setup_cfg(_cfg)
    return desired_runtime, _cfg


@pytest.fixture
def task(config):
    _, cfg = config
    relu_custom_task = load_kernelbench_task(
        cfg,
        "19_ReLU",
        from_db=False,
        as_custom_task=True,
        origin="KernelBench",
    )
    return relu_custom_task


# Test function to execute the test logic
def test_run_controller(task, config):
    desired_runtime, cfg = config

    # problem id
    test_problem = cfg.task_set.subset[0]

    # initialize controller
    controller = Controller(cfg, 0, 0)

    # run_single is a generator, and we want to check what's in the results
    controller.run_single(task)

    with open("runs/test_controller/results.json", "r") as f:
        results = json.load(f)

    results.pop("config", None)  # remove config for easier assertions

    assert len(results) == 1, f"Expected results for only one problem, but got {len(results)}"
    test_problem = list(results.keys())[0]

    # check
    assert len(results[test_problem]) == 1
    res_first_trial = results[test_problem][0]
    assert res_first_trial["status"] == "correct"
    assert res_first_trial["runtime"] > 0
    assert res_first_trial["perf_score"] == 5
    assert res_first_trial["runtime"] < desired_runtime
    print(results)
    print(results.keys())
    print(results[test_problem])

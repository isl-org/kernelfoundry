########################
# Utils Functions
########################

import pandas as pd
from typing import Tuple
from sqlalchemy import or_, func
from sqlalchemy.orm import load_only

import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.eval_pipeline.database.tables import Kernel, Task

################################################################################
# Scale up experiments in parallel
################################################################################


DEFAULT_COLUMNS_TO_LOAD = [
    "task_name",
    "Op_ID",
    "Level_ID",
    "job_name",
    "trial",
    "score",
    "status",
    "runtime",
    "runtime_stats",
    "improve_over_native",
    "language_model",
    "input_code",
    "output_language",
    "output_code",
    "eval_log",
    "gpu_arch",
    "optimization_profile",
]


def load_database(
    filter_by: dict,
    task_origin: str = None,
    max_improvement: float | None = None,
    min_improvement: float | None = None,
    job_options=None,
    limit: int | None = None,
):
    """Read database, filter with filter_by dict, and return as DataFrame"""
    # restrict to columns to reduce memory usage
    mapped_attr = [getattr(Kernel, f) for f in DEFAULT_COLUMNS_TO_LOAD]
    with db.SessionRO() as session:
        query = session.query(Kernel).options(load_only(*mapped_attr))

        if filter_by:
            query = query.filter_by(**filter_by)

        # Join with Task table and filter by task_origin if specified
        if task_origin:
            query = query.join(Task, Kernel.task_id == Task.id).filter(Task.task_origin == task_origin)

        if max_improvement is not None:
            query = query.filter(Kernel.improve_over_native <= max_improvement)

        if min_improvement is not None:
            query = query.filter(Kernel.improve_over_native >= min_improvement)

        if job_options:
            # For OR condition: job_name can be either A or B
            assert isinstance(job_options, list)
            query = query.filter(or_(*[Kernel.job_name == condition for condition in job_options]))

        if limit is not None:
            # apply random ordering then limit
            query = query.order_by(func.random()).limit(limit)

        # Fetch results
        kernel_results = pd.read_sql(query.statement, query.session.bind)

    return kernel_results


def load_best_kernel(
    problem_name: str, language: str, sortby: str = "improve_over_native", job_name: str = None
) -> Tuple[str, str]:
    """Load best kernel generated so far from database

    Args:
        problem_name (str): the name of the task
        language (str): output language
        sortby (str, optional): which field to use to determine best kernel. Defaults to "improve_over_native".

    Returns:
        Tuple[str, str]: code of best kernel, prior eval log
    """
    filtering = {"output_language": language, "task_name": problem_name}
    if job_name is not None:
        if "/" in job_name:
            job_name = job_name.split("/")
        else:
            job_name = [job_name]
    results_for_problem = load_database(filtering, job_options=job_name)

    if len(results_for_problem) == 0:
        # try with .py ending
        filtering["task_name"] = problem_name + ".py"
        results_for_problem = load_database(filtering, job_options=job_name)
        # still empty - error
        if len(results_for_problem) == 0:
            raise RuntimeError(f"Set kernels_iter_0_path=best, but no prior generation found for {problem_name}!")

    if sortby == "improve_over_native":
        results_for_problem.sort_values(by="improve_over_native", inplace=True, ascending=False)
    elif sortby == "runtime":
        results_for_problem.sort_values(by="runtime", inplace=True)
        if any(results_for_problem["runtime"] > 0):  # reduce to correct ones if there are any
            results_for_problem = results_for_problem[results_for_problem["runtime"] > 0]

    print("Runtime of best", results_for_problem.iloc[0][["runtime", "improve_over_native"]])
    return f"```\n{results_for_problem.iloc[0]['output_code']}```", results_for_problem.iloc[0]["eval_log"]

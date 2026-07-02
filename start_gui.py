from pathlib import Path
import tempfile
import autoroot
from nicegui import ui, app
from omegaconf import OmegaConf

import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.gui.job_logs import job_logs_detail_page, job_logs_main_page
from kernelfoundry.gui.kernel_graph import kernel_graph_page
from kernelfoundry.gui.kernel_detail import kernel_detail_page
from kernelfoundry.gui.roofline import roofline_page


def _init_database() -> None:
    cfg = OmegaConf.create({"paths": {"kernels_db_path": "sqlite:///runs/kernels.sqlite3"}})
    db.init(cfg)


def _load_favicon() -> str | None:
    favicon_path = Path(__file__).parent / "favicon.png"
    if not favicon_path.exists():
        return None
    try:
        import base64

        return "data:image/png;base64," + base64.b64encode(favicon_path.read_bytes()).decode()
    except (OSError, ValueError):
        return None


@ui.page("/")
def main_page() -> None:
    job_logs_main_page()


@ui.page("/job_logs/{job_id:int}")
def job_logs_page(job_id: int) -> None:
    job_logs_detail_page(job_id)


@ui.page("/graph")
def graph_page(
    job_name: str | None = None,
    task_name: str | None = None,
    user_id: str | None = None,
    job_id: str | None = None,
) -> None:
    kernel_graph_page(job_name=job_name, task_name=task_name, user_id=user_id, job_id=job_id)


@ui.page("/kernel/{kernel_id:int}")
def kernel_page(kernel_id: int) -> None:
    kernel_detail_page(kernel_id)


@ui.page("/roofline/{kernel_id:int}")
def roofline_page_route(kernel_id: int) -> None:
    roofline_page(kernel_id)


if __name__ == "__main__":
    _init_database()
    # ui.run(
    #     host="0.0.0.0",
    #     port=8889,
    #     reload=False,
    #     storage_secret="kernelfoundry_slim",
    #     title="KernelFoundry Viewer",
    #     show=False,
    #     loop="asyncio",
    #     favicon=_load_favicon(),
    # )

    with tempfile.TemporaryDirectory(prefix="webui_profiler_data_") as tmpdir:
        app.add_static_files("/profiler_data", tmpdir)

        from kernelfoundry.gui.perfetto import patch

        patch(tmpdir)

        ui.run(
            host="0.0.0.0",
            port=8885,
            reload=False,
            storage_secret="code_gen_kernel",
            title="KernelFoundry Viewer",
            show=False,
            loop="asyncio",  # use asyncio loop to avoid the buserror problem with uvloop
            favicon=_load_favicon(),
        )

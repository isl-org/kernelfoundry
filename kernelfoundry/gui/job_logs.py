"""Page for showing job logs (build logs, test logs, profiling logs) and a job overview page"""

import re
import traceback
from typing import Optional
from nicegui import ui
from nicegui.events import GenericEventArguments
from kernelfoundry.gui.utils import (
    get_jobs,
    get_jobs_by_user,
    get_job_logs_by_job_id,
    get_job_is_validate,
    get_total_jobs,
    get_total_jobs_by_user,
    archive_job_by_id,
    cancel_job_by_id,
)

PAGE_SIZE = 10

table_data = {
    "rows": [],
    "pagination": {
        "rowsPerPage": 10,
        "descending": True,
        "sortBy": "created_at",
        "page": 1,
        "rowsNumber": 0,
    },
}

ALL_STATUS_OPTIONS = ["INIT", "RUN", "COMPLETE", "FAIL", "VALIDATING", "VALIDATED", "CANCELED"]

current_statuses = None
current_user_id = None
jobs_container = None


def get_full_count():
    """Get the total count of jobs for pagination."""
    if current_user_id:
        full_count = get_total_jobs_by_user(current_user_id, statuses=current_statuses)
    else:
        full_count = get_total_jobs(statuses=current_statuses)
    table_data["pagination"]["rowsNumber"] = full_count
    jobs_table.refresh()


def get_rows(pagination):
    """Get paginated job rows."""
    page = pagination["page"]
    rpp = pagination["rowsPerPage"]
    sort_by = pagination.get("sortBy", "created_at")
    descending = pagination.get("descending", True)

    if current_user_id:
        jobs = get_jobs_by_user(
            current_user_id,
            statuses=current_statuses,
            offset=(page - 1) * rpp,
            limit=rpp,
            sort_by=sort_by,
            descending=descending,
        )
    else:
        jobs = get_jobs(
            statuses=current_statuses,
            offset=(page - 1) * rpp,
            limit=rpp,
            sort_by=sort_by,
            descending=descending,
        )

    rows = []
    for job in jobs:
        # Check if buttons should be enabled
        job_status = job.status or "Unknown"

        # Generate result link
        job_name = job.config.get("job_name") if job.config else None
        task_name = job.config.get("task_name") if job.config else None
        if job.config and (job_name is None or task_name is None):
            # fall back to old naming scheme:
            job_name = job.config.get("run_name") or job_name
            task_name = job.config.get("Op_Name") or task_name
        job_name, task_name = job_name or "N/A", task_name or "N/A"

        rows.append(
            {
                "id": job.id,
                "task_name": task_name,
                "job_name": job_name,
                "progress": f"{int(job.progress * 100)}%" if job.progress is not None else "N/A",
                "status": job_status,
                "created_at": job.created_at.strftime("%Y-%m-%d %H:%M") if job.created_at else "N/A",
                "started_at": job.started_at.strftime("%Y-%m-%d %H:%M") if job.started_at else "N/A",
                "finished_at": job.finished_at.strftime("%Y-%m-%d %H:%M") if job.finished_at else "N/A",
                "logs": (f"{job.id}"),
                "results": (f"{job.id}"),
                "archive": (f"{job.id}"),
            }
        )
    return rows


def on_request(e: GenericEventArguments) -> None:
    """Handle pagination requests."""
    new_pagination = e.args["pagination"]
    pagination = table_data["pagination"]
    pagination.update(new_pagination)
    new_rows = get_rows(pagination)
    table_data["rows"] = new_rows
    jobs_table.refresh()


@ui.refreshable
def jobs_table():
    """Refreshable jobs table with pagination."""
    columns = [
        {"name": "id", "label": "Job ID", "field": "id", "required": True, "align": "left", "sortable": True},
        {"name": "task_name", "label": "Task", "field": "task_name", "align": "left"},
        {"name": "job_name", "label": "Job Name", "field": "job_name", "align": "left"},
        {"name": "progress", "label": "Progress", "field": "progress", "align": "center"},
        {"name": "status", "label": "Status", "field": "status", "align": "center"},
        {"name": "created_at", "label": "Created", "field": "created_at", "align": "left", "sortable": True},
        {"name": "started_at", "label": "Started At", "field": "started_at", "align": "left", "sortable": True},
        {"name": "finished_at", "label": "Finished At", "field": "finished_at", "align": "center", "sortable": True},
        {"name": "logs", "label": "Logs", "field": "logs", "align": "center"},
        {"name": "results", "label": "Results", "field": "results", "align": "center"},
        {"name": "archive", "label": "Archive", "field": "archive", "align": "center"},
    ]

    table = ui.table(
        columns=columns, rows=table_data["rows"], row_key="id", pagination=table_data["pagination"]
    ).classes("w-full")
    table.on("request", on_request)

    # Add buttons to the table
    table.add_slot(
        "body-cell-logs",
        """
        <q-td :props="props">
            <q-btn 
                color="primary" 
                size="sm" 
                label="Logs"
                @click="$parent.$emit('logs-click', props.row.id)"
            />
        </q-td>
    """,
    )

    table.add_slot(
        "body-cell-results",
        """
        <q-td :props="props">
            <q-btn 
                color="secondary" 
                size="sm" 
                label="Results"
                @click="$parent.$emit('results-click', props.row.id)"
            />
        </q-td>
    """,
    )

    table.add_slot(
        "body-cell-finished_at",
        """
        <q-td :props="props">
            <q-btn 
                v-if="props.row.status === 'RUN' || props.row.status === 'VALIDATING'"
                color="warning" 
                size="sm" 
                icon="stop"
                round
                @click="$parent.$emit('cancel-click', props.row.id)"
            >
                <q-tooltip>Cancel Job</q-tooltip>
            </q-btn>
            <span v-else>{{ props.row.finished_at }}</span>
        </q-td>
    """,
    )

    table.add_slot(
        "body-cell-archive",
        """
        <q-td :props="props">
            <q-btn 
                color="negative" 
                size="sm" 
                icon="delete"
                round
                @click="$parent.$emit('archive-click', props.row.id)"
            >
                <q-tooltip>Archive Job</q-tooltip>
            </q-btn>
        </q-td>
    """,
    )

    # Handle button clicks
    def handle_logs_click(job_id):
        ui.navigate.to(f"/job_logs/{job_id}")

    def handle_cancel_click(job_id):
        """Cancel a job using the direct database function."""
        try:
            result = cancel_job_by_id(job_id)

            if result is None:
                ui.notify("Job not found", type="negative")
            elif result is False:
                ui.notify("Failed to cancel job", type="negative")
            else:
                ui.notify("Job canceled successfully", type="positive")
                # Refresh the table
                load_jobs()
        except Exception as e:
            ui.notify(f"Error canceling job: {str(e)}", type="negative")

    def handle_archive_click(job_id):
        """Archive a job using the direct database function."""
        try:
            result = archive_job_by_id(job_id)

            if result is None:
                ui.notify("Job not found", type="negative")
            elif result is False:
                ui.notify("Failed to archive job", type="negative")
            else:
                ui.notify("Job archived successfully", type="positive")
                # Refresh the table
                load_jobs()
        except Exception as e:
            ui.notify(f"Error archiving job: {str(e)}", type="negative")

    table.on("logs-click", lambda e: handle_logs_click(e.args))
    table.on(
        "results-click",
        lambda e: ui.navigate.to(f"/graph?job_id={e.args}"),
    )
    table.on("cancel-click", lambda e: handle_cancel_click(e.args))
    table.on("archive-click", lambda e: handle_archive_click(e.args))


def load_jobs():
    """Load jobs with pagination."""
    global current_statuses
    if current_statuses is None:
        current_statuses = ALL_STATUS_OPTIONS.copy()

    try:
        # Get total count and update pagination
        get_full_count()

        # Reset to first page and load initial data
        table_data["pagination"]["page"] = 1
        rows = get_rows(table_data["pagination"])
        table_data["rows"] = rows

        # Clear container and show table
        jobs_container.clear()
        with jobs_container:
            if not rows and table_data["pagination"]["rowsNumber"] == 0:
                ui.label("No jobs found with the selected status filter.").classes("text-gray-500")
            else:
                jobs_table()

    except Exception as e:
        jobs_container.clear()
        with jobs_container:
            ui.label(f"Error loading jobs: {str(e)}").classes("text-red-500")


def job_logs_main_page(user_id: Optional[str] = None):
    """Main page showing list of all jobs with logs."""
    global jobs_container, current_user_id

    current_user_id = user_id

    saved_statuses = current_statuses if current_statuses is not None else ALL_STATUS_OPTIONS.copy()

    def on_status_change():
        global current_statuses
        value = status_filter.value or []
        current_statuses = value if isinstance(value, list) else [value]
        load_jobs()

    # Header with title and action buttons
    with ui.row().classes("gap-50 items-center"):
        ui.label("Job Logs").classes("text-2xl font-bold")
        with ui.row().classes("gap-3 items-center"):
            # ui.button("Results page", icon="analytics", on_click=lambda: ui.navigate.to("/graph")).props(
            #     "color=primary outlined"
            # )

            # Filter button with popup menu
            with ui.button(icon="filter_list").props("color=primary outlined flat").classes("px-2"):
                with ui.menu().props("auto-close=false"):
                    with ui.card().classes("p-4 min-w-52"):
                        ui.label("Filter by status").classes("text-sm font-semibold mb-2")
                        status_filter = ui.select(
                            options=ALL_STATUS_OPTIONS,
                            value=saved_statuses,
                            multiple=True,
                            clearable=False,
                        ).classes("w-full")
                        status_filter.on_value_change(on_status_change)

    jobs_container = ui.column().classes("w-full")

    # Load jobs on page open using the persisted filter
    load_jobs()


def job_logs_detail_page(job_id: int):
    """Detailed page showing job logs for a specific job."""
    try:
        # Get job information to determine if it's a validation job
        is_validation_job = get_job_is_validate(job_id)

        logs = get_job_logs_by_job_id(job_id)

        if not logs:
            ui.label(f"No logs found for job ID {job_id}").classes("text-xl text-red-500")
            return

        ui.label(f"Job Logs - Job ID {job_id}").classes("text-2xl font-bold mb-4")

        # Group logs by message type with reference/custom separation
        log_groups = {
            "build": {"reference": [], "custom": []},
            "test_correctness": {"reference": [], "custom": []},
            "test_performance": {"reference": [], "custom": []},
            "profiling": {"reference": [], "custom": []},
            "session_log": {"custom": []},
        }
        other_logs = []

        for log in logs:
            message = log.message.lower() if log.message else ""
            log_entry = {"timestamp": log.timestamp, "level": log.level, "message": log.message}

            # Extract log content from extra data
            if log.extra and isinstance(log.extra, dict) and len(log.extra) > 0 and "log" in log.extra:
                log_content, worker_info = log.extra["log"], log.extra.get("worker_info", {})
                data_info = {
                    "content": log_content + f"\nWorker info: {worker_info}",
                    "trial": log.extra.get("trial", 0),
                    "version": log.extra.get("version", 0),
                    "host": worker_info.get("hostname", "unknown"),
                    "session_log": log.extra.get("log", {}),
                    "agent_session_id": getattr(log, "agent_session_id", "N/A"),
                }
                log_entry.update(data_info)
            else:
                other_logs.append(log_entry)
                continue

            # Define prefix mappings: prefix -> (category, type)
            prefix_mappings = {
                "build_reference": ("build", "reference"),
                "build": ("build", "custom"),
                "test_reference_correctness": ("test_correctness", "reference"),
                "test_custom_correctness": ("test_correctness", "custom"),
                "test_reference_performance": ("test_performance", "reference"),
                "test_custom_performance": ("test_performance", "custom"),
                "test_reference_trace": ("profiling", "reference"),
                "test_custom_trace": ("profiling", "custom"),
                "session_log": ("session_log", "custom"),
            }

            for prefix, (category, log_type) in prefix_mappings.items():
                if message.startswith(prefix):
                    log_groups[category][log_type].append(log_entry)
                    break

        # Sort logs by timestamp descending (most recent first)
        for category in log_groups:
            for log_type in log_groups[category]:
                log_groups[category][log_type] = sorted(
                    log_groups[category][log_type], key=lambda x: x["timestamp"] or ""
                )

        # Create summary section
        create_summary_section(log_groups, is_validation_job)

        # Create tabs for different log types
        with ui.tabs().classes("w-full") as tabs:
            main_log_tab = ui.tab("General logs")
            build_tab = ui.tab("Build")
            test_correctness_tab = ui.tab("Test Correctness")
            test_performance_tab = ui.tab("Test Performance")
            profiling_tab = ui.tab("Profiling")
            session_log_tab = ui.tab("Agent Session Logs")

        with ui.tab_panels(tabs, value=main_log_tab).classes("w-full"):
            with ui.tab_panel(main_log_tab):
                create_log_section("General", other_logs)

            with ui.tab_panel(build_tab):
                create_side_by_side_log_section("Build Logs", log_groups["build"])

            with ui.tab_panel(test_correctness_tab):
                create_side_by_side_log_section("Test Correctness Logs", log_groups["test_correctness"])

            with ui.tab_panel(test_performance_tab):
                create_side_by_side_log_section("Test Performance Logs", log_groups["test_performance"])

            with ui.tab_panel(profiling_tab):
                create_side_by_side_log_section("Profiling Logs", log_groups["profiling"])

            with ui.tab_panel(session_log_tab):
                create_agent_session_log_section("Agent Session Logs", log_groups["session_log"]["custom"])

    except Exception as e:
        ui.label(f"Error loading logs for job {job_id}: {str(e)}\n\n{traceback.format_exc()}").classes(
            "text-xl text-red-500 whitespace-pre-wrap"
        )


def create_log_section(title, logs):
    """Create a section displaying logs."""
    if not logs:
        ui.label(f"No {title.lower()} found").classes("text-gray-500")
        return

    ui.label(f"{title}").classes("text-lg font-bold mb-2")

    # Sort logs by timestamp (most recent first)
    sorted_logs = sorted(logs, key=lambda x: x["timestamp"] or "")

    # Create formatted log entries
    log_lines = []
    for log_entry in sorted_logs:
        timestamp_str = log_entry["timestamp"].strftime("%H:%M:%S") if log_entry["timestamp"] else "00:00:00"
        level_str = log_entry["level"] or "UNKNOWN"
        message_str = log_entry["message"] or "No message"
        log_lines.append(f"[{timestamp_str}] [{level_str}] {message_str}")

    # Display all logs in a single box
    with ui.card().classes("w-full"):
        with ui.card_section():
            if log_lines:
                ui.code("\n".join(log_lines)).classes("whitespace-pre text-sm w-full")
            else:
                ui.label("No log entries available").classes("text-gray-400 italic")


def create_agent_session_log_section(title, logs):
    """Create a section displaying logs."""
    if not logs:
        ui.label(f"No {title.lower()} found").classes("text-gray-500")
        return

    ui.label(f"{title} ({len(logs)} entries)").classes("text-lg font-bold mb-2")

    # Sort logs by timestamp (most recent first)
    sorted_logs = sorted(logs, key=lambda x: x["timestamp"] or "")

    # Create a separate card for each log entry
    for log_entry in sorted_logs:
        timestamp_str = log_entry["timestamp"].strftime("%H:%M:%S") if log_entry["timestamp"] else "00:00:00"
        session_id = log_entry["agent_session_id"]
        message_str = log_entry["session_log"][:10000]

        with ui.card().classes("w-full mb-2"):
            with ui.card_section():
                # Title with timestamp and session id
                title_str = f"{timestamp_str} | Session ID: {session_id}"
                ui.label(title_str).classes("text-sm font-medium text-gray-700 mb-2")

                # Log content spanning the whole card
                if message_str:
                    ui.code(message_str, language="markdown").classes("whitespace-pre-wrap text-sm")
                else:
                    ui.label("No log content available").classes("text-gray-400 italic")


def create_side_by_side_log_section(title, log_data):
    """Create a section displaying logs side by side (reference vs custom)."""
    reference_logs = log_data["reference"]
    custom_logs = log_data["custom"]

    if not reference_logs and not custom_logs:
        ui.label(f"No {title.lower()} found").classes("text-gray-500")
        return

    ui.label(f"{title} (Reference: {len(reference_logs)}, Custom: {len(custom_logs)} entries)").classes(
        "text-lg font-bold mb-2"
    )

    def make_title(log_entry):
        # Title with timestamp, trial, and version
        timestamp = log_entry["timestamp"].strftime("%H:%M:%S")
        host = log_entry.get("host", "unknown")
        title = f"{timestamp} | Trial: {log_entry['trial']} | Version: {log_entry['version']} | Host: {host}"
        return title

    with ui.row().classes("w-full gap-4"):
        # Reference logs on the left
        with ui.column().classes("flex-1"):
            ui.label("Reference").classes("text-lg font-semibold text-blue-600 mb-2")
            if reference_logs:
                for log_entry in reference_logs:
                    with ui.card().classes("w-full mb-2"):
                        with ui.card_section():
                            # Title with timestamp, trial, and version
                            title = make_title(log_entry)
                            ui.label(title).classes("text-sm font-medium text-gray-700 mb-2")

                            # Log content spanning the whole card
                            if log_entry["content"]:
                                ui.label(log_entry["content"]).classes("whitespace-pre-wrap text-sm")
                            else:
                                ui.label("No log content available").classes("text-gray-400 italic")
            else:
                ui.label("No reference logs available").classes("text-gray-400 italic")

        # Custom logs on the right
        with ui.column().classes("flex-1"):
            ui.label("Custom").classes("text-lg font-semibold text-green-600 mb-2")
            if custom_logs:
                for log_entry in custom_logs:
                    with ui.card().classes("w-full mb-2"):
                        with ui.card_section():
                            # Title with timestamp, trial, and version
                            title = make_title(log_entry)
                            ui.label(title).classes("text-sm font-medium text-gray-700 mb-2")

                            # Log content spanning the whole card
                            if log_entry["content"]:
                                ui.label(log_entry["content"]).classes("whitespace-pre-wrap text-sm")
                            else:
                                ui.label("No log content available").classes("text-gray-400 italic")
            else:
                ui.label("No custom logs available").classes("text-gray-400 italic")


def get_status_from_logs(logs, extract_runtime=False):
    """Determine status from log entries by checking for return codes."""
    if not logs:
        return {"status": "empty", "runtime": None} if extract_runtime else "empty"

    # Use only the first log entry for the summary status
    log_entry = logs[0]

    status = "success"  # Default if logs exist
    runtime = None

    # Look for return code patterns and runtime in log content
    content = log_entry.get("content", "").lower()
    if "returncode=0" in content or "return code: 0" in content or "completed successfully" in content:
        status = "success"
    elif (
        "returncode=1" in content
        or "return code: 1" in content
        or any(
            error_pattern in content.lower().replace("standard error", "")
            for error_pattern in ["error", "failed", "failure", "exception"]
        )
    ):
        status = "failed"

    # Extract runtime for performance logs
    if extract_runtime and content:
        runtime_pattern = r"Avg:\s*([0-9]+\.?[0-9]*)\s*ms"
        matches = re.findall(runtime_pattern, content)
        if matches:
            runtimes = [float(m) for m in matches]
            runtime = sum(runtimes) / len(runtimes)

    if extract_runtime:
        return {"status": status, "runtime": runtime}
    return status


def create_summary_section(log_groups, is_validation_job=False):
    """Create a summary section showing status for each log category."""
    categories = ["build", "test_correctness", "test_performance"]
    category_labels = ["Build", "Correctness", "Performance"]

    # find latest trial
    max_trial_ref, max_trial_custom = 0, 0
    for category in categories:
        trials = [entry.get("trial", 0) or 0 for entry in log_groups[category]["reference"]]
        max_trial_ref = max(max_trial_ref, max(trials) if trials else 0)
        trials = [entry.get("trial", 0) or 0 for entry in log_groups[category]["custom"]]
        max_trial_custom = max(max_trial_custom, max(trials) if trials else 0)

    # Pre-calculate all statuses and runtimes
    ref_results = {}
    custom_results = {}

    for category in categories:
        # Reference logs
        ref_logs = [log for log in log_groups[category]["reference"] if (log.get("trial", 0) or 0) == max_trial_ref]
        if category == "test_performance":
            result = get_status_from_logs(ref_logs, extract_runtime=True)
            ref_results[category] = {"status": result["status"], "runtime": result["runtime"]}
        else:
            ref_results[category] = {"status": get_status_from_logs(ref_logs), "runtime": None}

        # Custom logs
        custom_logs = [log for log in log_groups[category]["custom"] if log.get("trial", 0) == max_trial_custom]
        if category == "test_performance":
            result = get_status_from_logs(custom_logs, extract_runtime=True)
            custom_results[category] = {"status": result["status"], "runtime": result["runtime"]}
        else:
            custom_results[category] = {"status": get_status_from_logs(custom_logs), "runtime": None}

    # Set the title based on job type
    if is_validation_job:
        # if reference correct: Set to valid
        build_status = ref_results["build"]["status"]
        correctness_status = ref_results["test_correctness"]["status"]
        is_valid = build_status != "failed" and correctness_status != "failed"
        validation_result = "valid" if is_valid else "invalid"

        with ui.row().classes("items-center gap-2 mb-4"):
            color_class = "text-green-600" if is_valid else "text-red-600"
            ui.label(f"Summary: {validation_result}").classes(f"text-xl font-bold {color_class}")
            info_icon = ui.icon("info", size="sm").classes("text-gray-500 cursor-pointer")
            with info_icon:
                ui.tooltip("A job is valid when both the reference build and correctness tests pass successfully")
    else:
        ui.label("Summary (latest results)").classes("text-xl font-bold mb-4")

    with ui.card().classes("w-full mb-6"):
        with ui.card_section():
            # Create table-like layout using grid
            with ui.column().classes("w-full"):
                # Header row
                with ui.row().classes("w-full mb-4"):
                    with ui.column().classes("w-32"):  # Fixed width for row labels
                        ui.label("").classes("text-transparent")  # Empty corner
                    for label in category_labels:
                        with ui.column().classes("flex-1 text-center"):
                            ui.label(label).classes("font-semibold text-gray-700")

                # Reference row
                with ui.row().classes("w-full mb-3"):
                    with ui.column().classes("w-32"):  # Same fixed width
                        ui.label("Reference").classes("text-lg font-semibold text-blue-600")

                    for category in categories:
                        with ui.column().classes("flex-1 text-center"):
                            status = ref_results[category]["status"]
                            runtime = ref_results[category]["runtime"]

                            # Status icon
                            if status == "success" and not runtime:
                                ui.icon("check_circle", color="green").classes("text-xl mx-auto")
                            elif status == "success" and runtime:
                                ui.label(f"{runtime:.2f}ms").classes("text-gray-400 text-xl font-bold mx-auto")
                            elif status == "failed":
                                ui.icon("cancel", color="red").classes("text-xl mx-auto")
                            else:
                                ui.label("−").classes("text-gray-400 text-xl font-bold mx-auto")

                # Custom row
                with ui.row().classes("w-full"):
                    with ui.column().classes("w-32"):  # Same fixed width
                        ui.label("Custom").classes("text-lg font-semibold text-green-600")

                    for category in categories:
                        with ui.column().classes("flex-1 text-center"):
                            status = custom_results[category]["status"]
                            runtime = custom_results[category]["runtime"]

                            # Status icon
                            if status == "success" and not runtime:
                                ui.icon("check_circle", color="green").classes("text-xl mx-auto")
                            elif status == "success" and runtime:
                                ui.label(f"{runtime:.2f}ms").classes("text-gray-400 text-xl font-bold mx-auto")
                            elif status == "failed":
                                ui.icon("cancel", color="red").classes("text-xl mx-auto")
                            else:
                                ui.label("−").classes("text-gray-400 text-xl font-bold mx-auto")

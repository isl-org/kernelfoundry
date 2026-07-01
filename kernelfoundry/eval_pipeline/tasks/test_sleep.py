"""Simple test task that sleeps for a given duration."""

from celery import shared_task
import logging

from kernelfoundry.eval_pipeline.utils.sysinfo import get_worker_info


@shared_task(track_started=True)
def test_sleep(duration: float, message: str) -> str:
    import time

    time.sleep(duration)
    worker_info = get_worker_info()
    return message

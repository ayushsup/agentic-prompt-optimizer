"""Async utilities with retry logic."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


async def run_asyncio_task_with_retry(
    task_factory: Callable[[], Awaitable[T]],
    n_max_retries: int,
    sleep_base_seconds_for_retry: float,
    max_sleep_time_seconds: float,
    timeout_seconds: float | None,
) -> T:
    """Retry async task with exponential backoff.

    Args:
        task_factory: A callable that returns a fresh coroutine for each retry
        n_max_retries: Maximum number of retry attempts
        sleep_base_seconds_for_retry: Base sleep time between retries (multiplied by attempt number)
        max_sleep_time_seconds: Maximum sleep time between retries
        timeout_seconds: Timeout for each individual attempt
    """
    for i in range(n_max_retries):
        if i > 0:
            sleep_time = min(
                sleep_base_seconds_for_retry * i,
                max_sleep_time_seconds,
            )
            logger.info(f"Sleeping for {sleep_time} seconds before retry...")
            await asyncio.sleep(sleep_time)
        try:
            try:
                response = await asyncio.wait_for(
                    task_factory(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout after {timeout_seconds} seconds while waiting for task."
                )
                raise
            return response
        except Exception as e:
            logger.exception(f"Error in task: {e}")
            if i == n_max_retries - 1:
                raise
            logger.info(f"Retrying {i + 1} of {n_max_retries}...")
            continue

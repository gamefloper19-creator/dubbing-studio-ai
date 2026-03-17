"""
Batch video dubbing processor with queue management.

Supports processing 20-25 videos simultaneously with
progress monitoring and error recovery.
"""

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

from dubbing_studio.config import BatchConfig

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a batch job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Represents a single video dubbing job in the batch queue."""
    job_id: str
    video_path: str
    target_language: str
    narrator_style: str = "documentary"
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 to 1.0
    current_stage: str = ""
    output_path: str = ""
    error_message: str = ""
    retry_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    estimated_remaining: float = 0.0


@dataclass
class BatchProgress:
    """Overall batch processing progress."""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    active_jobs: int = 0
    queued_jobs: int = 0
    overall_progress: float = 0.0
    estimated_total_remaining: float = 0.0


class BatchProcessor:
    """
    Process multiple videos in batch with queue management.

    Features:
    - Concurrent processing (configurable)
    - Progress monitoring
    - Error recovery with retries
    - Automatic export
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self._jobs: dict[str, BatchJob] = {}
        self._lock = Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._progress_callback: Optional[Callable] = None

    def add_job(
        self,
        video_path: str,
        target_language: str,
        narrator_style: str = "documentary",
        job_id: Optional[str] = None,
    ) -> str:
        """
        Add a video to the processing queue.

        Args:
            video_path: Path to video file.
            target_language: Target language code.
            narrator_style: Narration style.
            job_id: Optional custom job ID.

        Returns:
            Job ID.
        """
        if job_id is None:
            job_id = f"job_{len(self._jobs) + 1:03d}_{int(time.time())}"

        job = BatchJob(
            job_id=job_id,
            video_path=video_path,
            target_language=target_language,
            narrator_style=narrator_style,
        )

        with self._lock:
            self._jobs[job_id] = job

        logger.info("Added job %s: %s -> %s", job_id, video_path, target_language)
        return job_id

    def add_videos(
        self,
        video_paths: list[str],
        target_language: str,
        narrator_style: str = "documentary",
    ) -> list[str]:
        """
        Add multiple videos to the queue.

        Args:
            video_paths: List of video file paths.
            target_language: Target language code.
            narrator_style: Narration style.

        Returns:
            List of job IDs.
        """
        job_ids = []
        for path in video_paths:
            job_id = self.add_job(path, target_language, narrator_style)
            job_ids.append(job_id)
        return job_ids

    def process_all(
        self,
        dubbing_function: Callable,
        progress_callback: Optional[Callable] = None,
    ) -> list[BatchJob]:
        """
        Process all queued jobs.

        Args:
            dubbing_function: Function that takes (video_path, target_language,
                            narrator_style, progress_callback) and returns output_path.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of completed BatchJob objects.
        """
        self._progress_callback = progress_callback
        max_workers = max(1, min(self.config.max_concurrent, len(self._jobs)))

        logger.info(
            "Starting batch processing: %d jobs, %d concurrent",
            len(self._jobs), max_workers,
        )

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}

        with self._lock:
            for job_id, job in self._jobs.items():
                if job.status == JobStatus.QUEUED:
                    future = self._executor.submit(
                        self._process_job,
                        job,
                        dubbing_function,
                    )
                    futures[future] = job_id

        # Wait for all to complete
        for future in as_completed(futures):
            job_id = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error("Job %s failed: %s", job_id, e)

        self._executor.shutdown(wait=True)

        return list(self._jobs.values())

    def _process_job(
        self,
        job: BatchJob,
        dubbing_function: Callable,
    ) -> None:
        """Process a single job with retry logic."""
        job.status = JobStatus.PROCESSING
        job.start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    job.status = JobStatus.RETRYING
                    job.retry_count = attempt
                    logger.info(
                        "Retrying job %s (attempt %d/%d)",
                        job.job_id, attempt + 1, self.config.max_retries + 1,
                    )

                def job_progress(stage: str, progress: float) -> None:
                    job.current_stage = stage
                    job.progress = progress
                    elapsed = time.time() - job.start_time
                    if progress > 0:
                        job.estimated_remaining = (elapsed / progress) * (1 - progress)
                    if self._progress_callback:
                        self._progress_callback(self.get_progress())

                output_path = dubbing_function(
                    job.video_path,
                    job.target_language,
                    job.narrator_style,
                    job_progress,
                )

                job.output_path = output_path
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.end_time = time.time()
                job.current_stage = "Completed"

                logger.info(
                    "Job %s completed in %.1fs: %s",
                    job.job_id,
                    job.end_time - job.start_time,
                    output_path,
                )

                if self._progress_callback:
                    self._progress_callback(self.get_progress())

                return

            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                logger.error("Job %s attempt %d failed: %s", job.job_id, attempt + 1, e)

                if attempt >= self.config.max_retries or not self.config.retry_on_failure:
                    job.status = JobStatus.FAILED
                    job.error_message = error_msg
                    job.end_time = time.time()

                    if self._progress_callback:
                        self._progress_callback(self.get_progress())
                    return

                time.sleep(2 ** attempt)  # exponential backoff

    def get_progress(self) -> BatchProgress:
        """Get current batch processing progress."""
        with self._lock:
            total = len(self._jobs)
            completed = sum(
                1 for j in self._jobs.values()
                if j.status == JobStatus.COMPLETED
            )
            failed = sum(
                1 for j in self._jobs.values()
                if j.status == JobStatus.FAILED
            )
            active = sum(
                1 for j in self._jobs.values()
                if j.status in (JobStatus.PROCESSING, JobStatus.RETRYING)
            )
            queued = sum(
                1 for j in self._jobs.values()
                if j.status == JobStatus.QUEUED
            )

            # Calculate overall progress
            if total > 0:
                progress_sum = sum(j.progress for j in self._jobs.values())
                overall = progress_sum / total
            else:
                overall = 0.0

            # Estimate remaining time
            remaining_estimates = [
                j.estimated_remaining
                for j in self._jobs.values()
                if j.status in (JobStatus.PROCESSING, JobStatus.RETRYING)
                and j.estimated_remaining > 0
            ]
            total_remaining = max(remaining_estimates) if remaining_estimates else 0.0

            return BatchProgress(
                total_jobs=total,
                completed_jobs=completed,
                failed_jobs=failed,
                active_jobs=active,
                queued_jobs=queued,
                overall_progress=overall,
                estimated_total_remaining=total_remaining,
            )

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a specific job by ID."""
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> list[BatchJob]:
        """Get all jobs."""
        return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                return True
        return False

    def clear_completed(self) -> None:
        """Remove completed and failed jobs from the queue."""
        with self._lock:
            self._jobs = {
                k: v for k, v in self._jobs.items()
                if v.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            }

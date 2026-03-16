"""Tests for batch processing module."""

from dubbing_studio.batch.processor import (
    BatchJob,
    BatchProcessor,
    BatchProgress,
    JobStatus,
)
from dubbing_studio.config import BatchConfig


class TestJobStatus:
    def test_statuses(self):
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.RETRYING.value == "retrying"
        assert JobStatus.CANCELLED.value == "cancelled"


class TestBatchJob:
    def test_defaults(self):
        job = BatchJob(
            job_id="test_001",
            video_path="/path/to/video.mp4",
            target_language="hi",
        )
        assert job.job_id == "test_001"
        assert job.status == JobStatus.QUEUED
        assert job.progress == 0.0
        assert job.retry_count == 0


class TestBatchProgress:
    def test_defaults(self):
        progress = BatchProgress()
        assert progress.total_jobs == 0
        assert progress.completed_jobs == 0
        assert progress.overall_progress == 0.0


class TestBatchProcessor:
    def setup_method(self):
        self.processor = BatchProcessor()

    def test_add_job(self):
        job_id = self.processor.add_job("/video.mp4", "hi")
        assert job_id is not None
        job = self.processor.get_job(job_id)
        assert job is not None
        assert job.video_path == "/video.mp4"
        assert job.target_language == "hi"

    def test_add_job_custom_id(self):
        job_id = self.processor.add_job("/video.mp4", "es", job_id="custom_id")
        assert job_id == "custom_id"

    def test_add_videos(self):
        paths = ["/v1.mp4", "/v2.mp4", "/v3.mp4"]
        job_ids = self.processor.add_videos(paths, "fr")
        assert len(job_ids) == 3

    def test_get_all_jobs(self):
        self.processor.add_job("/v1.mp4", "hi")
        self.processor.add_job("/v2.mp4", "es")
        jobs = self.processor.get_all_jobs()
        assert len(jobs) == 2

    def test_cancel_job(self):
        job_id = self.processor.add_job("/video.mp4", "hi")
        assert self.processor.cancel_job(job_id) is True
        job = self.processor.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    def test_cancel_non_queued_job(self):
        job_id = self.processor.add_job("/video.mp4", "hi")
        job = self.processor.get_job(job_id)
        job.status = JobStatus.PROCESSING
        assert self.processor.cancel_job(job_id) is False

    def test_cancel_nonexistent_job(self):
        assert self.processor.cancel_job("nonexistent") is False

    def test_clear_completed(self):
        j1 = self.processor.add_job("/v1.mp4", "hi")
        self.processor.add_job("/v2.mp4", "es")
        self.processor.get_job(j1).status = JobStatus.COMPLETED
        self.processor.clear_completed()
        assert len(self.processor.get_all_jobs()) == 1

    def test_get_progress_empty(self):
        progress = self.processor.get_progress()
        assert progress.total_jobs == 0
        assert progress.overall_progress == 0.0

    def test_get_progress_with_jobs(self):
        j1 = self.processor.add_job("/v1.mp4", "hi")
        self.processor.add_job("/v2.mp4", "es")
        self.processor.get_job(j1).status = JobStatus.COMPLETED
        self.processor.get_job(j1).progress = 1.0
        progress = self.processor.get_progress()
        assert progress.total_jobs == 2
        assert progress.completed_jobs == 1
        assert progress.queued_jobs == 1

    def test_process_all_success(self):
        self.processor.add_job("/v1.mp4", "hi")

        def mock_dub(video_path, target_lang, style, progress_cb):
            progress_cb("Testing", 0.5)
            progress_cb("Done", 1.0)
            return "/output/v1_dubbed.mp4"

        results = self.processor.process_all(mock_dub)
        assert len(results) == 1
        assert results[0].status == JobStatus.COMPLETED
        assert results[0].output_path == "/output/v1_dubbed.mp4"

    def test_process_all_failure_with_retry(self):
        config = BatchConfig(max_retries=1, retry_on_failure=True)
        processor = BatchProcessor(config)
        processor.add_job("/v1.mp4", "hi")

        call_count = 0

        def failing_dub(video_path, target_lang, style, progress_cb):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Test failure")

        results = processor.process_all(failing_dub)
        assert results[0].status == JobStatus.FAILED
        assert call_count == 2  # initial + 1 retry

    def test_custom_config(self):
        config = BatchConfig(max_concurrent=8, max_retries=5)
        processor = BatchProcessor(config)
        assert processor.config.max_concurrent == 8
        assert processor.config.max_retries == 5

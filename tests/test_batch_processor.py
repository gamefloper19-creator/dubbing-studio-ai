"""Tests for batch processing module."""

import time
from unittest.mock import MagicMock

from dubbing_studio.batch.processor import (
    BatchJob,
    BatchProcessor,
    BatchProgress,
    JobStatus,
)
from dubbing_studio.config import BatchConfig


class TestJobStatus:
    def test_all_statuses(self):
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
            video_path="/tmp/video.mp4",
            target_language="hi",
        )
        assert job.job_id == "test_001"
        assert job.video_path == "/tmp/video.mp4"
        assert job.target_language == "hi"
        assert job.narrator_style == "documentary"
        assert job.status == JobStatus.QUEUED
        assert job.progress == 0.0
        assert job.retry_count == 0


class TestBatchProgress:
    def test_defaults(self):
        p = BatchProgress()
        assert p.total_jobs == 0
        assert p.completed_jobs == 0
        assert p.failed_jobs == 0
        assert p.overall_progress == 0.0


class TestBatchProcessor:
    def test_add_job(self):
        bp = BatchProcessor()
        job_id = bp.add_job("/tmp/v1.mp4", "hi")
        assert job_id is not None
        assert bp.get_job(job_id) is not None
        assert bp.get_job(job_id).target_language == "hi"

    def test_add_job_custom_id(self):
        bp = BatchProcessor()
        job_id = bp.add_job("/tmp/v1.mp4", "es", job_id="custom_001")
        assert job_id == "custom_001"

    def test_add_videos(self):
        bp = BatchProcessor()
        ids = bp.add_videos(["/tmp/v1.mp4", "/tmp/v2.mp4"], "fr")
        assert len(ids) == 2
        assert len(bp.get_all_jobs()) == 2

    def test_cancel_job(self):
        bp = BatchProcessor()
        jid = bp.add_job("/tmp/v1.mp4", "hi")
        assert bp.cancel_job(jid) is True
        assert bp.get_job(jid).status == JobStatus.CANCELLED

    def test_cancel_nonexistent(self):
        bp = BatchProcessor()
        assert bp.cancel_job("nonexistent") is False

    def test_clear_completed(self):
        bp = BatchProcessor()
        j1 = bp.add_job("/tmp/v1.mp4", "hi")
        j2 = bp.add_job("/tmp/v2.mp4", "es")
        bp.get_job(j1).status = JobStatus.COMPLETED
        bp.clear_completed()
        assert len(bp.get_all_jobs()) == 1

    def test_get_progress_empty(self):
        bp = BatchProcessor()
        progress = bp.get_progress()
        assert progress.total_jobs == 0
        assert progress.overall_progress == 0.0

    def test_get_progress_with_jobs(self):
        bp = BatchProcessor()
        j1 = bp.add_job("/tmp/v1.mp4", "hi")
        j2 = bp.add_job("/tmp/v2.mp4", "es")
        bp.get_job(j1).status = JobStatus.COMPLETED
        bp.get_job(j1).progress = 1.0
        progress = bp.get_progress()
        assert progress.total_jobs == 2
        assert progress.completed_jobs == 1
        assert progress.queued_jobs == 1

    def test_process_all_success(self):
        bp = BatchProcessor(BatchConfig(max_concurrent=2))
        bp.add_job("/tmp/v1.mp4", "hi", job_id="j1")
        bp.add_job("/tmp/v2.mp4", "es", job_id="j2")

        def mock_dub(video, lang, style, progress_cb):
            progress_cb("test", 0.5)
            progress_cb("done", 1.0)
            return f"/tmp/output_{lang}.mp4"

        results = bp.process_all(mock_dub)
        assert len(results) == 2
        completed = [j for j in results if j.status == JobStatus.COMPLETED]
        assert len(completed) == 2

    def test_process_all_with_failure(self):
        bp = BatchProcessor(BatchConfig(max_concurrent=1, max_retries=0))
        bp.add_job("/tmp/v1.mp4", "hi", job_id="j1")

        def mock_dub_fail(video, lang, style, progress_cb):
            raise RuntimeError("Test error")

        results = bp.process_all(mock_dub_fail)
        assert results[0].status == JobStatus.FAILED
        assert "Test error" in results[0].error_message

    def test_process_with_retry(self):
        bp = BatchProcessor(BatchConfig(max_concurrent=1, max_retries=1))
        bp.add_job("/tmp/v1.mp4", "hi", job_id="j1")

        call_count = 0

        def mock_dub_retry(video, lang, style, progress_cb):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First attempt fails")
            return "/tmp/output.mp4"

        results = bp.process_all(mock_dub_retry)
        assert results[0].status == JobStatus.COMPLETED
        assert call_count == 2

"""Tests for evaluate.persistence module."""

import json
import os
import tempfile

import pytest

from evaluate.models import RunHistoryEntry
from evaluate.persistence import (
    ensure_results_folder,
    load_result_file,
    load_run_history,
    sanitize_filename,
    save_result_file,
    save_run_history,
)


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_alphanumeric_unchanged(self):
        assert sanitize_filename("model123") == "model123"

    def test_dash_and_underscore_kept(self):
        assert sanitize_filename("my-model_v2") == "my-model_v2"

    def test_dot_kept(self):
        assert sanitize_filename("model.v1") == "model.v1"

    def test_special_chars_replaced(self):
        assert sanitize_filename("model/v1:latest") == "model_v1_latest"

    def test_spaces_replaced(self):
        assert sanitize_filename("my model name") == "my_model_name"

    def test_unicode_replaced(self):
        assert sanitize_filename("mô_hình_tts") == "m__h_nh_tts"

    def test_multiple_special_chars(self):
        result = sanitize_filename("model@v1#2$3")
        assert result == "model_v1_2_3"


class TestSaveResultFile:
    """Tests for save_result_file function."""

    def test_creates_file(self, tmp_path):
        results = [
            {"sample_id": "s1", "value": 3.5, "text": "hello"},
            {"sample_id": "s2", "value": 4.0, "text": "world"},
        ]
        summary = {"mean": 3.75, "std": 0.25, "min": 3.5, "max": 4.0}

        file_path = save_result_file(
            "test_model", "mcd", results, summary, str(tmp_path)
        )

        assert os.path.exists(file_path)
        assert file_path.endswith("test_model_mcd.json")

    def test_json_structure(self, tmp_path):
        results = [{"sample_id": "s1", "value": 2.5, "text": "test"}]
        summary = {"mean": 2.5, "std": 0.0, "min": 2.5, "max": 2.5}

        file_path = save_result_file(
            "my_model", "pesq", results, summary, str(tmp_path)
        )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["model_name"] == "my_model"
        assert data["metric_name"] == "pesq"
        assert data["samples"] == results
        assert data["summary"] == summary

    def test_creates_output_dir(self, tmp_path):
        output_dir = str(tmp_path / "nested" / "dir")
        results = [{"sample_id": "s1", "value": 1.0, "text": "a"}]
        summary = {"mean": 1.0}

        save_result_file("model", "stoi", results, summary, output_dir)

        assert os.path.isdir(output_dir)

    def test_sanitizes_model_name(self, tmp_path):
        results = [{"sample_id": "s1", "value": 1.0, "text": "a"}]
        summary = {"mean": 1.0}

        file_path = save_result_file(
            "model/v1:latest", "mcd", results, summary, str(tmp_path)
        )

        assert "model_v1_latest_mcd.json" in file_path

    def test_none_value_in_results(self, tmp_path):
        results = [{"sample_id": "s1", "value": None, "text": "test"}]
        summary = {"mean": 0.0}

        file_path = save_result_file(
            "model", "pesq", results, summary, str(tmp_path)
        )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["samples"][0]["value"] is None


class TestLoadResultFile:
    """Tests for load_result_file function."""

    def test_loads_valid_file(self, tmp_path):
        data = {
            "model_name": "test",
            "metric_name": "mcd",
            "samples": [{"sample_id": "s1", "value": 3.0, "text": "hi"}],
            "summary": {"mean": 3.0},
        }
        file_path = str(tmp_path / "test.json")
        with open(file_path, "w") as f:
            json.dump(data, f)

        result = load_result_file(file_path)

        assert result == data

    def test_returns_none_for_missing_file(self):
        result = load_result_file("/nonexistent/path/file.json")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        file_path = str(tmp_path / "bad.json")
        with open(file_path, "w") as f:
            f.write("not valid json {{{")

        result = load_result_file(file_path)
        assert result is None


class TestSaveRunHistory:
    """Tests for save_run_history function."""

    def test_creates_new_history_file(self, tmp_path):
        history_path = str(tmp_path / "history.json")
        entry = RunHistoryEntry(
            model_name="model_a",
            metric_name="mcd",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:05:00",
            num_samples=10,
            status="completed",
        )

        save_run_history(entry, history_path)

        with open(history_path, "r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["model_name"] == "model_a"
        assert data[0]["status"] == "completed"

    def test_appends_to_existing_history(self, tmp_path):
        history_path = str(tmp_path / "history.json")
        # Create initial entry
        entry1 = RunHistoryEntry(
            model_name="model_a",
            metric_name="mcd",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:05:00",
            num_samples=10,
            status="completed",
        )
        save_run_history(entry1, history_path)

        # Append second entry
        entry2 = RunHistoryEntry(
            model_name="model_b",
            metric_name="pesq",
            start_time="2024-01-01T01:00:00",
            end_time="2024-01-01T01:10:00",
            num_samples=20,
            status="completed",
        )
        save_run_history(entry2, history_path)

        with open(history_path, "r") as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["model_name"] == "model_a"
        assert data[1]["model_name"] == "model_b"

    def test_preserves_existing_entries(self, tmp_path):
        history_path = str(tmp_path / "history.json")
        # Pre-populate with existing data
        existing = [{"model_name": "old_model", "metric_name": "stoi"}]
        with open(history_path, "w") as f:
            json.dump(existing, f)

        entry = RunHistoryEntry(
            model_name="new_model",
            metric_name="mcd",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:05:00",
            num_samples=5,
            status="completed",
        )
        save_run_history(entry, history_path)

        with open(history_path, "r") as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0] == {"model_name": "old_model", "metric_name": "stoi"}

    def test_creates_parent_directory(self, tmp_path):
        history_path = str(tmp_path / "nested" / "dir" / "history.json")
        entry = RunHistoryEntry(
            model_name="model",
            metric_name="mcd",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:05:00",
            num_samples=1,
            status="completed",
        )

        save_run_history(entry, history_path)

        assert os.path.exists(history_path)

    def test_handles_corrupt_existing_file(self, tmp_path):
        history_path = str(tmp_path / "history.json")
        with open(history_path, "w") as f:
            f.write("not valid json")

        entry = RunHistoryEntry(
            model_name="model",
            metric_name="mcd",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:05:00",
            num_samples=1,
            status="completed",
        )

        save_run_history(entry, history_path)

        with open(history_path, "r") as f:
            data = json.load(f)

        # Should start fresh with just the new entry
        assert len(data) == 1
        assert data[0]["model_name"] == "model"


class TestLoadRunHistory:
    """Tests for load_run_history function."""

    def test_loads_valid_history(self, tmp_path):
        history_path = str(tmp_path / "history.json")
        entries = [
            {"model_name": "m1", "metric_name": "mcd", "status": "completed"},
            {"model_name": "m2", "metric_name": "pesq", "status": "failed"},
        ]
        with open(history_path, "w") as f:
            json.dump(entries, f)

        result = load_run_history(history_path)

        assert len(result) == 2
        assert result[0]["model_name"] == "m1"
        assert result[1]["status"] == "failed"

    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_run_history(str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_returns_empty_for_invalid_json(self, tmp_path):
        history_path = str(tmp_path / "bad.json")
        with open(history_path, "w") as f:
            f.write("{invalid}")

        result = load_run_history(history_path)
        assert result == []

    def test_returns_empty_for_non_list_json(self, tmp_path):
        history_path = str(tmp_path / "obj.json")
        with open(history_path, "w") as f:
            json.dump({"key": "value"}, f)

        result = load_run_history(history_path)
        assert result == []


class TestEnsureResultsFolder:
    """Tests for ensure_results_folder function."""

    def test_creates_directory(self, tmp_path):
        results_dir = str(tmp_path / "new_results")
        ensure_results_folder(results_dir)
        assert os.path.isdir(results_dir)

    def test_no_error_if_exists(self, tmp_path):
        results_dir = str(tmp_path / "existing")
        os.makedirs(results_dir)
        # Should not raise
        ensure_results_folder(results_dir)
        assert os.path.isdir(results_dir)

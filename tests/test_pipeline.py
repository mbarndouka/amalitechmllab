"""Tests for pipeline/runner.py — stage dispatch and ordering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.runner import STAGES, run_all, run_stage


def test_stages_contains_expected_keys():
    expected = {"explore", "clean", "engineer", "eda", "train", "advanced", "tune", "interpret"}
    assert expected == set(STAGES.keys())


def test_run_stage_calls_correct_function():
    mock_fn = MagicMock()
    cfg = {"key": "value"}
    with patch.dict(STAGES, {"train": mock_fn}):
        run_stage("train", cfg)
    mock_fn.assert_called_once_with(cfg)


def test_run_stage_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        run_stage("nonexistent_stage", {})


def test_run_all_calls_every_stage_in_order():
    call_order = []
    mock_fns = {name: MagicMock(side_effect=lambda cfg, n=name: call_order.append(n)) for name in STAGES}
    with patch.dict(STAGES, mock_fns, clear=True):
        run_all({})
    assert call_order == list(mock_fns.keys())


def test_run_all_passes_cfg_to_each_stage():
    cfg = {"data": {"raw_path": "test.csv"}}
    mock_fns = {name: MagicMock() for name in STAGES}
    with patch.dict(STAGES, mock_fns, clear=True):
        run_all(cfg)
    for mock_fn in mock_fns.values():
        mock_fn.assert_called_once_with(cfg)


def test_run_stage_propagates_exception():
    def boom(cfg):
        raise RuntimeError("stage failed")

    with patch.dict(STAGES, {"train": boom}), pytest.raises(RuntimeError, match="stage failed"):
        run_stage("train", {})

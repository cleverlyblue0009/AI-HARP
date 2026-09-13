"""The keep-awake helper must be a harmless, reversible no-op when not wanted."""

from __future__ import annotations

import sys

from common.keep_awake import keep_awake


def test_disabled_is_a_no_op():
    with keep_awake(enabled=False) as active:
        assert active is False


def test_enters_and_exits_cleanly():
    with keep_awake() as active:
        assert isinstance(active, bool)
    if sys.platform != "win32":
        assert active is False


def test_exits_cleanly_when_the_body_raises():
    """The sleep request must be released even if training crashes."""
    try:
        with keep_awake():
            raise RuntimeError("simulated training crash")
    except RuntimeError:
        pass


def test_training_flag_is_wired_around_train_not_just_parsed():
    """A parsed-but-unused flag would pass a name check and do nothing."""
    import pytest

    pytest.importorskip("torch")
    import inspect

    import agents.train as train

    src = inspect.getsource(train.main)
    assert "--keep-awake" in src
    assert "keep_awake(args.keep_awake)" in src
    wrapper = src.index("keep_awake(args.keep_awake)")
    call = src.index("train(cfg, cfgs, updates, out")
    assert wrapper < call, "train() must run inside the keep_awake context"

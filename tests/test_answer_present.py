"""Regression test for #38: answer_present() must not gate on a magic byte size.

Before the fix, answer_present() required ANSWER.md size > 200, so a short-but-correct answer
(e.g. the 67-byte "The battery life of the Atlas-7 is 8 hours...") was reported absent and the
job's run_meta said done=false despite a correct answer in the source of truth (ANSWER.md).
The invariant: an answer is present iff ANSWER.md exists with non-whitespace content.
"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import run_searchbox as rs


def _work(content):
    w = Path(tempfile.mkdtemp())
    if content is not None:
        (w / "ANSWER.md").write_text(content)
    return w


def test_short_correct_answer_is_present():
    # the exact bug case: a 67-byte correct answer (would FAIL under the old size>200 gate)
    w = _work("The battery life of the Atlas-7 is 8 hours of continuous operation.")
    assert rs.answer_present(w) is True


def test_single_char_answer_is_present():
    assert rs.answer_present(_work("8")) is True


def test_whitespace_only_is_absent():
    assert rs.answer_present(_work("   \n\t ")) is False


def test_empty_file_is_absent():
    assert rs.answer_present(_work("")) is False


def test_missing_file_is_absent():
    assert rs.answer_present(_work(None)) is False


def test_long_answer_is_present():
    assert rs.answer_present(_work("x" * 500)) is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all answer_present regression tests passed")

"""Regression test for #39: fail-closed done + assistant-only answer capture.

When the LLM is unreachable every call errors (Connection error, output=0 tokens), but pi still
streams a message_end for the USER prompt. Before the fix the orchestrator captured that echo as
turn_text and wrote the QUESTION into ANSWER.md, then reported done=true — a failed run
masquerading as success (fail-OPEN). Two invariants:
  1. assistant_text() ignores non-assistant message_end events (user/tool echoes).
  2. run_is_done() requires output_tokens>0 (a real answer costs output; 0 => the LLM never ran).
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import run_searchbox as rs


def _msg_end(role, text):
    content = [{"type": "text", "text": text}] if text is not None else []
    return json.dumps({"type": "message_end", "message": {"role": role, "content": content}})


def test_user_echo_is_not_captured():
    # the exact bug: pi echoes the user prompt as a message_end -> must be ignored
    assert rs.assistant_text(_msg_end("user", "What is the battery life of the Atlas-7?")) == ""


def test_tool_message_is_not_captured():
    assert rs.assistant_text(_msg_end("tool", "some tool result")) == ""


def test_assistant_text_is_captured():
    assert rs.assistant_text(_msg_end("assistant", "8 hours")) == "8 hours"


def test_assistant_empty_content_is_blank():
    # the errored assistant turn (Connection error) has empty content
    assert rs.assistant_text(_msg_end("assistant", None)) == ""


def test_malformed_line_is_blank():
    assert rs.assistant_text("not json") == ""


def test_done_requires_output_tokens():
    # llama down: stop_reason=budget_spent, an answer file may exist, but output==0 => NOT done
    assert rs.run_is_done("budget_spent", True, 0) is False


def test_done_with_real_output():
    assert rs.run_is_done("budget_spent", True, 47) is True


def test_done_requires_answer():
    assert rs.run_is_done("budget_spent", False, 47) is False


def test_done_requires_natural_stop():
    assert rs.run_is_done("ceiling_seconds", True, 47) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all fail-closed-done regression tests passed")

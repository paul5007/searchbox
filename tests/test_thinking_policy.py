"""Regression test for #15: per-turn thinking-budget policy hook.

thinking_switch() decides, per turn, whether to append a Qwen3 thinking soft-switch to the prompt.
The invariant that matters most: policy=off must leave the prompt byte-identical to vanilla (no
behavior change until the default is flipped behind the eval gate in #17).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
import run_searchbox as rs


def test_off_is_noop_for_both_turn_kinds():
    assert rs.thinking_switch(is_nudge=False, policy="off") == ""
    assert rs.thinking_switch(is_nudge=True, policy="off") == ""


def test_unknown_policy_is_noop():
    assert rs.thinking_switch(is_nudge=False, policy="banana") == ""


def test_nudge_gate_gates_only_nudge_turns():
    # the question turn inherits run-global thinking (no switch)...
    assert rs.thinking_switch(is_nudge=False, policy="nudge_gate") == ""
    # ...the mechanical KEEP_GOING nudge turns are gated OFF
    assert rs.thinking_switch(is_nudge=True, policy="nudge_gate") == rs._THINK_OFF


def test_gate_all_gates_every_turn():
    assert rs.thinking_switch(is_nudge=False, policy="gate_all") == rs._THINK_OFF
    assert rs.thinking_switch(is_nudge=True, policy="gate_all") == rs._THINK_OFF


def test_off_keeps_prompt_byte_identical():
    q = rs.TASK_COMMAND.format(query="What is the battery life of the Atlas-7?")
    assert q + rs.thinking_switch(is_nudge=False, policy="off") == q
    assert rs.KEEP_GOING + rs.thinking_switch(is_nudge=True, policy="off") == rs.KEEP_GOING


def test_switch_is_appended_not_replacing():
    q = rs.TASK_COMMAND.format(query="hi")
    assert (q + rs.thinking_switch(is_nudge=False, policy="gate_all")).startswith(q)
    assert (q + rs.thinking_switch(is_nudge=False, policy="gate_all")).endswith("/no_think")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all thinking-policy tests passed")

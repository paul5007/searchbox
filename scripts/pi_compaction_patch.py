#!/usr/bin/env python3
"""Idempotent patch for pi auto-compaction re-entrancy race.

Bug: _runAutoCompaction stores the AbortController in the instance field
this._autoCompactionAbortController and reads `.signal` off it AFTER several
awaits. When two auto-compactions fire on the same threshold crossing (searchbox
re-nudge lands a queued msg as context crosses the threshold), the first one's
finally{} sets the field to undefined; the second then reads `.signal` on
undefined -> "Cannot read properties of undefined (reading 'signal')".

Fix:
  1. Re-entrancy guard + synchronous controller alloc at method entry (before any
     await) so a concurrent call bails immediately.
  2. Capture a local `ac` and use ac.signal everywhere instead of the instance
     field (immune to the field being nulled by a sibling).
  3. finally{} only clears the field if it still owns `ac`.
"""
import sys, re, pathlib

p = pathlib.Path(sys.argv[1])
s = p.read_text()

MARK = "/*OPENCLAW_COMPACTION_RACE_FIX*/"
if MARK in s:
    print("already patched")
    sys.exit(0)

# 1+2: at method entry add guard + synchronous controller, capture local ac.
old_head = (
    "    async _runAutoCompaction(reason, willRetry) {\n"
    "        const settings = this.settingsManager.getCompactionSettings();\n"
    "        let started = false;\n"
    "        try {\n"
)
new_head = (
    "    async _runAutoCompaction(reason, willRetry) {\n"
    "        const settings = this.settingsManager.getCompactionSettings();\n"
    "        let started = false;\n"
    "        " + MARK + "\n"
    "        if (this._autoCompactionAbortController !== undefined) { return false; }\n"
    "        const ac = new AbortController();\n"
    "        this._autoCompactionAbortController = ac;\n"
    "        started = true;\n"
    "        try {\n"
)
assert old_head in s, "head anchor not found"
s = s.replace(old_head, new_head, 1)

# Remove the original mid-body controller alloc (now redundant).
old_alloc = (
    "            this._emit({ type: \"compaction_start\", reason });\n"
    "            this._autoCompactionAbortController = new AbortController();\n"
    "            started = true;\n"
)
new_alloc = (
    "            this._emit({ type: \"compaction_start\", reason });\n"
)
assert old_alloc in s, "alloc anchor not found"
s = s.replace(old_alloc, new_alloc, 1)

# 2: replace the three signal reads with the local ac.
before = s.count("this._autoCompactionAbortController.signal")
s = s.replace("this._autoCompactionAbortController.signal", "ac.signal")
assert before == 3, f"expected 3 signal reads, found {before}"

# 3: ownership-checked clear in finally.
old_finally = (
    "        finally {\n"
    "            this._autoCompactionAbortController = undefined;\n"
    "        }"
)
new_finally = (
    "        finally {\n"
    "            if (this._autoCompactionAbortController === ac) { this._autoCompactionAbortController = undefined; }\n"
    "        }"
)
assert old_finally in s, "finally anchor not found"
s = s.replace(old_finally, new_finally, 1)

p.write_text(s)
print("patched OK")

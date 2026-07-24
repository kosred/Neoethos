from pathlib import Path

path = Path(__file__).with_name("010-smc-gate-semantics.py")
text = path.read_text(encoding="utf-8")

old = '''        if re.search(r"(?:pub\\s+)?struct\\s+$", prefix):
            continue
'''
new = '''        if re.search(r"(?:pub\\s+)?struct\\s+$", prefix):
            continue
        if re.search(r"(?:->|impl)\\s*$", prefix):
            continue
'''
if text.count(old) != 1:
    raise RuntimeError("could not harden literal migration helper exactly once")
text = text.replace(old, new, 1)

old_checkpoint = '''if checkpoint_literals != 1:
    raise RuntimeError(f"expected one checkpoint SearchResult literal, found {checkpoint_literals}")
write(checkpoint_path, checkpoint)
'''
new_checkpoint = '''if checkpoint_literals != 2:
    raise RuntimeError(f"expected two checkpoint SearchResult literals, found {checkpoint_literals}")
test_start = checkpoint.index("#[cfg(test)]")
test_region = checkpoint[test_start:]
fixture_value = "effective_smc_gate_threshold: self.effective_smc_gate_threshold"
if test_region.count(fixture_value) != 1:
    raise RuntimeError("checkpoint fixture gate was not inserted exactly once")
test_region = test_region.replace(
    fixture_value,
    "effective_smc_gate_threshold: 0.75",
    1,
)
checkpoint = checkpoint[:test_start] + test_region
write(checkpoint_path, checkpoint)
'''
if text.count(old_checkpoint) != 1:
    raise RuntimeError("could not patch checkpoint literal handling exactly once")
text = text.replace(old_checkpoint, new_checkpoint, 1)

path.write_text(text, encoding="utf-8")

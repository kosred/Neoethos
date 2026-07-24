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

# The original migration kept the first DiscoveryConfig/DiscoveryResult edits
# only in memory, then re-read discovery.rs while adding test literal fields.
# Persist those structural edits before the repository-wide literal pass.
marker = '''# Every synthetic/test literal gets an explicit non-production sentinel. The
# real finalize constructor is replaced with the actual gate below.
'''
replacement = '''# Persist the structural config/result changes before the repository-wide
# literal pass reads discovery.rs again.
write(discovery_path, discovery)

# Every synthetic/test literal gets an explicit non-production sentinel. The
# real finalize constructor is replaced with the actual gate below.
'''
if text.count(marker) != 1:
    raise RuntimeError("could not insert discovery structural write exactly once")
text = text.replace(marker, replacement, 1)

# The only direct test call to the now-expanded private finalizer needs an
# explicit fixture gate. Production passes the real SearchResult value.
insert_before = '''# Document the verified migration commit name for the workflow bot.
'''
test_patch = '''# Update the direct finalizer fixture with an explicit SMC gate.
discovery_tests_path = "crates/neoethos-search/src/discovery_tests.rs"
discovery_tests = read(discovery_tests_path)
discovery_tests = replace_exact(
    discovery_tests,
    "        &config,\\n        features.names.clone(),\\n        &mut funnel,\\n",
    "        &config,\\n        0.75,\\n        features.names.clone(),\\n        &mut funnel,\\n",
    label="finalizer test gate",
)
write(discovery_tests_path, discovery_tests)

# Document the verified migration commit name for the workflow bot.
'''
if text.count(insert_before) != 1:
    raise RuntimeError("could not insert finalizer fixture patch exactly once")
text = text.replace(insert_before, test_patch, 1)

path.write_text(text, encoding="utf-8")

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
path.write_text(text.replace(old, new, 1), encoding="utf-8")

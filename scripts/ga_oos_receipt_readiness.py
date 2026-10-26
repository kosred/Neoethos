#!/usr/bin/env python3
"""Read-only readiness gate for the NeoEthos REAL receipt-bound EURUSD research.

A public EURUSD Vortex/OHLCV release, nor a 100-bar test fixture, is NOT an
authenticated cTrader acquisition matrix, account-snapshot, symbol contract,
nor a finalist-specific quote replay. This script reads *metadata and paths*
only; it never opens the reserved OOS or changes research/trading settings.
Pass --strict to exit nonzero unless the complete local input set is present.
This only checks file presence and SHA *shape*; canonical-research MUST
subsequently validate all receipt hashes, lineage, account currency, contents
and locked-portfolio quote-replay compatibility. Never claim PASS from here.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_DIRS = ("authority_root", "data_root")
REQUIRED_FILES = ("broker_symbol_contract", "broker_account_snapshot", "settings_source")
QUOTE_FIELDS = ("quote_replay_bundle", "quote_replay_sha256", "quote_replay_store")
SHA = re.compile(r"^[a-fA-F0-9]{64}$")


def scan(args):
    missing = []
    mismatch = []
    evidence = {}
    for name in REQUIRED_DIRS:
        raw = getattr(args, name)
        ok = bool(raw) and Path(raw).is_dir()
        evidence[name] = "present_directory" if ok else "missing_directory"
        if not ok: missing.append(name)
    for name in REQUIRED_FILES:
        raw = getattr(args, name)
        ok = bool(raw) and Path(raw).is_file()
        evidence[name] = "present_file" if ok else "missing_file"
        if not ok: missing.append(name)
    for name in ("plan_sha256", "matrix_sha256"):
        good = bool(getattr(args, name)) and bool(SHA.fullmatch(getattr(args, name)))
        evidence[name] = "sha256_shape_valid" if good else "missing_or_invalid_sha256"
        if not good: missing.append(name)
    quote_present = [bool(getattr(args, k)) for k in QUOTE_FIELDS]
    if any(quote_present) and not all(quote_present):
        mismatch.append("quote_replay_requires_all_three_inputs")
    for name in QUOTE_FIELDS:
        raw = getattr(args, name)
        if name == "quote_replay_sha256":
            ok = bool(raw) and bool(SHA.fullmatch(raw))
            evidence[name] = "sha256_shape_valid" if ok else "not_supplied"
        elif name == "quote_replay_store":
            ok = bool(raw) and Path(raw).is_dir()
            evidence[name] = "present_directory" if ok else "not_supplied"
        else:
            ok = bool(raw) and Path(raw).is_file()
            evidence[name] = "present_file" if ok else "not_supplied"
        if all(quote_present) and not ok: missing.append(name)
    quote_ready = all(quote_present) and not any(k in missing for k in QUOTE_FIELDS)
    if not quote_ready: missing.append("reviewed_finalist_quote_replay_bundle")
    early_ready = all(k not in missing for k in
                      (*REQUIRED_DIRS, *REQUIRED_FILES, "plan_sha256", "matrix_sha256"))
    result = {
        "schema": "neoethos.research.oos-receipt-readiness.v1",
        "execution": "READ_ONLY",
        "final_oos_read": False,
        "historical_market_evaluation_ran": False,
        "ga_experiment_feature": "experimental-active-percentile",
        "preflight_early_stage_inputs_present": early_ready,
        "quote_replay_path_inputs_present": quote_ready,
        "status": "PRELIMINARY_PATHS_PRESENT_NOT_AUTHENTICATED"
                  if early_ready and quote_ready and not mismatch else "HOLD",
        "missing_inputs": sorted(set(missing)),
        "invalid_combinations": mismatch,
        "evidence": evidence,
        "next_gate": "neoethos-cli canonical-research checks receipt hashes, "
                     "selected generations, broker account/symbol/economics, "
                     "and locked portfolio before an actual quote OOS test",
        "limitations": [
            "Presence of an artifact cannot authenticate its contents",
            "A public EURUSD OHLCV archive cannot replace sealed cTrader receipts",
            "Final quote replay package must exactly match the selected finalists",
            "A successful screening-only holdout cannot certify executable OOS profitability",
        ]
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in (*REQUIRED_DIRS, *REQUIRED_FILES, "plan_sha256", "matrix_sha256",
              *QUOTE_FIELDS):
        ap.add_argument("--" + n.replace("_", "-"))
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 if verified input paths are absent")
    ap.add_argument("--out", help="optional local JSON report (new file only)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        empty = scan(a)
        assert empty["status"] == "HOLD"
        assert empty["final_oos_read"] is False
        assert "reviewed_finalist_quote_replay_bundle" in empty["missing_inputs"]
        assert not empty["preflight_early_stage_inputs_present"]
        print("OOS_READINESS_SELF_TEST_PASS missing_inputs_fail_closed=1")
        return 0
    report=scan(a)
    encoded=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if a.out:
        p=Path(a.out)
        if p.exists():
            ap.error("out report path already exists; refusing overwrite")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("x",encoding="utf-8") as dest: dest.write(encoded)
    print(encoded)
    return 2 if a.strict and report["status"] != "PRELIMINARY_PATHS_PRESENT_NOT_AUTHENTICATED" else 0


if __name__ == "__main__":
    sys.exit(main())

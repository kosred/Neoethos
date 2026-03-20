# Repo Audit Report

## Executive Summary

## Repository Census

## Static Verification Findings

## Runtime Findings

## File-By-File Findings

## Contract And Operational Findings

## Warning Inventory

## Recommended Fix Tranches

## Findings Ledger Schema

Each JSON line in `cache/audit/2026-03-20-findings.jsonl` must include:
- `category`
- `severity`
- `lane`
- `command`
- `file`
- `line`
- `summary`
- `evidence`
- `root_cause`
- `recommended_fix`

Allowed `category` values:
- `build breakage`
- `test failure`
- `lint/warning`
- `runtime breakage`
- `correctness bug`
- `contract mismatch`
- `dead or unreachable code`
- `observability gap`
- `performance risk`
- `architectural smell`

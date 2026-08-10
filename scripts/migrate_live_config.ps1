#requires -Version 7.0
<#
.SYNOPSIS
  Migrate the operator's live NeoEthos config — the ONLY file a run reads.

.DESCRIPTION
  %LOCALAPPDATA%\neoethos\config.yaml is the file every discovery run, every
  training run and every live trading session actually loads. Settings::load()
  reads $CONFIG_FILE, else this file if it exists, else the literal relative
  path "config.yaml". It exists. The two YAMLs in the repo are documentation.

  This one is dated 2026-07-31 and it is stale in three different ways:

    1. It carries keys whose Rust fields no longer exist. Today serde ignores
       them. Once deny_unknown_fields lands they become a HARD STARTUP FAILURE.
    2. It is missing every knob added since 2026-07-31 — exit_policy,
       gene_stop_bounds, the session spread curve, the cost band — so those
       silently take code defaults with nothing recording that they did.
    3. Several of its values contradict a default that has since been decided
       the other way, and FIVE OF THOSE ARE ON THE MONEY PATH.

  This tool NEVER decides a money value. It backs the file up, shows the whole
  diff before writing anything, separates money-path items into their own
  section, asks about each one individually, and refuses to run unattended.

  DEFAULT BEHAVIOUR IS REPORT-ONLY. Nothing is written without -Apply.

.PARAMETER ConfigPath
  The live store. Defaults to %LOCALAPPDATA%\neoethos\config.yaml.

.PARAMETER DefaultsPath
  The GENERATED defaults projection, desktop/src-tauri/resources/config.yaml.
  Used for the generic ADD / DIVERGENCE passes. It is trusted ONLY if it
  carries the marker line the generator writes; a stale hand-written seed is
  refused rather than used, because writing "defaults" taken from a file that
  is not the defaults is how this mess started.

.PARAMETER BackupDir
  Where the timestamped backup goes. Defaults to the session scratchpad.

.PARAMETER Apply
  Write changes. Without it, this is a report and touches nothing.

.EXAMPLE
  pwsh scripts/migrate_live_config.ps1
      Report only. Read this first. It changes nothing.

.EXAMPLE
  pwsh scripts/migrate_live_config.ps1 -Apply
      Backs up, shows the diff, then asks section by section and, for money,
      item by item. There is deliberately no -Force and no -Yes.
#>

[CmdletBinding()]
param(
    [string] $ConfigPath = (Join-Path $env:LOCALAPPDATA 'neoethos\config.yaml'),
    [string] $DefaultsPath,
    [string] $BackupDir,
    [switch] $Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The line the generator writes into desktop/src-tauri/resources/config.yaml.
# Keep in sync with crates/neoethos-core/tests/generated_seed_is_current.rs.
$script:GeneratedMarker = '# neoethos-generated-defaults: v1'

# ===========================================================================
#  THE CURATED PLAN
#  Everything here was established by the 2026-08-09 knob audit and verified
#  against this exact file. It needs no defaults file to be correct.
# ===========================================================================

# Keys whose Rust field no longer exists. Removing them changes NOTHING today
# (serde already ignores them) and prevents a hard startup failure later.
$script:Tombstones = @(
    @{ Path = 'models.export_onnx';            Why = 'Field deleted with the ONNX export path. Ignored today; fatal once deny_unknown_fields lands.' }
    @{ Path = 'news.news_kill_window_min';     Why = 'Field deleted. The news kill window is no longer configurable here.' }
    @{ Path = 'news.news_lookahead_minutes';   Why = 'Field deleted.' }
    @{ Path = 'news.perplexity_enabled';       Why = 'Field deleted with the Perplexity helper.' }
    # Derived, not configured. These three have ZERO readers — dropping them is
    # behaviour-neutral whether or not the field deletion has landed yet.
    @{ Path = 'system.n_jobs';                 Why = 'DERIVED, NOT CONFIGURED. This is the one field in the whole struct whose default is genuine probe output — available_parallelism() minus one, then narrowed by system.hardware.cpu_budget. Your 11 is that fingerprint on a 12-core box, pickled back in by Settings::save as if you had chosen it. Zero readers. To constrain parallelism, set system.hardware.cpu_budget.' }
    @{ Path = 'system.num_gpus';               Why = 'DERIVED, NOT CONFIGURED. Zero readers — every real consumer reads the hardware probe. Note yours says 0 on a machine with a 3090; that was never probe output, it is a static default that merely looks like one.' }
    @{ Path = 'models.inference_batch_size';   Why = 'Zero readers. The hardware execution plan produces 128/1024/2048/4096/8192 and never the 32 stored here.' }
)

# Blocks this file has NEVER had. Written at their DOCUMENTED DEFAULT values,
# which is behaviour-neutral by construction: serde already applies exactly
# these when the key is absent. Writing them makes the silence visible.
#
# ⚠ Each block records the default it is asserting. If the generated defaults
# file is available the script VERIFIES every one of these against it and
# refuses to write any that disagrees, rather than trusting this list.
$script:MissingBlocks = @(
    @{
        Parent = 'models'
        Name   = 'models.exit_policy'
        Why    = 'Discovery exit geometry. Until 2026-08-09 these were three literals in strategy_gene.rs with no config recipient, so no run in fourteen months could measure the search without them. The search reads THIS copy, not risk.trailing_*.'
        Lines  = @(
            '  exit_policy:'
            '    trailing_enabled: false'
            '    trailing_be_trigger_r: 1.0'
            '    trailing_stop_multiplier: 1.0'
            '    trailing_min_lock_pips: 2.0'
        )
    }
    @{
        Parent = 'models'
        Name   = 'models.gene_stop_bounds'
        Why    = 'The stop/target band the GA may draw and mutate within, in multiples of THIS dataset''s median ATR instead of absolute pips. The old literals (sl 6-20, tp 12-45, rr 1.5-2.5) are M5 numbers: on H1 the whole stop band sits inside one bar''s range.'
        Lines  = @(
            '  gene_stop_bounds:'
            '    atr_scaled: true'
            '    sl_min_atr: 1.0'
            '    sl_max_atr: 4.0'
            '    rr_min: 1.5'
            '    rr_max: 4.0'
            '    sl_min_pips: 6.0'
            '    sl_max_pips: 20.0'
            '    tp_min_pips: 12.0'
            '    tp_max_pips: 45.0'
        )
    }
    @{
        Parent = 'risk'
        Name   = 'risk.backtest_spread_pips_*'
        Why    = 'Session-aware backtest spread for the three UTC buckets (Asian 22-07, Overlap 07-16, Late NY 16-22). The per-bar lookup exists on the CPU path AND in the CUDA kernel and was populated only under #[cfg(test)] until 2026-08-09 — a flat spread was charged at 03:00 Tokyo and at the London open alike. null keeps that flat behaviour and makes the run WARN about it. Set all three or none.'
        Lines  = @(
            '  backtest_spread_pips_asian: null'
            '  backtest_spread_pips_overlap: null'
            '  backtest_spread_pips_late_ny: null'
        )
    }
    @{
        Parent = 'risk'
        Name   = 'risk.cost_band_*'
        Why    = 'Round-trip cost band, in pips, that every reported result is measured against. A result that clears the optimistic edge but not the pessimistic one is FLAGGED, not reported as a win.'
        Lines  = @(
            '  cost_band_optimistic_pips: 1.6'
            '  cost_band_pessimistic_pips: 2.4'
        )
    }
    @{
        Parent = 'risk'
        Name   = 'risk.max_trades_per_day_enabled'
        Why    = 'Arms the max_trades_per_day cap you already have set to 8. Default false = the cap is inert. Measured on the real journal: armed at 8 it would have refused 68.1% of historical entries.'
        Lines  = @('  max_trades_per_day_enabled: false')
    }
)

# ===========================================================================
#  MONEY. Reported individually, never changed without an explicit per-item
#  answer. The tool proposes; the operator disposes.
# ===========================================================================
$script:MoneyItems = @(
    @{
        Path    = 'models.prop_search_min_payoff_ratio'
        Live    = '0.0'
        Default = '2.0'
        Means   = 'THE REALIZED-PAYOFF FLOOR IS OFF. 0.0 admits any strategy regardless of how its average win compares to its average loss — the pre-2026-08-09 state that let the search buy trade volume with no payoff margin.'
        IfSet   = 'At 2.0 a strategy is refused unless its realized average win is at least twice its average loss. An empty portfolio at 2.0 is the honest "2RR is rare" answer, not a bug. NOTE the measured counter-evidence: payoff moved 2.8x across a trailing-stop sweep while expectancy stayed at about -4.15 pips/trade, so payoff describes the SHAPE of the win/loss split and not the direction of the money.'
    }
    @{
        Path    = 'models.discovery_runtime.prefilter_top_k'
        Live    = '50'
        Default = '240'
        Means   = 'The feature prefilter keeps 50 columns. The base feature set collapses from 217 columns to roughly 64, and the SMC, session and footprint families die first because, unlike regime_*, they have no force-keep.'
        IfSet   = 'At 240 the search sees the intended feature pool. This changes WHAT IS SEARCHED: artifacts produced before and after are not comparable.'
    }
    @{
        Path    = 'models.require_walkforward_for_export'
        Live    = 'false'
        Default = 'true'
        Means   = 'THE OUT-OF-SAMPLE EXPORT GATE IS OFF. A portfolio can be exported toward live money on the prop-firm window gate alone, without passing walk-forward. Walk-forward still runs and is recorded; it just does not block.'
        IfSet   = 'At true, in-sample-overfit strategies (IS Sharpe 3-11, PF up to 62) stop exporting. It also kills regime specialists — genes good in SOME market conditions but not all years — which is the documented 2026-06-06 reason it was turned off. This is a real trade-off, not a bug fix.'
    }
    @{
        Path    = 'risk.max_portfolio_risk'
        Live    = '0.0'
        Default = '(see repo profile: 0.34)'
        Means   = 'READ THIS TWICE: 0.0 on a knob named max_ means NO CAP AT ALL, not "no risk". There is no portfolio-wide ceiling on concurrent risk today.'
        IfSet   = 'Any positive value caps total concurrent risk. The ambiguity itself is being turned into a LOUD STARTUP ERROR naming both readings (shard A) rather than silently re-interpreted, because picking either meaning changes how much of the account can be at risk at once.'
    }
    @{
        Path    = 'risk.trailing_enabled'
        Live    = 'true'
        Default = '(no value this tool can write here changes anything)'
        Means   = 'THIS IS THE ORPHANED COPY. trailing_enabled / trailing_be_trigger_r / trailing_min_lock_pips exist on BOTH RiskConfig and ExitPolicyConfig. The search reads the ExitPolicy copy. Your hand-tuned values here move NOTHING. Live execution, separately, trails unconditionally with no config gate at all.'
        IfSet   = 'Nothing this tool can write to this key changes behaviour. The fix is to wire live execution to models.exit_policy and THEN delete these — that order, never the reverse. See the four-value table printed below.'
    }
)

# E's requirement, stated in full because getting this wrong loses the only
# record of what the operator intended. His four hand-tuned trailing values are
# inert; the search has been using the ExitPolicy defaults the whole time. The
# copy-across is offered as an OPT-IN and is presented as a BEHAVIOUR CHANGE,
# never as "restoring your settings".
$script:TrailingPairs = @(
    @{ Live = 'risk.trailing_enabled';         Target = 'models.exit_policy.trailing_enabled';        Note = '' }
    @{ Live = 'risk.trailing_atr_multiplier';  Target = 'models.exit_policy.trailing_stop_multiplier'; Note = 'RENAMED, AND THE OLD NAME LIED. Despite "atr_multiplier" this was NEVER an ATR multiple — it is a multiple of the position''s OWN STOP DISTANCE. Copying the number across under the assumption of "ATR x 0.4" changes what it means without telling you.' }
    @{ Live = 'risk.trailing_be_trigger_r';    Target = 'models.exit_policy.trailing_be_trigger_r';   Note = '' }
    @{ Live = 'risk.trailing_min_lock_pips';   Target = 'models.exit_policy.trailing_min_lock_pips';  Note = '' }
)

# Answer-changing but not money-path. Same treatment: reported, per-item.
$script:SearchItems = @(
    @{ Path = 'models.discovery_runtime.adaptive_thresholds'; Live = 'false'; Default = 'true';   Means = 'Static threshold ladder [0.10..0.90], documented as calibrated for z-score-normalised features, compared against raw magnitudes spanning ~1e5:1. A 0.35 threshold is unreachable for an ATR term and always-on for an RSI term.' }
    @{ Path = 'models.data_runtime.normalize_features';       Live = 'false'; Default = 'true';   Means = 'The other half of the same defect. The GA weight ladder spans 5:1 while raw features span 1e5:1, so a multi-indicator gene equals its single largest-magnitude term. Turning this on means anything fitted on the raw cube must be retrained.' }
    @{ Path = 'models.cpcv_max_rows';                         Live = '0';     Default = '200000'; Means = '0 = UNBOUNDED. CPCV is the heaviest serial validation in the run. Separately: the CPCV gate validates the TAIL only and reports no coverage figure, so 200000 against 1.05M bars is an OOS gate that saw 19% of history and reported a clean pass.' }
    @{ Path = 'models.discovery_runtime.prefilter_insample_frac'; Live = '0.7'; Default = '0.8';  Means = 'A 70/30 in-sample split instead of 80/20.' }
    @{ Path = 'models.prop_search_max_indicators';            Live = '0';     Default = '16';     Means = '0 MEANS ALL, NOT NONE. Genes may use every available indicator — measured at up to 58, which is severe overfitting. 16 caps the degrees of freedom.' }
    @{ Path = 'models.prop_search_generations';               Live = '1000';  Default = '20000';  Means = 'Generation ceiling. Time-bounded by max_hours below, which usually binds first.' }
    @{ Path = 'models.prop_search_max_hours';                 Live = '24.0';  Default = '1.0';    Means = 'Hours per combo. 24 is a deliberate long-run setting; leaving it is fine if that is what you meant.' }
)

# ===========================================================================
#  A minimal reader for serde-emitted YAML. The live store is written by
#  Settings::save(), so it is strictly 2-space-indented block style with no
#  anchors, no flow maps beyond {}, and no multi-line scalars. This reader
#  REFUSES anything it does not recognise rather than guessing — a parser that
#  guesses would be one more silent substitution.
# ===========================================================================

function Read-YamlOutline {
    param([string[]] $Lines)

    $records = New-Object System.Collections.Generic.List[object]
    $stack = New-Object System.Collections.Generic.List[object]   # {Indent, Key}

    for ($i = 0; $i -lt $Lines.Count; $i++) {
        $raw = $Lines[$i]
        if ($raw -match '^\s*$' -or $raw -match '^\s*#') { continue }

        if ($raw -match '^(?<ind>\s*)-\s') { continue }   # sequence item: belongs to the key above

        if ($raw -notmatch '^(?<ind>\s*)(?<key>[A-Za-z_][A-Za-z0-9_]*):(?<rest>.*)$') {
            throw "migrate_live_config: line $($i + 1) is not a form this tool understands and it will not guess:`n  $raw"
        }
        $indent = $Matches['ind'].Length
        $key = $Matches['key']
        $rest = $Matches['rest'].Trim()

        while ($stack.Count -gt 0 -and $stack[$stack.Count - 1].Indent -ge $indent) {
            $stack.RemoveAt($stack.Count - 1)
        }
        $parentPath = if ($stack.Count -gt 0) { $stack[$stack.Count - 1].Path } else { '' }
        $path = if ($parentPath) { "$parentPath.$key" } else { $key }

        $isContainer = ($rest -eq '')
        $records.Add([pscustomobject]@{
                Path      = $path
                Key       = $key
                Indent    = $indent
                LineIndex = $i
                Value     = $rest
                Container = $isContainer
            })
        $stack.Add([pscustomobject]@{ Indent = $indent; Path = $path })
    }
    return $records
}

function Get-Record {
    param($Records, [string] $Path)
    return ($Records | Where-Object { $_.Path -eq $Path } | Select-Object -First 1)
}

# The inclusive line span a record owns: itself, plus everything indented
# deeper, plus any sequence items and comments that belong to it.
function Get-RecordSpan {
    param($Records, [string[]] $Lines, $Record)
    $start = $Record.LineIndex
    $end = $start
    for ($i = $start + 1; $i -lt $Lines.Count; $i++) {
        $l = $Lines[$i]
        if ($l -match '^\s*$') { $end = $i; continue }
        $ind = ($l -replace '^(\s*).*$', '$1').Length
        if ($ind -gt $Record.Indent) { $end = $i; continue }
        break
    }
    # do not swallow trailing blank lines
    while ($end -gt $start -and $Lines[$end] -match '^\s*$') { $end-- }
    return @{ Start = $start; End = $end }
}

# ===========================================================================
#  Guards
# ===========================================================================

function Assert-Interactive {
    if (-not [Environment]::UserInteractive) {
        throw @'
REFUSING TO RUN UNATTENDED.

This tool edits the only configuration file your runs read, including the
values that size real positions. It requires a human at the keyboard for every
section and for every money-path item individually. There is deliberately no
-Force and no -Yes. Run it from an interactive PowerShell session.
'@
    }
    try { $null = $Host.UI.RawUI.WindowSize }
    catch {
        throw 'REFUSING TO RUN UNATTENDED: no interactive host UI is available.'
    }
}

function Confirm-Section {
    param([string] $Name, [string] $Consequence)
    Write-Host ''
    Write-Host "  To apply this section, type its name exactly: $Name" -ForegroundColor Yellow
    Write-Host "  Anything else skips it. $Consequence"
    $answer = Read-Host '  >'
    return ($answer -ceq $Name)
}

function Confirm-MoneyItem {
    param([string] $Path)
    Write-Host ''
    Write-Host "  This is a MONEY-PATH value. To change it, type the full key path:" -ForegroundColor Red
    Write-Host "    $Path"
    Write-Host '  Anything else leaves it exactly as it is.'
    $answer = Read-Host '  >'
    return ($answer -ceq $Path)
}

# ===========================================================================
#  Main
# ===========================================================================

Write-Host ''
Write-Host '=============================================================================' -ForegroundColor Cyan
Write-Host ' NeoEthos live config migration' -ForegroundColor Cyan
Write-Host '=============================================================================' -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "The live store does not exist at: $ConfigPath`nNothing to migrate. If your runs are reading a different file, pass -ConfigPath."
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (-not $DefaultsPath) {
    $DefaultsPath = Join-Path $repoRoot 'desktop\src-tauri\resources\config.yaml'
}
if (-not $BackupDir) {
    $BackupDir = Join-Path $env:TEMP 'neoethos-live-config-migration'
}

$lines = [System.IO.File]::ReadAllLines($ConfigPath)
$records = Read-YamlOutline -Lines $lines
$fileInfo = Get-Item -LiteralPath $ConfigPath

Write-Host ''
Write-Host "  File            : $ConfigPath"
Write-Host "  Last written    : $($fileInfo.LastWriteTime)"
Write-Host "  Keys            : $($records.Count)"
Write-Host "  Mode            : $(if ($Apply) { 'APPLY (interactive)' } else { 'REPORT ONLY — nothing will be written' })"

# --- is the generated defaults projection usable? --------------------------
$defaultsUsable = $false
if (Test-Path -LiteralPath $DefaultsPath) {
    $defaultsText = [System.IO.File]::ReadAllLines($DefaultsPath)
    $defaultsUsable = @($defaultsText | Where-Object { $_.TrimEnd() -eq $script:GeneratedMarker }).Count -gt 0
}
Write-Host "  Defaults source : $DefaultsPath"
if ($defaultsUsable) {
    Write-Host '                    OK — carries the generator marker.' -ForegroundColor Green
}
else {
    Write-Host '                    NOT TRUSTED — no generator marker.' -ForegroundColor Yellow
    Write-Host '                    Run: cargo test -p neoethos-core --test generated_seed_is_current'
    Write-Host '                    Until then the generic key-by-key passes are skipped. The curated'
    Write-Host '                    plan below is unaffected: it needs no defaults file to be correct.'
}

$defaultRecords = $null
if ($defaultsUsable) {
    $defaultRecords = Read-YamlOutline -Lines $defaultsText
}

# ---------------------------------------------------------------------------
# SECTION 1 — DROP
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '--- SECTION 1: DROP — keys whose field no longer exists ---------------------' -ForegroundColor Cyan

$toDrop = @()
foreach ($t in $script:Tombstones) {
    $r = Get-Record -Records $records -Path $t.Path
    if ($r) { $toDrop += [pscustomobject]@{ Record = $r; Why = $t.Why } }
}
if ($defaultsUsable) {
    foreach ($r in $records) {
        if ($r.Container) { continue }
        if ($toDrop.Record -contains $r) { continue }
        if (-not (Get-Record -Records $defaultRecords -Path $r.Path)) {
            $known = $script:Tombstones | Where-Object { $_.Path -eq $r.Path }
            if (-not $known) {
                $toDrop += [pscustomobject]@{
                    Record = $r
                    Why    = 'Not present in the generated defaults, so it is not a field of Settings. Either the field was deleted or the key is misspelled — today both are silently ignored.'
                }
            }
        }
    }
}

if ($toDrop.Count -eq 0) {
    Write-Host '  Nothing to drop.' -ForegroundColor Green
}
else {
    Write-Host "  $($toDrop.Count) key(s). Removing them changes NOTHING today — serde already ignores"
    Write-Host '  them. It prevents a hard startup failure once deny_unknown_fields lands.'
    foreach ($d in $toDrop) {
        Write-Host ''
        Write-Host ("    - {0}   (line {1}: {2})" -f $d.Record.Path, ($d.Record.LineIndex + 1), $lines[$d.Record.LineIndex].Trim()) -ForegroundColor Yellow
        Write-Host ("      {0}" -f $d.Why)
    }
}

# ---------------------------------------------------------------------------
# SECTION 2 — ADD
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '--- SECTION 2: ADD — knobs this file has never had ---------------------------' -ForegroundColor Cyan
Write-Host '  Every value below is the code default, so writing it changes NO behaviour:'
Write-Host '  serde already applies exactly these when the key is absent. What changes is'
Write-Host '  that the value stops being invisible.'

$toAdd = @()
foreach ($b in $script:MissingBlocks) {
    $firstKey = ($b.Lines[0] -replace '^\s*([A-Za-z_][A-Za-z0-9_]*):.*$', '$1')
    $probe = if ($b.Parent) { "$($b.Parent).$firstKey" } else { $firstKey }
    if (Get-Record -Records $records -Path $probe) { continue }

    # If we have trustworthy defaults, verify every scalar in the block against
    # them and refuse the block rather than write a value we only believe.
    $verified = $true
    $mismatch = @()
    if ($defaultsUsable) {
        foreach ($l in $b.Lines) {
            if ($l -notmatch '^(?<ind>\s*)(?<key>[A-Za-z_][A-Za-z0-9_]*):(?<rest>.*)$') { continue }
            $rest = $Matches['rest'].Trim()
            if ($rest -eq '') { continue }
            $k = $Matches['key']
            $ind = $Matches['ind'].Length
            $p = if ($ind -ge 4) { "$($b.Parent).$firstKey.$k" } else { "$($b.Parent).$k" }
            $dr = Get-Record -Records $defaultRecords -Path $p
            if (-not $dr) { $verified = $false; $mismatch += "$p is not in the generated defaults" ; continue }
            if ($dr.Value -ne $rest) { $verified = $false; $mismatch += "$p default is '$($dr.Value)', this block would write '$rest'" }
        }
    }

    $toAdd += [pscustomobject]@{ Block = $b; Verified = $verified; Mismatch = $mismatch }
}

if ($toAdd.Count -eq 0) {
    Write-Host '  Nothing to add.' -ForegroundColor Green
}
else {
    foreach ($a in $toAdd) {
        Write-Host ''
        Write-Host ("    + {0}  (under `"{1}:`")" -f $a.Block.Name, $a.Block.Parent) -ForegroundColor Green
        Write-Host ("      {0}" -f $a.Block.Why)
        foreach ($l in $a.Block.Lines) { Write-Host "        $l" -ForegroundColor DarkGray }
        if ($defaultsUsable -and -not $a.Verified) {
            Write-Host '      ⚠ REFUSED — does not match the generated defaults:' -ForegroundColor Red
            foreach ($m in $a.Mismatch) { Write-Host "          $m" -ForegroundColor Red }
            Write-Host '        This block will NOT be written. Fix the curated plan in this script.' -ForegroundColor Red
        }
        elseif (-not $defaultsUsable) {
            Write-Host '      (unverified: no generated defaults file — values are from the 2026-08-09 audit)' -ForegroundColor Yellow
        }
    }
}

# ---------------------------------------------------------------------------
# SECTION 3 — MONEY
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '=============================================================================' -ForegroundColor Red
Write-Host ' SECTION 3: MONEY PATH — five values only you can decide' -ForegroundColor Red
Write-Host '=============================================================================' -ForegroundColor Red
Write-Host ' Nothing here is changed unless you type the full key path for that one item.'

$moneyPresent = @()
foreach ($m in $script:MoneyItems) {
    $r = Get-Record -Records $records -Path $m.Path
    Write-Host ''
    if ($r) {
        Write-Host ("  {0}" -f $m.Path) -ForegroundColor Red
        Write-Host ("    your file : {0}" -f $r.Value)
        Write-Host ("    default   : {0}" -f $m.Default)
        Write-Host ("    MEANS NOW : {0}" -f $m.Means)
        Write-Host ("    IF CHANGED: {0}" -f $m.IfSet)
        $moneyPresent += [pscustomobject]@{ Item = $m; Record = $r }
    }
    else {
        Write-Host ("  {0}  — ABSENT, so it silently takes the default ({1})." -f $m.Path, $m.Default) -ForegroundColor Yellow
        Write-Host ("    MEANS NOW : {0}" -f $m.IfSet)
    }
}

# --- 3b. the four trailing values, side by side ----------------------------
Write-Host ''
Write-Host '  --- YOUR FOUR HAND-TUNED TRAILING VALUES, AND WHAT THE SEARCH ACTUALLY USED ---' -ForegroundColor Red
Write-Host '  Your file has no models.exit_policy block at all, and the search reads that'
Write-Host '  block exclusively. So every one of these numbers has moved nothing.'
Write-Host ''
Write-Host ('    {0,-34} {1,-10} {2}' -f 'key in your file', 'your value', 'what the search used instead')
foreach ($tp in $script:TrailingPairs) {
    $r = Get-Record -Records $records -Path $tp.Live
    $v = if ($r) { $r.Value } else { '(absent)' }
    Write-Host ('    {0,-34} {1,-10} {2}' -f $tp.Live, $v, "$($tp.Target)  [Rust default]") -ForegroundColor Yellow
    if ($tp.Note) { Write-Host ("        ⚠ {0}" -f $tp.Note) -ForegroundColor Red }
}
Write-Host ''
Write-Host '  THIS TOOL DOES NOT COPY THEM ACROSS, and that is the safe default, not laziness:' -ForegroundColor Red
Write-Host '  copying trailing_enabled: true into models.exit_policy turns the trail ON for'
Write-Host '  every future search. Measured on real EURUSD bars, the trail is applied BEFORE'
Write-Host '  the take-profit check on every bar, which made the take-profit dead code and'
Write-Host '  pinned realised payoff near 1.08 against a configured floor of 2.0. It is also a'
Write-Host '  reward hack under a payoff floor: a trail multiplier of 3 produces payoff 2.53,'
Write-Host '  clears a 2.0 floor, and loses 4.18 pips per trade.'
Write-Host ''
Write-Host '  The four keys are LEFT IN YOUR FILE. They are inert, they still parse, and they'
Write-Host '  are the only surviving record of what you intended. They are not deleted until'
Write-Host '  live execution reads models.exit_policy — that order, never the reverse.'
Write-Host '  To adopt them deliberately, edit models.exit_policy by hand after reading the'
Write-Host '  paragraph above. This tool will not present that as a repair.'

# ---------------------------------------------------------------------------
# SECTION 4 — ANSWER-CHANGING (not money)
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '--- SECTION 4: values that change WHAT IS SEARCHED ---------------------------' -ForegroundColor Cyan
Write-Host '  Not money, but not cosmetic: runs before and after are not comparable.'
foreach ($s in $script:SearchItems) {
    $r = Get-Record -Records $records -Path $s.Path
    if (-not $r) { continue }
    if ($r.Value -eq $s.Default) { continue }
    Write-Host ''
    Write-Host ("  {0}" -f $s.Path) -ForegroundColor Yellow
    Write-Host ("    your file : {0}     default: {1}" -f $r.Value, $s.Default)
    Write-Host ("    {0}" -f $s.Means)
}

# ---------------------------------------------------------------------------
# SECTION 5 — PRESERVED
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '--- SECTION 5: PRESERVED — your choices, untouched ---------------------------' -ForegroundColor Cyan
$preserveRoots = @(
    'system.symbol', 'system.watchlist', 'system.account_currency', 'system.data_dir',
    'system.ui_locale', 'system.trading_mode', 'system.base_timeframe',
    'system.higher_timeframes', 'system.required_timeframes', 'system.multi_resolution_timeframes',
    'system.broker_timezone', 'system.metrics_db_path', 'system.cache_dir',
    'risk.preset', 'risk.initial_balance', 'risk.risk_per_trade', 'risk.max_risk_per_trade',
    'risk.risky_max_risk_per_trade', 'risk.prop_firm_max_risk_per_trade',
    'risk.daily_drawdown_limit', 'risk.total_drawdown_limit', 'risk.max_lot_size',
    'risk.commission_per_lot', 'risk.backtest_spread_pips', 'risk.slippage_pips',
    'models.ml_models', 'models.prop_search_population', 'models.prop_search_portfolio_size',
    'app_runtime.server_bind', 'secrets_file'
)
foreach ($p in $preserveRoots) {
    $r = Get-Record -Records $records -Path $p
    if ($r) {
        $shown = if ($r.Container) { '(block)' } else { $r.Value }
        Write-Host ("    {0,-52} {1}" -f $p, $shown) -ForegroundColor DarkGray
    }
}
Write-Host ''
Write-Host '  risk.preset deserves a note: PropFirmPreset selects a firm NAME while' -ForegroundColor Yellow
Write-Host '  #[serde(default)] on RiskConfig builds Default FIRST, so the preset: key lands'
Write-Host '  AFTER the six fields it is documented to seed. Your drawdown numbers may not be'
Write-Host '  the ones your preset implies. That is being turned into a LOUD LOAD ERROR naming'
Write-Host '  both readings (shard A), not silently re-derived. This tool changes neither.'

# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
if (-not $Apply) {
    Write-Host ''
    Write-Host '=============================================================================' -ForegroundColor Cyan
    Write-Host ' REPORT ONLY. Nothing was written.' -ForegroundColor Cyan
    Write-Host ' Re-run with -Apply to be asked, section by section and money item by money' -ForegroundColor Cyan
    Write-Host ' item, what to change. Your file is untouched.' -ForegroundColor Cyan
    Write-Host '=============================================================================' -ForegroundColor Cyan
    return
}

Assert-Interactive

New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = Join-Path $BackupDir "config.yaml.$stamp.bak"
Copy-Item -LiteralPath $ConfigPath -Destination $backup
Write-Host ''
Write-Host "  BACKED UP -> $backup" -ForegroundColor Green
if (-not (Test-Path -LiteralPath $backup)) {
    throw 'Backup did not appear on disk. Refusing to modify the live store.'
}

$work = [System.Collections.Generic.List[string]]::new()
$lines | ForEach-Object { $work.Add($_) }
$changeLog = New-Object System.Collections.Generic.List[string]

# --- 1. DROP ---
if ($toDrop.Count -gt 0) {
    Write-Host ''
    Write-Host "SECTION 1 (DROP): $($toDrop.Count) dead key(s). Behaviour-neutral today; required before deny_unknown_fields." -ForegroundColor Cyan
    if (Confirm-Section -Name 'DROP' -Consequence 'Skipping leaves them, and they become fatal later.') {
        foreach ($d in ($toDrop | Sort-Object { $_.Record.LineIndex } -Descending)) {
            $span = Get-RecordSpan -Records $records -Lines $lines -Record $d.Record
            for ($i = $span.End; $i -ge $span.Start; $i--) { $work.RemoveAt($i) }
            $changeLog.Add("DROP    $($d.Record.Path)")
        }
        Write-Host '  applied.' -ForegroundColor Green
    }
    else { Write-Host '  skipped.' -ForegroundColor Yellow }
}

# --- 2. ADD ---
$addable = @($toAdd | Where-Object { $_.Verified -or -not $defaultsUsable })
if ($addable.Count -gt 0) {
    Write-Host ''
    Write-Host "SECTION 2 (ADD): $($addable.Count) block(s), all at their default value." -ForegroundColor Cyan
    if (Confirm-Section -Name 'ADD' -Consequence 'Skipping leaves these knobs invisible; behaviour is identical either way.') {
        # Recompute outline against the working copy so indices stay honest.
        foreach ($a in $addable) {
            $cur = Read-YamlOutline -Lines $work.ToArray()
            $parent = Get-Record -Records $cur -Path $a.Block.Parent
            if (-not $parent) { throw "Cannot add $($a.Block.Name): parent section '$($a.Block.Parent)' not found." }
            $span = Get-RecordSpan -Records $cur -Lines $work.ToArray() -Record $parent
            $work.InsertRange($span.End + 1, [string[]]$a.Block.Lines)
            $changeLog.Add("ADD     $($a.Block.Name)  (default values — no behaviour change)")
        }
        Write-Host '  applied.' -ForegroundColor Green
    }
    else { Write-Host '  skipped.' -ForegroundColor Yellow }
}

# --- 3. MONEY, item by item ---
if ($moneyPresent.Count -gt 0) {
    Write-Host ''
    Write-Host '=============================================================================' -ForegroundColor Red
    Write-Host ' SECTION 3 (MONEY): asked one at a time. Default answer is ALWAYS "leave it".' -ForegroundColor Red
    Write-Host '=============================================================================' -ForegroundColor Red
    foreach ($mp in $moneyPresent) {
        $item = $mp.Item
        if ($item.Default -notmatch '^[-0-9]') {
            Write-Host ''
            Write-Host ("  {0}: there is no value this tool can write that changes anything." -f $item.Path) -ForegroundColor Yellow
            Write-Host '  Left exactly as it is.'
            continue
        }
        Write-Host ''
        Write-Host ("  {0}   {1}  ->  {2}" -f $item.Path, $mp.Record.Value, $item.Default) -ForegroundColor Red
        Write-Host ("    now : {0}" -f $item.Means)
        Write-Host ("    then: {0}" -f $item.IfSet)
        if (Confirm-MoneyItem -Path $item.Path) {
            $cur = Read-YamlOutline -Lines $work.ToArray()
            $r = Get-Record -Records $cur -Path $item.Path
            if (-not $r) { throw "Lost track of $($item.Path) — refusing to write." }
            $old = $work[$r.LineIndex]
            $work[$r.LineIndex] = ($old -replace '(:\s*).*$', ('${1}' + $item.Default))
            $changeLog.Add("MONEY   $($item.Path): $($mp.Record.Value) -> $($item.Default)")
            Write-Host '    CHANGED.' -ForegroundColor Red
        }
        else { Write-Host '    left as it is.' -ForegroundColor Green }
    }
}

# --- write ---
if ($changeLog.Count -eq 0) {
    Write-Host ''
    Write-Host '  Nothing was accepted. The live store is unchanged.' -ForegroundColor Green
    return
}

Write-Host ''
Write-Host '--- ABOUT TO WRITE ----------------------------------------------------------' -ForegroundColor Cyan
foreach ($c in $changeLog) { Write-Host "  $c" }
Write-Host ''
Write-Host "  Backup: $backup"
if (-not (Confirm-Section -Name 'WRITE' -Consequence 'Skipping discards everything above; your file stays as it was.')) {
    Write-Host '  Discarded. The live store is unchanged.' -ForegroundColor Green
    return
}

[System.IO.File]::WriteAllLines($ConfigPath, $work.ToArray())
$log = Join-Path $BackupDir "migration.$stamp.log"
[System.IO.File]::WriteAllLines($log, @(
        "NeoEthos live config migration $stamp",
        "file:   $ConfigPath",
        "backup: $backup",
        ''
    ) + $changeLog.ToArray())

Write-Host ''
Write-Host "  WRITTEN. $($changeLog.Count) change(s)." -ForegroundColor Green
Write-Host "  Log:    $log"
Write-Host "  Revert: Copy-Item -LiteralPath '$backup' -Destination '$ConfigPath'"

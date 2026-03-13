# HPC Unified Mode - Zero Configuration

**No separate scripts. No manual configuration. Just run `forex-ai.py` and it automatically detects your HPC hardware and runs everything.**

## Quick Start

On your Hyperstack N3 instance, simply run:

```bash
python forex-ai.py --train
```

That's it. The system automatically:
1. Detects 8× A6000 GPUs + 504 threads
2. Runs unified discovery + training
3. Saves checkpoints every 15 minutes
4. Pushes results to GitHub every 30 minutes
5. Resumes automatically if interrupted

---

## What It Does

```
┌─────────────────────────────────────────────────────────────────┐
│  YOU RUN:  python forex-ai.py --train                           │
│                     ↓                                            │
│  AUTO-DETECT: 8 GPUs + 504 threads = HPC Mode                    │
│                     ↓                                            │
│  UNIFIED PIPELINE:                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────┐ │
│  │ Features    │→ │ GPU Discovery│→ │ CPU Validate│→ │ Train  │ │
│  │ (504 thrd)  │  │ (8×A6000)    │  │ (480 thrd)  │  │ Models │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └────────┘ │
│         ↓                ↓                ↓              ↓      │
│    Auto-save        Auto-save        Auto-save      Auto-save   │
│    (15 min)         (15 min)         (15 min)       (15 min)    │
│         ↓                ↓                ↓              ↓      │
│    GitHub sync      GitHub sync      GitHub sync    GitHub sync │
│    (30 min)         (30 min)         (30 min)       (30 min)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Environment Variables (Optional)

Only set these if you want to customize behavior:

```bash
# GitHub backup (highly recommended!)
export FOREX_BOT_GITHUB_REPO="https://github.com/yourusername/forex-strategies"
export FOREX_BOT_GIT_NAME="HPC Bot"
export FOREX_BOT_GIT_EMAIL="bot@yourdomain.com"

# Save intervals
export FOREX_BOT_CHECKPOINT_MINUTES=15    # Local checkpoints
export FOREX_BOT_BACKUP_MINUTES=30        # GitHub backups

# Low credit threshold (auto-save and exit)
export FOREX_BOT_LOW_CREDIT_THRESHOLD=5.0  # $5 remaining

# Population size (auto-set to 500K for HPC, but you can override)
export FOREX_BOT_HPC_POPULATION=500000

# Run!
python forex-ai.py --train
```

---

## Resume After Interruption

If your instance runs out of credits or is terminated:

```bash
# On new instance, clone your repo with the backup branch
git clone -b hpc-results https://github.com/yourusername/forex-strategies
cd forex-strategies

# Run again - it automatically resumes from last checkpoint
python forex-ai.py --train
```

**Resume happens automatically.** The system:
1. Checks for local checkpoints in `checkpoints/hpc/`
2. If not found, checks the GitHub backup
3. Restores population and continues from last generation

---

## What Gets Saved

### Local Checkpoints (`checkpoints/hpc/`)
- Every 15 minutes during discovery
- Last 10 checkpoints kept (automatic cleanup)
- Format: `{session_id}_{phase}_gen{generation}.pkl`

### GitHub Backup (`hpc-results` branch)
- Every 30 minutes
- `strategies.json` - Top 1000 strategies
- `models/` - Trained model files
- `checkpoints/` - Latest checkpoint

### Final Results (`results/`)
- `hpc_final_{timestamp}.json` - Complete results when finished

---

## Monitoring Progress

### Live Logs
```bash
tail -f logs/forex_bot.log | grep -E "(Generation|strategies|HPC|checkpoint|GitHub)"
```

### Check Current Status
```bash
# View checkpoints
ls -la checkpoints/hpc/

# View latest population size
python -c "import joblib; cp = joblib.load('checkpoints/hpc/latest.pkl'); print(f'Gen {cp[\"generation\"]}: {len(cp[\"population\"])} strategies')"
```

### GitHub Backup Status
```bash
# Check last backup time
git log -1 --format=%ci origin/hpc-results
```

---

## Cost Protection

The system automatically saves and exits when:

1. **Low Credits Detected**: <$5 remaining (configurable)
2. **Termination Signal**: SIGTERM or cloud pre-emption warning
3. **Keyboard Interrupt**: Ctrl+C (graceful shutdown)

```
💳 Credits low ($4.50), preparing to save and exit...
💾 Checkpoint saved: checkpoints/hpc/session_20240223_120000_discovery_gen150.pkl
✅ Backup pushed to GitHub: hpc-results
📁 Final results saved: results/hpc_final_session_20240223_120000.json
👋 Safe to terminate. Run again to resume.
```

---

## Hardware Requirements

HPC mode auto-activates when:

| Resource | Minimum | Optimal (Hyperstack N3) |
|----------|---------|------------------------|
| GPUs | 4× | 8× RTX A6000 |
| GPU VRAM | 24GB | 48GB per GPU |
| CPU Threads | 128 | 504 (252 cores × 2 SMT) |
| RAM | 128GB | 464GB |

---

## Performance Expectations

| Phase | Duration | Output |
|-------|----------|--------|
| Feature Compute | ~5 min | Cached features for all symbols |
| GPU Discovery (200 gen) | ~16 hours | 500K+ strategies evaluated |
| CPU Validation | ~2 hours | Top 10K validated |
| Model Training | ~1 hour | Ensemble of 8 models |
| **Total** | **~20 hours** | **Production-ready system** |

---

## Troubleshooting

### "No HPC mode detected"
```bash
# Force HPC mode
export FOREX_BOT_HPC_MODE=1
python forex-ai.py --train
```

### "GitHub backup failing"
```bash
# Check git config
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
export FOREX_BOT_GITHUB_REPO="https://github.com/user/repo"

# Or disable GitHub backup
export FOREX_BOT_BACKUP_MINUTES=99999  # Effectively disabled
```

### "Out of disk space"
```bash
# Check checkpoint size
du -sh checkpoints/hpc/

# Clean old checkpoints (keep only last 3)
ls -t checkpoints/hpc/*.pkl | tail -n +4 | xargs rm
```

### Resume from specific checkpoint
```bash
# List available
ls -lt checkpoints/hpc/

# Copy specific checkpoint as 'latest'
cp checkpoints/hpc/session_XYZ_gen100.pkl checkpoints/hpc/resume.pkl

# It will auto-resume from the most recent
python forex-ai.py --train
```

---

## Architecture

```python
# Single entry point in forex-ai.py
async def main_async():
    if _detect_hpc_mode():
        # HPC: Unified training + discovery + auto-save
        from forex_bot.hpc_coordinator import run_hpc_unified
        results = run_hpc_unified(settings, symbols)
    else:
        # Standard: Separate training and discovery
        await _run_global_training(...)
```

**Components:**
- `HPCCheckpointManager`: Saves progress every 15 min
- `GitHubBackupManager`: Pushes to GitHub every 30 min  
- `CloudCostMonitor`: Detects low credits / termination
- `HPCUnifiedRunner`: Orchestrates the pipeline

---

## Migration from Old Scripts

If you were using the old `run_hyperstack_n3.sh`:

| Old Way | New Way |
|---------|---------|
| `./run_hyperstack_n3.sh full` | `python forex-ai.py --train` |
| `./run_hyperstack_n3.sh check` | Auto-detected, no action needed |
| Manual Git commits | Automatic every 30 min |
| Manual resume | Automatic on restart |

---

## Summary

**Before (complicated):**
```bash
./scripts/run_hyperstack_n3.sh check
./scripts/run_hyperstack_n3.sh full
# Hope it doesn't crash
# Manual git commit if it does
```

**After (simple):**
```bash
export FOREX_BOT_GITHUB_REPO="https://github.com/..."
python forex-ai.py --train
# Automatically saves, backs up, resumes
```

**Zero configuration. Maximum safety. Never lose progress.**

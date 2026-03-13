# HPC Unified Mode - Implementation Summary

## What You Asked For

✅ **One command**: `python forex-ai.py --train`  
✅ **Auto-detection**: Detects 8 GPUs + 504 threads automatically  
✅ **Unified pipeline**: Training + discovery in one run  
✅ **Auto-save**: Every 15 minutes locally  
✅ **GitHub backup**: Every 30 minutes to prevent data loss  
✅ **Resume**: Automatically continues from last checkpoint  
✅ **Cost protection**: Saves and exits when credits run low

---

## Files Created

### 1. `src/forex_bot/hpc_coordinator.py` (New)
The unified orchestrator that manages everything:
- `HPCCheckpointManager` - Saves progress every 15 minutes
- `GitHubBackupManager` - Pushes to GitHub every 30 minutes  
- `CloudCostMonitor` - Detects low credits / termination signals
- `HPCUnifiedRunner` - Runs the full pipeline with auto-resume

### 2. Modified `src/forex_bot/main.py`
Added HPC detection and unified mode integration:
- `_detect_hpc_mode()` - Auto-detects HPC hardware
- Integration in `main_async()` - Calls unified runner when HPC detected

### 3. `crates/forex-search/src/hpc.rs` (Updated)
Now properly handles 504 threads (252 cores × 2 SMT):
- `get_gpu_cpu_affinity()` - Maps GPUs to NUMA nodes with SMT
- `get_validation_cpu_cores()` - Uses 480 threads, reserves 24 for coordination

---

## How It Works

### 1. Run Command
```bash
# Set your GitHub repo (optional but recommended)
export FOREX_BOT_GITHUB_REPO="https://github.com/yourusername/forex-strategies"

# Run - that's it!
python forex-ai.py --train
```

### 2. Auto-Detection
The system checks:
- 8+ GPUs? ✓
- 500+ CPU threads? ✓
→ **Activates HPC unified mode automatically**

### 3. Pipeline Execution
```
Features (504 threads)
    ↓
GPU Discovery (8×A6000, 200 generations)
    ↓ [auto-save every 15 min]
CPU Validation (480 threads)
    ↓ [auto-save every 15 min]
Model Training
    ↓ [auto-save every 15 min]
Ensemble Construction
    ↓
Done!
```

### 4. Continuous Backup
- **Local**: `checkpoints/hpc/session_XYZ_gen{N}.pkl`
- **GitHub**: `hpc-results` branch with strategies + models

### 5. Resume on Restart
If the instance dies:
```bash
# On new instance
python forex-ai.py --train
# Automatically finds checkpoint and continues
```

---

## Configuration Options

All optional - system works with zero config:

```bash
# GitHub backup (HIGHLY RECOMMENDED)
export FOREX_BOT_GITHUB_REPO="https://github.com/..."
export FOREX_BOT_GIT_NAME="HPC Bot"
export FOREX_BOT_GIT_EMAIL="bot@example.com"

# Intervals (defaults are good)
export FOREX_BOT_CHECKPOINT_MINUTES=15
export FOREX_BOT_BACKUP_MINUTES=30

# Credit threshold for auto-exit
export FOREX_BOT_LOW_CREDIT_THRESHOLD=5.0

# Force HPC mode (if auto-detect fails)
export FOREX_BOT_HPC_MODE=1
```

---

## Thread Configuration (Automatic)

| Resource | Count | Usage |
|----------|-------|-------|
| Primary threads | 0-251 | Rayon compute, backtesting |
| SMT threads | 252-503 | I/O, GPU coordination |
| Reserved | 12 physical (24 logical) | OS, monitoring |
| **Total used** | **480** | **Maximum throughput** |

---

## What Gets Saved

### Local (`checkpoints/hpc/`)
- `session_YYYYMMDD_HHMMSS_{phase}_gen{N}.pkl`
- Contains: population, metrics, metadata
- Last 10 checkpoints kept (auto-cleanup)

### GitHub (`hpc-results` branch)
- `strategies.json` - Top strategies
- `models/` - Trained models
- `checkpoints/` - Latest checkpoint

### Final (`results/`)
- `hpc_final_{session_id}.json` - Complete results

---

## Resume Scenarios

### Scenario 1: Credits Run Out
```
... running generation 150/200 ...
💳 Credits low ($4.50), preparing to save and exit...
💾 Checkpoint saved: checkpoints/hpc/session_XYZ_gen150.pkl
✅ Backup pushed to GitHub: hpc-results
👋 Safe to terminate.
```

**Resume:**
```bash
# Same or new instance
python forex-ai.py --train
# Detects checkpoint, resumes from generation 150
```

### Scenario 2: Instance Pre-empted
```
🚨 Shutdown signal received, saving progress...
💾 Checkpoint saved
✅ Backup pushed to GitHub
```

**Resume:**
```bash
# Clone backup on new instance
git clone -b hpc-results https://github.com/.../forex-strategies
cd forex-strategies
python forex-ai.py --train
# Resumes from GitHub backup
```

---

## Performance

| Metric | Target |
|--------|--------|
| GPU evaluation | 600K strategies/sec/GPU |
| Total (8 GPUs) | 4.8M strategies/sec |
| CPU validation | 100K strategies/sec |
| Full pipeline | ~20 hours → ~6-8 hours |

---

## Safety Features

1. **Signal Handling**: SIGTERM, SIGINT, SIGUSR1 (pre-emption warning)
2. **Credit Monitoring**: Auto-save when <$5 credits
3. **Atomic Writes**: Checkpoints written to temp file then renamed
4. **Retry Logic**: GitHub push retries 3 times
5. **Cleanup**: Old checkpoints auto-deleted (keep last 10)

---

## No More Scripts!

**Before (fragmented):**
- `run_hyperstack_n3.sh` - Hardware check
- `run_hyperstack_n3.sh discovery` - Run discovery
- `run_hyperstack_n3.sh validate` - Run validation
- Manual git commits
- Hope nothing crashes

**After (unified):**
- `python forex-ai.py --train` - Everything automatic

---

## Testing

Test the integration without HPC hardware:

```python
# Test checkpoint manager
from forex_bot.hpc_coordinator import HPCCheckpointManager
cp = HPCCheckpointManager()
cp.save_checkpoint("test", 1, [{"strategy": "test"}], {"metric": 1.0})

# Test GitHub backup
from forex_bot.hpc_coordinator import GitHubBackupManager
gh = GitHubBackupManager("https://github.com/...")
gh.auto_backup([{"strategy": "test"}], force=True)
```

---

## Next Steps

1. **Set your GitHub repo:**
   ```bash
   export FOREX_BOT_GITHUB_REPO="https://github.com/yourusername/yourrepo"
   ```

2. **Run on Hyperstack:**
   ```bash
   python forex-ai.py --train
   ```

3. **Monitor:**
   ```bash
   tail -f logs/forex_bot.log | grep -E "(HPC|checkpoint|GitHub|Generation)"
   ```

4. **Relax** - it will auto-save, backup, and resume if needed.

---

## Files Modified/Created

```
src/forex_bot/
├── hpc_coordinator.py          [NEW] - Unified orchestrator
└── main.py                     [MOD] - HPC detection + integration

crates/forex-search/src/
├── hpc.rs                      [MOD] - 504 thread support
├── hpc_gpu_discovery.rs        [NEW] - Island model GA
└── hpc_simd.rs                 [NEW] - AVX2 optimizations

Documentation/
├── HPC_UNIFIED_MODE.md         [NEW] - User guide
├── HPC_INTEGRATION_SUMMARY.md  [NEW] - This file
└── SMT_THREAD_CONFIGURATION.md [NEW] - Thread details
```

---

**Bottom line: One command. Zero configuration. Never lose progress.**

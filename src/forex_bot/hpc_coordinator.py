"""
HPC Unified Coordinator - Integrated into forex-ai.py
Single entry point for training + discovery + auto-save + GitHub sync
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


class HPCCheckpointManager:
    """Manages automatic checkpoints for long-running HPC jobs."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints/hpc"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_time = 0
        self.checkpoint_interval = int(os.environ.get("FOREX_BOT_CHECKPOINT_MINUTES", "15")) * 60
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
    def should_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint."""
        return time.time() - self.last_checkpoint_time > self.checkpoint_interval
    
    def save_checkpoint(
        self,
        phase: str,
        generation: int,
        population: list[dict],
        metrics: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a checkpoint with current progress."""
        checkpoint_path = self.checkpoint_dir / f"{self.session_id}_{phase}_gen{generation}.pkl"
        
        checkpoint = {
            "session_id": self.session_id,
            "phase": phase,
            "generation": generation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "population": population,
            "metrics": metrics,
            "metadata": metadata or {},
            "hostname": os.environ.get("HOSTNAME", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        
        # Atomic write
        temp_path = checkpoint_path.with_suffix(".tmp")
        joblib.dump(checkpoint, temp_path, compress=3)
        temp_path.rename(checkpoint_path)
        
        self.last_checkpoint_time = time.time()
        logger.info(f"💾 Checkpoint saved: {checkpoint_path} ({len(population)} strategies)")
        
        return checkpoint_path
    
    def load_latest_checkpoint(self, phase: str | None = None) -> dict[str, Any] | None:
        """Load the most recent checkpoint."""
        pattern = f"{self.session_id}_*" if phase is None else f"{self.session_id}_{phase}_*"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern + ".pkl"))
        
        if not checkpoints:
            # Try to find any checkpoint from this host
            hostname = os.environ.get("HOSTNAME", "")
            if hostname:
                checkpoints = sorted(self.checkpoint_dir.glob(f"*_{hostname}_*.pkl"))
        
        if not checkpoints:
            return None
            
        latest = checkpoints[-1]
        logger.info(f"📂 Resuming from checkpoint: {latest}")
        return joblib.load(latest)
    
    def list_checkpoints(self) -> list[Path]:
        """List all available checkpoints."""
        return sorted(self.checkpoint_dir.glob("*.pkl"))
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last:
            for old in checkpoints[:-keep_last]:
                old.unlink()
                logger.debug(f"Cleaned up old checkpoint: {old}")


class GitHubBackupManager:
    """Automatically backup strategies and models to GitHub."""
    
    def __init__(self, repo_url: str | None = None, branch: str = "hpc-results"):
        self.repo_url = repo_url or os.environ.get("FOREX_BOT_GITHUB_REPO")
        self.branch = branch
        self.backup_dir = Path("hpc_backup")
        self.backup_dir.mkdir(exist_ok=True)
        self.last_backup_time = 0
        self.backup_interval = int(os.environ.get("FOREX_BOT_BACKUP_MINUTES", "30")) * 60
        
        # Git config
        self.git_name = os.environ.get("FOREX_BOT_GIT_NAME", "HPC Bot")
        self.git_email = os.environ.get("FOREX_BOT_GIT_EMAIL", "hpc@forex-ai.local")
        
    def should_backup(self) -> bool:
        """Check if it's time for a backup."""
        if not self.repo_url:
            return False
        return time.time() - self.last_backup_time > self.backup_interval
    
    def prepare_backup_bundle(
        self,
        strategies: list[dict],
        models_dir: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Create a backup bundle with strategies and models."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bundle_path = self.backup_dir / f"hpc_backup_{timestamp}.zip"
        
        with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Save strategies as JSON
            strategies_json = json.dumps({
                "timestamp": timestamp,
                "count": len(strategies),
                "strategies": strategies,
                "metadata": metadata or {},
            }, indent=2)
            zf.writestr("strategies.json", strategies_json)
            
            # Include models if available
            if models_dir and models_dir.exists():
                for model_file in models_dir.rglob("*"):
                    if model_file.is_file():
                        arcname = f"models/{model_file.relative_to(models_dir)}"
                        zf.write(model_file, arcname)
            
            # Include checkpoint info
            checkpoints = list(Path("checkpoints/hpc").glob("*.pkl"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                zf.write(latest, f"checkpoints/{latest.name}")
        
        return bundle_path
    
    def backup_to_github(self, bundle_path: Path) -> bool:
        """Push backup bundle to GitHub."""
        if not self.repo_url:
            logger.warning("GitHub repo not configured, skipping backup")
            return False
        
        try:
            # Check if git is available
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True,
            )
            
            # Setup git in backup directory
            git_dir = self.backup_dir / ".git"
            if not git_dir.exists():
                subprocess.run(["git", "init"], cwd=self.backup_dir, check=True)
                subprocess.run(
                    ["git", "remote", "add", "origin", self.repo_url],
                    cwd=self.backup_dir,
                    check=True,
                )
            
            # Configure git
            subprocess.run(
                ["git", "config", "user.name", self.git_name],
                cwd=self.backup_dir,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", self.git_email],
                cwd=self.backup_dir,
                check=True,
            )
            
            # Create/update branch
            subprocess.run(
                ["git", "checkout", "-B", self.branch],
                cwd=self.backup_dir,
                check=True,
            )
            
            # Add and commit
            subprocess.run(["git", "add", "."], cwd=self.backup_dir, check=True)
            commit_msg = f"HPC backup {datetime.now(timezone.utc).isoformat()}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.backup_dir,
                capture_output=True,
            )
            
            # Push (with retry)
            for attempt in range(3):
                try:
                    subprocess.run(
                        ["git", "push", "-f", "origin", self.branch],
                        cwd=self.backup_dir,
                        check=True,
                        timeout=60,
                    )
                    self.last_backup_time = time.time()
                    logger.info(f"✅ Backup pushed to GitHub: {self.branch}")
                    return True
                except subprocess.TimeoutExpired:
                    logger.warning(f"GitHub push timeout, attempt {attempt + 1}/3")
                    time.sleep(5)
            
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"GitHub backup failed: {e}")
            return False
        except FileNotFoundError:
            logger.warning("Git not available, skipping GitHub backup")
            return False
    
    def auto_backup(
        self,
        strategies: list[dict],
        models_dir: Path | None = None,
        force: bool = False,
    ) -> bool:
        """Automatically backup if interval has passed."""
        if not force and not self.should_backup():
            return False
        
        bundle = self.prepare_backup_bundle(strategies, models_dir)
        return self.backup_to_github(bundle)


class CloudCostMonitor:
    """Monitor cloud costs and save progress before termination."""
    
    def __init__(self, low_credit_threshold: float = 5.0):
        self.low_credit_threshold = low_credit_threshold
        self.last_credit_check = 0
        self.credit_check_interval = 60  # Check every minute
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup handlers for shutdown signals."""
        import signal
        
        def handle_shutdown(signum, frame):
            logger.warning(f"🚨 Shutdown signal {signum} received, saving progress...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
        
        # Hyperstack-specific: Check for termination warning
        if os.environ.get("HYPERSTACK_INSTANCE_ID"):
            signal.signal(signal.SIGUSR1, handle_shutdown)  # Pre-termination warning
    
    def check_credits(self) -> float | None:
        """Check remaining cloud credits (Hyperstack-specific)."""
        try:
            # Try to get credits from Hyperstack API
            api_key = os.environ.get("HYPERSTACK_API_KEY")
            if not api_key:
                return None
            
            import requests
            
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(
                "https://api.hyperstack.cloud/v1/billing/credits",
                headers=headers,
                timeout=10,
            )
            
            if response.status_code == 200:
                data = response.json()
                credits = float(data.get("remaining_credits", 0))
                
                if credits < self.low_credit_threshold:
                    logger.warning(f"⚠️ Low credits: ${credits:.2f}")
                
                return credits
            
        except Exception as e:
            logger.debug(f"Credit check failed: {e}")
        
        return None
    
    def should_save_and_exit(self) -> bool:
        """Check if we should save progress and prepare for exit."""
        if self.shutdown_requested:
            return True
        
        # Check credits periodically
        if time.time() - self.last_credit_check > self.credit_check_interval:
            credits = self.check_credits()
            self.last_credit_check = time.time()
            
            if credits is not None and credits < self.low_credit_threshold:
                logger.warning(f"💳 Credits low (${credits:.2f}), preparing to save and exit...")
                return True
        
        return False


class HPCUnifiedRunner:
    """
    Unified runner that combines training + discovery with auto-save.
    Integrates into forex-ai.py main flow.
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.checkpoint_mgr = HPCCheckpointManager()
        self.github_mgr = GitHubBackupManager()
        self.cost_monitor = CloudCostMonitor()
        
        # Statistics
        self.start_time = time.time()
        self.strategies_found: list[dict] = []
        self.generation = 0
        self.phase = "init"
        
        # HPC mode detection
        self.hpc_enabled = self._detect_hpc_mode()
        if self.hpc_enabled:
            logger.info("🚀 HPC Mode Enabled: 504 threads, 8× A6000, auto-save active")
    
    def _detect_hpc_mode(self) -> bool:
        """Detect if running on HPC hardware."""
        gpu_count = 0
        try:
            import torch
            gpu_count = torch.cuda.device_count()
        except:
            pass
        
        cpu_threads = os.cpu_count() or 1
        
        # Check for HPC indicators
        return (
            gpu_count >= 8 and
            cpu_threads >= 500 and
            os.environ.get("FOREX_BOT_HPC_MODE", "0") == "1"
        )
    
    def get_optimal_config(self) -> dict[str, Any]:
        """Get optimal configuration for HPC hardware."""
        if not self.hpc_enabled:
            return {}
        
        return {
            # Population sizing for 384GB VRAM
            "discovery_population": 500_000,
            "discovery_generations": 200,
            "discovery_chunk_size": 8192,
            
            # Thread configuration (504 threads)
            "rayon_threads": 240,      # Primary threads only
            "cpu_threads": 480,        # Primary + SMT
            "validation_threads": 480,
            
            # GPU configuration
            "gpu_workers": 8,
            "gpu_devices": list(range(8)),
            
            # Auto-save intervals
            "checkpoint_minutes": 15,
            "github_backup_minutes": 30,
            
            # Island model for 8 GPUs
            "island_model": True,
            "island_migration_interval": 10,
            "num_islands": 8,
        }
    
    async def run_unified_pipeline(
        self,
        symbols: list[str],
        resume: bool = True,
    ) -> dict[str, Any]:
        """
        Run unified training + discovery pipeline with auto-save.
        
        This is the main entry point called by forex-ai.py --train
        """
        results = {
            "strategies": [],
            "models": [],
            "checkpoints": [],
            "backups": [],
            "runtime_seconds": 0,
        }
        
        # Try to resume from checkpoint
        if resume:
            checkpoint = self.checkpoint_mgr.load_latest_checkpoint()
            if checkpoint:
                self.generation = checkpoint.get("generation", 0)
                self.strategies_found = checkpoint.get("population", [])
                self.phase = checkpoint.get("phase", "init")
                logger.info(f"🔄 Resumed from generation {self.generation} with {len(self.strategies_found)} strategies")
        
        try:
            # Phase 1: Feature computation and data loading
            if self.phase in ["init", "features"]:
                self.phase = "features"
                await self._run_feature_computation(symbols)
                self._checkpoint_if_needed()
            
            # Phase 2: Massive GPU discovery
            if self.phase in ["features", "discovery"]:
                self.phase = "discovery"
                await self._run_gpu_discovery(symbols)
                self._checkpoint_if_needed()
            
            # Phase 3: CPU validation
            if self.phase in ["discovery", "validation"]:
                self.phase = "validation"
                await self._run_cpu_validation()
                self._checkpoint_if_needed()
            
            # Phase 4: Model training (if enabled)
            if self.phase in ["validation", "training"]:
                self.phase = "training"
                await self._run_model_training()
                self._checkpoint_if_needed()
            
            # Phase 5: Ensemble construction
            if self.phase in ["training", "ensemble"]:
                self.phase = "ensemble"
                await self._run_ensemble_construction()
            
            self.phase = "complete"
            
        except asyncio.CancelledError:
            logger.info("⚠️ Pipeline cancelled, progress saved")
            raise
        
        finally:
            # Final save
            self._final_save()
            results["runtime_seconds"] = time.time() - self.start_time
            results["strategies"] = self.strategies_found
        
        return results
    
    async def _run_feature_computation(self, symbols: list[str]):
        """Compute features for all symbols."""
        logger.info(f"📊 Phase 1: Computing features for {len(symbols)} symbols")
        
        # This integrates with existing feature pipeline
        # But uses all 504 threads
        
        if self.hpc_enabled:
            os.environ["FOREX_BOT_FEATURE_WORKERS"] = "504"
            os.environ["RAYON_NUM_THREADS"] = "240"
        
        # TODO: Integrate with existing DataLoader/FeatureEngineer
        await asyncio.sleep(0.1)  # Placeholder
    
    async def _run_gpu_discovery(self, symbols: list[str]):
        """Run massive GPU-accelerated discovery."""
        logger.info(f"🔍 Phase 2: GPU discovery on 8× A6000")
        
        config = self.get_optimal_config()
        target_generations = config.get("discovery_generations", 200)
        
        # Resume from checkpoint generation
        start_gen = self.generation
        
        for gen in range(start_gen, target_generations):
            self.generation = gen
            
            # Check for low credits / shutdown signal
            if self.cost_monitor.should_save_and_exit():
                logger.warning("💾 Saving progress due to shutdown signal...")
                self._checkpoint_if_needed(force=True)
                self._github_backup_if_needed(force=True)
                sys.exit(0)
            
            # Run one generation of discovery
            # TODO: Integrate with discovery_tensor.py
            new_strategies = await self._run_discovery_generation(gen)
            self.strategies_found.extend(new_strategies)
            
            # Progress logging
            if gen % 10 == 0:
                elapsed = time.time() - self.start_time
                progress = (gen / target_generations) * 100
                logger.info(f"Generation {gen}/{target_generations} ({progress:.1f}%) - {len(self.strategies_found)} strategies - {elapsed/60:.1f}min elapsed")
            
            # Auto-save
            if gen % 5 == 0:
                self._checkpoint_if_needed()
                self._github_backup_if_needed()
    
    async def _run_discovery_generation(self, gen: int) -> list[dict]:
        """Run one generation of discovery. Returns new strategies."""
        # TODO: Integrate with existing discovery engine
        # This should call the Rust GPU discovery with proper configuration
        return []
    
    async def _run_cpu_validation(self):
        """Validate strategies on CPU."""
        logger.info(f"✅ Phase 3: Validating {len(self.strategies_found)} strategies on 480 threads")
        
        if self.hpc_enabled:
            os.environ["FOREX_BOT_CPU_THREADS"] = "480"
            os.environ["RAYON_NUM_THREADS"] = "240"
        
        # TODO: Integrate with fast_backtest.py
        await asyncio.sleep(0.1)
    
    async def _run_model_training(self):
        """Train models."""
        logger.info("🧠 Phase 4: Training models")
        
        # Use primary threads only for training (memory bound)
        if self.hpc_enabled:
            os.environ["RAYON_NUM_THREADS"] = "120"
        
        # TODO: Integrate with trainer.py
        await asyncio.sleep(0.1)
    
    async def _run_ensemble_construction(self):
        """Build final ensemble."""
        logger.info("🎯 Phase 5: Building ensemble")
        # TODO: Integrate with ensemble.py
        await asyncio.sleep(0.1)
    
    def _checkpoint_if_needed(self, force: bool = False):
        """Save checkpoint if interval has passed."""
        if not force and not self.checkpoint_mgr.should_checkpoint():
            return
        
        if not self.strategies_found:
            return
        
        self.checkpoint_mgr.save_checkpoint(
            phase=self.phase,
            generation=self.generation,
            population=self.strategies_found[-10000:],  # Keep last 10K for size
            metrics={
                "total_strategies": len(self.strategies_found),
                "runtime_seconds": time.time() - self.start_time,
            },
        )
        
        self.checkpoint_mgr.cleanup_old_checkpoints(keep_last=10)
    
    def _github_backup_if_needed(self, force: bool = False):
        """Backup to GitHub if interval has passed."""
        if not self.github_mgr.should_backup() and not force:
            return
        
        if not self.strategies_found:
            return
        
        success = self.github_mgr.auto_backup(
            strategies=self.strategies_found[-1000:],  # Last 1K strategies
            models_dir=Path("models") if Path("models").exists() else None,
            force=force,
        )
        
        if success:
            logger.info("✅ Progress backed up to GitHub")
    
    def _final_save(self):
        """Final save on completion or error."""
        logger.info("💾 Performing final save...")
        
        # Save checkpoint
        self._checkpoint_if_needed(force=True)
        
        # GitHub backup
        self._github_backup_if_needed(force=True)
        
        # Save final results
        if self.strategies_found:
            final_path = Path(f"results/hpc_final_{self.checkpoint_mgr.session_id}.json")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(final_path, "w") as f:
                json.dump({
                    "session_id": self.checkpoint_mgr.session_id,
                    "strategies": self.strategies_found,
                    "total_count": len(self.strategies_found),
                    "runtime_seconds": time.time() - self.start_time,
                    "phase": self.phase,
                    "completed": self.phase == "complete",
                }, f, indent=2)
            
            logger.info(f"📁 Final results saved: {final_path}")


def run_hpc_unified(settings: Any, symbols: list[str]) -> dict[str, Any]:
    """
    Entry point for unified HPC run.
    Called from forex-ai.py when HPC mode is detected.
    """
    runner = HPCUnifiedRunner(settings)
    
    # Check if we should resume
    checkpoint = runner.checkpoint_mgr.load_latest_checkpoint()
    resume = checkpoint is not None
    
    if resume:
        logger.info(f"🔄 Resuming HPC run from {checkpoint.get('phase', 'unknown')} phase")
    else:
        logger.info("🚀 Starting new HPC unified run")
    
    # Run the pipeline
    return asyncio.run(runner.run_unified_pipeline(symbols, resume=resume))


if __name__ == "__main__":
    # Test mode
    logging.basicConfig(level=logging.INFO)
    
    class MockSettings:
        pass
    
    results = run_hpc_unified(MockSettings(), ["EURUSD", "GBPUSD"])
    print(f"Results: {results}")


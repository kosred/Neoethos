# Generic Scheduler / Small Files Refactor Note

Created: 2026-05-04 Europe/Berlin
Repository: kosred/forex-ai
Scope: record the decision that Hyperstack-specific logic should be replaced by a generic hardware scheduler, and that files should stay small and focused.

## User idea

The user pointed out that the Hyperstack-specific code can eventually be removed completely, because the final runtime logic should work regardless of GPU type or GPU count.

The user also emphasized that files should remain small, clean, and focused.

Both points are correct and should guide the refactor.

## Core decision

Do not build the future architecture around a special Hyperstack path.

Instead, build a generic hardware/runtime scheduler that works with:

- CPU-only machines
- one GPU
- multiple consumer GPUs
- multiple workstation/server GPUs
- mixed GPU sizes
- PCIe-only topologies
- NVLink topologies when available
- future accelerators

Hyperstack N3 / 8xA6000 can exist only as one detected hardware profile or preset, not as a separate hardcoded execution path.

## Why Hyperstack-specific logic should go away

The current `hpc.rs` encodes assumptions such as:

- 8x RTX A6000
- 48GB VRAM per GPU
- 252 physical cores / 504 logical threads
- 464GB RAM
- two NUMA sockets
- fixed NVLink pairs
- fixed chunk size
- fixed large population

This does not generalize to hardware such as 16x RTX 4060 cards, single-GPU systems, or other future machines.

The correct abstraction is:

```rust
HardwareProfile
HardwareTopology
GpuTopology
NumaTopology
ResolvedRuntimeConfig
DeviceAssignment
SchedulerProfile
WorkUnit
ShardingStrategy
```

The scheduler should decide how to distribute work based on detected hardware, not based on a hardcoded vendor/server profile.

## What to preserve from `hpc.rs`

The file contains useful ideas that should be preserved as generic concepts:

- detect available GPUs
- detect VRAM
- detect CPU threads
- detect RAM
- model NUMA locality
- model GPU peer links when available
- reserve coordination threads
- choose chunk sizes from hardware capability
- choose population/search scale from hardware capability

These should be moved into generic hardware detection and scheduler modules.

## What should be removed later

After generic scheduler replacement exists, remove or retire:

- hardcoded Hyperstack N3 activation path
- hardcoded A6000 assumptions
- hardcoded NVLink pair list
- hardcoded CPU affinity ranges
- hardcoded population/chunk-size decisions tied to one machine

Deletion should happen only after callers use the generic scheduler.

## Small and clean file rule

The refactor should avoid large files that own too many concepts.

Preferred structure:

```text
runtime/
  hardware_profile.rs
  hardware_probe.rs
  topology.rs
  device_assignment.rs
  scheduler.rs
  work_unit.rs
  precision_policy.rs
  fallback_policy.rs
```

Each file should have one clear reason to exist.

Examples:

- `hardware_probe.rs`: detects hardware.
- `topology.rs`: represents CPU/GPU/NUMA/peer-link topology.
- `device_assignment.rs`: describes assigned devices for a work unit.
- `scheduler.rs`: maps work units to devices.
- `work_unit.rs`: defines work unit kinds and metadata.
- `precision_policy.rs`: defines fp32/bf16/fp16 policy.
- `fallback_policy.rs`: defines fail-closed vs fallback behavior.

## Refactor rule

If a file starts owning unrelated concepts, split it.

Bad pattern:

```text
one file does hardware detection + scheduling + env parsing + kernel launch + artifact metadata
```

Good pattern:

```text
small files, one concept each, connected by typed structs
```

## Bottom line

The Hyperstack code should not be the foundation of the future system.

The future system should be hardware-agnostic:

```text
hardware detection -> runtime config -> scheduler -> work units -> backend kernels
```

This lets the bot run correctly on any hardware layout while keeping files small, clean, and maintainable.

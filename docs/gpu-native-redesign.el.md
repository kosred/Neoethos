# GPU-native pipeline ανακάλυψης — Stage 1

**Κατάσταση:** Εγκεκριμένο αρχιτεκτονικό αρχείο  
**Στάδιο:** Θεμέλια και ετοιμότητα benchmark  
**Baseline commit:** `2be1408ee3986026fdbb2a5a74aaaf6ac67e5209`  
**Πεδίο:** pipeline discovery/backtesting του `neoethos-search`

Το παρόν έγγραφο είναι η δεσμευτική προδιαγραφή για το Stage 1 του GPU-native redesign του NeoEthos. Ενσωματώνει το εγκεκριμένο σχέδιο και όλες τις μεταγενέστερες διορθώσεις. Προηγούμενες σημειώσεις σχεδιασμού και addenda αντικαθίστανται από αυτή την ενοποιημένη έκδοση.

## 1. Στόχος

Το NeoEthos πρέπει να υποστηρίζει αυστηρό GPU execution mode στο οποίο καμία υπολογιστική εργασία που εξαρτάται από candidates δεν μεταφέρεται σιωπηρά στη CPU. Το τελικό σύστημα κρίνεται από τον συνολικό wall time, την ορθότητα, την παραμονή των δεδομένων στη συσκευή και τα ρητά στοιχεία εκτέλεσης — όχι απλώς από το αν ο host υπέβαλε ένα ή περισσότερα GPU kernels.

Το Stage 1 **δεν** ισχυρίζεται ότι έχει επιλεγεί η τελική GPU αρχιτεκτονική. Δημιουργεί τα contracts, τους μηχανισμούς ορθότητας, το instrumentation, τα εκτελέσιμα prototypes και το remote benchmark kit που απαιτούνται ώστε η απόφαση να βασιστεί σε μετρήσεις πάνω σε νοικιασμένη NVIDIA RTX A6000.

Ο τελικός επιδιωκόμενος κανόνας είναι:

```text
GPU-required work unit:
    CPU strategy-compute executions      = 0
    CPU candidate backtests              = 0
    CPU validation simulations           = 0
    silent CPU fallbacks                  = 0
    intermediate full-result D2H copies  = 0
```

Η CPU επιτρέπεται να χρησιμοποιείται για disk I/O, configuration, launch orchestration, UI/progress reporting, serialization artifacts και compact χειρισμό τελικών αποτελεσμάτων.

## 2. Υφιστάμενα στοιχεία και working hypotheses

Η σημερινή υλοποίηση χρησιμοποιεί έναν worker ανά gene για το population backtest, με σειριακό και branch-heavy equity walk πάνω στα bars. Στα πυκνά timeframes, ο hybrid scheduler έχει μετρήσει τη CPU lane ως ταχύτερη και κατά συνέπεια έχει δρομολογήσει σημαντικό μέρος της εργασίας στη CPU. Σε προηγούμενο νοικιασμένο GPU σύστημα πληρώθηκε GPU χρόνος ενώ το discovery workload εκτελέστηκε κυρίως στη CPU.

Οι αιτίες της χαμηλής GPU απόδοσης είναι **working hypotheses** και όχι αποδεδειγμένα συμπεράσματα. Το Nsight profiling πρέπει να ποσοτικοποιήσει τη συμβολή των εξής:

- σειριακό per-gene bar walk,
- ανεπαρκές candidate-axis occupancy για κανονικά population sizes,
- materialization του signal matrix,
- host-to-device uploads και device-to-host readbacks,
- μικρά ή κατακερματισμένα batches,
- συμπεριφορά allocations και memory pools,
- process-wide ή device-wide launch serialization,
- divergence μεταξύ strategy paths και register pressure,
- CPU-only post-search και validation στάδια.

Καμία αρχιτεκτονική επιλογή ή δήλωση επιτάχυνσης δεν επιτρέπεται να βασιστεί μόνο σε αυτές τις υποθέσεις.

## 3. Όρια των stages

### Stage 1 — η παρούσα προδιαγραφή

Το Stage 1 παραδίδει card-independent και compile-verifiable θεμέλια, καθώς και benchmark readiness:

- ρητό backend και fallback policy,
- fail-fast capability checks,
- audited CPU strategy execution,
- versioned trading και discovery semantics,
- field-specific parity harness δώδεκα επιπέδων,
- device-safe FFI layouts και device-resident engine contracts,
- NVTX και benchmark instrumentation,
- εκτελέσιμα Prototype A, B και C,
- native CUDA scaffolding,
- fail-fast rented-A6000 run kit.

### Stage 2 — μετά τις μετρήσεις στην A6000

Το Stage 2 μεταφέρει **όλη την υπολειπόμενη candidate-dependent CPU computation** που θα εντοπίσει το capability inventory. Το inventory είναι η αυθεντική πηγή· η ακόλουθη λίστα είναι το ελάχιστο και όχι πλήρες όριο:

- signal και minimum-trade filtering,
- quality screening και risk diagnostics,
- prop-firm window simulation,
- candidate-pool correlation και portfolio selection,
- PBO, ranking και host-side validation reductions,
- canonical και forward-tail replay,
- robustness permutation και plateau evaluation,
- host-side fold gathering ή result sorting,
- GA generation, selection, crossover, mutation, deduplication και archive management,
- κάθε μελλοντικό candidate-dependent CPU evaluator που θα εντοπίσει το audit.

### Stage 3 — ξεχωριστό workstream

Το GPU-native ML training και inference είναι ρητά εκτός Stage 1 και Stage 2. Απαιτούν ξεχωριστή αρχιτεκτονική και benchmark plan.

Το mesh παραμένει λειτουργικό σε όλη τη διάρκεια, αλλά δεν αποτελεί προτεραιότητα αυτού του redesign.

## 4. Απαράβατα stop gates

Πριν από πραγματικές μετρήσεις σε discrete NVIDIA GPU απαγορεύονται:

- τελική επιλογή Prototype A, B ή C,
- A6000-specific tuning,
- δημοσίευση speedup claims,
- οριστική διαγραφή υπάρχοντος engine,
- χρήση iGPU measurements ως proxy για A6000,
- αλλαγή strategy semantics για καλύτερα benchmarks,
- απόκρυψη unsupported capabilities μέσω CPU fallback.

Το CubeCL παραμένει portable correctness/reference backend και η διαδρομή για non-NVIDIA hardware.

## 5. Ορισμός GPU-native

Ένα work unit θεωρείται computationally GPU-native μόνο όταν όλη η candidate-dependent αριθμητική εργασία που ζητά εκτελείται σε accelerator backend και ο host δεν την επανυπολογίζει ή ολοκληρώνει στη CPU.

Επιτρεπόμενες host ευθύνες:

- disk και network I/O,
- parsing και validation configuration,
- δημιουργία device sessions και submission εργασίας,
- ελαφρύ progress reporting,
- εγγραφή artifacts και logs,
- ρητό compact readback τελικών survivors ή debug traces.

Απαγορευμένα σε strict GPU mode:

- CPU signal synthesis,
- CPU candidate backtesting,
- CPU Monte Carlo evaluation,
- CPU walk-forward/CPCV/PBO computation,
- CPU risk ή prop-firm simulation,
- CPU candidate correlation ή ranking,
- CPU robustness και replay backtests,
- κρυφό fallback μετά από GPU failure.

## 6. Backend model

Η επιλογή backend αναπαρίσταται από ανεξάρτητους policy axes και όχι από ένα αμφίσημο string.

```rust
pub enum DevicePreference {
    Cpu,
    Auto,
    Gpu,
}

pub enum FallbackPolicy {
    AllowCpu,
    ForbidCpu,
}

pub enum AcceleratorHint {
    Any,
    Cuda,
    Wgpu,
    Vulkan,
    Rocm,
}

pub struct EvaluationBackend {
    pub device: DevicePreference,
    pub fallback: FallbackPolicy,
    pub accelerator_hint: AcceleratorHint,
}
```

Canonical configuration mapping:

| Configuration | Device | Fallback |
|---|---:|---:|
| `cpu` | CPU | επιτρέπεται |
| `auto` | αυτόματο | επιτρέπεται |
| `gpu` | GPU preferred | επιτρέπεται |
| `gpu_required` | GPU required | απαγορεύεται |

Για το discovery, το `models.prop_search_device` υπερισχύει του global `system.enable_gpu_preference`. Το `NEOETHOS_REQUIRE_GPU` μπορεί μόνο να κλιμακώσει την εκτέλεση σε `Gpu + ForbidCpu` και δεν μπορεί ποτέ να υποβαθμίσει ήδη αυστηρό mode.

Το environment boolean parsing είναι ρητό:

```text
true:  1, true, yes, on
false: 0, false, no, off, empty, unset
```

Άκυροι συνδυασμοί όπως `Cpu + ForbidCpu` αποτυγχάνουν στο configuration ή στο preflight.

## 7. GPU failure policy

Ο χειρισμός GPU failures είναι typed και mode-aware.

```rust
pub enum GpuAction {
    RetryOnGpu,
    FallbackToCpu,
    FailLoud,
}
```

Κανόνες:

- `ParityViolation` αποτυγχάνει πάντα φανερά.
- `WrongShape` είναι internal/correctness failure και αποτυγχάνει πάντα φανερά.
- `NoAdapter`, `UnsupportedBackend` και `DeviceLost` μπορούν να κάνουν fallback μόνο όταν αυτό επιτρέπεται ρητά.
- `AllocationPressure` προκαλεί πρώτα bounded GPU-only rebatching ή μείωση workspace.
- Το strict GPU mode αποτυγχάνει μόνο αφού εξαντληθούν τα deterministic GPU retries και δεν κάνει ποτέ CPU fallback.

Το rebatching πρέπει να διατηρεί ακριβώς:

- candidate IDs,
- scenario IDs,
- candidate και scenario ordering,
- deterministic RNG counters,
- τελικό output ordering και semantics.

Τα retry limits, τα batch sizes που δοκιμάστηκαν και η τελική αιτία αποτυχίας αναφέρονται ρητά.

## 8. CPU strategy-compute audit

Όλοι οι CPU strategy evaluators δρομολογούνται μέσω κεντρικού audited wrapper αντί να βασίζονται σε διάσπαρτα guards.

```rust
cpu_strategy::run(
    backend,
    audit_context,
    category,
    call_site,
    || { /* CPU strategy computation */ },
)
```

Το audit είναι scoped ανά work unit και ποτέ process-global.

```rust
pub struct CpuStrategyAuditContext {
    pub work_unit_id: WorkUnitId,
    pub attempted_by_category: Counters,
    pub executed_by_category: Counters,
}
```

Οι ελάχιστες κατηγορίες περιλαμβάνουν population evaluation, signal synthesis, candidate backtest, validation simulation, risk diagnostics, correlation/ranking και robustness/replay.

Σε `ForbidCpu`, μια attempted CPU call καταγράφεται και απορρίπτεται πριν εκτελεστεί. Ένα καθαρό strict-mode run απαιτεί μηδενικές executed CPU strategy calls.

Τα CPU reference/parity runs εκτελούνται σε ξεχωριστό validation mode και δεν αναμειγνύονται ποτέ με production GPU-required timing runs.

## 9. Full-pipeline capability preflight

Πριν ξεκινήσει η GA, το strict GPU mode εκτελεί typed capability preflight σε κάθε απαιτούμενο stage.

Αν οποιοδήποτε stage δεν έχει GPU implementation, το run αποτυγχάνει αμέσως με πλήρη λίστα unsupported stages. Δεν επιτρέπεται να ξοδεύει ώρες στη GA και να αποτυγχάνει αργότερα σε CPU-only gate.

Κατά το Stage 1, το full discovery σε strict mode αναμένεται να αναφέρει unsupported Stage 2 στάδια. Τα engine-only και ήδη GPU validation benchmarks μπορούν να ελέγχουν προσωρινά το zero-CPU invariant.

Το capability manifest παράγεται από το πραγματικό pipeline inventory και περιλαμβάνει backend, engine, strategy-feature και scenario support.

## 10. Versioned semantics

Δύο ανεξάρτητοι canonical descriptors αποτρέπουν τη σύγχυση των execution semantics με το search/selection policy.

### 10.1 Trading semantics

```rust
pub const TRADING_SEMANTICS_VERSION: u32 = ...;
```

Το `TRADING_SEMANTICS_HASH` παράγεται από versioned canonical `TradingSemanticsDescriptor` και όχι από documentation text ή source bytes. Καλύπτει τουλάχιστον:

- signal threshold και direction rules,
- SMC gate behaviour,
- entry timing,
- same-bar SL/TP precedence,
- fixed και adaptive stops,
- trailing και break-even rules,
- spread, commission, swap και conversion costs,
- confidence και position sizing,
- equity και drawdown state,
- daily/monthly/prop-firm calendar boundaries,
- maximum-hold και forced-exit behaviour.

### 10.2 Discovery semantics

```rust
pub const DISCOVERY_SEMANTICS_VERSION: u32 = ...;
```

Το `DISCOVERY_SEMANTICS_HASH` παράγεται από versioned canonical `DiscoverySemanticsDescriptor`. Καλύπτει τουλάχιστον:

- fitness fields και formulas,
- integer/quantized ranking policy,
- tie-breaking και candidate identity,
- filtering rules,
- validation verdict rules,
- PBO/WF/CPCV selection policy,
- survivor και portfolio selection policy.

Και τα δύο ζεύγη version/hash γράφονται σε benchmark reports και validation artifacts.

Το γνωστό SMC mismatch επιλύεται πριν επεκταθεί το parity. Ένα legacy-before έναντι canonical-after fixture καταγράφει κάθε σκόπιμη αλλαγή survivors. Από εκείνο το σημείο και μετά, κάθε backend πρέπει να αναπαράγει τα νέα canonical semantics· δεν υποσχόμαστε διατήρηση των legacy survivors.

## 11. Canonical ranking και identity

Το floating-point tolerance δεν μπορεί να συνυπάρχει με exact survivor ordering χωρίς κοινό canonical key.

CPU και GPU κατασκευάζουν το ίδιο integer/quantized `RankKey`. Η quantization policy αποτελεί μέρος του `DISCOVERY_SEMANTICS_HASH`.

Τελική σειρά tie-breaking:

1. canonical primary rank fields,
2. canonical secondary rank fields,
3. gene signature hash ως γρήγορος discriminator,
4. canonical serialized gene bytes,
5. stable candidate/trial ID.

Η canonical gene serialization πρέπει:

- να ταξινομεί indicator-weight pairs με καθορισμένη σειρά,
- να κανονικοποιεί το negative zero,
- να απορρίπτει non-finite values,
- να χρησιμοποιεί ρητά field widths και endianness,
- να περιλαμβάνει όλα τα semantic fields,
- να μη βασίζεται αποκλειστικά στο υπάρχον quantized hash.

## 12. Parity hierarchy δώδεκα επιπέδων

Το parity αξιολογείται σε αιτιώδη σειρά. Η ταύτιση μόνο του τελικού PnL δεν είναι επαρκής.

1. score πριν από threshold,
2. signal direction,
3. confidence,
4. candidate entry events,
5. candidate exit bar και reason,
6. accepted-trade sequence,
7. position size και costs,
8. equity μετά από κάθε accepted trade,
9. daily/monthly/prop-firm state,
10. final metrics,
11. validation verdict,
12. final survivor ordering.

Απαιτείται exact comparison για discrete state: signals, bars, reasons, IDs, accepted-trade order, verdicts και survivor order.

Scores, confidence, equity, PnL και derived floating metrics χρησιμοποιούν δηλωμένες field-specific absolute, relative και ULP policies. Τα tolerances είναι versioned και καταγράφονται· δεν υπάρχει ένα global magic epsilon.

Το `compare_traces()` αναφέρει το πρώτο επίπεδο απόκλισης.

Τα levels 1–9 εκτελούνται με tiny direct engine fixtures. Τα levels 10–12 εκτελούνται με deterministic Stage 1 integration fixture. Επιτυχία στα levels 10–12 κατά το Stage 1 δεν αποδεικνύει ότι ολόκληρο το production pipeline είναι GPU-native.

Το GPU trace path είναι ξεχωριστό compile-time kernel specialization με ξεχωριστά trace buffers και όχι runtime branch στο production kernel.

Τα integrated-GPU tests καλούν απευθείας το CubeCL/WGPU backend μέσω test-only override και δεν εξαρτώνται από τον production scheduler που σκόπιμα παραλείπει iGPUs.

## 13. Contracts και FFI layouts

Το `neoethos-gpu-contracts` περιέχει δύο ανεξάρτητα representation layers.

### Host DTOs

Ergonomic Rust/Serde structures μπορούν να χρησιμοποιούν `Vec`, `String` και typed enums.

### Device και FFI POD

Οι device-facing structures χρησιμοποιούν `#[repr(C)]`, fixed-width primitives και ρητά offsets/counts. Δεν περιέχουν Rust `Vec`, `String`, native-layout enums ή unbounded `bool`.

Rust και C++ compile-time assertions επαληθεύουν:

- συνολικό size,
- alignment,
- field offsets,
- enum/tag values,
- κοινό `ABI_VERSION`.

Το contract καλύπτει datasets, gene CSR arrays, scenarios, outcomes, trades, metrics, prop-firm state και index maps.

## 14. Scenario descriptors και deterministic RNG

Το global scenario batching χρησιμοποιεί compact device descriptors αντί για cloned genes.

Ένας descriptor περιλαμβάνει:

- base candidate ID,
- scenario ID και type,
- seed/counter,
- window/index-map descriptor,
- cost overrides,
- parameter perturbation descriptor,
- segmented-reduction key.

Ένα counter-based Philox-style RNG contract παράγει parameters στη συσκευή από stable identifiers. Η αλλαγή ordering ή rebatching δεν επιτρέπεται να μεταβάλλει τα scenarios.

## 15. Device-resident BacktestEngine

Το engine API δεν πρέπει να επιβάλλει host readback μεταξύ pipeline stages.

Εννοιολογικό contract:

```rust
trait BacktestEngine {
    fn upload_session(...) -> Result<DatasetHandle>;
    fn upload_genes(...) -> Result<GeneBufferHandle>;
    fn upload_scenarios(...) -> Result<ScenarioBufferHandle>;

    fn evaluate(
        &self,
        dataset: &DatasetHandle,
        genes: &GeneBufferHandle,
        scenarios: &ScenarioBufferHandle,
    ) -> Result<DeviceMetricsHandle>;

    fn filter(
        &self,
        metrics: &DeviceMetricsHandle,
        policy: &DeviceFilterPolicy,
    ) -> Result<DeviceSelectionHandle>;

    fn readback_compact(
        &self,
        selection: &DeviceSelectionHandle,
    ) -> Result<HostSurvivorSummary>;
}
```

Τα opaque handles συνδέονται με:

- session ID,
- backend ID,
- physical/logical device ID,
- workspace generation/version,
- buffer kind,
- parent buffer/session relationships όπου απαιτείται.

Runtime validation απορρίπτει stale, cross-session, cross-device και cross-backend handles.

Τα synchronization semantics είναι ρητά. Οι operations είτε καταναλώνουν/επιστρέφουν event ή fence handles είτε ορίζουν καθαρά blocking behaviour. Το trait δεν επιτρέπεται να βασίζεται σε κρυφό global synchronization.

Το device-transfer instrumentation καταγράφει:

- dataset uploads ανά session,
- gene/scenario uploads,
- full και compact D2H copies,
- reuploads μεταξύ chained stages,
- synchronization events,
- transferred bytes.

Η αποδοχή του Prototype A απαιτεί ένα dataset upload ανά session, κανένα intermediate full-metric D2H readback, κανένα reupload μεταξύ chained stages και μόνο explicit compact readback.

## 16. Ownership του GpuDiscoverySession

Ένα session κατέχει το accelerator context και τα reusable allocations για ένα symbol/timeframe work unit.

```text
GpuDiscoverySession
  dataset και feature buffers
  SMC και calendar arrays
  gene buffers
  scenario descriptors
  validation index maps
  reusable workspaces
  device metrics και selections
  stream/event resources
  transfer counters
```

Ο τελικός scheduler χρησιμοποιεί bounded streams, reusable workspaces, explicit dependencies και memory-aware backpressure. Process-wide launch mutex επιτρέπεται μόνο ως documented migration guard και απαγορεύεται στα benchmarked prototype paths.

## 17. Benchmark methodology

Ο benchmark runner υποστηρίζει δύο fixture classes:

- deterministic tiny fixtures για correctness,
- hashed representative real-data snapshots για H1, M30, M15, M5 και M1.

Εκτελεί ξεχωριστά passes για:

1. clean wall-time measurements,
2. diagnostic counters,
3. Nsight Systems,
4. Nsight Compute.

Diagnostics και traces δεν επιτρέπεται να επηρεάζουν clean timing runs.

Τα sweeps περιλαμβάνουν:

- population και GPU batch size,
- αριθμό bars,
- αριθμό features,
- scenario count και density,
- fixed-row workloads,
- fixed-calendar-duration workloads.

Κάθε report περιέχει:

- Git SHA,
- legacy/canonical baseline identity,
- dataset και configuration hashes,
- trading και discovery semantics versions/hashes,
- seed,
- backend και prototype,
- CPU, RAM, GPU και device class,
- driver, runtime και CUDA toolkit versions,
- clocks, power και thermal state όπου διατίθενται,
- warm-up count και measured repetitions,
- median, P95 και variance,
- candidates/s, candidate-bars/s και trades/s,
- peak VRAM,
- event density και hold-length distribution,
- transfer counts και bytes,
- parity status και capability coverage.

Unsupported metrics γράφονται ως `null` και δεν κατασκευάζονται τεχνητά.

## 18. Σύγκριση prototypes

### Prototype A — fused exact bar walk

Persistent και exact GPU baseline που συνδυάζει signal synthesis, SMC gating, position state και metrics, διατηρώντας dataset και intermediates στη συσκευή.

Acceptance:

- levels 1–9 περνούν απευθείας από το engine,
- levels 10–12 περνούν από το Stage 1 integration harness,
- ένα dataset upload ανά session,
- κανένα dense signal readback,
- κανένα CPU candidate post-processing,
- μόνο compact output.

Αποτελεί correctness baseline και όχι προεπιλεγμένο νικητή.

### Prototype B — warp/subwarp cooperative walk

Πραγματικό εκτελέσιμο prototype στο οποίο warp ή supported subgroup συνεργάζεται σε candidate work όπου είναι χρήσιμο. Δεν ισχυρίζεται όφελος πριν από μετρήσεις.

Η εκτέλεση σε WGPU/iGPU εξαρτάται από capabilities. Η υλοποίηση ελέγχει subgroup operations και width· unsupported devices επιστρέφουν typed `UnsupportedCapability`. Το πραγματικό correctness/performance test εκτελείται στην A6000 όπου απαιτείται.

### Prototype C — sparse event, first hit και device stitch

Πραγματικό εκτελέσιμο minimal engine για δηλωμένο strategy subset. Εκτελεί event compaction, fixed/adaptive barrier first-hit και exact device-side stitching. Unsupported trailing ή path-dependent semantics επιστρέφουν typed status και δεν κάνουν panic ή σιωπηλό CPU fallback.

### Κανόνες δίκαιης σύγκρισης

Όλες οι A/B/C μετρήσεις χρησιμοποιούν ίδια datasets, genes, scenarios και strategy subsets.

Τα reports διαχωρίζουν:

- performance στο common capability intersection,
- coverage και unsupported percentage στο πλήρες workload.

Η algorithm comparison χρησιμοποιεί το ίδιο backend. Η σύγκριση CubeCL έναντι native CUDA γίνεται ως ξεχωριστός άξονας, ώστε να μη συγχέονται backend και architecture differences.

## 19. Native CUDA scaffold

Το `neoethos-gpu-cuda` παρέχει CUDA C++/CCCL κώδικα πίσω από stable C ABI πάνω στα κοινά POD layouts.

Card-independent checks:

- CUDA C++ compilation όπου υπάρχει toolkit,
- Rust/C++ link και symbol checks,
- ABI version validation,
- size/alignment/offset static assertions.

Real-GPU-gated checks:

- Rust → C ABI call,
- CUDA allocation και upload,
- ένα πραγματικό kernel ή CCCL primitive,
- readback και parity με CPU reference,
- Compute Sanitizer.

Το scaffold δεν χαρακτηρίζεται production-ready ή ταχύτερο. Experimental Rust-to-PTX infrastructure δεν αποτελεί production dependency στο Stage 1.

## 20. Rented A6000 run kit

Το `scripts/gpu-bench/` εκτελεί fail-fast validation πριν από ακριβή εργασία:

- GPU visibility και identity,
- driver/toolkit/runtime compatibility,
- container GPU access,
- CUDA smoke test,
- CUPTI και Nsight permissions,
- strict backend/preflight behaviour,
- zero-CPU engine audit,
- dataset και config hashes,
- επαρκές disk/RAM/VRAM,
- output persistence.

Το kit δημιουργεί pinned legacy και candidate worktrees/images. Διατηρεί δύο references:

- historical legacy baseline στο `2be1408...`,
- canonical Stage 1 baseline μετά το semantics fix.

Εκτελεί baseline και prototypes σε ξεχωριστά clean, diagnostic και profiler passes.

Το τελικό report παρουσιάζει Pareto surface ανά timeframe, population, scenario density, coverage, VRAM και correctness. Δεν επιλέγει αυτόματα engine από ένα aggregate score. Μετά τις μετρήσεις ακολουθεί recorded human decision gate.

## 21. Implementation plan

### Phase 0 — backend axes, preflight και CPU audit

- **0.0:** δημοσίευση του παρόντος authoritative αγγλικού και ελληνικού design record.
- **0.1:** device preference, fallback policy, accelerator hint, configuration mapping και precedence.
- **0.2:** typed failure actions, WrongShape reclassification και deterministic GPU rebatching.
- **0.3:** κεντρικοποίηση CPU strategy evaluation πίσω από per-work-unit audited wrapper.
- **0.4:** full-pipeline GPU capability inventory και fail-fast preflight.

### Phase 1 — canonical semantics και parity

- **1.0:** επίλυση SMC semantics, trading/discovery descriptors, versions και hashes, καταγραφή legacy/canonical delta.
- **1.1:** field-specific parity policies, canonical RankKey και deterministic identity/order.
- **1.2:** ξεχωριστά CPU/GPU trace specializations.
- **1.3:** levels 1–9 direct fixtures και levels 10–12 integration fixtures, με direct iGPU backend invocation.

### Phase 2 — contracts και device residency

- **2.1:** `neoethos-gpu-contracts` host DTO και device POD layers με ABI assertions.
- **2.2:** scenario descriptors, counter-based RNG contract και segmented keys.
- **2.3:** session-bound device handles, explicit events/fences, transfer instrumentation και CubeCL session implementation.

### Phase 3 — instrumentation και benchmarks

- **3.1:** optional NVTX ranges για όλα τα discovery stages.
- **3.2:** multi-pass benchmark runner, snapshots, sweeps και machine-readable reports.

### Phase 4 — εκτελέσιμα prototypes

- **4.1:** το Prototype A ικανοποιεί persistence, transfer και parity acceptance.
- **4.2:** εκτελέσιμα minimal B και C engines με typed capability reporting και fair-comparison fixtures.

### Phase 5 — CUDA scaffold και remote kit

- **5.1:** CUDA/CCCL FFI scaffold και διαχωρισμός compile-time από real-GPU verification.
- **5.2:** pinned, fail-fast A6000 benchmark/profiling kit και Pareto report.

## 22. Verification ανά phase

### Phase 0

```bash
cargo test -p neoethos-search
cargo check --features gpu-vulkan
```

Τα tests καλύπτουν configuration mapping/precedence, πραγματικό boolean parsing, failure-action matrix, WrongShape fail-loud behaviour, rebatch determinism, per-work-unit audit counters και typed preflight output.

### Phase 1

Τα tests καλύπτουν canonical SMC semantics, τα δύο semantic hashes, field-specific parity, canonical ranking, collision fixtures, trace separation και direct-backend iGPU parity.

### Phase 2

Τα tests καλύπτουν POD layout assertions, handle ownership/staleness, explicit synchronization, transfer counters και device-to-device chaining χωρίς forced D2H.

### Phase 3

Το CLI benchmark παράγει έγκυρο JSON για tiny και snapshot fixtures. Unsupported iGPU-only metrics είναι `null`. Timing, diagnostics και profiler passes είναι ξεχωριστά.

### Phase 4

Το Prototype A περνά direct και integration parity όπως ορίζεται. Τα minimal B/C engines εκτελούνται σε supported backends και επιστρέφουν typed unsupported status αλλού. Τα common-intersection fixtures είναι ίδια για όλα τα engines.

### Phase 5

Τα card-independent CUDA compile/link/ABI checks περνούν όπου υπάρχει toolkit. Το runtime smoke test και το Compute Sanitizer είναι ρητά GPU-gated. Το run kit περνά dry-run linting πριν από την ενοικίαση.

## 23. Delivery protocol

Κάθε implementation commit παραδίδεται ξεχωριστά με:

- commit SHA,
- changed files και affected APIs,
- σύντομο change report,
- ακριβή test/verification αποτελέσματα,
- γνωστούς περιορισμούς και typed unsupported capabilities,
- ρητή σημείωση για κάθε απόκλιση από το εγκεκριμένο scope.

Νέος architecture-planning γύρος ανοίγει μόνο αν ο κώδικας, τα tests ή οι πραγματικές μετρήσεις αποκαλύψουν blocker που αλλάζει εγκεκριμένο contract.

## 24. Αρχή λήψης απόφασης

Τα correctness fixtures και τα μετρημένα end-to-end αποτελέσματα είναι η τελική αρχή. Kernel microbenchmarks, θεωρητικό occupancy και προσωπική προτίμηση υλοποίησης δεν υπερισχύουν των full-pipeline στοιχείων.

Δεν ενσωματώνεται fixed candidate target στην αρχιτεκτονική. Το search breadth παραμένει συνάρτηση του μετρημένου throughput, του διαθέσιμου VRAM, του time budget και του απαιτούμενου validation depth.

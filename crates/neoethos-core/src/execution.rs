//! Lease-bound execution for CPU-parallel work.
//!
//! Rayon pools are execution mechanisms, not capacity authorities. Every call
//! accepts ownership of a lease issued by `neoethos-execution-budget`, builds
//! or checks out a pool of exactly that width, and retains the lease until the
//! installed work has completed. Idle cached worker threads own no permits and
//! cannot be reached without another lease transfer.

use crate::execution_budget::{
    CpuLease, CpuLeaseTransfer, CpuPermitBroker, WorkerLimit, enter_lease_bound_worker_scope,
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone)]
pub struct BudgetedCpuExecutor {
    inner: Arc<ExecutorInner>,
}

struct ExecutorInner {
    authority: CpuPermitBroker,
    max_cached_worker_threads: usize,
    next_pool_id: AtomicU64,
    cache: Mutex<PoolCache>,
}

#[derive(Default)]
struct PoolCache {
    by_width: BTreeMap<usize, Vec<ThreadPool>>,
    idle_worker_threads: usize,
}

#[derive(Debug)]
pub enum BudgetedCpuExecutorError {
    MismatchedLeaseAuthority,
    OutsideLeasedPool,
    PoolBuild {
        width: usize,
        source: rayon::ThreadPoolBuildError,
    },
}

impl BudgetedCpuExecutorError {
    pub fn width(&self) -> Option<usize> {
        match self {
            Self::MismatchedLeaseAuthority | Self::OutsideLeasedPool => None,
            Self::PoolBuild { width, .. } => Some(*width),
        }
    }
}

impl fmt::Display for BudgetedCpuExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedLeaseAuthority => write!(
                formatter,
                "transferred CPU lease was issued by a different capacity authority"
            ),
            Self::OutsideLeasedPool => formatter
                .write_str("CPU work is outside the exact private pool holding its active lease"),
            Self::PoolBuild { width, source } => write!(
                formatter,
                "failed to build a {width}-worker budgeted Rayon pool: {source}"
            ),
        }
    }
}

impl Error for BudgetedCpuExecutorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MismatchedLeaseAuthority | Self::OutsideLeasedPool => None,
            Self::PoolBuild { source, .. } => Some(source),
        }
    }
}

/// Borrowed proof of the exact reservation and private worker pool executing
/// this work. Only the executor creates it, and its lifetime cannot outlive
/// the accepted lease. Equal worker counts or thread names are not authority.
pub struct BudgetedCpuScope<'scope> {
    lease: &'scope CpuLease,
    pool: &'scope ThreadPool,
}

impl BudgetedCpuScope<'_> {
    pub fn lease(&self) -> &CpuLease {
        self.lease
    }

    pub fn worker_limit(&self) -> WorkerLimit {
        self.lease.width()
    }

    /// Check the specific pool, without consulting or initializing Rayon's
    /// global pool. Nested parallel work in this same pool remains valid.
    pub fn require_current_pool(&self) -> Result<(), BudgetedCpuExecutorError> {
        if self.pool.current_thread_index().is_none()
            || self.pool.current_num_threads() != self.lease.width().get()
        {
            return Err(BudgetedCpuExecutorError::OutsideLeasedPool);
        }
        Ok(())
    }
}

impl BudgetedCpuExecutor {
    /// Create a reusable executor with a bounded idle-thread cache.
    ///
    /// This limit controls only how many already-created worker threads may
    /// remain idle. It does not grant CPU capacity and it does not narrow a
    /// valid transferred lease. A wider pool is dropped rather than cached.
    pub fn new_for_broker(
        authority: CpuPermitBroker,
        max_cached_worker_threads: WorkerLimit,
    ) -> Self {
        Self {
            inner: Arc::new(ExecutorInner {
                authority,
                max_cached_worker_threads: max_cached_worker_threads.get(),
                next_pool_id: AtomicU64::new(0),
                cache: Mutex::new(PoolCache::default()),
            }),
        }
    }

    /// Execute work on a pool whose total worker width exactly matches the
    /// transferred lease. The active lease scope also makes any fresh nested
    /// broker acquisition fail; nested work must receive a split lease.
    pub fn execute<R, Work>(
        &self,
        transfer: CpuLeaseTransfer,
        work: Work,
    ) -> Result<R, BudgetedCpuExecutorError>
    where
        R: Send,
        Work: FnOnce() -> R + Send,
    {
        self.execute_with_scope(transfer, |_| work())
    }

    /// Execute work on the matching private pool while lending the accepted
    /// lease to the work item. This is the lease-bearing form required by
    /// APIs that must verify or split the exact reservation they execute
    /// under; the lease remains owned by the executor until all work returns.
    pub fn execute_with_lease<R, Work>(
        &self,
        transfer: CpuLeaseTransfer,
        work: Work,
    ) -> Result<R, BudgetedCpuExecutorError>
    where
        R: Send,
        Work: FnOnce(&CpuLease) -> R + Send,
    {
        self.execute_with_scope(transfer, |scope| work(scope.lease()))
    }

    /// Lend proof of both the reservation and its exact executing pool to
    /// consumers which must reject accidental global/foreign-pool execution.
    pub fn execute_with_scope<R, Work>(
        &self,
        transfer: CpuLeaseTransfer,
        work: Work,
    ) -> Result<R, BudgetedCpuExecutorError>
    where
        R: Send,
        Work: FnOnce(&BudgetedCpuScope<'_>) -> R + Send,
    {
        if !self.inner.authority.owns_transfer(&transfer) {
            return Err(BudgetedCpuExecutorError::MismatchedLeaseAuthority);
        }
        let lease = transfer.accept();
        let width = lease.width().get();
        let checkout = self.checkout(width)?;
        let pool = checkout.pool();
        let result = pool.install(|| {
            lease.scope(|| {
                work(&BudgetedCpuScope {
                    lease: &lease,
                    pool,
                })
            })
        });

        drop(lease);
        drop(checkout);
        Ok(result)
    }

    /// Execute fork-join work on the matching private pool and wait for every
    /// task spawned through the supplied Rayon scope before returning the
    /// lease. Callers must use this scope instead of detached `rayon::spawn`.
    pub fn execute_scoped<'scope, R, Work>(
        &self,
        transfer: CpuLeaseTransfer,
        work: Work,
    ) -> Result<R, BudgetedCpuExecutorError>
    where
        R: Send,
        Work: FnOnce(&rayon::Scope<'scope>) -> R + Send,
    {
        if !self.inner.authority.owns_transfer(&transfer) {
            return Err(BudgetedCpuExecutorError::MismatchedLeaseAuthority);
        }
        let lease = transfer.accept();
        let width = lease.width().get();
        let checkout = self.checkout(width)?;
        let result = checkout.pool().scope(|scope| lease.scope(|| work(scope)));

        // Return capacity only after every scoped Rayon task is complete.
        // Then make the now-idle pool eligible for bounded reuse.
        drop(lease);
        drop(checkout);
        Ok(result)
    }

    /// Runtime proof of the pool width visible to the current work item.
    pub fn current_pool_width() -> usize {
        rayon::current_num_threads()
    }

    /// Number of cached idle OS worker threads. These threads own no permits.
    pub fn cached_idle_worker_threads(&self) -> usize {
        lock_unpoisoned(&self.inner.cache).idle_worker_threads
    }

    fn checkout(&self, width: usize) -> Result<PoolCheckout, BudgetedCpuExecutorError> {
        if let Some(pool) = self.take_cached(width) {
            return Ok(PoolCheckout::new(Arc::clone(&self.inner), width, pool));
        }

        let pool_id = self.inner.next_pool_id.fetch_add(1, Ordering::Relaxed);
        let pool = ThreadPoolBuilder::new()
            .num_threads(width)
            .thread_name(move |index| format!("neoethos-cpu-{pool_id}-{index}"))
            .spawn_handler(|thread| {
                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(stack_size) = thread.stack_size() {
                    builder = builder.stack_size(stack_size);
                }
                builder.spawn(move || {
                    let _worker_scope = enter_lease_bound_worker_scope();
                    thread.run();
                })?;
                Ok(())
            })
            .build()
            .map_err(|source| BudgetedCpuExecutorError::PoolBuild { width, source })?;
        Ok(PoolCheckout::new(Arc::clone(&self.inner), width, pool))
    }

    fn take_cached(&self, width: usize) -> Option<ThreadPool> {
        let mut cache = lock_unpoisoned(&self.inner.cache);
        let pool = cache.by_width.get_mut(&width)?.pop()?;
        cache.idle_worker_threads = cache
            .idle_worker_threads
            .checked_sub(width)
            .expect("cached worker count tracks every cached pool");
        if cache.by_width.get(&width).is_some_and(Vec::is_empty) {
            cache.by_width.remove(&width);
        }
        Some(pool)
    }
}

impl fmt::Debug for BudgetedCpuExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BudgetedCpuExecutor")
            .field("authority", &self.inner.authority.snapshot())
            .field(
                "max_cached_worker_threads",
                &self.inner.max_cached_worker_threads,
            )
            .field(
                "cached_idle_worker_threads",
                &self.cached_idle_worker_threads(),
            )
            .finish()
    }
}

struct PoolCheckout {
    owner: Arc<ExecutorInner>,
    width: usize,
    pool: Option<ThreadPool>,
}

impl PoolCheckout {
    fn new(owner: Arc<ExecutorInner>, width: usize, pool: ThreadPool) -> Self {
        Self {
            owner,
            width,
            pool: Some(pool),
        }
    }

    fn pool(&self) -> &ThreadPool {
        self.pool
            .as_ref()
            .expect("checked-out pool exists until checkout is dropped")
    }
}

impl Drop for PoolCheckout {
    fn drop(&mut self) {
        let mut cache = lock_unpoisoned(&self.owner.cache);
        let Some(next_total) = cache.idle_worker_threads.checked_add(self.width) else {
            return;
        };
        if next_total > self.owner.max_cached_worker_threads {
            return;
        }
        let Some(pool) = self.pool.take() else {
            return;
        };
        cache.by_width.entry(self.width).or_default().push(pool);
        cache.idle_worker_threads = next_total;
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_budget::CpuPermitRequest;

    #[test]
    fn execute_binds_nested_rayon_work_to_the_exact_lease_width() {
        let width = WorkerLimit::new(3).expect("positive worker width");
        let broker = CpuPermitBroker::new(width);
        let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), width);
        let lease = broker
            .acquire(CpuPermitRequest::local(width))
            .expect("acquire the complete test budget");

        let mut observations = executor
            .execute(lease.into_transfer(), || {
                rayon::broadcast(|context| {
                    (
                        context.index(),
                        rayon::current_num_threads(),
                        std::thread::current().name().map(str::to_owned),
                    )
                })
            })
            .expect("execute on the lease-bound pool");
        observations.sort_by_key(|observation| observation.0);

        assert_eq!(observations.len(), width.get());
        assert_eq!(
            observations
                .iter()
                .map(|observation| observation.0)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(
            observations
                .iter()
                .all(|observation| observation.1 == width.get())
        );
        assert!(observations.iter().all(|observation| {
            observation
                .2
                .as_deref()
                .is_some_and(|name| name.starts_with("neoethos-cpu-"))
        }));
        assert_eq!(broker.snapshot().available_permits, width.get());
    }

    #[test]
    fn scope_accepts_every_worker_of_its_exact_leased_pool() {
        let width = WorkerLimit::new(3).unwrap();
        let broker = CpuPermitBroker::new(width);
        let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), width);
        let lease = broker.acquire(CpuPermitRequest::local(width)).unwrap();
        let observations = executor
            .execute_with_scope(lease.into_transfer(), |scope| {
                scope.require_current_pool().unwrap();
                assert_eq!(scope.worker_limit(), width);
                rayon::broadcast(|context| {
                    scope.require_current_pool().unwrap();
                    (context.index(), rayon::current_num_threads())
                })
            })
            .unwrap();
        assert_eq!(observations.len(), 3);
        assert!(observations.iter().all(|(_, workers)| *workers == 3));
        assert_eq!(broker.snapshot().live_reserved_sum, 0);
    }

    #[test]
    fn scope_rejects_foreign_equal_width_pool_and_plain_thread() {
        let width = WorkerLimit::new(2).unwrap();
        let broker = CpuPermitBroker::new(width);
        let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), width);
        let foreign = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let lease = broker.acquire(CpuPermitRequest::local(width)).unwrap();
        executor
            .execute_with_scope(lease.into_transfer(), |scope| {
                let rejected = foreign.install(|| scope.require_current_pool());
                assert!(matches!(
                    rejected,
                    Err(BudgetedCpuExecutorError::OutsideLeasedPool)
                ));
                std::thread::scope(|threads| {
                    let rejected = threads
                        .spawn(|| scope.require_current_pool())
                        .join()
                        .unwrap();
                    assert!(matches!(
                        rejected,
                        Err(BudgetedCpuExecutorError::OutsideLeasedPool)
                    ));
                });
                scope.require_current_pool().unwrap();
            })
            .unwrap();
        assert_eq!(broker.snapshot().live_reserved_sum, 0);
    }
}

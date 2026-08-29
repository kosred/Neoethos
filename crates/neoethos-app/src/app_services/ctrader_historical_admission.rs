use std::error::Error;
use std::fmt;
use std::io::{self, Read, Write};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// cTrader documents at most five historical requests per second on one
/// connection. The 225 ms spacing is both strictly slower than that ceiling
/// and the fastest interval proven clean by the 2026-08-18 Demo live probe.
pub(crate) const CTRADER_HISTORICAL_REQUEST_MIN_INTERVAL: Duration = Duration::from_millis(225);

/// A synchronous socket read is bounded to this interval so cancellation is
/// observed promptly while waiting for a cTrader response.
pub(crate) const CTRADER_RESPONSE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// DNS, every resolved-address TCP attempt, TLS, and the WebSocket upgrade all
/// consume this one budget. No stage receives a fresh timeout.
pub(crate) const CTRADER_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Absolute budget for one request or Pong write. `DeadlineIo` further slices
/// the underlying syscall waits to `CTRADER_RESPONSE_POLL_INTERVAL` so a stop
/// is observed without granting each partial write a fresh five seconds.
pub(crate) const CTRADER_SOCKET_WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// One request gets one absolute response deadline. Ping/pong and unrelated
/// frames are protocol progress, not permission to wait forever.
pub(crate) const CTRADER_RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) const CTRADER_BLOCKED_PAYLOAD_TYPE_ERROR_CODE: &str = "BLOCKED_PAYLOAD_TYPE";

/// `BLOCKED_PAYLOAD_TYPE` already carries the broker's `retryAfter` decision.
/// Retrying it inside the generic cold-connection loop would ignore that
/// decision and can extend the block. Other errors retain the existing retry
/// policy; this helper deliberately answers only the rate-limit exception.
pub(crate) fn should_retry_ctrader_error(error_code: &str) -> bool {
    !error_code
        .trim()
        .eq_ignore_ascii_case(CTRADER_BLOCKED_PAYLOAD_TYPE_ERROR_CODE)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HistoricalAdmissionClockOverflow;

impl fmt::Display for HistoricalAdmissionClockOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cTrader historical admission deadline overflowed")
    }
}

impl Error for HistoricalAdmissionClockOverflow {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HistoricalRequestCancelled;

impl fmt::Display for HistoricalRequestCancelled {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cTrader historical request was cancelled")
    }
}

impl Error for HistoricalRequestCancelled {}

/// A fresh, one-way cancellation flag owned by exactly one historical fetch.
/// The condition variable wakes a connection-local admission wait immediately;
/// a cancelled flag is never reset or stored on a reusable transport.
#[derive(Clone, Debug, Default)]
pub(crate) struct HistoricalRequestCancellation {
    state: Arc<(Mutex<bool>, Condvar)>,
}

impl HistoricalRequestCancellation {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[cfg(any(test, feature = "broker-history-service"))]
    pub(crate) fn cancel(&self) {
        let (state, wake) = &*self.state;
        let mut cancelled = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *cancelled = true;
        wake.notify_all();
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        let (state, _) = &*self.state;
        *state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn wait_for_cancellation(&self, timeout: Duration) -> bool {
        let (state, wake) = &*self.state;
        let cancelled = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *cancelled {
            return true;
        }
        let (cancelled, _) = wake
            .wait_timeout_while(cancelled, timeout, |cancelled| !*cancelled)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *cancelled
    }
}

pub(crate) trait CTraderMonotonicClock: Clone {
    fn now(&self) -> Instant;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SystemCTraderMonotonicClock;

impl CTraderMonotonicClock for SystemCTraderMonotonicClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CTraderIoPhase {
    Dns,
    TcpConnect,
    TlsWebSocketHandshake,
    RequestWrite,
    ResponseRead,
}

impl fmt::Display for CTraderIoPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Dns => "DNS resolution",
            Self::TcpConnect => "TCP connect",
            Self::TlsWebSocketHandshake => "TLS/WebSocket handshake",
            Self::RequestWrite => "request write",
            Self::ResponseRead => "response read",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CTraderOperationBudgetOverflow;

impl fmt::Display for CTraderOperationBudgetOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cTrader network operation deadline overflowed")
    }
}

impl Error for CTraderOperationBudgetOverflow {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CTraderIoBoundaryError {
    Cancelled {
        phase: CTraderIoPhase,
    },
    DeadlineExceeded {
        phase: CTraderIoPhase,
        timeout: Duration,
    },
}

impl fmt::Display for CTraderIoBoundaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { phase } => {
                write!(formatter, "cTrader {phase} was cancelled")
            }
            Self::DeadlineExceeded { phase, timeout } => write!(
                formatter,
                "cTrader {phase} exceeded its absolute {} ms budget",
                timeout.as_millis()
            ),
        }
    }
}

impl Error for CTraderIoBoundaryError {}

/// Detect an operator cancellation through every typed wrapper used by the
/// historical network path. This deliberately does not inspect display text:
/// transport errors retain `CTraderIoBoundaryError` as an `io::Error` source,
/// and `anyhow` contexts retain the same standard error chain.
#[cfg(feature = "broker-history-service")]
pub(crate) fn is_historical_request_cancelled(mut error: &(dyn Error + 'static)) -> bool {
    loop {
        if error.is::<HistoricalRequestCancelled>()
            || matches!(
                error.downcast_ref::<CTraderIoBoundaryError>(),
                Some(CTraderIoBoundaryError::Cancelled { .. })
            )
        {
            return true;
        }
        if let Some(tungstenite::Error::Io(io_error)) = error.downcast_ref::<tungstenite::Error>()
            && let Some(inner) = io_error.get_ref()
            && is_historical_request_cancelled(inner)
        {
            return true;
        }
        if let Some(io_error) = error.downcast_ref::<io::Error>()
            && let Some(inner) = io_error.get_ref()
            && is_historical_request_cancelled(inner)
        {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

#[derive(Clone)]
pub(crate) struct CTraderOperationBudget<C: CTraderMonotonicClock> {
    clock: C,
    deadline: Instant,
    timeout: Duration,
    cancellation: Option<HistoricalRequestCancellation>,
}

impl<C: CTraderMonotonicClock> CTraderOperationBudget<C> {
    pub(crate) fn new(
        clock: C,
        timeout: Duration,
        cancellation: Option<HistoricalRequestCancellation>,
    ) -> Result<Self, CTraderOperationBudgetOverflow> {
        let started_at = clock.now();
        Self::new_at(clock, started_at, timeout, cancellation)
    }

    pub(crate) fn new_at(
        clock: C,
        started_at: Instant,
        timeout: Duration,
        cancellation: Option<HistoricalRequestCancellation>,
    ) -> Result<Self, CTraderOperationBudgetOverflow> {
        let deadline = started_at
            .checked_add(timeout)
            .ok_or(CTraderOperationBudgetOverflow)?;
        Ok(Self {
            clock,
            deadline,
            timeout,
            cancellation,
        })
    }

    pub(crate) fn check(&self, phase: CTraderIoPhase) -> Result<(), CTraderIoBoundaryError> {
        self.check_at(phase, self.clock.now())
    }

    fn check_at(&self, phase: CTraderIoPhase, now: Instant) -> Result<(), CTraderIoBoundaryError> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(HistoricalRequestCancellation::is_cancelled)
        {
            return Err(CTraderIoBoundaryError::Cancelled { phase });
        }
        if now >= self.deadline {
            return Err(CTraderIoBoundaryError::DeadlineExceeded {
                phase,
                timeout: self.timeout,
            });
        }
        Ok(())
    }

    pub(crate) fn check_io(&self, phase: CTraderIoPhase) -> io::Result<()> {
        self.check(phase).map_err(c_trader_boundary_io_error)
    }

    pub(crate) fn remaining_io(&self, phase: CTraderIoPhase) -> io::Result<Duration> {
        let now = self.clock.now();
        self.check_at(phase, now)
            .map_err(c_trader_boundary_io_error)?;
        Ok(self.deadline.duration_since(now))
    }

    pub(crate) fn wait_for_poll(&self, phase: CTraderIoPhase, maximum: Duration) -> io::Result<()> {
        let wait = maximum.min(self.remaining_io(phase)?);
        if let Some(cancellation) = &self.cancellation {
            cancellation.wait_for_cancellation(wait);
        } else {
            std::thread::sleep(wait);
        }
        self.check_io(phase)
    }

    pub(crate) fn capped_from_now(
        &self,
        timeout: Duration,
        phase: CTraderIoPhase,
    ) -> io::Result<Self> {
        let now = self.clock.now();
        self.check_at(phase, now)
            .map_err(c_trader_boundary_io_error)?;
        let cap = now.checked_add(timeout).ok_or_else(|| {
            c_trader_boundary_io_error(CTraderIoBoundaryError::DeadlineExceeded { phase, timeout })
        })?;
        let deadline = self.deadline.min(cap);
        Ok(Self {
            clock: self.clock.clone(),
            deadline,
            timeout: deadline.duration_since(now),
            cancellation: self.cancellation.clone(),
        })
    }
}

fn c_trader_boundary_io_error(error: CTraderIoBoundaryError) -> io::Error {
    let kind = match error {
        CTraderIoBoundaryError::Cancelled { .. } => io::ErrorKind::Interrupted,
        CTraderIoBoundaryError::DeadlineExceeded { .. } => io::ErrorKind::TimedOut,
    };
    io::Error::new(kind, error)
}

#[derive(Clone, Copy)]
pub(crate) enum CTraderIoDirection {
    Read,
    Write,
}

type CTraderTimeoutConfigurer<S> = fn(&mut S, CTraderIoDirection, Duration) -> io::Result<()>;

/// Enforces one immutable absolute budget below TLS and WebSocket parsing.
/// Checks happen before and after every underlying operation, so successful
/// one-byte reads/writes and fragmented frames cannot extend the deadline.
pub(crate) struct DeadlineIo<S, C: CTraderMonotonicClock> {
    inner: S,
    budget: CTraderOperationBudget<C>,
    phase: CTraderIoPhase,
    poll_interval: Duration,
    configure_timeout: Option<CTraderTimeoutConfigurer<S>>,
}

impl<S, C: CTraderMonotonicClock> DeadlineIo<S, C> {
    #[cfg(test)]
    pub(crate) fn new(
        inner: S,
        budget: CTraderOperationBudget<C>,
        phase: CTraderIoPhase,
        poll_interval: Duration,
    ) -> Self {
        Self {
            inner,
            budget,
            phase,
            poll_interval,
            configure_timeout: None,
        }
    }

    pub(crate) fn with_timeout_configurer(
        inner: S,
        budget: CTraderOperationBudget<C>,
        phase: CTraderIoPhase,
        poll_interval: Duration,
        configure_timeout: CTraderTimeoutConfigurer<S>,
    ) -> Self {
        Self {
            inner,
            budget,
            phase,
            poll_interval,
            configure_timeout: Some(configure_timeout),
        }
    }

    pub(crate) fn arm(
        &mut self,
        budget: CTraderOperationBudget<C>,
        phase: CTraderIoPhase,
        poll_interval: Duration,
    ) {
        self.budget = budget;
        self.phase = phase;
        self.poll_interval = poll_interval;
    }

    fn prepare(&mut self, direction: CTraderIoDirection) -> io::Result<()> {
        let remaining = self.budget.remaining_io(self.phase)?;
        if let Some(configure_timeout) = self.configure_timeout {
            configure_timeout(
                &mut self.inner,
                direction,
                self.poll_interval.min(remaining),
            )?;
        }
        Ok(())
    }

    fn finish<T>(&self, result: io::Result<T>) -> io::Result<T> {
        match result {
            Ok(value) => {
                self.budget.check_io(self.phase)?;
                Ok(value)
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                self.budget.check_io(self.phase)?;
                Err(io::Error::new(io::ErrorKind::WouldBlock, error))
            }
            Err(error) => {
                self.budget.check_io(self.phase)?;
                Err(error)
            }
        }
    }
}

impl<S: Read, C: CTraderMonotonicClock> Read for DeadlineIo<S, C> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.prepare(CTraderIoDirection::Read)?;
        let result = self.inner.read(buffer);
        self.finish(result)
    }
}

impl<S: Write, C: CTraderMonotonicClock> Write for DeadlineIo<S, C> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.prepare(CTraderIoDirection::Write)?;
        let result = self.inner.write(buffer);
        self.finish(result)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.prepare(CTraderIoDirection::Write)?;
        let result = self.inner.flush();
        self.finish(result)
    }
}

pub(crate) trait CTraderSocketConnector<C: CTraderMonotonicClock> {
    type Resolved;
    type Stream;
    type Socket;

    fn resolve(&mut self, budget: &CTraderOperationBudget<C>) -> io::Result<Self::Resolved>;
    fn connect_tcp(
        &mut self,
        resolved: Self::Resolved,
        budget: &CTraderOperationBudget<C>,
    ) -> io::Result<Self::Stream>;
    fn tls_websocket_handshake(
        &mut self,
        stream: Self::Stream,
        budget: &CTraderOperationBudget<C>,
    ) -> io::Result<Self::Socket>;
}

pub(crate) fn establish_ctrader_socket_with_connector<C, Connector>(
    connector: &mut Connector,
    budget: &CTraderOperationBudget<C>,
) -> io::Result<Connector::Socket>
where
    C: CTraderMonotonicClock,
    Connector: CTraderSocketConnector<C>,
{
    budget.check_io(CTraderIoPhase::Dns)?;
    let resolved = connector.resolve(budget)?;
    budget.check_io(CTraderIoPhase::Dns)?;

    budget.check_io(CTraderIoPhase::TcpConnect)?;
    let stream = connector.connect_tcp(resolved, budget)?;
    budget.check_io(CTraderIoPhase::TcpConnect)?;

    budget.check_io(CTraderIoPhase::TlsWebSocketHandshake)?;
    let socket = connector.tls_websocket_handshake(stream, budget)?;
    budget.check_io(CTraderIoPhase::TlsWebSocketHandshake)?;
    Ok(socket)
}

#[cfg(any(test, feature = "broker-history-service"))]
mod historical_fetch_registry {
    use super::*;
    use std::sync::OnceLock;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct HistoricalFetchConflict {
        pub(crate) active_run_id: u64,
    }

    impl fmt::Display for HistoricalFetchConflict {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "cTrader historical fetch {} is already active",
                self.active_run_id
            )
        }
    }

    impl Error for HistoricalFetchConflict {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum HistoricalFetchStartError {
        AlreadyActive(HistoricalFetchConflict),
        RunIdOverflow,
    }

    impl fmt::Display for HistoricalFetchStartError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::AlreadyActive(conflict) => conflict.fmt(formatter),
                Self::RunIdOverflow => {
                    formatter.write_str("cTrader historical fetch id overflowed")
                }
            }
        }
    }

    impl Error for HistoricalFetchStartError {}

    #[derive(Debug)]
    pub(crate) enum HistoricalFetchQueueStartError<QueueError> {
        Fetch(HistoricalFetchStartError),
        Queue(QueueError),
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum HistoricalFetchCancelOutcome {
        Cancelled {
            run_id: u64,
        },
        PublicationInProgress {
            run_id: u64,
        },
        StaleRun {
            requested_run_id: u64,
            active_run_id: u64,
        },
        NoActiveFetch,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum HistoricalFetchPhase {
        Capturing,
        CancellationRequested,
        PublicationInProgress,
    }

    impl HistoricalFetchPhase {
        pub(crate) const fn as_str(self) -> &'static str {
            match self {
                Self::Capturing => "capturing",
                Self::CancellationRequested => "cancellation_requested",
                Self::PublicationInProgress => "publication_in_progress",
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct HistoricalFetchStatus {
        pub(crate) run_id: u64,
        pub(crate) phase: HistoricalFetchPhase,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum HistoricalPublicationStartError {
        Cancelled(HistoricalRequestCancelled),
        RegistrationLost { run_id: u64 },
        AlreadyInProgress { run_id: u64 },
    }

    impl fmt::Display for HistoricalPublicationStartError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Cancelled(error) => error.fmt(formatter),
                Self::RegistrationLost { run_id } => {
                    write!(
                        formatter,
                        "cTrader historical fetch {run_id} lost registration"
                    )
                }
                Self::AlreadyInProgress { run_id } => write!(
                    formatter,
                    "cTrader historical fetch {run_id} publication is already in progress"
                ),
            }
        }
    }

    impl Error for HistoricalPublicationStartError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            match self {
                Self::Cancelled(error) => Some(error),
                Self::RegistrationLost { .. } | Self::AlreadyInProgress { .. } => None,
            }
        }
    }

    struct RegisteredHistoricalFetch {
        run_id: u64,
        cancellation: HistoricalRequestCancellation,
        phase: HistoricalFetchPhase,
    }

    struct HistoricalFetchRegistryState {
        next_run_id: u64,
        active: Option<RegisteredHistoricalFetch>,
    }

    /// Serializes the production `/data/fetch` operation. The registered token is
    /// fresh per run and removed by an RAII guard on every normal or unwind path.
    pub(crate) struct HistoricalFetchRegistry {
        state: Mutex<HistoricalFetchRegistryState>,
    }

    impl HistoricalFetchRegistry {
        pub(crate) fn new() -> Self {
            Self {
                state: Mutex::new(HistoricalFetchRegistryState {
                    next_run_id: 1,
                    active: None,
                }),
            }
        }

        pub(crate) fn try_start(
            &self,
        ) -> Result<ActiveHistoricalFetch<'_>, HistoricalFetchStartError> {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(active) = &state.active {
                return Err(HistoricalFetchStartError::AlreadyActive(
                    HistoricalFetchConflict {
                        active_run_id: active.run_id,
                    },
                ));
            }
            let run_id = state.next_run_id;
            state.next_run_id = run_id
                .checked_add(1)
                .ok_or(HistoricalFetchStartError::RunIdOverflow)?;
            let cancellation = HistoricalRequestCancellation::new();
            state.active = Some(RegisteredHistoricalFetch {
                run_id,
                cancellation: cancellation.clone(),
                phase: HistoricalFetchPhase::Capturing,
            });
            Ok(ActiveHistoricalFetch {
                registry: self,
                run_id,
                cancellation,
            })
        }

        /// Register the one process fetch before touching the shared CPU queue.
        /// A conflicting caller therefore fails without submitting work that could
        /// wake after the current fetch has finished and silently become a second
        /// fetch. If queue submission fails, the newly registered run is released
        /// by its RAII guard before the error is returned.
        pub(crate) fn try_start_queued<Pending, QueueError>(
            &self,
            submit: impl FnOnce() -> Result<Pending, QueueError>,
        ) -> Result<QueuedHistoricalFetch<'_, Pending>, HistoricalFetchQueueStartError<QueueError>>
        {
            let active = self
                .try_start()
                .map_err(HistoricalFetchQueueStartError::Fetch)?;
            let pending = submit().map_err(HistoricalFetchQueueStartError::Queue)?;
            Ok(QueuedHistoricalFetch { pending, active })
        }

        pub(crate) fn cancel_run(&self, requested_run_id: u64) -> HistoricalFetchCancelOutcome {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(active) = state.active.as_mut() else {
                return HistoricalFetchCancelOutcome::NoActiveFetch;
            };
            if active.run_id != requested_run_id {
                return HistoricalFetchCancelOutcome::StaleRun {
                    requested_run_id,
                    active_run_id: active.run_id,
                };
            }
            match active.phase {
                HistoricalFetchPhase::PublicationInProgress => {
                    HistoricalFetchCancelOutcome::PublicationInProgress {
                        run_id: active.run_id,
                    }
                }
                HistoricalFetchPhase::Capturing | HistoricalFetchPhase::CancellationRequested => {
                    active.cancellation.cancel();
                    active.phase = HistoricalFetchPhase::CancellationRequested;
                    HistoricalFetchCancelOutcome::Cancelled {
                        run_id: active.run_id,
                    }
                }
            }
        }

        pub(crate) fn status(&self) -> Option<HistoricalFetchStatus> {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.active.as_ref().map(|active| HistoricalFetchStatus {
                run_id: active.run_id,
                phase: active.phase,
            })
        }
    }

    pub(crate) struct ActiveHistoricalFetch<'a> {
        registry: &'a HistoricalFetchRegistry,
        run_id: u64,
        cancellation: HistoricalRequestCancellation,
    }

    impl ActiveHistoricalFetch<'_> {
        pub(crate) fn run_id(&self) -> u64 {
            self.run_id
        }

        pub(crate) fn cancellation(&self) -> &HistoricalRequestCancellation {
            &self.cancellation
        }

        /// Last fail-closed boundary before settings, network, or publication work
        /// begins after a queued CPU grant. The download itself retains its finer
        /// cancellation checks, including the atomic publication transition.
        pub(crate) fn execute_if_not_cancelled<Output>(
            &self,
            work: impl FnOnce(&Self) -> Output,
        ) -> Result<Output, HistoricalRequestCancelled> {
            if self.cancellation.is_cancelled() {
                return Err(HistoricalRequestCancelled);
            }
            Ok(work(self))
        }

        pub(crate) fn begin_publication(
            &self,
        ) -> Result<HistoricalPublicationPermit<'_>, HistoricalPublicationStartError> {
            let mut state = self
                .registry
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(active) = state.active.as_mut() else {
                return Err(HistoricalPublicationStartError::RegistrationLost {
                    run_id: self.run_id,
                });
            };
            if active.run_id != self.run_id {
                return Err(HistoricalPublicationStartError::RegistrationLost {
                    run_id: self.run_id,
                });
            }
            match active.phase {
                HistoricalFetchPhase::Capturing if !self.cancellation.is_cancelled() => {
                    active.phase = HistoricalFetchPhase::PublicationInProgress;
                    Ok(HistoricalPublicationPermit {
                        registry: self.registry,
                        run_id: self.run_id,
                    })
                }
                HistoricalFetchPhase::Capturing | HistoricalFetchPhase::CancellationRequested => {
                    Err(HistoricalPublicationStartError::Cancelled(
                        HistoricalRequestCancelled,
                    ))
                }
                HistoricalFetchPhase::PublicationInProgress => {
                    Err(HistoricalPublicationStartError::AlreadyInProgress {
                        run_id: self.run_id,
                    })
                }
            }
        }
    }

    pub(crate) struct HistoricalPublicationPermit<'a> {
        registry: &'a HistoricalFetchRegistry,
        run_id: u64,
    }

    pub(crate) struct QueuedHistoricalFetch<'a, Pending> {
        // Keep the queued admission before the run guard so cancellation/error
        // teardown releases the queue request before it releases the run id.
        pending: Pending,
        active: ActiveHistoricalFetch<'a>,
    }

    impl<'a, Pending> QueuedHistoricalFetch<'a, Pending> {
        pub(crate) fn run_id(&self) -> u64 {
            self.active.run_id()
        }

        pub(crate) fn cancellation(&self) -> &HistoricalRequestCancellation {
            self.active.cancellation()
        }

        pub(crate) fn into_parts(self) -> (ActiveHistoricalFetch<'a>, Pending) {
            (self.active, self.pending)
        }
    }

    impl HistoricalPublicationPermit<'_> {
        pub(crate) fn run_id(&self) -> u64 {
            self.run_id
        }
    }

    impl Drop for HistoricalPublicationPermit<'_> {
        fn drop(&mut self) {
            let mut state = self
                .registry
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(active) = state.active.as_mut()
                && active.run_id == self.run_id
                && active.phase == HistoricalFetchPhase::PublicationInProgress
            {
                active.phase = HistoricalFetchPhase::Capturing;
            }
        }
    }

    impl Drop for ActiveHistoricalFetch<'_> {
        fn drop(&mut self) {
            let mut state = self
                .registry
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state
                .active
                .as_ref()
                .is_some_and(|active| active.run_id == self.run_id)
            {
                state.active = None;
            }
        }
    }

    static PROCESS_HISTORICAL_FETCH_REGISTRY: OnceLock<HistoricalFetchRegistry> = OnceLock::new();

    fn process_historical_fetch_registry() -> &'static HistoricalFetchRegistry {
        PROCESS_HISTORICAL_FETCH_REGISTRY.get_or_init(HistoricalFetchRegistry::new)
    }

    pub(crate) fn begin_process_historical_fetch_queued<Pending, QueueError>(
        submit: impl FnOnce() -> Result<Pending, QueueError>,
    ) -> Result<QueuedHistoricalFetch<'static, Pending>, HistoricalFetchQueueStartError<QueueError>>
    {
        process_historical_fetch_registry().try_start_queued(submit)
    }

    pub(crate) fn cancel_process_historical_fetch(run_id: u64) -> HistoricalFetchCancelOutcome {
        process_historical_fetch_registry().cancel_run(run_id)
    }

    pub(crate) fn process_historical_fetch_status() -> Option<HistoricalFetchStatus> {
        process_historical_fetch_registry().status()
    }
}

#[cfg(any(test, feature = "broker-history-service"))]
pub(crate) use historical_fetch_registry::{
    HistoricalFetchCancelOutcome, HistoricalFetchQueueStartError, HistoricalFetchStartError,
    begin_process_historical_fetch_queued, cancel_process_historical_fetch,
    process_historical_fetch_status,
};

#[cfg(feature = "broker-history-service")]
pub(crate) use historical_fetch_registry::ActiveHistoricalFetch;

#[cfg(test)]
pub(crate) use historical_fetch_registry::{
    HistoricalFetchConflict, HistoricalFetchPhase, HistoricalFetchRegistry, HistoricalFetchStatus,
    HistoricalPublicationStartError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CTraderResponseDeadlineOverflow;

impl fmt::Display for CTraderResponseDeadlineOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cTrader response deadline overflowed")
    }
}

impl Error for CTraderResponseDeadlineOverflow {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CTraderResponseWaitError {
    Cancelled(HistoricalRequestCancelled),
    DeadlineExceeded { timeout: Duration },
}

impl fmt::Display for CTraderResponseWaitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled(error) => error.fmt(formatter),
            Self::DeadlineExceeded { timeout } => write!(
                formatter,
                "cTrader response deadline exceeded after {} ms",
                timeout.as_millis()
            ),
        }
    }
}

impl Error for CTraderResponseWaitError {}

/// Absolute monotonic deadline for one sent request. This type exposes no
/// reset operation, so heartbeat or unrelated frames cannot extend the wait.
pub(crate) struct ConnectionResponseDeadline {
    deadline: Instant,
    timeout: Duration,
}

impl ConnectionResponseDeadline {
    pub(crate) fn new(timeout: Duration) -> Result<Self, CTraderResponseDeadlineOverflow> {
        Self::new_at(Instant::now(), timeout)
    }

    fn new_at(
        started_at: Instant,
        timeout: Duration,
    ) -> Result<Self, CTraderResponseDeadlineOverflow> {
        let deadline = started_at
            .checked_add(timeout)
            .ok_or(CTraderResponseDeadlineOverflow)?;
        Ok(Self { deadline, timeout })
    }

    pub(crate) fn check(
        &self,
        cancellation: Option<&HistoricalRequestCancellation>,
    ) -> Result<(), CTraderResponseWaitError> {
        self.check_at(Instant::now(), cancellation)
    }

    fn check_at(
        &self,
        now: Instant,
        cancellation: Option<&HistoricalRequestCancellation>,
    ) -> Result<(), CTraderResponseWaitError> {
        if cancellation.is_some_and(HistoricalRequestCancellation::is_cancelled) {
            return Err(CTraderResponseWaitError::Cancelled(
                HistoricalRequestCancelled,
            ));
        }
        if now >= self.deadline {
            return Err(CTraderResponseWaitError::DeadlineExceeded {
                timeout: self.timeout,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) enum HistoricalAdmissionSendError<E> {
    ClockOverflow(HistoricalAdmissionClockOverflow),
    Cancelled(HistoricalRequestCancelled),
    Send(E),
}

/// State for exactly one open cTrader socket. It is intentionally neither
/// static nor shared: the official quota is per connection, so a fresh
/// `send_sequence` must start with an immediate first historical request.
pub(crate) struct ConnectionHistoricalAdmission {
    minimum_interval: Duration,
    last_send: Option<Instant>,
}

impl ConnectionHistoricalAdmission {
    pub(crate) fn new(minimum_interval: Duration) -> Self {
        Self {
            minimum_interval,
            last_send: None,
        }
    }

    /// Wait, run the actual send, then anchor the next deadline to the
    /// completed send's monotonic time.
    /// Anchoring after the closure prevents catch-up bursts after a delayed
    /// sender. Failed sends consume no slot.
    pub(crate) fn admit_and_send<E>(
        &mut self,
        cancellation: Option<&HistoricalRequestCancellation>,
        send: impl FnOnce() -> Result<(), E>,
    ) -> Result<Instant, HistoricalAdmissionSendError<E>> {
        if cancellation.is_some_and(HistoricalRequestCancellation::is_cancelled) {
            return Err(HistoricalAdmissionSendError::Cancelled(
                HistoricalRequestCancelled,
            ));
        }
        if let Some(last_send) = self.last_send {
            let deadline = last_send
                .checked_add(self.minimum_interval)
                .ok_or_else(|| {
                    HistoricalAdmissionSendError::ClockOverflow(HistoricalAdmissionClockOverflow)
                })?;
            let now = Instant::now();
            if now < deadline {
                let remaining = deadline.duration_since(now);
                if cancellation
                    .is_some_and(|cancellation| cancellation.wait_for_cancellation(remaining))
                {
                    return Err(HistoricalAdmissionSendError::Cancelled(
                        HistoricalRequestCancelled,
                    ));
                }
                if cancellation.is_none() {
                    std::thread::sleep(remaining);
                }
            }
        }
        if cancellation.is_some_and(HistoricalRequestCancellation::is_cancelled) {
            return Err(HistoricalAdmissionSendError::Cancelled(
                HistoricalRequestCancelled,
            ));
        }
        send().map_err(HistoricalAdmissionSendError::Send)?;
        let sent_at = Instant::now();
        self.last_send = Some(sent_at);
        Ok(sent_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::convert::Infallible;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use tungstenite::protocol::Role;
    use tungstenite::{WebSocket, protocol::WebSocketConfig};

    #[derive(Clone, Debug)]
    struct ManualMonotonicClock {
        now: Arc<Mutex<Instant>>,
    }

    impl ManualMonotonicClock {
        fn new(now: Instant) -> Self {
            Self {
                now: Arc::new(Mutex::new(now)),
            }
        }

        fn advance(&self, duration: Duration) {
            let mut now = self
                .now
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *now = now.checked_add(duration).expect("manual clock overflow");
        }
    }

    impl CTraderMonotonicClock for ManualMonotonicClock {
        fn now(&self) -> Instant {
            *self
                .now
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }
    }

    /// A peer that never blocks or closes: it produces a valid initial text
    /// fragment followed by an endless sequence of unfinished continuation
    /// frames. Every byte advances the injected monotonic clock, so a test can
    /// prove the boundary checks elapsed time even while the parser progresses.
    struct ProgressingFragmentStream {
        clock: ManualMonotonicClock,
        cancellation: Option<HistoricalRequestCancellation>,
        cancel_after_reads: Option<usize>,
        reads: usize,
    }

    impl ProgressingFragmentStream {
        fn new(clock: ManualMonotonicClock) -> Self {
            Self {
                clock,
                cancellation: None,
                cancel_after_reads: None,
                reads: 0,
            }
        }

        fn cancelling_after(
            clock: ManualMonotonicClock,
            cancellation: HistoricalRequestCancellation,
            reads: usize,
        ) -> Self {
            Self {
                clock,
                cancellation: Some(cancellation),
                cancel_after_reads: Some(reads),
                reads: 0,
            }
        }

        fn next_byte(&self) -> u8 {
            const INITIAL_TEXT_FRAGMENT: [u8; 3] = [0x01, 0x01, b'{'];
            const CONTINUATION_FRAGMENT: [u8; 3] = [0x00, 0x01, b' '];
            if self.reads < INITIAL_TEXT_FRAGMENT.len() {
                INITIAL_TEXT_FRAGMENT[self.reads]
            } else {
                CONTINUATION_FRAGMENT
                    [(self.reads - INITIAL_TEXT_FRAGMENT.len()) % CONTINUATION_FRAGMENT.len()]
            }
        }
    }

    impl Read for ProgressingFragmentStream {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            if buffer.is_empty() {
                return Ok(0);
            }
            buffer[0] = self.next_byte();
            self.reads += 1;
            self.clock.advance(Duration::from_millis(1));
            if self.cancel_after_reads == Some(self.reads) {
                self.cancellation
                    .as_ref()
                    .expect("cancelling stream owns its cancellation")
                    .cancel();
            }
            Ok(1)
        }
    }

    impl Write for ProgressingFragmentStream {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            if buffer.is_empty() {
                return Ok(0);
            }
            self.clock.advance(Duration::from_millis(1));
            Ok(1)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct AdvancingConnector {
        calls: Vec<CTraderIoPhase>,
        clock: Option<ManualMonotonicClock>,
        cancellation: Option<HistoricalRequestCancellation>,
        cancel_at: Option<CTraderIoPhase>,
    }

    impl AdvancingConnector {
        fn new(clock: ManualMonotonicClock) -> Self {
            Self {
                calls: Vec::new(),
                clock: Some(clock),
                cancellation: None,
                cancel_at: None,
            }
        }

        fn cancelling_at(
            clock: ManualMonotonicClock,
            cancellation: HistoricalRequestCancellation,
            phase: CTraderIoPhase,
        ) -> Self {
            Self {
                calls: Vec::new(),
                clock: Some(clock),
                cancellation: Some(cancellation),
                cancel_at: Some(phase),
            }
        }

        fn advance(&self, duration: Duration) {
            self.clock
                .as_ref()
                .expect("connector clock")
                .advance(duration);
        }

        fn complete_stage(
            &mut self,
            phase: CTraderIoPhase,
            budget: &CTraderOperationBudget<ManualMonotonicClock>,
            duration: Duration,
        ) -> std::io::Result<()> {
            self.calls.push(phase);
            self.advance(duration);
            if self.cancel_at == Some(phase) {
                self.cancellation
                    .as_ref()
                    .expect("cancelling connector owns its cancellation")
                    .cancel();
            }
            budget.check_io(phase)
        }
    }

    impl CTraderSocketConnector<ManualMonotonicClock> for AdvancingConnector {
        type Resolved = ();
        type Stream = ();
        type Socket = ();

        fn resolve(
            &mut self,
            budget: &CTraderOperationBudget<ManualMonotonicClock>,
        ) -> std::io::Result<Self::Resolved> {
            self.complete_stage(CTraderIoPhase::Dns, budget, Duration::from_millis(11))?;
            Ok(())
        }

        fn connect_tcp(
            &mut self,
            _resolved: Self::Resolved,
            budget: &CTraderOperationBudget<ManualMonotonicClock>,
        ) -> std::io::Result<Self::Stream> {
            self.complete_stage(
                CTraderIoPhase::TcpConnect,
                budget,
                Duration::from_millis(11),
            )?;
            Ok(())
        }

        fn tls_websocket_handshake(
            &mut self,
            _stream: Self::Stream,
            budget: &CTraderOperationBudget<ManualMonotonicClock>,
        ) -> std::io::Result<Self::Socket> {
            self.complete_stage(
                CTraderIoPhase::TlsWebSocketHandshake,
                budget,
                Duration::from_millis(11),
            )?;
            Ok(())
        }
    }

    struct RegistryStoppingFragmentStream<'a> {
        inner: ProgressingFragmentStream,
        registry: &'a HistoricalFetchRegistry,
        run_id: u64,
        stop_after_reads: usize,
    }

    impl Read for RegistryStoppingFragmentStream<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let read = self.inner.read(buffer)?;
            if self.inner.reads == self.stop_after_reads {
                assert_eq!(
                    self.registry.cancel_run(self.run_id),
                    HistoricalFetchCancelOutcome::Cancelled {
                        run_id: self.run_id,
                    }
                );
            }
            Ok(read)
        }
    }

    impl Write for RegistryStoppingFragmentStream<'_> {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.inner.write(buffer)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.inner.flush()
        }
    }

    struct FakeAdmittedCpuLease {
        dropped: Rc<Cell<bool>>,
    }

    impl Drop for FakeAdmittedCpuLease {
        fn drop(&mut self) {
            self.dropped.set(true);
        }
    }

    struct FakePendingAdmission {
        dropped: Rc<Cell<bool>>,
    }

    impl Drop for FakePendingAdmission {
        fn drop(&mut self) {
            self.dropped.set(true);
        }
    }

    #[test]
    fn six_messages_on_one_connection_are_strictly_spaced() {
        assert_eq!(
            CTRADER_HISTORICAL_REQUEST_MIN_INTERVAL,
            Duration::from_millis(225)
        );
        let interval = Duration::from_millis(15);
        let mut admission = ConnectionHistoricalAdmission::new(interval);
        let admitted_at = (0..6)
            .map(|_| {
                admission
                    .admit_and_send(None, || Ok::<_, Infallible>(()))
                    .expect("historical send")
            })
            .collect::<Vec<_>>();

        for pair in admitted_at.windows(2) {
            assert!(
                pair[1].duration_since(pair[0]) >= interval,
                "one connection granted a historical burst: {admitted_at:?}"
            );
        }
    }

    #[test]
    fn separate_connections_do_not_share_a_deadline() {
        let interval = Duration::from_secs(2);
        let mut first = ConnectionHistoricalAdmission::new(interval);
        let mut second = ConnectionHistoricalAdmission::new(interval);
        let started = Instant::now();

        first
            .admit_and_send(None, || Ok::<_, Infallible>(()))
            .expect("first socket");
        second
            .admit_and_send(None, || Ok::<_, Infallible>(()))
            .expect("independent socket");

        assert!(
            started.elapsed() < Duration::from_secs(1),
            "an independent connection inherited another socket's deadline"
        );
    }

    #[test]
    fn cancellation_interrupts_wait_and_does_not_consume_a_send_slot() {
        let interval = Duration::from_millis(500);
        let mut admission = ConnectionHistoricalAdmission::new(interval);
        let cancellation = HistoricalRequestCancellation::new();
        let first_send = admission
            .admit_and_send(Some(&cancellation), || Ok::<_, Infallible>(()))
            .expect("first historical send");

        let canceller = cancellation.clone();
        let cancel_thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            canceller.cancel();
        });
        let started = Instant::now();
        let mut send_calls = 0;
        let error = admission
            .admit_and_send(Some(&cancellation), || {
                send_calls += 1;
                Ok::<_, Infallible>(())
            })
            .expect_err("cancelled wait must not send");
        cancel_thread.join().expect("canceller thread");

        assert!(matches!(
            error,
            HistoricalAdmissionSendError::Cancelled(HistoricalRequestCancelled)
        ));
        assert_eq!(send_calls, 0);
        assert_eq!(admission.last_send, Some(first_send));
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "cancellation did not interrupt the 500ms admission wait"
        );
    }

    #[test]
    fn pre_cancelled_fresh_run_never_sends() {
        let mut admission = ConnectionHistoricalAdmission::new(Duration::ZERO);
        let cancellation = HistoricalRequestCancellation::new();
        cancellation.cancel();
        let mut send_calls = 0;

        let error = admission
            .admit_and_send(Some(&cancellation), || {
                send_calls += 1;
                Ok::<_, Infallible>(())
            })
            .expect_err("pre-cancelled run must fail before send");

        assert!(matches!(
            error,
            HistoricalAdmissionSendError::Cancelled(HistoricalRequestCancelled)
        ));
        assert_eq!(send_calls, 0);
        assert_eq!(admission.last_send, None);
    }

    #[test]
    fn cancellation_during_response_wait_fails_before_another_socket_poll() {
        let started = Instant::now();
        let response_deadline =
            ConnectionResponseDeadline::new_at(started, Duration::from_secs(30))
                .expect("response deadline");
        let cancellation = HistoricalRequestCancellation::new();

        response_deadline
            .check_at(started + Duration::from_secs(1), Some(&cancellation))
            .expect("uncancelled response wait");
        cancellation.cancel();

        assert!(matches!(
            response_deadline.check_at(started + Duration::from_secs(1), Some(&cancellation)),
            Err(CTraderResponseWaitError::Cancelled(
                HistoricalRequestCancelled
            ))
        ));
    }

    #[test]
    fn heartbeat_progress_never_extends_the_absolute_response_deadline() {
        let started = Instant::now();
        let timeout = Duration::from_secs(30);
        let response_deadline =
            ConnectionResponseDeadline::new_at(started, timeout).expect("response deadline");

        for heartbeat_at in [5, 10, 20, 29] {
            response_deadline
                .check_at(started + Duration::from_secs(heartbeat_at), None)
                .expect("heartbeat before absolute deadline");
        }

        assert!(matches!(
            response_deadline.check_at(started + timeout, None),
            Err(CTraderResponseWaitError::DeadlineExceeded {
                timeout: observed_timeout
            }) if observed_timeout == Duration::from_secs(30)
        ));
    }

    #[test]
    fn connect_stages_share_one_cumulative_absolute_budget() {
        let started = Instant::now();
        let clock = ManualMonotonicClock::new(started);
        let cancellation = HistoricalRequestCancellation::new();
        let budget = CTraderOperationBudget::new_at(
            clock.clone(),
            started,
            Duration::from_millis(30),
            Some(cancellation),
        )
        .expect("connect budget");
        let mut connector = AdvancingConnector::new(clock);

        let error = establish_ctrader_socket_with_connector(&mut connector, &budget)
            .expect_err("three individually-fast stages exceed one cumulative budget");

        let boundary = error
            .get_ref()
            .and_then(|source| source.downcast_ref::<CTraderIoBoundaryError>());
        assert!(matches!(
            boundary,
            Some(CTraderIoBoundaryError::DeadlineExceeded {
                phase: CTraderIoPhase::TlsWebSocketHandshake,
                timeout,
            }) if *timeout == Duration::from_millis(30)
        ));
        assert_eq!(
            connector.calls,
            vec![
                CTraderIoPhase::Dns,
                CTraderIoPhase::TcpConnect,
                CTraderIoPhase::TlsWebSocketHandshake,
            ],
            "the staged connector must not retry or reconnect"
        );
    }

    #[test]
    fn cancellation_terminates_each_connect_stage_without_a_later_stage_or_retry() {
        for cancelled_phase in [
            CTraderIoPhase::Dns,
            CTraderIoPhase::TcpConnect,
            CTraderIoPhase::TlsWebSocketHandshake,
        ] {
            let started = Instant::now();
            let clock = ManualMonotonicClock::new(started);
            let cancellation = HistoricalRequestCancellation::new();
            let budget = CTraderOperationBudget::new_at(
                clock.clone(),
                started,
                Duration::from_secs(30),
                Some(cancellation.clone()),
            )
            .expect("connect budget");
            let mut connector =
                AdvancingConnector::cancelling_at(clock, cancellation, cancelled_phase);

            let error = establish_ctrader_socket_with_connector(&mut connector, &budget)
                .expect_err("cancelled connect stage must fail closed");

            let boundary = error
                .get_ref()
                .and_then(|source| source.downcast_ref::<CTraderIoBoundaryError>());
            assert!(matches!(
                boundary,
                Some(CTraderIoBoundaryError::Cancelled { phase }) if *phase == cancelled_phase
            ));
            let expected_calls = match cancelled_phase {
                CTraderIoPhase::Dns => vec![CTraderIoPhase::Dns],
                CTraderIoPhase::TcpConnect => {
                    vec![CTraderIoPhase::Dns, CTraderIoPhase::TcpConnect]
                }
                CTraderIoPhase::TlsWebSocketHandshake => vec![
                    CTraderIoPhase::Dns,
                    CTraderIoPhase::TcpConnect,
                    CTraderIoPhase::TlsWebSocketHandshake,
                ],
                phase => panic!("unexpected connect test phase: {phase:?}"),
            };
            assert_eq!(connector.calls, expected_calls);
        }
    }

    #[test]
    fn continuously_progressing_fragments_cannot_extend_response_deadline() {
        let started = Instant::now();
        let clock = ManualMonotonicClock::new(started);
        let budget =
            CTraderOperationBudget::new_at(clock.clone(), started, Duration::from_millis(12), None)
                .expect("response budget");
        let stream = ProgressingFragmentStream::new(clock);
        let io = DeadlineIo::new(
            stream,
            budget,
            CTraderIoPhase::ResponseRead,
            Duration::from_millis(3),
        );
        let config = WebSocketConfig::default()
            .max_message_size(Some(1024))
            .max_frame_size(Some(128));
        let mut socket = WebSocket::from_raw_socket(io, Role::Client, Some(config));

        let error = socket
            .read()
            .expect_err("fragment progress must not reset the absolute deadline");

        let boundary = match &error {
            tungstenite::Error::Io(error) => error
                .get_ref()
                .and_then(|source| source.downcast_ref::<CTraderIoBoundaryError>()),
            _ => None,
        };
        assert!(matches!(
            boundary,
            Some(CTraderIoBoundaryError::DeadlineExceeded {
                phase: CTraderIoPhase::ResponseRead,
                timeout,
            }) if *timeout == Duration::from_millis(12)
        ));
    }

    #[test]
    fn continuously_progressing_websocket_write_cannot_extend_five_second_budget() {
        let started = Instant::now();
        let clock = ManualMonotonicClock::new(started);
        let budget =
            CTraderOperationBudget::new_at(clock.clone(), started, Duration::from_millis(5), None)
                .expect("write budget");
        let stream = ProgressingFragmentStream::new(clock);
        let io = DeadlineIo::new(
            stream,
            budget,
            CTraderIoPhase::RequestWrite,
            Duration::from_millis(2),
        );
        let mut socket = WebSocket::from_raw_socket(io, Role::Client, None);

        let error = socket
            .send(tungstenite::Message::Text(
                "bounded-write".repeat(32).into(),
            ))
            .expect_err("partial write progress must not reset the absolute budget");

        let boundary = match &error {
            tungstenite::Error::Io(error) => error
                .get_ref()
                .and_then(|source| source.downcast_ref::<CTraderIoBoundaryError>()),
            _ => None,
        };
        assert!(matches!(
            boundary,
            Some(CTraderIoBoundaryError::DeadlineExceeded {
                phase: CTraderIoPhase::RequestWrite,
                timeout,
            }) if *timeout == Duration::from_millis(5)
        ));
    }

    #[test]
    fn cancellation_interrupts_a_continuously_progressing_fragment() {
        let started = Instant::now();
        let clock = ManualMonotonicClock::new(started);
        let cancellation = HistoricalRequestCancellation::new();
        let budget = CTraderOperationBudget::new_at(
            clock.clone(),
            started,
            Duration::from_secs(30),
            Some(cancellation.clone()),
        )
        .expect("response budget");
        let stream = ProgressingFragmentStream::cancelling_after(clock, cancellation, 6);
        let io = DeadlineIo::new(
            stream,
            budget,
            CTraderIoPhase::ResponseRead,
            Duration::from_millis(3),
        );
        let mut socket = WebSocket::from_raw_socket(io, Role::Client, None);

        let error = socket
            .read()
            .expect_err("cancellation must interrupt parser progress");

        let boundary = match &error {
            tungstenite::Error::Io(error) => error
                .get_ref()
                .and_then(|source| source.downcast_ref::<CTraderIoBoundaryError>()),
            _ => None,
        };
        assert!(matches!(
            boundary,
            Some(CTraderIoBoundaryError::Cancelled {
                phase: CTraderIoPhase::ResponseRead,
            })
        ));
    }

    #[test]
    fn route_stop_during_progressing_read_releases_run_and_cpu_lease() {
        let registry = HistoricalFetchRegistry::new();
        let cpu_lease_dropped = Rc::new(Cell::new(false));
        let run_id = {
            let active = registry.try_start().expect("active fetch");
            let run_id = active.run_id();
            let started = Instant::now();
            let clock = ManualMonotonicClock::new(started);
            let budget = CTraderOperationBudget::new_at(
                clock.clone(),
                started,
                Duration::from_secs(30),
                Some(active.cancellation().clone()),
            )
            .expect("response budget");
            let stream = RegistryStoppingFragmentStream {
                inner: ProgressingFragmentStream::new(clock),
                registry: &registry,
                run_id,
                stop_after_reads: 6,
            };
            let io = DeadlineIo::new(
                stream,
                budget,
                CTraderIoPhase::ResponseRead,
                Duration::from_millis(3),
            );
            let mut socket = WebSocket::from_raw_socket(io, Role::Client, None);
            let lease = FakeAdmittedCpuLease {
                dropped: Rc::clone(&cpu_lease_dropped),
            };

            let error = active
                .execute_if_not_cancelled(|_| {
                    let _lease = lease;
                    socket.read()
                })
                .expect("the worker started before stop")
                .expect_err("the exact run stop must interrupt the socket read");
            let boundary = match &error {
                tungstenite::Error::Io(error) => error
                    .get_ref()
                    .and_then(|source| source.downcast_ref::<CTraderIoBoundaryError>()),
                _ => None,
            };
            assert!(matches!(
                boundary,
                Some(CTraderIoBoundaryError::Cancelled {
                    phase: CTraderIoPhase::ResponseRead,
                })
            ));
            run_id
        };

        assert!(
            cpu_lease_dropped.get(),
            "stop leaked the admitted CPU lease"
        );
        assert_eq!(
            registry.status(),
            None,
            "stop left the completed exact run registered"
        );
        let fresh = registry
            .try_start()
            .expect("a route-level stop must release capacity for a new run");
        assert_ne!(fresh.run_id(), run_id, "the released run id was reused");
    }

    #[test]
    fn persistent_historical_connect_threads_route_cancellation_into_socket_boundary() {
        let source = include_str!("ctrader_data.rs");
        let persistent_wire = source
            .split("impl CTraderPersistentHistoricalWire for ProductionCTraderPersistentHistoricalWire")
            .nth(1)
            .and_then(|tail| tail.split("fn application_auth").next())
            .expect("persistent historical connect implementation");

        assert!(
            persistent_wire.contains("connect_session(Some(&self.cancellation))"),
            "DNS/TCP/TLS connect cannot observe the route's exact-run stop token"
        );
        assert_eq!(
            persistent_wire.matches("connect_session(").count(),
            1,
            "the persistent wire must not reconnect around a cancelled connect"
        );
    }

    #[test]
    fn production_connect_uses_owned_hickory_runtime_and_deadline_io_below_tls() {
        let source = include_str!("ctrader_messages.rs");
        assert!(source.contains("Resolver::builder_tokio()"));
        assert!(source.contains("tokio::time::timeout(poll, lookup.as_mut())"));
        assert!(source.contains("tokio::net::TcpStream::connect(address)"));
        assert!(source.matches("drop(runtime);").count() >= 2);
        assert!(!source.contains("ToSocketAddrs"));
        assert!(!source.contains("connect(url.as_str())"));

        let handshake = source
            .split("fn handshake_ctrader_websocket")
            .nth(1)
            .and_then(|tail| {
                tail.split("impl CTraderSocketConnector<SystemCTraderMonotonicClock>")
                    .next()
            })
            .expect("production TLS/WebSocket handshake");
        let deadline_io = handshake
            .find("DeadlineIo::with_timeout_configurer")
            .expect("deadline wrapper");
        let tls = handshake
            .find("client_tls_with_config")
            .expect("caller-owned TLS/WebSocket handshake");
        assert!(deadline_io < tls, "DeadlineIo must sit below TLS/WebSocket");
        assert!(handshake.contains("HandshakeError::Interrupted"));
        assert!(handshake.contains("mid_handshake.handshake()"));
    }

    #[test]
    fn production_response_poll_and_deadline_configuration_is_live() {
        assert_eq!(CTRADER_RESPONSE_POLL_INTERVAL, Duration::from_millis(100));
        assert_eq!(CTRADER_RESPONSE_TIMEOUT, Duration::from_secs(30));

        let deadline = ConnectionResponseDeadline::new(CTRADER_RESPONSE_TIMEOUT)
            .expect("production response deadline");
        deadline.check(None).expect("fresh response deadline");
    }

    #[test]
    fn production_socket_write_is_bounded_and_fails_the_session_closed() {
        assert_eq!(CTRADER_SOCKET_WRITE_TIMEOUT, Duration::from_secs(5));

        let source = include_str!("ctrader_messages.rs");
        let connection = source
            .split("pub(crate) fn connect_session")
            .nth(1)
            .and_then(|tail| tail.split("pub fn build_application_auth_json").next())
            .expect("session connection source");
        assert!(connection.contains("CTRADER_CONNECT_TIMEOUT"));
        assert!(connection.contains("establish_ctrader_socket_with_connector"));

        let send_one = source
            .split("pub(crate) fn send_one")
            .nth(1)
            .and_then(|tail| {
                tail.split("impl Drop for ProductionCTraderOpenApiSession")
                    .next()
            })
            .expect("authoritative send-one source");
        assert!(send_one.contains("CTRADER_SOCKET_WRITE_TIMEOUT"));
        assert!(send_one.contains("CTraderIoPhase::RequestWrite"));
        assert!(send_one.contains("arm_ctrader_socket_budget"));
        assert!(send_one.contains("failed to send cTrader open api message"));
        assert!(send_one.contains("failed to reply to cTrader ping"));
        assert!(!send_one.contains("connect_session"));
        assert!(!send_one.contains("retry"));
    }

    #[test]
    fn one_active_fetch_conflicts_and_stop_cancels_only_that_run() {
        let registry = HistoricalFetchRegistry::new();
        let first = registry.try_start().expect("first fetch");
        let first_run_id = first.run_id();

        assert!(matches!(
            registry.try_start(),
            Err(HistoricalFetchStartError::AlreadyActive(
                HistoricalFetchConflict { active_run_id }
            )) if active_run_id == first_run_id
        ));
        assert_eq!(
            registry.cancel_run(first_run_id),
            HistoricalFetchCancelOutcome::Cancelled {
                run_id: first_run_id
            }
        );
        assert!(first.cancellation().is_cancelled());

        drop(first);
        assert_eq!(
            registry.cancel_run(first_run_id),
            HistoricalFetchCancelOutcome::NoActiveFetch
        );
        let second = registry.try_start().expect("fresh second fetch");
        assert_ne!(second.run_id(), first_run_id);
        assert!(!second.cancellation().is_cancelled());
    }

    #[test]
    fn active_queued_fetch_rejects_a_second_caller_before_cpu_submission() {
        let registry = HistoricalFetchRegistry::new();
        let first_pending_dropped = Rc::new(Cell::new(false));
        let first = registry
            .try_start_queued(|| {
                Ok::<_, Infallible>(FakePendingAdmission {
                    dropped: Rc::clone(&first_pending_dropped),
                })
            })
            .expect("first fetch waits for a saturated CPU queue");
        let second_submit_called = Cell::new(false);

        let second = registry.try_start_queued(|| {
            second_submit_called.set(true);
            Ok::<_, Infallible>(())
        });

        assert!(matches!(
            second,
            Err(HistoricalFetchQueueStartError::Fetch(
                HistoricalFetchStartError::AlreadyActive(HistoricalFetchConflict {
                    active_run_id
                })
            )) if active_run_id == first.run_id()
        ));
        assert!(
            !second_submit_called.get(),
            "the conflicting caller entered the CPU queue"
        );
        drop(first);
        assert!(first_pending_dropped.get());
        assert_eq!(registry.status(), None);
    }

    #[test]
    fn queue_and_execution_errors_release_the_registered_run() {
        let registry = HistoricalFetchRegistry::new();
        let queue_error = registry.try_start_queued(|| Err::<(), _>("synthetic queue error"));
        assert!(matches!(
            queue_error,
            Err(HistoricalFetchQueueStartError::Queue(
                "synthetic queue error"
            ))
        ));
        assert_eq!(registry.status(), None);

        let execute = || -> Result<(), &'static str> {
            let queued = registry
                .try_start_queued(|| Ok::<_, Infallible>(()))
                .expect("registered execution");
            let (active, ()) = queued.into_parts();
            let work = active
                .execute_if_not_cancelled(|_| Err("synthetic execution error"))
                .map_err(|_| "unexpected cancellation")?;
            work?;
            Ok(())
        };
        assert_eq!(execute(), Err("synthetic execution error"));
        assert_eq!(registry.status(), None);
        registry
            .try_start()
            .expect("error paths must release the process fetch slot");
    }

    #[test]
    fn cancellation_while_cpu_queued_drops_pending_and_never_executes_work() {
        let registry = HistoricalFetchRegistry::new();
        let pending_dropped = Rc::new(Cell::new(false));
        let queued = registry
            .try_start_queued(|| {
                Ok::<_, Infallible>(FakePendingAdmission {
                    dropped: Rc::clone(&pending_dropped),
                })
            })
            .expect("queued fetch");
        assert!(!queued.cancellation().is_cancelled());
        assert_eq!(
            registry.cancel_run(queued.run_id()),
            HistoricalFetchCancelOutcome::Cancelled {
                run_id: queued.run_id()
            }
        );
        assert!(queued.cancellation().is_cancelled());
        let (active, pending) = queued.into_parts();
        drop(pending);

        let settings_read = Cell::new(false);
        let network_started = Cell::new(false);
        let published = Cell::new(false);
        let result = active.execute_if_not_cancelled(|_| {
            settings_read.set(true);
            network_started.set(true);
            published.set(true);
        });

        assert_eq!(result, Err(HistoricalRequestCancelled));
        assert!(pending_dropped.get(), "queued CPU request was not dropped");
        assert!(!settings_read.get());
        assert!(!network_started.get());
        assert!(!published.get());
    }

    #[test]
    fn accepted_exact_run_cancel_prevents_publication() {
        let registry = HistoricalFetchRegistry::new();
        let active = registry.try_start().expect("active fetch");

        assert_eq!(
            registry.cancel_run(active.run_id()),
            HistoricalFetchCancelOutcome::Cancelled {
                run_id: active.run_id()
            }
        );
        let mut published = false;
        if active.begin_publication().is_ok() {
            published = true;
        }

        assert!(!published, "an accepted cancel reached publication");
        assert!(matches!(
            active.begin_publication(),
            Err(HistoricalPublicationStartError::Cancelled(
                HistoricalRequestCancelled
            ))
        ));
    }

    #[test]
    fn publication_first_returns_typed_in_progress_instead_of_false_cancel() {
        let registry = HistoricalFetchRegistry::new();
        let active = registry.try_start().expect("active fetch");
        let publication = active
            .begin_publication()
            .expect("atomic publication transition");
        assert_eq!(publication.run_id(), active.run_id());

        assert_eq!(
            registry.cancel_run(active.run_id()),
            HistoricalFetchCancelOutcome::PublicationInProgress {
                run_id: active.run_id()
            }
        );
        assert!(!active.cancellation().is_cancelled());
    }

    #[test]
    fn completed_publication_returns_the_same_run_to_capturing_for_the_next_cell() {
        let registry = HistoricalFetchRegistry::new();
        let active = registry.try_start().expect("active fetch");
        let first = active
            .begin_publication()
            .expect("first atomic publication transition");

        assert_eq!(
            registry.status(),
            Some(HistoricalFetchStatus {
                run_id: active.run_id(),
                phase: HistoricalFetchPhase::PublicationInProgress,
            })
        );
        drop(first);
        assert_eq!(
            registry.status(),
            Some(HistoricalFetchStatus {
                run_id: active.run_id(),
                phase: HistoricalFetchPhase::Capturing,
            })
        );

        let second = active
            .begin_publication()
            .expect("second atomic publication transition");
        assert_eq!(second.run_id(), active.run_id());
    }

    #[test]
    fn stale_run_stop_cannot_cancel_a_fresh_run() {
        let registry = HistoricalFetchRegistry::new();
        let first = registry.try_start().expect("first fetch");
        let stale_run_id = first.run_id();
        drop(first);
        let second = registry.try_start().expect("fresh fetch");

        assert_eq!(
            registry.cancel_run(stale_run_id),
            HistoricalFetchCancelOutcome::StaleRun {
                requested_run_id: stale_run_id,
                active_run_id: second.run_id(),
            }
        );
        assert!(!second.cancellation().is_cancelled());
    }

    #[test]
    fn active_fetch_raii_cleanup_runs_during_unwind() {
        let registry = HistoricalFetchRegistry::new();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let active = registry.try_start().expect("active fetch");
            let _publication = active.begin_publication().expect("publication transition");
            panic!("synthetic fetch panic");
        }));
        assert!(unwind.is_err());

        registry
            .try_start()
            .expect("panic path must release the process fetch slot");
    }

    #[test]
    fn fetch_status_reports_the_exact_run_and_atomic_phase() {
        let registry = HistoricalFetchRegistry::new();
        assert_eq!(registry.status(), None);

        let cancelled = registry.try_start().expect("cancelled fetch");
        assert_eq!(
            registry.status(),
            Some(HistoricalFetchStatus {
                run_id: cancelled.run_id(),
                phase: HistoricalFetchPhase::Capturing,
            })
        );
        assert_eq!(HistoricalFetchPhase::Capturing.as_str(), "capturing");
        assert!(matches!(
            registry.cancel_run(cancelled.run_id()),
            HistoricalFetchCancelOutcome::Cancelled { .. }
        ));
        assert_eq!(
            registry.status(),
            Some(HistoricalFetchStatus {
                run_id: cancelled.run_id(),
                phase: HistoricalFetchPhase::CancellationRequested,
            })
        );
        assert_eq!(
            HistoricalFetchPhase::CancellationRequested.as_str(),
            "cancellation_requested"
        );
        drop(cancelled);
        assert_eq!(registry.status(), None);

        let publishing = registry.try_start().expect("publishing fetch");
        let permit = publishing
            .begin_publication()
            .expect("publication transition");
        assert_eq!(permit.run_id(), publishing.run_id());
        assert_eq!(
            registry.status(),
            Some(HistoricalFetchStatus {
                run_id: publishing.run_id(),
                phase: HistoricalFetchPhase::PublicationInProgress,
            })
        );
        assert_eq!(
            HistoricalFetchPhase::PublicationInProgress.as_str(),
            "publication_in_progress"
        );
        drop(permit);
        assert_eq!(
            registry.status(),
            Some(HistoricalFetchStatus {
                run_id: publishing.run_id(),
                phase: HistoricalFetchPhase::Capturing,
            })
        );
        drop(publishing);
        assert_eq!(registry.status(), None);
    }

    #[test]
    fn process_fetch_registry_wrappers_share_the_registered_run() {
        let queued = begin_process_historical_fetch_queued(|| Ok::<_, Infallible>(()))
            .expect("process fetch");
        let run_id = queued.run_id();
        assert_eq!(
            process_historical_fetch_status(),
            Some(HistoricalFetchStatus {
                run_id,
                phase: HistoricalFetchPhase::Capturing,
            })
        );
        let (active, ()) = queued.into_parts();
        assert_eq!(
            cancel_process_historical_fetch(run_id),
            HistoricalFetchCancelOutcome::Cancelled { run_id }
        );
        drop(active);
        assert_eq!(process_historical_fetch_status(), None);
        assert_eq!(
            cancel_process_historical_fetch(u64::MAX),
            HistoricalFetchCancelOutcome::NoActiveFetch
        );
    }

    #[test]
    fn data_fetch_route_owns_raii_guard_and_stop_targets_the_registered_run() {
        let data_control = include_str!("../server/data_control.rs");
        let fetch = data_control
            .split("pub async fn fetch(")
            .nth(1)
            .and_then(|tail| tail.split("pub async fn fetch_status").next())
            .expect("data fetch handler source");
        let registered_start = fetch
            .find("begin_process_historical_capture()")
            .expect("shared typed fetch registration before CPU submission");
        let submit = fetch
            .find(".submit(broker_fetch_cpu_demand())")
            .expect("cancellable pending CPU admission");
        let wait = fetch
            .find("pending_admission.wait()")
            .expect("queued admission wait");
        let spawn = fetch
            .find("tokio::task::spawn_blocking")
            .expect("blocking fetch boundary");
        let download = fetch
            .find("download_history_blocking(")
            .expect("broker download");
        assert!(registered_start < submit && submit < wait && wait < spawn && spawn < download);
        assert!(fetch.contains("tokio::select!"));
        assert!(fetch.contains("active_fetch.is_cancelled()"));
        assert!(fetch.contains("dataset_selection.as_ref(),"));
        assert!(fetch.contains("&active_fetch,"));
        assert!(!fetch.contains(".admit(broker_fetch_cpu_demand())"));
        assert!(fetch.contains("HistoricalFetchStartFailure::AlreadyActive"));
        assert!(fetch.contains("StatusCode::CONFLICT"));

        let status = data_control
            .split("pub async fn fetch_status")
            .nth(1)
            .and_then(|tail| tail.split("pub async fn stop_fetch").next())
            .expect("typed data fetch status handler source");
        assert!(status.contains("process_historical_capture_status()"));
        assert!(status.contains("run_id: Some(status.run_id)"));
        assert!(status.contains("phase: Some(status.phase)"));

        let stop = data_control
            .split("pub async fn stop_fetch")
            .nth(1)
            .and_then(|tail| tail.split("// ─── POST /data/import").next())
            .expect("data fetch stop handler source");
        assert!(
            data_control
                .contains("pub async fn stop_fetch(Json(body): Json<StopFetchBody>) -> Response")
        );
        assert!(!data_control.contains("pub async fn stop_fetch()"));
        assert!(stop.contains("cancel_process_historical_capture(body.run_id)"));
        assert!(stop.contains("HistoricalFetchCancelResult::Cancelled"));
        assert!(stop.contains("HistoricalFetchCancelResult::PublicationInProgress"));
        assert!(stop.contains("HistoricalFetchCancelResult::StaleRun"));
        assert!(stop.contains("HistoricalFetchCancelResult::NoActiveFetch"));

        let routes = include_str!("../server/mod.rs");
        assert!(routes.contains("/data/fetch/status"));
        assert!(routes.contains("get(data_control::fetch_status)"));
        assert!(routes.contains("/data/fetch/stop"));
        assert!(routes.contains("post(data_control::stop_fetch)"));
    }

    #[test]
    fn production_session_uses_bounded_socket_polls_under_one_response_deadline() {
        let source = include_str!("ctrader_messages.rs");
        let connection = source
            .split("pub(crate) fn connect_session")
            .nth(1)
            .and_then(|tail| tail.split("pub fn build_application_auth_json").next())
            .expect("session connection source");
        assert!(connection.contains("CTRADER_CONNECT_TIMEOUT"));
        assert!(source.contains("DeadlineIo::with_timeout_configurer"));
        assert!(source.contains("configure_ctrader_tcp_timeout"));

        let send_one = source
            .split("pub(crate) fn send_one")
            .nth(1)
            .and_then(|tail| {
                tail.split("impl Drop for ProductionCTraderOpenApiSession")
                    .next()
            })
            .expect("authoritative send-one source");
        assert_eq!(
            send_one.matches("ConnectionResponseDeadline::new").count(),
            1,
            "a response or heartbeat must not reset the absolute deadline"
        );
        assert!(send_one.contains("is_ctrader_socket_poll_timeout"));
        assert!(send_one.contains("response_deadline.check(cancellation)"));
        assert!(send_one.contains("CTraderIoPhase::ResponseRead"));
        assert!(send_one.contains("response_budget.clone()"));
    }

    #[test]
    fn blocked_payload_type_is_never_retried() {
        let mut attempts = 0;
        for _ in 0..3 {
            attempts += 1;
            if !should_retry_ctrader_error("BLOCKED_PAYLOAD_TYPE") {
                break;
            }
        }

        assert_eq!(attempts, 1, "BLOCKED_PAYLOAD_TYPE was blindly retried");
        assert!(should_retry_ctrader_error("CANT_ROUTE"));
    }

    #[test]
    fn production_session_owns_one_admission_and_the_exact_payload_set() {
        let source = include_str!("ctrader_messages.rs");
        let classifier = source
            .split("fn is_ctrader_historical_request")
            .nth(1)
            .and_then(|tail| {
                tail.split("pub struct ProductionCTraderOpenApiTransport")
                    .next()
            })
            .expect("historical payload classifier source");
        for constant in [
            "CTRADER_OA_DEAL_LIST_REQUEST_PAYLOAD_TYPE",
            "CTRADER_OA_GET_TRENDBARS_REQUEST_PAYLOAD_TYPE",
            "CTRADER_OA_CASH_FLOW_HISTORY_LIST_REQUEST_PAYLOAD_TYPE",
            "CTRADER_OA_GET_TICK_DATA_REQUEST_PAYLOAD_TYPE",
        ] {
            assert_eq!(
                classifier.matches(constant).count(),
                1,
                "historical classifier drifted for {constant}"
            );
        }

        let connection = source
            .split("pub(crate) fn connect_session")
            .nth(1)
            .and_then(|tail| tail.split("pub fn build_application_auth_json").next())
            .expect("session connection source");
        let socket = connection
            .find("establish_ctrader_socket_with_connector")
            .expect("socket connection");
        let admission = connection
            .find("ConnectionHistoricalAdmission::new")
            .expect("connection-local admission construction");
        assert!(socket < admission);

        let session = source
            .split("impl ProductionCTraderOpenApiSession")
            .nth(1)
            .and_then(|tail| {
                tail.split("impl Drop for ProductionCTraderOpenApiSession")
                    .next()
            })
            .expect("authoritative session implementation");
        let send = session
            .find(".admit_and_send(cancellation")
            .expect("historical admission at socket send boundary");
        assert!(!session[..send].contains("OnceLock"));
        assert!(!session[..send].contains("static "));
    }

    #[test]
    fn production_transport_delegates_to_one_authoritative_socket_session() {
        let source = include_str!("ctrader_messages.rs");
        assert!(source.contains("struct ProductionCTraderOpenApiSession"));
        let transport = source
            .split("impl CTraderOpenApiTransport for ProductionCTraderOpenApiTransport")
            .nth(1)
            .expect("production transport implementation");
        let send_sequence = transport
            .split("fn send_sequence")
            .nth(1)
            .expect("send_sequence body");

        assert!(send_sequence.contains("self.connect_session(None)"));
        assert!(send_sequence.contains("session.send_one("));
        assert!(!send_sequence.contains("connect(url.as_str())"));
        assert!(!send_sequence.contains("socket.send("));
        assert!(!send_sequence.contains("socket.read("));
    }

    #[test]
    fn resilient_transport_returns_on_blocked_before_retry_sleep() {
        let source = include_str!("ctrader_messages.rs");
        let resilient = source
            .split("pub fn send_sequence_resilient")
            .nth(1)
            .and_then(|tail| tail.split("/// Snapshot of a").next())
            .expect("resilient transport source");
        let decision = resilient
            .find("!should_retry_ctrader_error(&error.error_code)")
            .expect("non-retryable broker-error decision");
        let immediate_return = resilient[decision..]
            .find("return Err")
            .map(|offset| decision + offset)
            .expect("immediate blocked return");
        let retry_sleep = resilient
            .find("std::thread::sleep")
            .expect("transient retry sleep");
        assert!(decision < immediate_return && immediate_return < retry_sleep);
    }
}

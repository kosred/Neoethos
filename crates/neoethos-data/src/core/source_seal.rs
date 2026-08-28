//! Platform-enforced exclusion and mutation detection for mutable import paths.

use anyhow::Result;
use neoethos_core::execution_budget::AuxiliarySlotLease;
use std::fs::File;
use std::path::Path;

const SOURCE_SEAL_SLOT_COUNT: usize = 8;

/// Number of coordinated auxiliary slots required by path-source sealing.
///
/// This logical limit is platform-stable so admission can be configured before
/// platform-specific source-seal initialization runs.
pub const fn source_seal_slot_limit() -> usize {
    SOURCE_SEAL_SLOT_COUNT
}

/// Initializes path-source sealing before application threads or an async runtime exist.
///
/// Linux callers must invoke this synchronously during process startup. Windows has no
/// signal-backed source seal, so the same startup hook is a deterministic no-op there.
pub fn initialize_source_seal_before_runtime() -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        linux::initialize_before_runtime()
    }
    #[cfg(windows)]
    {
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        anyhow::bail!(
            "path import source sealing is unsupported on this platform; use a streamed upload"
        )
    }
}

#[cfg(all(test, target_os = "linux"))]
static PROCESS_SIGNAL_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(test, target_os = "linux"))]
pub(crate) fn process_signal_test_guard() -> std::sync::MutexGuard<'static, ()> {
    let guard = PROCESS_SIGNAL_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    linux::initialize_for_tests().expect("initialize Linux source-seal test runtime");
    guard
}

#[cfg(all(test, windows))]
mod windows_preflight_tests {
    use super::initialize_source_seal_before_runtime;

    #[test]
    fn source_seal_preflight_is_a_deterministic_idempotent_noop_on_windows() {
        initialize_source_seal_before_runtime().expect("first Windows source-seal preflight");
        initialize_source_seal_before_runtime().expect("repeated Windows source-seal preflight");
    }

    #[test]
    fn source_seal_preflight_remains_supported_after_a_windows_thread_has_existed() {
        std::thread::spawn(|| {})
            .join()
            .expect("short-lived Windows thread");
        initialize_source_seal_before_runtime()
            .expect("Windows has no process signal-mask startup ordering requirement");
    }
}

pub(crate) struct SourceSeal {
    #[cfg(target_os = "linux")]
    inner: linux::LinuxSourceSeal,
}

impl SourceSeal {
    pub(crate) fn acquire(
        file: &File,
        path: &Path,
        auxiliary_slot: &AuxiliarySlotLease,
    ) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            Ok(Self {
                inner: linux::LinuxSourceSeal::acquire_exact(file, path, auxiliary_slot.index())?,
            })
        }
        #[cfg(windows)]
        {
            let _ = (file, path, auxiliary_slot);
            Ok(Self {})
        }
        #[cfg(not(any(target_os = "linux", windows)))]
        {
            let _ = (file, path, auxiliary_slot);
            anyhow::bail!(
                "path import source sealing is unsupported on this platform; use a streamed upload"
            )
        }
    }

    pub(crate) fn check_clean(&mut self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.inner.check_clean()
        }
        #[cfg(windows)]
        {
            Ok(())
        }
        #[cfg(not(any(target_os = "linux", windows)))]
        {
            unreachable!("unsupported platforms cannot construct SourceSeal")
        }
    }

    pub(crate) fn release(self, file: &File) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.inner.release(file)
        }
        #[cfg(windows)]
        {
            let _ = file;
            Ok(())
        }
        #[cfg(not(any(target_os = "linux", windows)))]
        {
            let _ = file;
            unreachable!("unsupported platforms cannot construct SourceSeal")
        }
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::SOURCE_SEAL_SLOT_COUNT;
    use anyhow::{Context, Result, anyhow, bail};
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::fs::{self, File};
    use std::io;
    use std::os::fd::{AsRawFd, RawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::time::Duration;

    const F_SETOWN_EX: libc::c_int = 15;
    const F_SETSIG: libc::c_int = 10;
    const F_OWNER_TID: libc::c_int = 0;
    const COORDINATOR_ACK_TIMEOUT: Duration = Duration::from_secs(5);
    const INOTIFY_BUFFER_BYTES: usize = 16 * 1024;
    const WATCH_MASK: u32 = libc::IN_MODIFY
        | libc::IN_ATTRIB
        | libc::IN_CLOSE_WRITE
        | libc::IN_MOVED_FROM
        | libc::IN_MOVED_TO
        | libc::IN_CREATE
        | libc::IN_DELETE
        | libc::IN_DELETE_SELF
        | libc::IN_MOVE_SELF
        | libc::IN_Q_OVERFLOW;

    static STARTUP: Mutex<StartupState> = Mutex::new(StartupState::Uninitialized);
    static SIGIO_EVENT_FD: AtomicI32 = AtomicI32::new(-1);

    enum StartupState {
        Uninitialized,
        Ready(Arc<Coordinator>),
        Failed(String),
    }

    #[repr(C)]
    struct FileOwnerEx {
        owner_type: libc::c_int,
        pid: libc::pid_t,
    }

    struct Entry {
        fd: RawFd,
        generation: u64,
        cancelled: Arc<AtomicBool>,
    }

    struct CoordinatorState {
        entries: HashMap<libc::c_int, Entry>,
        next_generation: u64,
    }

    enum Command {
        Unregister {
            signal: libc::c_int,
            fd: RawFd,
            generation: u64,
            acknowledgement: Sender<Result<()>>,
        },
    }

    struct Registration {
        signal: libc::c_int,
        fd: RawFd,
        generation: u64,
        cancelled: Arc<AtomicBool>,
    }

    struct Coordinator {
        state: Arc<Mutex<CoordinatorState>>,
        tid: libc::pid_t,
        signals: Vec<libc::c_int>,
        command_tx: Sender<Command>,
        command_event_fd: RawFd,
        fallback_event_fd: RawFd,
        healthy: Arc<AtomicBool>,
    }

    struct CoordinatorHealth(Arc<AtomicBool>);

    impl Drop for CoordinatorHealth {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }

    pub(super) fn initialize_before_runtime() -> Result<()> {
        let mut startup = startup_state()?;
        match &*startup {
            StartupState::Ready(coordinator) => {
                let coordinator = Arc::clone(coordinator);
                drop(startup);
                coordinator.verify_ready()?;
                return verify_current_signal_mask(&coordinator.signals);
            }
            StartupState::Failed(message) => return Err(anyhow!(message.clone())),
            StartupState::Uninitialized => {}
        }

        match initialize_coordinator(true) {
            Ok(coordinator) => {
                *startup = StartupState::Ready(coordinator);
                Ok(())
            }
            Err(error) => {
                let message = format!("{error:#}");
                *startup = StartupState::Failed(message.clone());
                Err(anyhow!(message))
            }
        }
    }

    #[cfg(test)]
    pub(super) fn initialize_for_tests() -> Result<()> {
        let mut startup = startup_state()?;
        match &*startup {
            StartupState::Ready(coordinator) => {
                let coordinator = Arc::clone(coordinator);
                drop(startup);
                block_source_signals_on_current_thread(&coordinator.signals)?;
                coordinator.verify_ready()?;
                return verify_current_signal_mask(&coordinator.signals);
            }
            StartupState::Failed(message) => return Err(anyhow!(message.clone())),
            StartupState::Uninitialized => {}
        }

        match initialize_coordinator(false) {
            Ok(coordinator) => {
                *startup = StartupState::Ready(coordinator);
                Ok(())
            }
            Err(error) => {
                let message = format!("{error:#}");
                *startup = StartupState::Failed(message.clone());
                Err(anyhow!(message))
            }
        }
    }

    fn initialized_coordinator() -> Result<Arc<Coordinator>> {
        let mut startup = startup_state()?;
        let coordinator = match &*startup {
            StartupState::Ready(coordinator) => Arc::clone(coordinator),
            StartupState::Failed(message) => return Err(anyhow!(message.clone())),
            StartupState::Uninitialized => {
                let message = concat!(
                    "Linux source sealing was used before synchronous startup initialization; ",
                    "call initialize_source_seal_before_runtime before spawning any other thread ",
                    "or constructing an async runtime"
                )
                .to_owned();
                *startup = StartupState::Failed(message.clone());
                return Err(anyhow!(message));
            }
        };
        drop(startup);
        coordinator.verify_ready()?;
        verify_current_signal_mask(&coordinator.signals)?;
        Ok(coordinator)
    }

    fn startup_state() -> Result<MutexGuard<'static, StartupState>> {
        STARTUP
            .lock()
            .map_err(|_| anyhow!("Linux source-seal startup state lock is poisoned"))
    }

    fn initialize_coordinator(require_initial_thread: bool) -> Result<Arc<Coordinator>> {
        if require_initial_thread {
            verify_initial_process_thread()?;
        }
        let signals = reserved_signals()?;
        let previous_mask = block_source_signals_on_current_thread(&signals)?;
        let initialized = (|| {
            verify_current_signal_mask(&signals)?;
            if require_initial_thread {
                verify_initial_process_thread()?;
            }
            let coordinator = Coordinator::start(signals)?;
            coordinator.verify_ready()?;
            Ok(coordinator)
        })();
        match initialized {
            Ok(coordinator) => Ok(coordinator),
            Err(error) => match restore_current_signal_mask(&previous_mask) {
                Ok(()) => Err(error),
                Err(restore_error) => Err(error.context(format!(
                    "also failed to restore the startup thread signal mask: {restore_error:#}"
                ))),
            },
        }
    }

    impl Coordinator {
        fn start(signals: Vec<libc::c_int>) -> Result<Arc<Self>> {
            let fallback_event_fd = create_event_fd("SIGIO fallback")?;
            let command_event_fd = create_event_fd("source-seal command")?;
            install_sigio_handler(fallback_event_fd)?;
            verify_sigio_handler(fallback_event_fd)?;
            let state = Arc::new(Mutex::new(CoordinatorState {
                entries: HashMap::new(),
                next_generation: 1,
            }));
            let healthy = Arc::new(AtomicBool::new(false));
            let (command_tx, command_rx) = mpsc::channel();
            let (startup_tx, startup_rx) = mpsc::sync_channel(1);
            let thread_state = Arc::clone(&state);
            let thread_signals = signals.clone();
            let thread_health = Arc::clone(&healthy);
            std::thread::Builder::new()
                .name("neoethos-source-seal".to_owned())
                .spawn(move || {
                    coordinator_main(
                        thread_state,
                        thread_signals,
                        command_rx,
                        command_event_fd,
                        fallback_event_fd,
                        startup_tx,
                        thread_health,
                    );
                })
                .context("spawn Linux source-seal coordinator")?;
            let tid = startup_rx
                .recv_timeout(COORDINATOR_ACK_TIMEOUT)
                .context("Linux source-seal coordinator did not start")??;
            Ok(Arc::new(Self {
                state,
                tid,
                signals,
                command_tx,
                command_event_fd,
                fallback_event_fd,
                healthy,
            }))
        }

        fn verify_ready(&self) -> Result<()> {
            if !self.healthy.load(Ordering::Acquire) {
                bail!("Linux source-seal coordinator is not running")
            }
            verify_sigio_handler(self.fallback_event_fd)
        }

        #[cfg(test)]
        fn register_automatic(&self, fd: RawFd) -> Result<Registration> {
            self.verify_ready()?;
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow!("Linux source-seal coordinator state lock is poisoned"))?;
            let signal = self
                .signals
                .iter()
                .copied()
                .find(|signal| !state.entries.contains_key(signal))
                .context("all Linux source-seal signal slots are occupied")?;
            Self::insert_registration(&mut state, signal, fd)
        }

        fn register_exact(&self, fd: RawFd, slot_index: usize) -> Result<Registration> {
            self.verify_ready()?;
            let signal = self.signals.get(slot_index).copied().with_context(|| {
                format!(
                    "Linux source-seal slot {slot_index} is outside the {} reserved Linux signal slots",
                    self.signals.len()
                )
            })?;
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow!("Linux source-seal coordinator state lock is poisoned"))?;
            if state.entries.contains_key(&signal) {
                bail!("Linux source-seal slot {slot_index} is already occupied")
            }
            Self::insert_registration(&mut state, signal, fd)
        }

        fn insert_registration(
            state: &mut CoordinatorState,
            signal: libc::c_int,
            fd: RawFd,
        ) -> Result<Registration> {
            let generation = state.next_generation;
            state.next_generation = state
                .next_generation
                .checked_add(1)
                .context("Linux source-seal generation overflow")?;
            let cancelled = Arc::new(AtomicBool::new(false));
            state.entries.insert(
                signal,
                Entry {
                    fd,
                    generation,
                    cancelled: Arc::clone(&cancelled),
                },
            );
            Ok(Registration {
                signal,
                fd,
                generation,
                cancelled,
            })
        }

        fn abort_registration(&self, registration: &Registration) -> Result<()> {
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow!("Linux source-seal coordinator state lock is poisoned"))?;
            if state
                .entries
                .get(&registration.signal)
                .is_some_and(|entry| {
                    entry.fd == registration.fd && entry.generation == registration.generation
                })
            {
                state.entries.remove(&registration.signal);
                Ok(())
            } else {
                cancel_all(&mut state);
                bail!("Linux source-seal registration changed before lease acquisition")
            }
        }

        fn unregister(&self, registration: &Registration) -> Result<()> {
            let (acknowledgement, wait) = mpsc::channel();
            self.command_tx
                .send(Command::Unregister {
                    signal: registration.signal,
                    fd: registration.fd,
                    generation: registration.generation,
                    acknowledgement,
                })
                .context("queue Linux source-seal unregister barrier")?;
            write_event_fd(self.command_event_fd)?;
            wait.recv_timeout(COORDINATOR_ACK_TIMEOUT)
                .context("Linux source-seal unregister/drain barrier timed out")??;
            Ok(())
        }
    }

    pub(super) struct LinuxSourceSeal {
        coordinator: Arc<Coordinator>,
        registration: Option<Registration>,
        inotify_fd: RawFd,
        watch_descriptor: libc::c_int,
        file_name: Vec<u8>,
        released: bool,
    }

    impl LinuxSourceSeal {
        pub(super) fn acquire_exact(file: &File, path: &Path, slot_index: usize) -> Result<Self> {
            Self::acquire_registered(file, path, |coordinator, fd| {
                coordinator.register_exact(fd, slot_index)
            })
        }

        #[cfg(test)]
        fn acquire_automatic(file: &File, path: &Path) -> Result<Self> {
            Self::acquire_registered(file, path, Coordinator::register_automatic)
        }

        fn acquire_registered(
            file: &File,
            path: &Path,
            register: impl FnOnce(&Coordinator, RawFd) -> Result<Registration>,
        ) -> Result<Self> {
            let coordinator = initialized_coordinator()?;
            let (inotify_fd, watch_descriptor, file_name) = install_parent_watch(path)?;
            let registration = match register(&coordinator, file.as_raw_fd()) {
                Ok(registration) => registration,
                Err(error) => {
                    close_inotify(inotify_fd, watch_descriptor);
                    return Err(error);
                }
            };
            let owner = FileOwnerEx {
                owner_type: F_OWNER_TID,
                pid: coordinator.tid,
            };
            if unsafe { libc::fcntl(file.as_raw_fd(), F_SETOWN_EX, &owner) } != 0 {
                let error = io::Error::last_os_error();
                coordinator.abort_registration(&registration)?;
                close_inotify(inotify_fd, watch_descriptor);
                return Err(error).context("set Linux source-seal signal owner thread");
            }
            if unsafe { libc::fcntl(file.as_raw_fd(), F_SETSIG, registration.signal) } != 0 {
                let error = io::Error::last_os_error();
                coordinator.abort_registration(&registration)?;
                close_inotify(inotify_fd, watch_descriptor);
                return Err(error).context("set Linux source-seal real-time signal slot");
            }
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETLEASE, libc::F_RDLCK) } != 0 {
                let error = io::Error::last_os_error();
                coordinator.abort_registration(&registration)?;
                close_inotify(inotify_fd, watch_descriptor);
                return Err(error).context(
                    "establish required Linux read lease; a pre-existing writer or unsupported filesystem is fail-closed",
                );
            }
            let mut seal = Self {
                coordinator,
                registration: Some(registration),
                inotify_fd,
                watch_descriptor,
                file_name,
                released: false,
            };
            seal.check_clean()?;
            Ok(seal)
        }

        pub(super) fn check_clean(&mut self) -> Result<()> {
            let registration = self
                .registration
                .as_ref()
                .context("Linux source seal is no longer registered")?;
            if registration.cancelled.load(Ordering::Acquire) {
                bail!(
                    "Linux source seal was broken or signal attribution became ambiguous; refusing publication"
                );
            }
            if drain_inotify(self.inotify_fd, self.watch_descriptor, &self.file_name)? {
                registration.cancelled.store(true, Ordering::Release);
                bail!("Linux source path changed while its immutable snapshot was being copied");
            }
            Ok(())
        }

        pub(super) fn release(mut self, file: &File) -> Result<()> {
            self.check_clean()?;
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETLEASE, libc::F_UNLCK) } != 0 {
                return Err(io::Error::last_os_error())
                    .context("release Linux import source lease");
            }
            let registration = self
                .registration
                .take()
                .context("Linux source seal registration was already released")?;
            self.coordinator.unregister(&registration)?;
            close_inotify(self.inotify_fd, self.watch_descriptor);
            self.inotify_fd = -1;
            self.watch_descriptor = -1;
            self.released = true;
            Ok(())
        }
    }

    impl Drop for LinuxSourceSeal {
        fn drop(&mut self) {
            if self.released {
                return;
            }
            if let Some(registration) = self.registration.take() {
                registration.cancelled.store(true, Ordering::Release);
                unsafe {
                    libc::fcntl(registration.fd, libc::F_SETLEASE, libc::F_UNLCK);
                }
                let _ = self.coordinator.unregister(&registration);
            }
            close_inotify(self.inotify_fd, self.watch_descriptor);
        }
    }

    fn coordinator_main(
        state: Arc<Mutex<CoordinatorState>>,
        signals: Vec<libc::c_int>,
        command_rx: Receiver<Command>,
        command_event_fd: RawFd,
        fallback_event_fd: RawFd,
        startup_tx: mpsc::SyncSender<Result<libc::pid_t>>,
        healthy: Arc<AtomicBool>,
    ) {
        let health = CoordinatorHealth(healthy);
        let setup = setup_signal_fd(&signals);
        let (signal_fd, tid) = match setup {
            Ok(value) => value,
            Err(error) => {
                let _ = startup_tx.send(Err(error));
                return;
            }
        };
        health.0.store(true, Ordering::Release);
        if startup_tx.send(Ok(tid)).is_err() {
            unsafe {
                libc::close(signal_fd);
            }
            return;
        }
        loop {
            let mut descriptors = [
                libc::pollfd {
                    fd: signal_fd,
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: fallback_event_fd,
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: command_event_fd,
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            let polled =
                unsafe { libc::poll(descriptors.as_mut_ptr(), descriptors.len() as _, -1) };
            if polled < 0 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                if let Ok(mut state) = state.lock() {
                    cancel_all(&mut state);
                }
                return;
            }
            if descriptors[0].revents & libc::POLLIN != 0 {
                drain_signal_fd(signal_fd, &state);
            }
            if descriptors[1].revents & libc::POLLIN != 0 {
                drain_event_fd(fallback_event_fd);
                if let Ok(mut state) = state.lock() {
                    cancel_all(&mut state);
                }
            }
            if descriptors[2].revents & libc::POLLIN != 0 {
                drain_event_fd(command_event_fd);
                while let Ok(command) = command_rx.try_recv() {
                    match command {
                        Command::Unregister {
                            signal,
                            fd,
                            generation,
                            acknowledgement,
                        } => {
                            drain_signal_fd(signal_fd, &state);
                            let result = unregister_entry(&state, signal, fd, generation);
                            let _ = acknowledgement.send(result);
                        }
                    }
                }
            }
        }
    }

    fn unregister_entry(
        state: &Arc<Mutex<CoordinatorState>>,
        signal: libc::c_int,
        fd: RawFd,
        generation: u64,
    ) -> Result<()> {
        let mut state = state
            .lock()
            .map_err(|_| anyhow!("Linux source-seal coordinator state lock is poisoned"))?;
        if state
            .entries
            .get(&signal)
            .is_some_and(|entry| entry.fd == fd && entry.generation == generation)
        {
            state.entries.remove(&signal);
            Ok(())
        } else {
            cancel_all(&mut state);
            bail!("Linux source-seal unregister identity mismatch")
        }
    }

    fn drain_signal_fd(signal_fd: RawFd, state: &Arc<Mutex<CoordinatorState>>) {
        loop {
            let mut information = std::mem::MaybeUninit::<libc::signalfd_siginfo>::uninit();
            let read = unsafe {
                libc::read(
                    signal_fd,
                    information.as_mut_ptr().cast(),
                    std::mem::size_of::<libc::signalfd_siginfo>(),
                )
            };
            if read < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::WouldBlock {
                    return;
                }
                if let Ok(mut state) = state.lock() {
                    cancel_all(&mut state);
                }
                return;
            }
            if usize::try_from(read).ok() != Some(std::mem::size_of::<libc::signalfd_siginfo>()) {
                if let Ok(mut state) = state.lock() {
                    cancel_all(&mut state);
                }
                return;
            }
            let information = unsafe { information.assume_init() };
            let Ok(mut state) = state.lock() else {
                return;
            };
            if information.ssi_signo == libc::SIGIO as u32 {
                cancel_all(&mut state);
                continue;
            }
            let signal = information.ssi_signo as libc::c_int;
            let fd = information.ssi_fd;
            match state.entries.get(&signal) {
                Some(entry) if entry.fd == fd => {
                    entry.cancelled.store(true, Ordering::Release);
                }
                _ => cancel_all(&mut state),
            }
        }
    }

    fn cancel_all(state: &mut CoordinatorState) {
        for entry in state.entries.values() {
            entry.cancelled.store(true, Ordering::Release);
        }
    }

    fn setup_signal_fd(signals: &[libc::c_int]) -> Result<(RawFd, libc::pid_t)> {
        verify_current_signal_mask(signals)?;
        let mask = source_signal_mask(signals)?;
        let signal_fd =
            unsafe { libc::signalfd(-1, &mask, libc::SFD_CLOEXEC | libc::SFD_NONBLOCK) };
        if signal_fd < 0 {
            return Err(io::Error::last_os_error()).context("create source-seal signalfd");
        }
        let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        Ok((signal_fd, tid))
    }

    fn source_signal_mask(signals: &[libc::c_int]) -> Result<libc::sigset_t> {
        let mut mask = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
        if unsafe { libc::sigemptyset(mask.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error()).context("initialize source-seal signal mask");
        }
        let mut mask = unsafe { mask.assume_init() };
        for signal in signals.iter().copied().chain(std::iter::once(libc::SIGIO)) {
            if unsafe { libc::sigaddset(&mut mask, signal) } != 0 {
                return Err(io::Error::last_os_error()).context("populate source-seal signal mask");
            }
        }
        Ok(mask)
    }

    fn block_source_signals_on_current_thread(signals: &[libc::c_int]) -> Result<libc::sigset_t> {
        let mask = source_signal_mask(signals)?;
        let mut previous = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
        let result =
            unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &mask, previous.as_mut_ptr()) };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result)).context(
                "block source-seal signals on the startup thread before spawning any other thread",
            );
        }
        Ok(unsafe { previous.assume_init() })
    }

    fn restore_current_signal_mask(previous: &libc::sigset_t) -> Result<()> {
        let result =
            unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, previous, std::ptr::null_mut()) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(result))
                .context("restore source-seal startup thread signal mask")
        }
    }

    fn verify_current_signal_mask(signals: &[libc::c_int]) -> Result<()> {
        let mut current = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
        let result = unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), current.as_mut_ptr())
        };
        if result != 0 {
            return Err(io::Error::from_raw_os_error(result))
                .context("inspect current source-seal signal mask");
        }
        let current = unsafe { current.assume_init() };
        for signal in signals.iter().copied().chain(std::iter::once(libc::SIGIO)) {
            match unsafe { libc::sigismember(&current, signal) } {
                1 => {}
                0 => bail!(
                    "source-seal signal {signal} is not blocked on the current thread; startup initialization did not precede every thread"
                ),
                _ => {
                    return Err(io::Error::last_os_error())
                        .context("inspect source-seal signal-mask member");
                }
            }
        }
        Ok(())
    }

    fn verify_initial_process_thread() -> Result<()> {
        let process_id = unsafe { libc::getpid() };
        let thread_id = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        let thread_count = fs::read_dir("/proc/self/task")
            .context("inspect /proc/self/task before Linux source-seal startup")?
            .try_fold(0_usize, |count, entry| {
                entry.context("inspect Linux process thread entry")?;
                count
                    .checked_add(1)
                    .context("Linux process thread count overflow")
            })?;
        if thread_id != process_id || thread_count != 1 {
            bail!(
                "initialize_source_seal_before_runtime must run on the initial process thread before spawning any other thread or constructing an async runtime (pid={process_id}, tid={thread_id}, observed_threads={thread_count})"
            )
        }
        Ok(())
    }

    fn reserved_signals() -> Result<Vec<libc::c_int>> {
        let minimum = libc::SIGRTMIN();
        let maximum = libc::SIGRTMAX();
        let available = maximum
            .checked_sub(minimum)
            .and_then(|value| value.checked_add(1))
            .context("invalid Linux real-time signal range")?;
        if usize::try_from(available)? < SOURCE_SEAL_SLOT_COUNT {
            bail!("Linux exposes fewer than {SOURCE_SEAL_SLOT_COUNT} real-time signal slots");
        }
        (0..SOURCE_SEAL_SLOT_COUNT)
            .map(|offset| {
                minimum
                    .checked_add(libc::c_int::try_from(offset)?)
                    .context("source-seal signal number overflow")
            })
            .collect()
    }

    fn install_sigio_handler(event_fd: RawFd) -> Result<()> {
        let mut old_action = std::mem::MaybeUninit::<libc::sigaction>::uninit();
        if unsafe { libc::sigaction(libc::SIGIO, std::ptr::null(), old_action.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error()).context("inspect existing SIGIO disposition");
        }
        let old_action = unsafe { old_action.assume_init() };
        if old_action.sa_sigaction != libc::SIG_DFL && old_action.sa_sigaction != libc::SIG_IGN {
            bail!("SIGIO already has a process handler; Linux path import sealing is unsupported");
        }
        SIGIO_EVENT_FD.store(event_fd, Ordering::Release);
        let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
        action.sa_sigaction = sigio_handler as *const () as usize;
        action.sa_flags = libc::SA_RESTART;
        if unsafe { libc::sigemptyset(&mut action.sa_mask) } != 0 {
            return Err(io::Error::last_os_error()).context("initialize SIGIO handler mask");
        }
        if unsafe { libc::sigaction(libc::SIGIO, &action, std::ptr::null_mut()) } != 0 {
            return Err(io::Error::last_os_error()).context("install fail-closed SIGIO handler");
        }
        Ok(())
    }

    fn verify_sigio_handler(event_fd: RawFd) -> Result<()> {
        if SIGIO_EVENT_FD.load(Ordering::Acquire) != event_fd {
            bail!("Linux source-seal SIGIO fallback eventfd identity changed")
        }
        let mut action = std::mem::MaybeUninit::<libc::sigaction>::uninit();
        if unsafe { libc::sigaction(libc::SIGIO, std::ptr::null(), action.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error())
                .context("verify installed source-seal SIGIO fallback handler");
        }
        let action = unsafe { action.assume_init() };
        if action.sa_sigaction != sigio_handler as *const () as usize {
            bail!("Linux source-seal SIGIO fallback handler changed after initialization")
        }
        Ok(())
    }

    extern "C" fn sigio_handler(_signal: libc::c_int) {
        let fd = SIGIO_EVENT_FD.load(Ordering::Relaxed);
        if fd < 0 {
            return;
        }
        let value = 1_u64.to_ne_bytes();
        unsafe {
            libc::write(fd, value.as_ptr().cast(), value.len());
        }
    }

    fn create_event_fd(label: &str) -> Result<RawFd> {
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 {
            Err(io::Error::last_os_error()).with_context(|| format!("create {label} eventfd"))
        } else {
            Ok(fd)
        }
    }

    fn write_event_fd(fd: RawFd) -> Result<()> {
        let value = 1_u64.to_ne_bytes();
        let written = unsafe { libc::write(fd, value.as_ptr().cast(), value.len()) };
        if written == value.len() as isize {
            Ok(())
        } else {
            Err(io::Error::last_os_error()).context("wake Linux source-seal coordinator")
        }
    }

    fn drain_event_fd(fd: RawFd) {
        loop {
            let mut value = 0_u64;
            let read = unsafe {
                libc::read(
                    fd,
                    (&mut value as *mut u64).cast(),
                    std::mem::size_of::<u64>(),
                )
            };
            if read < 0 && io::Error::last_os_error().kind() == io::ErrorKind::WouldBlock {
                return;
            }
            if read <= 0 {
                return;
            }
        }
    }

    fn install_parent_watch(path: &Path) -> Result<(RawFd, libc::c_int, Vec<u8>)> {
        let parent = path
            .parent()
            .context("import source has no canonical parent directory")?;
        let file_name = path
            .file_name()
            .context("import source has no file name")?
            .as_bytes()
            .to_vec();
        let parent = CString::new(parent.as_os_str().as_bytes())
            .context("source parent path contains an interior NUL")?;
        let inotify_fd = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
        if inotify_fd < 0 {
            return Err(io::Error::last_os_error()).context("initialize Linux source-path watch");
        }
        let watch_descriptor =
            unsafe { libc::inotify_add_watch(inotify_fd, parent.as_ptr(), WATCH_MASK) };
        if watch_descriptor < 0 {
            let error = io::Error::last_os_error();
            unsafe {
                libc::close(inotify_fd);
            }
            return Err(error).context("watch Linux source parent directory");
        }
        Ok((inotify_fd, watch_descriptor, file_name))
    }

    fn drain_inotify(fd: RawFd, watch_descriptor: libc::c_int, file_name: &[u8]) -> Result<bool> {
        let mut buffer = [0_u8; INOTIFY_BUFFER_BYTES];
        let mut changed = false;
        loop {
            let read = unsafe { libc::read(fd, buffer.as_mut_ptr().cast(), buffer.len()) };
            if read < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::WouldBlock {
                    return Ok(changed);
                }
                return Err(error).context("read Linux source-path watch");
            }
            if read == 0 {
                return Ok(changed);
            }
            let read = usize::try_from(read)?;
            let mut offset = 0_usize;
            while offset < read {
                let header_bytes = std::mem::size_of::<libc::inotify_event>();
                if read - offset < header_bytes {
                    bail!("truncated Linux inotify event header");
                }
                let event = unsafe {
                    std::ptr::read_unaligned(
                        buffer[offset..].as_ptr().cast::<libc::inotify_event>(),
                    )
                };
                let name_bytes = usize::try_from(event.len)?;
                let event_bytes = header_bytes
                    .checked_add(name_bytes)
                    .context("Linux inotify event length overflow")?;
                if event_bytes > read - offset {
                    bail!("truncated Linux inotify event name");
                }
                let name = &buffer[offset + header_bytes..offset + event_bytes];
                let name = &name[..name
                    .iter()
                    .position(|byte| *byte == 0)
                    .unwrap_or(name.len())];
                if event.mask & libc::IN_Q_OVERFLOW != 0
                    || (event.wd == watch_descriptor
                        && ((event.mask & (libc::IN_DELETE_SELF | libc::IN_MOVE_SELF)) != 0
                            || name == file_name))
                {
                    changed = true;
                }
                offset = offset
                    .checked_add(event_bytes)
                    .context("Linux inotify buffer offset overflow")?;
            }
        }
    }

    fn close_inotify(fd: RawFd, watch_descriptor: libc::c_int) {
        if fd < 0 {
            return;
        }
        if watch_descriptor >= 0 {
            unsafe {
                libc::inotify_rm_watch(fd, watch_descriptor);
            }
        }
        unsafe {
            libc::close(fd);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs::OpenOptions;
        use std::io::Write;
        use std::process::Command;
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        const UNINITIALIZED_PROBE: &str = "NEOETHOS_SOURCE_SEAL_UNINITIALIZED_PROBE";
        const LATE_PREFLIGHT_PROBE: &str = "NEOETHOS_SOURCE_SEAL_LATE_PREFLIGHT_PROBE";

        #[test]
        fn lazy_first_import_without_startup_preflight_fails_closed() {
            let output =
                Command::new(std::env::current_exe().expect("current unit-test executable"))
                    .env(UNINITIALIZED_PROBE, "1")
                    .arg("lazy_first_import_without_startup_preflight_probe")
                    .arg("--nocapture")
                    .arg("--test-threads=1")
                    .output()
                    .expect("run isolated uninitialized source-seal probe");
            assert!(
                output.status.success(),
                "isolated uninitialized probe failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        #[test]
        fn lazy_first_import_without_startup_preflight_probe() {
            if std::env::var_os(UNINITIALIZED_PROBE).is_none() {
                return;
            }
            let temporary = tempfile::tempdir().expect("tempdir");
            let source = temporary.path().join("uninitialized.csv");
            std::fs::write(&source, b"source").expect("write source");
            let file = File::open(&source).expect("open source");
            let error = LinuxSourceSeal::acquire_automatic(&file, &source)
                .err()
                .expect("first import must not initialize the coordinator lazily");
            assert!(
                format!("{error:#}").contains("initialize_source_seal_before_runtime"),
                "{error:#}"
            );
        }

        #[test]
        fn startup_preflight_after_threads_exist_fails_closed_permanently() {
            let output =
                Command::new(std::env::current_exe().expect("current unit-test executable"))
                    .env(LATE_PREFLIGHT_PROBE, "1")
                    .arg("startup_preflight_after_threads_exist_probe")
                    .arg("--nocapture")
                    .arg("--test-threads=1")
                    .output()
                    .expect("run isolated late-preflight source-seal probe");
            assert!(
                output.status.success(),
                "isolated late-preflight probe failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        #[test]
        fn startup_preflight_after_threads_exist_probe() {
            if std::env::var_os(LATE_PREFLIGHT_PROBE).is_none() {
                return;
            }
            let (release_tx, release_rx) = mpsc::sync_channel::<()>(0);
            let existing_thread = std::thread::spawn(move || {
                release_rx.recv().expect("release existing thread");
            });
            let first_error = crate::core::source_seal::initialize_source_seal_before_runtime()
                .expect_err("late source-seal preflight must fail closed");
            let second_error = crate::core::source_seal::initialize_source_seal_before_runtime()
                .expect_err("failed startup ordering must remain failed");
            release_tx.send(()).expect("release existing thread");
            existing_thread.join().expect("existing thread");
            assert!(
                format!("{first_error:#}").contains("before spawning any other thread"),
                "{first_error:#}"
            );
            assert_eq!(format!("{first_error:#}"), format!("{second_error:#}"));
        }

        #[test]
        fn test_preflight_establishes_a_verified_signal_runtime_before_acquire() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let coordinator = initialized_coordinator().expect("initialized test coordinator");
            coordinator
                .verify_ready()
                .expect("coordinator, signalfd, and SIGIO fallback are ready");
            verify_current_signal_mask(&coordinator.signals)
                .expect("source-seal signals are blocked on the acquiring thread");
        }

        #[test]
        fn threads_spawned_after_preflight_inherit_the_entire_source_signal_mask() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let signals = initialized_coordinator()
                .expect("initialized test coordinator")
                .signals
                .clone();
            std::thread::spawn(move || verify_current_signal_mask(&signals))
                .join()
                .expect("inherited-mask thread")
                .expect("new thread inherited every reserved signal and SIGIO as blocked");
        }

        #[test]
        fn exact_slot_registration_uses_the_matching_reserved_signal() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let temporary = tempfile::tempdir().expect("tempdir");
            let source = temporary.path().join("exact-slot.csv");
            std::fs::write(&source, b"source").expect("write source");
            let file = File::open(&source).expect("open source");
            let coordinator = initialized_coordinator().expect("initialized coordinator");
            let expected_signal = coordinator.signals[3];

            let seal = LinuxSourceSeal::acquire_exact(&file, &source, 3).expect("exact slot seal");

            assert_eq!(
                seal.registration.as_ref().expect("registration").signal,
                expected_signal
            );
            seal.release(&file).expect("release exact slot seal");
        }

        #[test]
        fn occupied_exact_slot_is_rejected_instead_of_substituting_another_signal() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let temporary = tempfile::tempdir().expect("tempdir");
            let source_a = temporary.path().join("occupied-a.csv");
            let source_b = temporary.path().join("occupied-b.csv");
            std::fs::write(&source_a, b"source-a").expect("write source a");
            std::fs::write(&source_b, b"source-b").expect("write source b");
            let file_a = File::open(&source_a).expect("open source a");
            let file_b = File::open(&source_b).expect("open source b");
            let seal_a =
                LinuxSourceSeal::acquire_exact(&file_a, &source_a, 2).expect("first exact slot");

            let error = LinuxSourceSeal::acquire_exact(&file_b, &source_b, 2)
                .err()
                .expect("occupied exact slot must not fall back to another signal");

            assert!(format!("{error:#}").contains("slot 2 is already occupied"));
            seal_a.release(&file_a).expect("release first exact slot");
        }

        #[test]
        fn exact_slot_index_outside_the_reserved_signal_set_is_rejected() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let temporary = tempfile::tempdir().expect("tempdir");
            let source = temporary.path().join("out-of-range.csv");
            std::fs::write(&source, b"source").expect("write source");
            let file = File::open(&source).expect("open source");

            let error = LinuxSourceSeal::acquire_exact(&file, &source, SOURCE_SEAL_SLOT_COUNT)
                .err()
                .expect("out-of-range slot must fail closed");

            assert!(format!("{error:#}").contains("outside the 8 reserved Linux signal slots"));
        }

        #[test]
        fn concurrent_writer_breaks_only_the_target_source_seal() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let temporary = tempfile::tempdir().expect("tempdir");
            let source_a = temporary.path().join("a.csv");
            let source_b = temporary.path().join("b.csv");
            std::fs::write(&source_a, b"source-a").expect("write source a");
            std::fs::write(&source_b, b"source-b").expect("write source b");

            let file_a = File::open(&source_a).expect("open source a");
            let file_b = File::open(&source_b).expect("open source b");
            let mut seal_a =
                LinuxSourceSeal::acquire_automatic(&file_a, &source_a).expect("seal a");
            let mut seal_b =
                LinuxSourceSeal::acquire_automatic(&file_b, &source_b).expect("seal b");

            let (attempted_tx, attempted_rx) = mpsc::sync_channel(1);
            let writer_path = source_a.clone();
            let writer = std::thread::spawn(move || {
                attempted_tx.send(()).expect("announce write attempt");
                let mut file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&writer_path)
                    .expect("writer proceeds after broken lease is released");
                file.write_all(b"mutated").expect("mutate source");
                file.sync_all().expect("sync mutation");
            });
            attempted_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("writer attempted mutation");

            let deadline = Instant::now() + Duration::from_secs(5);
            let break_error = loop {
                match seal_a.check_clean() {
                    Ok(()) if Instant::now() < deadline => {
                        std::thread::yield_now();
                    }
                    Ok(()) => panic!("target lease break was not attributed before timeout"),
                    Err(error) => break error,
                }
            };
            assert!(
                format!("{break_error:#}").contains("broken")
                    || format!("{break_error:#}").contains("changed")
            );
            seal_b
                .check_clean()
                .expect("unrelated source seal stays clean");

            drop(seal_a);
            drop(file_a);
            writer.join().expect("writer thread");
            seal_b.release(&file_b).expect("release unrelated seal");
            assert_eq!(std::fs::read(&source_a).expect("read mutation"), b"mutated");
            assert_eq!(
                std::fs::read(&source_b).expect("read source b"),
                b"source-b"
            );
        }

        #[test]
        fn preexisting_writer_is_rejected_before_source_read() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let temporary = tempfile::tempdir().expect("tempdir");
            let source = temporary.path().join("open-writer.csv");
            std::fs::write(&source, b"source").expect("write source");
            let _writer = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&source)
                .expect("hold writable descriptor");
            let reader = File::open(&source).expect("open reader");
            let error = LinuxSourceSeal::acquire_automatic(&reader, &source)
                .err()
                .expect("pre-existing writer must prevent a read lease");
            assert!(
                format!("{error:#}").contains("pre-existing writer"),
                "{error:#}"
            );
        }

        #[test]
        fn rename_and_same_path_replacement_are_detected_by_the_parent_watch() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let temporary = tempfile::tempdir().expect("tempdir");
            let source = temporary.path().join("replace.csv");
            let moved = temporary.path().join("moved.csv");
            std::fs::write(&source, b"original").expect("write source");
            let reader = File::open(&source).expect("open reader");
            let mut seal =
                LinuxSourceSeal::acquire_automatic(&reader, &source).expect("seal source");

            std::fs::rename(&source, &moved).expect("rename sealed path");
            std::fs::write(&source, b"replacement").expect("replace same path");
            let deadline = Instant::now() + Duration::from_secs(5);
            let error = loop {
                match seal.check_clean() {
                    Ok(()) if Instant::now() < deadline => std::thread::yield_now(),
                    Ok(()) => panic!("rename/replacement event was not observed"),
                    Err(error) => break error,
                }
            };
            assert!(format!("{error:#}").contains("changed"), "{error:#}");
            drop(seal);
            drop(reader);
            assert_eq!(std::fs::read(&moved).expect("moved bytes"), b"original");
            assert_eq!(
                std::fs::read(&source).expect("replacement bytes"),
                b"replacement"
            );
        }

        #[test]
        fn ambiguous_sigio_fallback_cancels_all_active_source_seals() {
            let _process_signal_guard = crate::core::source_seal::process_signal_test_guard();
            let temporary = tempfile::tempdir().expect("tempdir");
            let source_a = temporary.path().join("fallback-a.csv");
            let source_b = temporary.path().join("fallback-b.csv");
            std::fs::write(&source_a, b"source-a").expect("write source a");
            std::fs::write(&source_b, b"source-b").expect("write source b");
            let file_a = File::open(&source_a).expect("open source a");
            let file_b = File::open(&source_b).expect("open source b");
            let mut seal_a =
                LinuxSourceSeal::acquire_automatic(&file_a, &source_a).expect("seal a");
            let mut seal_b =
                LinuxSourceSeal::acquire_automatic(&file_b, &source_b).expect("seal b");

            sigio_handler(libc::SIGIO);
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let a_cancelled = seal_a.check_clean().is_err();
                let b_cancelled = seal_b.check_clean().is_err();
                if a_cancelled && b_cancelled {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "ambiguous SIGIO did not cancel every seal"
                );
                std::thread::yield_now();
            }
        }
    }
}

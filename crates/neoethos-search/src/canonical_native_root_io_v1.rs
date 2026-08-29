#[cfg(not(target_os = "linux"))]
use neoethos_core::Settings;

#[cfg(not(target_os = "linux"))]
use crate::canonical_native_discovery_request_v1::CanonicalNativeDiscoveryRequestErrorV1;

pub const MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1: usize = 8 * 1024 * 1024;

#[cfg(not(target_os = "linux"))]
pub struct SealedCanonicalRootV1 {
    _unsupported: (),
}

#[cfg(not(target_os = "linux"))]
impl SealedCanonicalRootV1 {
    pub fn from_startup_settings(
        _settings: &Settings,
    ) -> Result<Self, CanonicalNativeDiscoveryRequestErrorV1> {
        return Err(CanonicalNativeDiscoveryRequestErrorV1::UnsupportedPlatform);
    }
}

#[cfg(target_os = "linux")]
mod linux {
    #![cfg_attr(
        all(not(test), not(feature = "gpu-cuda")),
        expect(
            dead_code,
            reason = "the atomic publisher is consumed only by the gpu-cuda generation-zero adapter"
        )
    )]

    use std::ffi::CString;
    use std::fs::{self, File};
    use std::io::{Read, Write};
    use std::mem::size_of;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::FileExt;
    use std::path::{Component, Path, PathBuf};

    use neoethos_core::Settings;

    use super::MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1;
    use crate::canonical_native_discovery_request_v1::CanonicalNativeDiscoveryRequestErrorV1;

    const RESOLVE_POLICY_V1: u64 = 0x01 | 0x02 | 0x04 | 0x08;
    const MAX_CANONICAL_ATOMIC_PUBLISH_BYTES_V1: u64 =
        crate::canonical_native_discovery_request_v1::MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1;
    const _: () = assert!(MAX_CANONICAL_ATOMIC_PUBLISH_BYTES_V1 == 512 * 1024 * 1024);
    const EXACT_COMPARE_BUFFER_BYTES_V1: usize = 64 * 1024;
    #[cfg(test)]
    const _: () = assert!(RESOLVE_POLICY_V1 == 0x0f);

    #[repr(C)]
    struct OpenHow {
        flags: u64,
        mode: u64,
        resolve: u64,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct RootIdentity {
        device: u64,
        inode: u64,
        mode: u32,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct ArtifactIdentity {
        root: RootIdentity,
        size: i64,
        links: u64,
        modified_seconds: i64,
        modified_nanoseconds: i64,
        changed_seconds: i64,
        changed_nanoseconds: i64,
    }

    pub struct SealedCanonicalRootV1 {
        fd: OwnedFd,
        identity: RootIdentity,
        physical_path: PathBuf,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum CanonicalArtifactPublishGateRejectionV1 {
        Cancelled,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum CanonicalArtifactPublishDispositionV1 {
        Installed,
        ExistingIdentical,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum CanonicalArtifactFinalStateV1 {
        NotInstalled,
        InstalledSyncPending,
        InstalledDurable,
        ExistingIdentical,
        ExistingDifferent,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum CanonicalArtifactTemporaryStateV1 {
        NotCreated,
        Present,
        RemovedSyncPending,
        RemovedDurable,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct CanonicalArtifactPublishStateV1 {
        final_state: CanonicalArtifactFinalStateV1,
        temporary_state: CanonicalArtifactTemporaryStateV1,
    }

    impl CanonicalArtifactPublishStateV1 {
        pub(crate) const fn new(
            final_state: CanonicalArtifactFinalStateV1,
            temporary_state: CanonicalArtifactTemporaryStateV1,
        ) -> Self {
            Self {
                final_state,
                temporary_state,
            }
        }

        pub(crate) const fn final_state(self) -> CanonicalArtifactFinalStateV1 {
            self.final_state
        }

        pub(crate) const fn temporary_state(self) -> CanonicalArtifactTemporaryStateV1 {
            self.temporary_state
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum CanonicalArtifactPublishErrorKindV1 {
        InvalidRelativePath,
        SecureResolutionUnavailable(String),
        UnsafeLink,
        EscapeOrMount,
        RaceDetected,
        NonRegularArtifact,
        ArtifactTooLarge {
            maximum: u64,
            attempted: u64,
        },
        PreInstallRejected(CanonicalArtifactPublishGateRejectionV1),
        ExistingContentMismatch,
        Io {
            operation: &'static str,
            message: String,
        },
    }

    #[derive(Debug, PartialEq, Eq)]
    pub(crate) struct CanonicalArtifactPublishErrorV1 {
        kind: CanonicalArtifactPublishErrorKindV1,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<String>,
    }

    impl CanonicalArtifactPublishErrorV1 {
        pub(crate) const fn kind(&self) -> &CanonicalArtifactPublishErrorKindV1 {
            &self.kind
        }

        pub(crate) const fn state(&self) -> CanonicalArtifactPublishStateV1 {
            self.state
        }
    }

    impl std::fmt::Display for CanonicalArtifactPublishErrorV1 {
        fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(output, "canonical artifact publish {:?}: ", self.state)?;
            match &self.kind {
                CanonicalArtifactPublishErrorKindV1::InvalidRelativePath => {
                    output.write_str("invalid relative path")
                }
                CanonicalArtifactPublishErrorKindV1::SecureResolutionUnavailable(reason) => {
                    write!(output, "secure resolution unavailable: {reason}")
                }
                CanonicalArtifactPublishErrorKindV1::UnsafeLink => output.write_str("unsafe link"),
                CanonicalArtifactPublishErrorKindV1::EscapeOrMount => {
                    output.write_str("root escape or mount crossing")
                }
                CanonicalArtifactPublishErrorKindV1::RaceDetected => {
                    output.write_str("root/artifact race detected")
                }
                CanonicalArtifactPublishErrorKindV1::NonRegularArtifact => {
                    output.write_str("artifact is not regular")
                }
                CanonicalArtifactPublishErrorKindV1::ArtifactTooLarge { maximum, attempted } => {
                    write!(output, "artifact too large: {attempted}>{maximum}")
                }
                CanonicalArtifactPublishErrorKindV1::PreInstallRejected(reason) => {
                    write!(output, "pre-install gate rejected: {reason:?}")
                }
                CanonicalArtifactPublishErrorKindV1::ExistingContentMismatch => {
                    output.write_str("existing artifact bytes differ")
                }
                CanonicalArtifactPublishErrorKindV1::Io { operation, message } => {
                    write!(output, "{operation}: {message}")
                }
            }
        }
    }

    impl std::error::Error for CanonicalArtifactPublishErrorV1 {}

    #[derive(Debug, PartialEq, Eq)]
    pub(crate) struct CanonicalArtifactPublishReceiptV1 {
        relative_path: String,
        bytes_written: u64,
        disposition: CanonicalArtifactPublishDispositionV1,
    }

    impl CanonicalArtifactPublishReceiptV1 {
        pub(crate) fn relative_path(&self) -> &str {
            &self.relative_path
        }

        pub(crate) const fn bytes_written(&self) -> u64 {
            self.bytes_written
        }

        pub(crate) const fn disposition(&self) -> CanonicalArtifactPublishDispositionV1 {
            self.disposition
        }
    }

    pub(crate) struct BoundedCanonicalArtifactWriterV1<'a> {
        sink: &'a mut dyn Write,
        maximum_bytes: u64,
        bytes_written: u64,
        overflow_attempted_bytes: Option<u64>,
    }

    impl<'a> BoundedCanonicalArtifactWriterV1<'a> {
        fn new(sink: &'a mut dyn Write, maximum_bytes: u64) -> Self {
            Self {
                sink,
                maximum_bytes,
                bytes_written: 0,
                overflow_attempted_bytes: None,
            }
        }

        pub(crate) const fn bytes_written(&self) -> u64 {
            self.bytes_written
        }

        pub(crate) const fn overflow_attempted_bytes(&self) -> Option<u64> {
            self.overflow_attempted_bytes
        }
    }

    impl Write for BoundedCanonicalArtifactWriterV1<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let byte_count = u64::try_from(bytes.len()).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::FileTooLarge, "write length overflow")
            })?;
            let attempted = self.bytes_written.checked_add(byte_count).ok_or_else(|| {
                self.overflow_attempted_bytes = Some(u64::MAX);
                std::io::Error::new(std::io::ErrorKind::FileTooLarge, "artifact size overflow")
            })?;
            if attempted > self.maximum_bytes {
                self.overflow_attempted_bytes = Some(attempted);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::FileTooLarge,
                    "artifact exceeds absolute byte cap",
                ));
            }
            let written = self.sink.write(bytes)?;
            let written_bytes = u64::try_from(written)
                .map_err(|_| std::io::Error::other("written byte count overflow"))?;
            self.bytes_written = self
                .bytes_written
                .checked_add(written_bytes)
                .ok_or_else(|| std::io::Error::other("written byte counter overflow"))?;
            Ok(written)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.sink.flush()
        }
    }

    struct VerifiedDirectoryV1 {
        fd: OwnedFd,
        identity: RootIdentity,
        expected_path: PathBuf,
    }

    struct ExactStagedSnapshotV1 {
        file: File,
        identity: ArtifactIdentity,
        name: Option<CString>,
    }

    impl SealedCanonicalRootV1 {
        pub fn from_startup_settings(
            settings: &Settings,
        ) -> Result<Self, CanonicalNativeDiscoveryRequestErrorV1> {
            let configured = CString::new(settings.system.data_dir.as_os_str().as_bytes())
                .map_err(|_| root_error("canonical root contains NUL"))?;
            let raw_fd = unsafe {
                libc::open(
                    configured.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            if raw_fd < 0 {
                return Err(root_error(&std::io::Error::last_os_error().to_string()));
            }
            let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
            let stat = fstat(fd.as_raw_fd()).map_err(|error| root_error(&error.to_string()))?;
            if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
                return Err(root_error("canonical root is not a directory"));
            }
            let physical_path = handle_path(fd.as_raw_fd()).map_err(|error| {
                CanonicalNativeDiscoveryRequestErrorV1::SecureResolutionUnavailable(
                    error.to_string(),
                )
            })?;
            if !physical_path.is_absolute() {
                return Err(root_error(
                    "canonical root handle did not resolve absolutely",
                ));
            }
            Ok(Self {
                fd,
                identity: root_identity(&stat),
                physical_path,
            })
        }

        fn verify(&self) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
            let stat = fstat(self.fd.as_raw_fd()).map_err(|_| race())?;
            let path = handle_path(self.fd.as_raw_fd()).map_err(|_| race())?;
            if root_identity(&stat) != self.identity || path != self.physical_path {
                return Err(race());
            }
            Ok(())
        }
    }

    pub(crate) fn read_canonical_artifact_exact_v1(
        root: &SealedCanonicalRootV1,
        relative_path: &str,
    ) -> Result<Vec<u8>, CanonicalNativeDiscoveryRequestErrorV1> {
        read_canonical_artifact_exact_impl_v1(root, relative_path, (|| {}, || {}))
    }

    #[cfg(test)]
    pub(crate) fn read_canonical_artifact_exact_with_test_hook_v1(
        root: &SealedCanonicalRootV1,
        relative_path: &str,
        hooks: (impl FnOnce(), impl FnOnce()),
    ) -> Result<Vec<u8>, CanonicalNativeDiscoveryRequestErrorV1> {
        read_canonical_artifact_exact_impl_v1(root, relative_path, hooks)
    }

    fn read_canonical_artifact_exact_impl_v1(
        root: &SealedCanonicalRootV1,
        relative_path: &str,
        hooks: (impl FnOnce(), impl FnOnce()),
    ) -> Result<Vec<u8>, CanonicalNativeDiscoveryRequestErrorV1> {
        root.verify()?;
        let relative = CString::new(relative_path).map_err(|_| {
            CanonicalNativeDiscoveryRequestErrorV1::InvalidArtifactReference(
                "relative path contains NUL".to_owned(),
            )
        })?;
        let how = OpenHow {
            flags: (libc::O_RDONLY
                | libc::O_NONBLOCK
                | libc::O_NOCTTY
                | libc::O_CLOEXEC
                | libc::O_NOFOLLOW) as u64,
            mode: 0,
            resolve: RESOLVE_POLICY_V1,
        };
        let raw_fd = unsafe {
            libc::syscall(
                libc::SYS_openat2,
                root.fd.as_raw_fd(),
                relative.as_ptr(),
                &how,
                size_of::<OpenHow>(),
            ) as libc::c_int
        };
        if raw_fd < 0 {
            return Err(map_open_error(std::io::Error::last_os_error()));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        (hooks.0)();
        let before_stat = fstat(fd.as_raw_fd()).map_err(|error| artifact_io("fstat", error))?;
        if before_stat.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err(CanonicalNativeDiscoveryRequestErrorV1::NonRegularArtifact);
        }
        let before = artifact_identity(&before_stat);
        if before.root.device != root.identity.device {
            return Err(CanonicalNativeDiscoveryRequestErrorV1::EscapeOrMount);
        }
        if before.size < 0 || before.size as u64 > MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1 as u64 {
            return Err(CanonicalNativeDiscoveryRequestErrorV1::ArtifactTooLarge {
                maximum: MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1 as u64,
                observed: before.size.max(0) as u64,
            });
        }
        let expected_path = root.physical_path.join(Path::new(relative_path));
        require_exact_path(root, fd.as_raw_fd(), &expected_path)?;
        (hooks.1)();

        let file = File::from(fd);
        let mut bytes = Vec::with_capacity(before.size as usize + 1);
        let mut bounded = (&file).take(MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1 as u64 + 1);
        bounded
            .read_to_end(&mut bytes)
            .map_err(|error| artifact_io("read", error))?;
        if bytes.len() > MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1 {
            return Err(CanonicalNativeDiscoveryRequestErrorV1::ArtifactTooLarge {
                maximum: MAX_CANONICAL_RESEARCH_CONTRACT_BYTES_V1 as u64,
                observed: bytes.len() as u64,
            });
        }
        let after_stat = fstat(file.as_raw_fd()).map_err(|_| race())?;
        require_exact_path(root, file.as_raw_fd(), &expected_path)?;
        root.verify()?;
        if artifact_identity(&after_stat) != before || after_stat.st_size as usize != bytes.len() {
            return Err(race());
        }
        Ok(bytes)
    }

    pub(crate) fn publish_canonical_artifact_create_new_v1(
        root: &SealedCanonicalRootV1,
        relative_path: &str,
        emit: impl FnOnce(&mut BoundedCanonicalArtifactWriterV1<'_>) -> std::io::Result<()>,
        pre_install_gate: impl FnOnce() -> Result<(), CanonicalArtifactPublishGateRejectionV1>,
    ) -> Result<CanonicalArtifactPublishReceiptV1, CanonicalArtifactPublishErrorV1> {
        publish_canonical_artifact_create_new_impl_v1(
            root,
            relative_path,
            emit,
            pre_install_gate,
            || {},
        )
    }

    #[cfg(test)]
    pub(crate) fn publish_canonical_artifact_create_new_with_pre_link_test_hook_v1(
        root: &SealedCanonicalRootV1,
        relative_path: &str,
        emit: impl FnOnce(&mut BoundedCanonicalArtifactWriterV1<'_>) -> std::io::Result<()>,
        pre_install_gate: impl FnOnce() -> Result<(), CanonicalArtifactPublishGateRejectionV1>,
        pre_link_hook: impl FnOnce(),
    ) -> Result<CanonicalArtifactPublishReceiptV1, CanonicalArtifactPublishErrorV1> {
        publish_canonical_artifact_create_new_impl_v1(
            root,
            relative_path,
            emit,
            pre_install_gate,
            pre_link_hook,
        )
    }

    fn publish_canonical_artifact_create_new_impl_v1(
        root: &SealedCanonicalRootV1,
        relative_path: &str,
        emit: impl FnOnce(&mut BoundedCanonicalArtifactWriterV1<'_>) -> std::io::Result<()>,
        pre_install_gate: impl FnOnce() -> Result<(), CanonicalArtifactPublishGateRejectionV1>,
        pre_link_hook: impl FnOnce(),
    ) -> Result<CanonicalArtifactPublishReceiptV1, CanonicalArtifactPublishErrorV1> {
        let initial_state = CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::NotInstalled,
            CanonicalArtifactTemporaryStateV1::NotCreated,
        );
        verify_publish_root_v1(root, initial_state, None)?;
        let (directory_components, final_name) =
            checked_publish_path_v1(relative_path, initial_state)?;
        let mut parent = duplicate_verified_root_v1(root, initial_state)?;
        sync_directory_v1(
            parent.fd.as_raw_fd(),
            "fsync canonical root",
            initial_state,
            None,
        )?;
        for component in directory_components {
            parent =
                open_or_create_verified_directory_v1(root, &parent, &component, initial_state)?;
        }
        verify_publish_directory_v1(root, &parent, initial_state, None)?;

        let (temporary_name, mut temporary_file) = create_unique_temporary_v1(
            root,
            &parent,
            CanonicalArtifactPublishStateV1::new(
                CanonicalArtifactFinalStateV1::NotInstalled,
                CanonicalArtifactTemporaryStateV1::Present,
            ),
        )?;
        let temporary_name_text = temporary_name.to_string_lossy().into_owned();
        let staged_state = CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::NotInstalled,
            CanonicalArtifactTemporaryStateV1::Present,
        );

        let (emit_result, written, overflow) = {
            let mut bounded = BoundedCanonicalArtifactWriterV1::new(
                &mut temporary_file,
                MAX_CANONICAL_ATOMIC_PUBLISH_BYTES_V1,
            );
            let result = emit(&mut bounded);
            (
                result,
                bounded.bytes_written(),
                bounded.overflow_attempted_bytes(),
            )
        };
        if let Some(attempted) = overflow {
            return fail_before_install_and_cleanup_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                temporary_identity_v1(
                    root,
                    &parent,
                    &temporary_name,
                    &temporary_file,
                    staged_state,
                )?,
                publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::ArtifactTooLarge {
                        maximum: MAX_CANONICAL_ATOMIC_PUBLISH_BYTES_V1,
                        attempted,
                    },
                    staged_state,
                    Some(&temporary_name_text),
                ),
            );
        }
        if let Err(error) = emit_result {
            let identity = temporary_identity_v1(
                root,
                &parent,
                &temporary_name,
                &temporary_file,
                staged_state,
            )?;
            return fail_before_install_and_cleanup_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                identity,
                publish_io_error_v1(
                    "stream temporary artifact",
                    error,
                    staged_state,
                    Some(&temporary_name_text),
                ),
            );
        }
        if let Err(error) = temporary_file.flush() {
            let identity = temporary_identity_v1(
                root,
                &parent,
                &temporary_name,
                &temporary_file,
                staged_state,
            )?;
            return fail_before_install_and_cleanup_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                identity,
                publish_io_error_v1(
                    "flush temporary artifact",
                    error,
                    staged_state,
                    Some(&temporary_name_text),
                ),
            );
        }
        let before_sync = temporary_identity_v1(
            root,
            &parent,
            &temporary_name,
            &temporary_file,
            staged_state,
        )?;
        if before_sync.size < 0 || before_sync.size as u64 != written || before_sync.links != 1 {
            return fail_before_install_and_cleanup_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                before_sync,
                publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    staged_state,
                    Some(&temporary_name_text),
                ),
            );
        }
        if let Err(error) = temporary_file.sync_all() {
            return fail_before_install_and_cleanup_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                before_sync,
                publish_io_error_v1(
                    "fsync temporary artifact",
                    error,
                    staged_state,
                    Some(&temporary_name_text),
                ),
            );
        }
        let staged_identity = temporary_identity_v1(
            root,
            &parent,
            &temporary_name,
            &temporary_file,
            staged_state,
        )?;
        if staged_identity != before_sync {
            return fail_before_install_and_cleanup_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                staged_identity,
                publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    staged_state,
                    Some(&temporary_name_text),
                ),
            );
        }
        let temporary_expected = parent
            .expected_path
            .join(std::ffi::OsStr::from_bytes(temporary_name.to_bytes()));
        if let Err(error) = verify_named_identity_v1(
            root,
            &parent,
            &temporary_name,
            &temporary_expected,
            staged_identity,
            staged_state,
            Some(&temporary_name_text),
        ) {
            return fail_before_install_and_cleanup_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                staged_identity,
                error,
            );
        }
        let staged_snapshot = match create_exact_staged_snapshot_v1(
            root,
            &parent,
            &temporary_name,
            &temporary_file,
            staged_identity,
            staged_state,
            Some(&temporary_name_text),
        ) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return fail_before_install_and_cleanup_v1(
                    parent.fd.as_raw_fd(),
                    &temporary_name,
                    staged_identity,
                    error,
                );
            }
        };

        if let Err(rejection) = pre_install_gate() {
            return fail_before_install_and_cleanup_with_snapshot_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                staged_identity,
                &staged_snapshot,
                publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::PreInstallRejected(rejection),
                    staged_state,
                    Some(&temporary_name_text),
                ),
            );
        }
        if let Err(error) = verify_publish_root_v1(root, staged_state, Some(&temporary_name_text)) {
            return fail_before_install_and_cleanup_with_snapshot_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                staged_identity,
                &staged_snapshot,
                error,
            );
        }
        if let Err(error) =
            verify_publish_directory_v1(root, &parent, staged_state, Some(&temporary_name_text))
        {
            return fail_before_install_and_cleanup_with_snapshot_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                staged_identity,
                &staged_snapshot,
                error,
            );
        }
        if let Err(error) = verify_staged_after_gate_v1(
            root,
            &parent,
            &temporary_name,
            &temporary_file,
            &staged_snapshot,
            staged_state,
            Some(&temporary_name_text),
            staged_identity,
        ) {
            return fail_before_install_and_cleanup_with_snapshot_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                staged_identity,
                &staged_snapshot,
                error,
            );
        }
        pre_link_hook();
        let post_hook_verification =
            verify_publish_root_v1(root, staged_state, Some(&temporary_name_text))
                .and_then(|()| {
                    verify_publish_directory_v1(
                        root,
                        &parent,
                        staged_state,
                        Some(&temporary_name_text),
                    )
                })
                .and_then(|()| {
                    verify_staged_after_gate_v1(
                        root,
                        &parent,
                        &temporary_name,
                        &temporary_file,
                        &staged_snapshot,
                        staged_state,
                        Some(&temporary_name_text),
                        staged_identity,
                    )
                });
        if let Err(error) = post_hook_verification {
            return fail_before_install_and_cleanup_with_snapshot_v1(
                parent.fd.as_raw_fd(),
                &temporary_name,
                staged_identity,
                &staged_snapshot,
                error,
            );
        }

        let link_result =
            link_snapshot_no_replace_v1(&staged_snapshot.file, parent.fd.as_raw_fd(), &final_name);
        if link_result.is_ok() {
            return finish_new_install_v1(
                root,
                &parent,
                relative_path,
                &final_name,
                &temporary_name,
                staged_identity,
                &staged_snapshot,
                written,
            );
        }
        let link_error = link_result.unwrap_err();
        if link_error.raw_os_error() == Some(libc::EEXIST) {
            return finish_existing_winner_v1(
                root,
                &parent,
                relative_path,
                &final_name,
                &temporary_name,
                staged_identity,
                &staged_snapshot,
                written,
            );
        }
        fail_before_install_and_cleanup_with_snapshot_v1(
            parent.fd.as_raw_fd(),
            &temporary_name,
            staged_identity,
            &staged_snapshot,
            publish_io_error_v1(
                "linkat no-replace",
                link_error,
                staged_state,
                Some(&temporary_name_text),
            ),
        )
    }

    fn checked_publish_path_v1(
        relative_path: &str,
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<(Vec<CString>, CString), CanonicalArtifactPublishErrorV1> {
        let path = Path::new(relative_path);
        let mut components = Vec::new();
        let mut rebuilt = PathBuf::new();
        for component in path.components() {
            let Component::Normal(component) = component else {
                return Err(publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::InvalidRelativePath,
                    state,
                    None,
                ));
            };
            let name = CString::new(component.as_bytes()).map_err(|_| {
                publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::InvalidRelativePath,
                    state,
                    None,
                )
            })?;
            rebuilt.push(component);
            components.push(name);
        }
        if components.is_empty() || rebuilt.as_os_str().as_bytes() != relative_path.as_bytes() {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::InvalidRelativePath,
                state,
                None,
            ));
        }
        let final_name = components.pop().expect("nonempty checked above");
        Ok((components, final_name))
    }

    fn duplicate_verified_root_v1(
        root: &SealedCanonicalRootV1,
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<VerifiedDirectoryV1, CanonicalArtifactPublishErrorV1> {
        let raw_fd = unsafe { libc::fcntl(root.fd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3) };
        if raw_fd < 0 {
            return Err(publish_io_error_v1(
                "duplicate canonical root handle",
                std::io::Error::last_os_error(),
                state,
                None,
            ));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        let directory = VerifiedDirectoryV1 {
            fd,
            identity: root.identity,
            expected_path: root.physical_path.clone(),
        };
        verify_publish_directory_v1(root, &directory, state, None)?;
        Ok(directory)
    }

    fn open_or_create_verified_directory_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        component: &CString,
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<VerifiedDirectoryV1, CanonicalArtifactPublishErrorV1> {
        verify_publish_directory_v1(root, parent, state, None)?;
        let created =
            if unsafe { libc::mkdirat(parent.fd.as_raw_fd(), component.as_ptr(), 0o700) } == 0 {
                true
            } else {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EEXIST) {
                    return Err(map_publish_open_error_v1("mkdirat", error, state, None));
                }
                false
            };
        let how = OpenHow {
            flags: (libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC) as u64,
            mode: 0,
            resolve: RESOLVE_POLICY_V1,
        };
        let fd = openat2_owned_v1(parent.fd.as_raw_fd(), component, &how)
            .map_err(|error| map_publish_open_error_v1("openat2 directory", error, state, None))?;
        let stat = fstat(fd.as_raw_fd())
            .map_err(|error| publish_io_error_v1("fstat directory", error, state, None))?;
        if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::NonRegularArtifact,
                state,
                None,
            ));
        }
        let identity = root_identity(&stat);
        if identity.device != root.identity.device {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::EscapeOrMount,
                state,
                None,
            ));
        }
        let expected_path = parent
            .expected_path
            .join(std::ffi::OsStr::from_bytes(component.to_bytes()));
        require_exact_publish_path_v1(root, fd.as_raw_fd(), &expected_path, state, None)?;
        let directory = VerifiedDirectoryV1 {
            fd,
            identity,
            expected_path,
        };
        if created {
            sync_directory_v1(directory.fd.as_raw_fd(), "fsync new directory", state, None)?;
            sync_directory_v1(parent.fd.as_raw_fd(), "fsync directory parent", state, None)?;
        }
        verify_publish_directory_v1(root, parent, state, None)?;
        verify_publish_directory_v1(root, &directory, state, None)?;
        Ok(directory)
    }

    fn create_unique_temporary_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<(CString, File), CanonicalArtifactPublishErrorV1> {
        for _ in 0..32 {
            let name = random_temporary_name_v1(state)?;
            let how = OpenHow {
                flags: (libc::O_RDWR
                    | libc::O_CREAT
                    | libc::O_EXCL
                    | libc::O_NOFOLLOW
                    | libc::O_CLOEXEC) as u64,
                mode: 0o600,
                resolve: RESOLVE_POLICY_V1,
            };
            match openat2_owned_v1(parent.fd.as_raw_fd(), &name, &how) {
                Ok(fd) => {
                    let file = File::from(fd);
                    let identity = temporary_identity_v1(root, parent, &name, &file, state)?;
                    if identity.size != 0 || identity.links != 1 {
                        return Err(publish_error_v1(
                            CanonicalArtifactPublishErrorKindV1::RaceDetected,
                            state,
                            name.to_str().ok(),
                        ));
                    }
                    return Ok((name, file));
                }
                Err(error) if error.raw_os_error() == Some(libc::EEXIST) => continue,
                Err(error) => {
                    return Err(map_publish_open_error_v1(
                        "openat2 create-new temporary artifact",
                        error,
                        state,
                        name.to_str().ok(),
                    ));
                }
            }
        }
        Err(publish_io_message_v1(
            "create unique temporary artifact",
            "exhausted collision retries",
            state,
            None,
        ))
    }

    fn random_temporary_name_v1(
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<CString, CanonicalArtifactPublishErrorV1> {
        let mut random = [0_u8; 16];
        let mut filled = 0;
        while filled < random.len() {
            let result = unsafe {
                libc::getrandom(
                    random[filled..].as_mut_ptr().cast(),
                    random.len() - filled,
                    0,
                )
            };
            if result < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(publish_io_error_v1(
                    "getrandom temporary name",
                    error,
                    state,
                    None,
                ));
            }
            if result == 0 {
                return Err(publish_io_message_v1(
                    "getrandom temporary name",
                    "returned zero bytes",
                    state,
                    None,
                ));
            }
            filled += result as usize;
        }
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut name = String::with_capacity(32 + 30);
        name.push_str(".neoethos-canonical-tmp-");
        for byte in random {
            name.push(HEX[(byte >> 4) as usize] as char);
            name.push(HEX[(byte & 0x0f) as usize] as char);
        }
        CString::new(name).map_err(|_| {
            publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::InvalidRelativePath,
                state,
                None,
            )
        })
    }

    fn random_snapshot_name_v1(
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<CString, CanonicalArtifactPublishErrorV1> {
        const TEMPORARY_PREFIX: &[u8] = b".neoethos-canonical-tmp-";
        const SNAPSHOT_PREFIX: &[u8] = b".neoethos-canonical-snapshot-";
        let temporary = random_temporary_name_v1(state)?;
        let suffix = temporary
            .to_bytes()
            .strip_prefix(TEMPORARY_PREFIX)
            .ok_or_else(|| {
                publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    state,
                    None,
                )
            })?;
        let mut name = Vec::with_capacity(SNAPSHOT_PREFIX.len() + suffix.len());
        name.extend_from_slice(SNAPSHOT_PREFIX);
        name.extend_from_slice(suffix);
        CString::new(name).map_err(|_| {
            publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::InvalidRelativePath,
                state,
                None,
            )
        })
    }

    fn temporary_identity_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        name: &CString,
        file: &File,
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<ArtifactIdentity, CanonicalArtifactPublishErrorV1> {
        let display_name = name.to_string_lossy();
        let stat = fstat(file.as_raw_fd()).map_err(|error| {
            publish_io_error_v1(
                "fstat temporary artifact",
                error,
                state,
                Some(&display_name),
            )
        })?;
        if stat.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::NonRegularArtifact,
                state,
                Some(&display_name),
            ));
        }
        let identity = artifact_identity(&stat);
        if identity.root.device != root.identity.device {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::EscapeOrMount,
                state,
                Some(&display_name),
            ));
        }
        let expected = parent
            .expected_path
            .join(std::ffi::OsStr::from_bytes(name.to_bytes()));
        require_exact_publish_path_v1(
            root,
            file.as_raw_fd(),
            &expected,
            state,
            Some(&display_name),
        )?;
        Ok(identity)
    }

    fn create_exact_staged_snapshot_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        temporary_name: &CString,
        temporary_file: &File,
        staged_identity: ArtifactIdentity,
        state: CanonicalArtifactPublishStateV1,
        temporary_name_text: Option<&str>,
    ) -> Result<ExactStagedSnapshotV1, CanonicalArtifactPublishErrorV1> {
        let dot = CString::new(".").map_err(|_| {
            publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::InvalidRelativePath,
                state,
                temporary_name_text,
            )
        })?;
        let how = OpenHow {
            flags: (libc::O_RDWR | libc::O_TMPFILE | libc::O_CLOEXEC) as u64,
            mode: 0o600,
            resolve: RESOLVE_POLICY_V1,
        };
        let (mut snapshot, snapshot_name) =
            match openat2_owned_v1(parent.fd.as_raw_fd(), &dot, &how) {
                Ok(fd) => (File::from(fd), None),
                Err(error) if error.raw_os_error() == Some(libc::EOPNOTSUPP) => {
                    let (name, file) = create_named_staged_snapshot_v1(root, parent, state)?;
                    (file, Some(name))
                }
                Err(error) => {
                    return Err(map_publish_open_error_v1(
                        "openat2 anonymous staged snapshot",
                        error,
                        state,
                        temporary_name_text,
                    ));
                }
            };
        let initial =
            checked_publish_artifact_identity_v1(root, &snapshot, state, temporary_name_text)?;
        let expected_initial_links = u64::from(snapshot_name.is_some());
        if initial.size != 0 || initial.links != expected_initial_links {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name_text,
            ));
        }

        let expected_size = u64::try_from(staged_identity.size).map_err(|_| {
            publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name_text,
            )
        })?;
        let mut offset = 0_u64;
        let mut buffer = [0_u8; EXACT_COMPARE_BUFFER_BYTES_V1];
        while offset < expected_size {
            let remaining =
                usize::try_from((expected_size - offset).min(EXACT_COMPARE_BUFFER_BYTES_V1 as u64))
                    .map_err(|_| {
                        publish_error_v1(
                            CanonicalArtifactPublishErrorKindV1::RaceDetected,
                            state,
                            temporary_name_text,
                        )
                    })?;
            let read = temporary_file
                .read_at(&mut buffer[..remaining], offset)
                .map_err(|error| {
                    publish_io_error_v1(
                        "pread staged snapshot source",
                        error,
                        state,
                        temporary_name_text,
                    )
                })?;
            if read == 0 {
                return Err(publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    state,
                    temporary_name_text,
                ));
            }
            snapshot.write_all(&buffer[..read]).map_err(|error| {
                publish_io_error_v1(
                    "write anonymous staged snapshot",
                    error,
                    state,
                    temporary_name_text,
                )
            })?;
            offset = offset
                .checked_add(u64::try_from(read).map_err(|_| {
                    publish_error_v1(
                        CanonicalArtifactPublishErrorKindV1::RaceDetected,
                        state,
                        temporary_name_text,
                    )
                })?)
                .ok_or_else(|| {
                    publish_error_v1(
                        CanonicalArtifactPublishErrorKindV1::RaceDetected,
                        state,
                        temporary_name_text,
                    )
                })?;
        }
        snapshot.flush().map_err(|error| {
            publish_io_error_v1(
                "flush anonymous staged snapshot",
                error,
                state,
                temporary_name_text,
            )
        })?;
        snapshot.sync_all().map_err(|error| {
            publish_io_error_v1(
                "fsync exact staged snapshot",
                error,
                state,
                temporary_name_text,
            )
        })?;

        let snapshot_identity =
            checked_publish_artifact_identity_v1(root, &snapshot, state, temporary_name_text)?;
        let source_identity =
            temporary_identity_v1(root, parent, temporary_name, temporary_file, state)?;
        if source_identity != staged_identity
            || snapshot_identity.links != expected_initial_links
            || snapshot_identity.size != staged_identity.size
            || !exact_open_files_equal_v1(
                temporary_file,
                &snapshot,
                staged_identity.size,
                state,
                temporary_name_text,
            )?
        {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name_text,
            ));
        }
        if let Some(name) = snapshot_name.as_ref() {
            let expected = parent
                .expected_path
                .join(std::ffi::OsStr::from_bytes(name.to_bytes()));
            verify_named_identity_v1(
                root,
                parent,
                name,
                &expected,
                snapshot_identity,
                state,
                temporary_name_text,
            )?;
        }
        Ok(ExactStagedSnapshotV1 {
            file: snapshot,
            identity: snapshot_identity,
            name: snapshot_name,
        })
    }

    fn create_named_staged_snapshot_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<(CString, File), CanonicalArtifactPublishErrorV1> {
        for _ in 0..32 {
            let name = random_snapshot_name_v1(state)?;
            let how = OpenHow {
                flags: (libc::O_RDWR
                    | libc::O_CREAT
                    | libc::O_EXCL
                    | libc::O_NOFOLLOW
                    | libc::O_CLOEXEC) as u64,
                mode: 0o600,
                resolve: RESOLVE_POLICY_V1,
            };
            match openat2_owned_v1(parent.fd.as_raw_fd(), &name, &how) {
                Ok(fd) => {
                    let file = File::from(fd);
                    let identity = temporary_identity_v1(root, parent, &name, &file, state)?;
                    if identity.size != 0 || identity.links != 1 {
                        return Err(publish_error_v1(
                            CanonicalArtifactPublishErrorKindV1::RaceDetected,
                            state,
                            name.to_str().ok(),
                        ));
                    }
                    return Ok((name, file));
                }
                Err(error) if error.raw_os_error() == Some(libc::EEXIST) => continue,
                Err(error) => {
                    return Err(map_publish_open_error_v1(
                        "openat2 create-new staged snapshot",
                        error,
                        state,
                        name.to_str().ok(),
                    ));
                }
            }
        }
        Err(publish_io_message_v1(
            "create unique staged snapshot",
            "exhausted collision retries",
            state,
            None,
        ))
    }

    fn verify_snapshot_name_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        snapshot: &ExactStagedSnapshotV1,
        state: CanonicalArtifactPublishStateV1,
        temporary_name_text: Option<&str>,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        let Some(name) = snapshot.name.as_ref() else {
            return Ok(());
        };
        let expected = parent
            .expected_path
            .join(std::ffi::OsStr::from_bytes(name.to_bytes()));
        verify_named_identity_v1(
            root,
            parent,
            name,
            &expected,
            snapshot.identity,
            state,
            temporary_name_text,
        )
    }

    fn verify_staged_after_gate_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        temporary_name: &CString,
        temporary_file: &File,
        snapshot: &ExactStagedSnapshotV1,
        state: CanonicalArtifactPublishStateV1,
        temporary_name_text: Option<&str>,
        staged_identity: ArtifactIdentity,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        let temporary_expected = parent
            .expected_path
            .join(std::ffi::OsStr::from_bytes(temporary_name.to_bytes()));
        let before = temporary_identity_v1(root, parent, temporary_name, temporary_file, state)?;
        let snapshot_before =
            checked_publish_artifact_identity_v1(root, &snapshot.file, state, temporary_name_text)?;
        if before != staged_identity || snapshot_before != snapshot.identity {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name_text,
            ));
        }
        verify_snapshot_name_v1(root, parent, snapshot, state, temporary_name_text)?;
        verify_named_identity_v1(
            root,
            parent,
            temporary_name,
            &temporary_expected,
            staged_identity,
            state,
            temporary_name_text,
        )?;
        if !exact_open_files_equal_v1(
            temporary_file,
            &snapshot.file,
            staged_identity.size,
            state,
            temporary_name_text,
        )? {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name_text,
            ));
        }
        let after = temporary_identity_v1(root, parent, temporary_name, temporary_file, state)?;
        let snapshot_after =
            checked_publish_artifact_identity_v1(root, &snapshot.file, state, temporary_name_text)?;
        if after != staged_identity || snapshot_after != snapshot.identity {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name_text,
            ));
        }
        verify_snapshot_name_v1(root, parent, snapshot, state, temporary_name_text)?;
        verify_named_identity_v1(
            root,
            parent,
            temporary_name,
            &temporary_expected,
            staged_identity,
            state,
            temporary_name_text,
        )
    }

    fn finish_new_install_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        relative_path: &str,
        final_name: &CString,
        temporary_name: &CString,
        staged_identity: ArtifactIdentity,
        snapshot: &ExactStagedSnapshotV1,
        bytes_written: u64,
    ) -> Result<CanonicalArtifactPublishReceiptV1, CanonicalArtifactPublishErrorV1> {
        let temporary_name_text = temporary_name.to_string_lossy();
        let sync_pending = CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::InstalledSyncPending,
            CanonicalArtifactTemporaryStateV1::Present,
        );
        let linked_identity = checked_publish_artifact_identity_v1(
            root,
            &snapshot.file,
            sync_pending,
            Some(&temporary_name_text),
        )?;
        let expected_link_count = snapshot.identity.links.checked_add(1).ok_or_else(|| {
            publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                sync_pending,
                Some(&temporary_name_text),
            )
        })?;
        if linked_identity.links != expected_link_count
            || !same_content_object_lineage_v1(snapshot.identity, linked_identity)
        {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                sync_pending,
                Some(&temporary_name_text),
            ));
        }
        let final_expected = root.physical_path.join(relative_path);
        verify_named_identity_v1(
            root,
            parent,
            final_name,
            &final_expected,
            linked_identity,
            sync_pending,
            Some(&temporary_name_text),
        )?;
        sync_directory_v1(
            parent.fd.as_raw_fd(),
            "fsync installed artifact parent",
            sync_pending,
            Some(&temporary_name_text),
        )?;
        let durable = CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::InstalledDurable,
            CanonicalArtifactTemporaryStateV1::Present,
        );
        cleanup_snapshot_name_v1(parent.fd.as_raw_fd(), snapshot, linked_identity, durable)?;
        let cleaned = cleanup_temporary_name_inode_safe_v1(
            parent.fd.as_raw_fd(),
            temporary_name,
            staged_identity,
            durable,
        )?;
        let final_identity =
            artifact_identity(&fstat(snapshot.file.as_raw_fd()).map_err(|error| {
                publish_io_error_v1(
                    "restat installed artifact",
                    error,
                    cleaned,
                    Some(&temporary_name_text),
                )
            })?);
        if final_identity.links != 1
            || !same_content_object_lineage_v1(snapshot.identity, final_identity)
        {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                cleaned,
                Some(&temporary_name_text),
            ));
        }
        verify_named_identity_v1(
            root,
            parent,
            final_name,
            &final_expected,
            final_identity,
            cleaned,
            Some(&temporary_name_text),
        )?;
        verify_publish_root_v1(root, cleaned, Some(&temporary_name_text))?;
        verify_publish_directory_v1(root, parent, cleaned, Some(&temporary_name_text))?;
        Ok(CanonicalArtifactPublishReceiptV1 {
            relative_path: relative_path.to_owned(),
            bytes_written,
            disposition: CanonicalArtifactPublishDispositionV1::Installed,
        })
    }

    fn finish_existing_winner_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        relative_path: &str,
        final_name: &CString,
        temporary_name: &CString,
        staged_identity: ArtifactIdentity,
        snapshot: &ExactStagedSnapshotV1,
        bytes_written: u64,
    ) -> Result<CanonicalArtifactPublishReceiptV1, CanonicalArtifactPublishErrorV1> {
        let temporary_name_text = temporary_name.to_string_lossy();
        let staged_state = CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::NotInstalled,
            CanonicalArtifactTemporaryStateV1::Present,
        );
        let comparison = compare_existing_winner_exact_v1(
            root,
            parent,
            final_name,
            &snapshot.file,
            snapshot.identity,
            staged_state,
            Some(&temporary_name_text),
        );
        let (identical, winner_identity) = match comparison {
            Ok(comparison) => comparison,
            Err(error) => {
                return fail_before_install_and_cleanup_with_snapshot_v1(
                    parent.fd.as_raw_fd(),
                    temporary_name,
                    staged_identity,
                    snapshot,
                    error,
                );
            }
        };
        let final_state = if identical {
            CanonicalArtifactFinalStateV1::ExistingIdentical
        } else {
            CanonicalArtifactFinalStateV1::ExistingDifferent
        };
        let winner_state = CanonicalArtifactPublishStateV1::new(
            final_state,
            CanonicalArtifactTemporaryStateV1::Present,
        );
        cleanup_snapshot_name_v1(
            parent.fd.as_raw_fd(),
            snapshot,
            snapshot.identity,
            winner_state,
        )?;
        let cleaned = cleanup_temporary_name_inode_safe_v1(
            parent.fd.as_raw_fd(),
            temporary_name,
            staged_identity,
            winner_state,
        )?;
        validate_existing_winner_after_cleanup_v1(
            root,
            parent,
            final_name,
            &snapshot.file,
            winner_identity,
            identical,
            cleaned,
            Some(&temporary_name_text),
        )?;
        if !identical {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::ExistingContentMismatch,
                cleaned,
                Some(&temporary_name_text),
            ));
        }
        Ok(CanonicalArtifactPublishReceiptV1 {
            relative_path: relative_path.to_owned(),
            bytes_written,
            disposition: CanonicalArtifactPublishDispositionV1::ExistingIdentical,
        })
    }

    fn validate_existing_winner_after_cleanup_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        final_name: &CString,
        staged_file: &File,
        initial_winner: ArtifactIdentity,
        expected_identical: bool,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        let final_expected = parent
            .expected_path
            .join(std::ffi::OsStr::from_bytes(final_name.to_bytes()));
        for _ in 0..4 {
            let winner = open_regular_for_publish_v1(
                root,
                parent,
                final_name,
                &final_expected,
                state,
                temporary_name,
            )?;
            let before =
                checked_publish_artifact_identity_v1(root, &winner, state, temporary_name)?;
            if !same_content_object_lineage_v1(initial_winner, before) {
                return Err(publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    state,
                    temporary_name,
                ));
            }
            let identical = exact_open_files_equal_v1(
                &winner,
                staged_file,
                before.size,
                state,
                temporary_name,
            )?;
            let after = checked_publish_artifact_identity_v1(root, &winner, state, temporary_name)?;
            let reopened = open_regular_for_publish_v1(
                root,
                parent,
                final_name,
                &final_expected,
                state,
                temporary_name,
            )?;
            let reopened_identity =
                checked_publish_artifact_identity_v1(root, &reopened, state, temporary_name)?;
            if before == after && after == reopened_identity {
                if identical != expected_identical {
                    return Err(publish_error_v1(
                        CanonicalArtifactPublishErrorKindV1::RaceDetected,
                        state,
                        temporary_name,
                    ));
                }
                return Ok(());
            }
            if !same_content_object_lineage_v1(initial_winner, after)
                || !same_content_object_lineage_v1(initial_winner, reopened_identity)
            {
                return Err(publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    state,
                    temporary_name,
                ));
            }
        }
        Err(publish_error_v1(
            CanonicalArtifactPublishErrorKindV1::RaceDetected,
            state,
            temporary_name,
        ))
    }

    fn same_content_object_lineage_v1(
        expected: ArtifactIdentity,
        actual: ArtifactIdentity,
    ) -> bool {
        expected.root == actual.root
            && expected.size == actual.size
            && expected.modified_seconds == actual.modified_seconds
            && expected.modified_nanoseconds == actual.modified_nanoseconds
    }

    fn exact_open_files_equal_v1(
        left: &File,
        right: &File,
        expected_size: i64,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<bool, CanonicalArtifactPublishErrorV1> {
        if expected_size < 0 {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name,
            ));
        }
        let expected_size = expected_size as u64;
        let mut offset = 0_u64;
        let mut left_buffer = [0_u8; EXACT_COMPARE_BUFFER_BYTES_V1];
        let mut right_buffer = [0_u8; EXACT_COMPARE_BUFFER_BYTES_V1];
        while offset < expected_size {
            let remaining =
                usize::try_from((expected_size - offset).min(EXACT_COMPARE_BUFFER_BYTES_V1 as u64))
                    .map_err(|_| {
                        publish_error_v1(
                            CanonicalArtifactPublishErrorKindV1::RaceDetected,
                            state,
                            temporary_name,
                        )
                    })?;
            let left_read = left
                .read_at(&mut left_buffer[..remaining], offset)
                .map_err(|error| {
                    publish_io_error_v1("pread existing winner", error, state, temporary_name)
                })?;
            let right_read = right
                .read_at(&mut right_buffer[..remaining], offset)
                .map_err(|error| {
                    publish_io_error_v1("pread staged artifact", error, state, temporary_name)
                })?;
            if left_read == 0 || right_read == 0 {
                return Err(publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    state,
                    temporary_name,
                ));
            }
            if left_read != right_read || left_buffer[..left_read] != right_buffer[..right_read] {
                return Ok(false);
            }
            let left_read = u64::try_from(left_read).map_err(|_| {
                publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    state,
                    temporary_name,
                )
            })?;
            offset = offset.checked_add(left_read).ok_or_else(|| {
                publish_error_v1(
                    CanonicalArtifactPublishErrorKindV1::RaceDetected,
                    state,
                    temporary_name,
                )
            })?;
        }
        Ok(true)
    }

    fn compare_existing_winner_exact_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        final_name: &CString,
        staged_file: &File,
        staged_identity: ArtifactIdentity,
        state: CanonicalArtifactPublishStateV1,
        temporary_name_text: Option<&str>,
    ) -> Result<(bool, ArtifactIdentity), CanonicalArtifactPublishErrorV1> {
        let final_expected = parent
            .expected_path
            .join(std::ffi::OsStr::from_bytes(final_name.to_bytes()));
        let winner = open_regular_for_publish_v1(
            root,
            parent,
            final_name,
            &final_expected,
            state,
            temporary_name_text,
        )?;
        let winner_before =
            checked_publish_artifact_identity_v1(root, &winner, state, temporary_name_text)?;
        if winner_before.size < 0
            || winner_before.size as u64 > MAX_CANONICAL_ATOMIC_PUBLISH_BYTES_V1
        {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::ArtifactTooLarge {
                    maximum: MAX_CANONICAL_ATOMIC_PUBLISH_BYTES_V1,
                    attempted: winner_before.size.max(0) as u64,
                },
                state,
                temporary_name_text,
            ));
        }
        let staged_before =
            checked_publish_artifact_identity_v1(root, staged_file, state, temporary_name_text)?;
        if staged_before != staged_identity {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name_text,
            ));
        }
        let identical = winner_before.size == staged_before.size
            && exact_open_files_equal_v1(
                &winner,
                staged_file,
                winner_before.size,
                state,
                temporary_name_text,
            )?;
        let winner_after =
            checked_publish_artifact_identity_v1(root, &winner, state, temporary_name_text)?;
        let staged_after =
            checked_publish_artifact_identity_v1(root, staged_file, state, temporary_name_text)?;
        if winner_after != winner_before || staged_after != staged_before {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name_text,
            ));
        }
        verify_named_identity_v1(
            root,
            parent,
            final_name,
            &final_expected,
            winner_before,
            state,
            temporary_name_text,
        )?;
        verify_publish_root_v1(root, state, temporary_name_text)?;
        verify_publish_directory_v1(root, parent, state, temporary_name_text)?;
        Ok((identical, winner_before))
    }

    fn open_regular_for_publish_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        name: &CString,
        expected_path: &Path,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<File, CanonicalArtifactPublishErrorV1> {
        let how = OpenHow {
            flags: (libc::O_RDONLY
                | libc::O_NONBLOCK
                | libc::O_NOCTTY
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC) as u64,
            mode: 0,
            resolve: RESOLVE_POLICY_V1,
        };
        let fd = openat2_owned_v1(parent.fd.as_raw_fd(), name, &how).map_err(|error| {
            map_publish_open_error_v1("openat2 published artifact", error, state, temporary_name)
        })?;
        let file = File::from(fd);
        checked_publish_artifact_identity_v1(root, &file, state, temporary_name)?;
        require_exact_publish_path_v1(
            root,
            file.as_raw_fd(),
            expected_path,
            state,
            temporary_name,
        )?;
        Ok(file)
    }

    fn checked_publish_artifact_identity_v1(
        root: &SealedCanonicalRootV1,
        file: &File,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<ArtifactIdentity, CanonicalArtifactPublishErrorV1> {
        let stat = fstat(file.as_raw_fd()).map_err(|error| {
            publish_io_error_v1("fstat published artifact", error, state, temporary_name)
        })?;
        if stat.st_mode & libc::S_IFMT != libc::S_IFREG {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::NonRegularArtifact,
                state,
                temporary_name,
            ));
        }
        let identity = artifact_identity(&stat);
        if identity.root.device != root.identity.device {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::EscapeOrMount,
                state,
                temporary_name,
            ));
        }
        Ok(identity)
    }

    fn verify_named_identity_v1(
        root: &SealedCanonicalRootV1,
        parent: &VerifiedDirectoryV1,
        name: &CString,
        expected_path: &Path,
        expected_identity: ArtifactIdentity,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        let reopened =
            open_regular_for_publish_v1(root, parent, name, expected_path, state, temporary_name)?;
        let actual = checked_publish_artifact_identity_v1(root, &reopened, state, temporary_name)?;
        if actual != expected_identity {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name,
            ));
        }
        Ok(())
    }

    fn cleanup_temporary_name_inode_safe_v1(
        parent_fd: libc::c_int,
        temporary_name: &CString,
        expected_identity: ArtifactIdentity,
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<CanonicalArtifactPublishStateV1, CanonicalArtifactPublishErrorV1> {
        let temporary_name_text = temporary_name.to_string_lossy();
        let how = OpenHow {
            flags: (libc::O_RDONLY
                | libc::O_NONBLOCK
                | libc::O_NOCTTY
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC) as u64,
            mode: 0,
            resolve: RESOLVE_POLICY_V1,
        };
        let reopened = openat2_owned_v1(parent_fd, temporary_name, &how).map_err(|error| {
            map_publish_open_error_v1(
                "openat2 temporary cleanup",
                error,
                state,
                Some(&temporary_name_text),
            )
        })?;
        let stat = fstat(reopened.as_raw_fd()).map_err(|error| {
            publish_io_error_v1(
                "fstat temporary cleanup",
                error,
                state,
                Some(&temporary_name_text),
            )
        })?;
        let actual_identity = artifact_identity(&stat);
        if stat.st_mode & libc::S_IFMT != libc::S_IFREG
            || actual_identity.root.device != expected_identity.root.device
            || actual_identity.root.inode != expected_identity.root.inode
        {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                Some(&temporary_name_text),
            ));
        }
        if unsafe { libc::unlinkat(parent_fd, temporary_name.as_ptr(), 0) } != 0 {
            return Err(publish_io_error_v1(
                "unlinkat temporary artifact",
                std::io::Error::last_os_error(),
                state,
                Some(&temporary_name_text),
            ));
        }
        let removal_pending = CanonicalArtifactPublishStateV1::new(
            state.final_state(),
            CanonicalArtifactTemporaryStateV1::RemovedSyncPending,
        );
        sync_directory_v1(
            parent_fd,
            "fsync temporary removal",
            removal_pending,
            Some(&temporary_name_text),
        )?;
        Ok(CanonicalArtifactPublishStateV1::new(
            state.final_state(),
            CanonicalArtifactTemporaryStateV1::RemovedDurable,
        ))
    }

    fn cleanup_snapshot_name_v1(
        parent_fd: libc::c_int,
        snapshot: &ExactStagedSnapshotV1,
        expected_identity: ArtifactIdentity,
        state: CanonicalArtifactPublishStateV1,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        let Some(name) = snapshot.name.as_ref() else {
            return Ok(());
        };
        cleanup_temporary_name_inode_safe_v1(parent_fd, name, expected_identity, state).map(|_| ())
    }

    fn fail_before_install_and_cleanup_with_snapshot_v1<T>(
        parent_fd: libc::c_int,
        temporary_name: &CString,
        expected_identity: ArtifactIdentity,
        snapshot: &ExactStagedSnapshotV1,
        original: CanonicalArtifactPublishErrorV1,
    ) -> Result<T, CanonicalArtifactPublishErrorV1> {
        let snapshot_error =
            cleanup_snapshot_name_v1(parent_fd, snapshot, snapshot.identity, original.state).err();
        let temporary_cleanup = cleanup_temporary_name_inode_safe_v1(
            parent_fd,
            temporary_name,
            expected_identity,
            original.state,
        );
        if let Some(error) = snapshot_error {
            return Err(error);
        }
        let cleaned = temporary_cleanup?;
        Err(CanonicalArtifactPublishErrorV1 {
            state: cleaned,
            ..original
        })
    }

    fn fail_before_install_and_cleanup_v1<T>(
        parent_fd: libc::c_int,
        temporary_name: &CString,
        expected_identity: ArtifactIdentity,
        original: CanonicalArtifactPublishErrorV1,
    ) -> Result<T, CanonicalArtifactPublishErrorV1> {
        cleanup_temporary_name_inode_safe_v1(
            parent_fd,
            temporary_name,
            expected_identity,
            original.state,
        )?;
        Err(CanonicalArtifactPublishErrorV1 {
            state: CanonicalArtifactPublishStateV1::new(
                original.state.final_state(),
                CanonicalArtifactTemporaryStateV1::RemovedDurable,
            ),
            ..original
        })
    }

    fn openat2_owned_v1(
        directory_fd: libc::c_int,
        name: &CString,
        how: &OpenHow,
    ) -> std::io::Result<OwnedFd> {
        let raw_fd = unsafe {
            libc::syscall(
                libc::SYS_openat2,
                directory_fd,
                name.as_ptr(),
                how,
                size_of::<OpenHow>(),
            ) as libc::c_int
        };
        if raw_fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(raw_fd) })
    }

    fn link_snapshot_no_replace_v1(
        snapshot: &File,
        parent_fd: libc::c_int,
        final_name: &CString,
    ) -> std::io::Result<()> {
        let source = CString::new(format!("/proc/self/fd/{}", snapshot.as_raw_fd()))
            .map_err(|_| std::io::Error::other("snapshot fd path contains NUL"))?;
        if unsafe {
            libc::linkat(
                libc::AT_FDCWD,
                source.as_ptr(),
                parent_fd,
                final_name.as_ptr(),
                libc::AT_SYMLINK_FOLLOW,
            )
        } != 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    fn sync_directory_v1(
        directory_fd: libc::c_int,
        operation: &'static str,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        if unsafe { libc::fsync(directory_fd) } != 0 {
            return Err(publish_io_error_v1(
                operation,
                std::io::Error::last_os_error(),
                state,
                temporary_name,
            ));
        }
        Ok(())
    }

    fn verify_publish_root_v1(
        root: &SealedCanonicalRootV1,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        let stat = fstat(root.fd.as_raw_fd()).map_err(|error| {
            publish_io_error_v1("fstat canonical root", error, state, temporary_name)
        })?;
        let path = handle_path(root.fd.as_raw_fd()).map_err(|error| {
            publish_io_error_v1(
                "resolve canonical root handle",
                error,
                state,
                temporary_name,
            )
        })?;
        if root_identity(&stat) != root.identity || path != root.physical_path {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name,
            ));
        }
        Ok(())
    }

    fn verify_publish_directory_v1(
        root: &SealedCanonicalRootV1,
        directory: &VerifiedDirectoryV1,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        verify_publish_root_v1(root, state, temporary_name)?;
        let stat = fstat(directory.fd.as_raw_fd()).map_err(|error| {
            publish_io_error_v1("fstat output directory", error, state, temporary_name)
        })?;
        if stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::NonRegularArtifact,
                state,
                temporary_name,
            ));
        }
        let identity = root_identity(&stat);
        if identity.device != root.identity.device {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::EscapeOrMount,
                state,
                temporary_name,
            ));
        }
        if identity != directory.identity {
            return Err(publish_error_v1(
                CanonicalArtifactPublishErrorKindV1::RaceDetected,
                state,
                temporary_name,
            ));
        }
        require_exact_publish_path_v1(
            root,
            directory.fd.as_raw_fd(),
            &directory.expected_path,
            state,
            temporary_name,
        )
    }

    fn require_exact_publish_path_v1(
        root: &SealedCanonicalRootV1,
        fd: libc::c_int,
        expected: &Path,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> Result<(), CanonicalArtifactPublishErrorV1> {
        let actual = handle_path(fd).map_err(|error| {
            publish_io_error_v1("resolve artifact handle", error, state, temporary_name)
        })?;
        if actual == expected {
            return Ok(());
        }
        let kind = if actual.starts_with(&root.physical_path) {
            CanonicalArtifactPublishErrorKindV1::RaceDetected
        } else {
            CanonicalArtifactPublishErrorKindV1::EscapeOrMount
        };
        Err(publish_error_v1(kind, state, temporary_name))
    }

    fn map_publish_open_error_v1(
        operation: &'static str,
        error: std::io::Error,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> CanonicalArtifactPublishErrorV1 {
        let kind = match error.raw_os_error() {
            Some(libc::ENOSYS | libc::EINVAL | libc::E2BIG) => {
                CanonicalArtifactPublishErrorKindV1::SecureResolutionUnavailable(error.to_string())
            }
            Some(libc::ELOOP | libc::ENOTDIR) => CanonicalArtifactPublishErrorKindV1::UnsafeLink,
            Some(libc::EXDEV) => CanonicalArtifactPublishErrorKindV1::EscapeOrMount,
            Some(libc::EAGAIN) => CanonicalArtifactPublishErrorKindV1::RaceDetected,
            _ => {
                return publish_io_error_v1(operation, error, state, temporary_name);
            }
        };
        publish_error_v1(kind, state, temporary_name)
    }

    fn publish_io_error_v1(
        operation: &'static str,
        error: std::io::Error,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> CanonicalArtifactPublishErrorV1 {
        publish_io_message_v1(operation, &error.to_string(), state, temporary_name)
    }

    fn publish_io_message_v1(
        operation: &'static str,
        message: &str,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> CanonicalArtifactPublishErrorV1 {
        publish_error_v1(
            CanonicalArtifactPublishErrorKindV1::Io {
                operation,
                message: message.to_owned(),
            },
            state,
            temporary_name,
        )
    }

    fn publish_error_v1(
        kind: CanonicalArtifactPublishErrorKindV1,
        state: CanonicalArtifactPublishStateV1,
        temporary_name: Option<&str>,
    ) -> CanonicalArtifactPublishErrorV1 {
        CanonicalArtifactPublishErrorV1 {
            kind,
            state,
            temporary_name: temporary_name.map(str::to_owned),
        }
    }

    fn require_exact_path(
        root: &SealedCanonicalRootV1,
        fd: libc::c_int,
        expected: &Path,
    ) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
        let actual = handle_path(fd).map_err(|_| race())?;
        if actual == expected {
            return Ok(());
        }
        if !actual.starts_with(&root.physical_path) {
            return Err(CanonicalNativeDiscoveryRequestErrorV1::EscapeOrMount);
        }
        Err(race())
    }

    fn handle_path(fd: libc::c_int) -> std::io::Result<PathBuf> {
        fs::read_link(format!("/proc/self/fd/{fd}"))
    }

    fn fstat(fd: libc::c_int) -> std::io::Result<libc::stat> {
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(unsafe { stat.assume_init() })
    }

    fn root_identity(stat: &libc::stat) -> RootIdentity {
        RootIdentity {
            device: stat.st_dev,
            inode: stat.st_ino,
            mode: stat.st_mode,
        }
    }

    fn artifact_identity(stat: &libc::stat) -> ArtifactIdentity {
        ArtifactIdentity {
            root: root_identity(stat),
            size: stat.st_size,
            links: stat.st_nlink,
            modified_seconds: stat.st_mtime,
            modified_nanoseconds: stat.st_mtime_nsec,
            changed_seconds: stat.st_ctime,
            changed_nanoseconds: stat.st_ctime_nsec,
        }
    }

    fn map_open_error(error: std::io::Error) -> CanonicalNativeDiscoveryRequestErrorV1 {
        match error.raw_os_error() {
            Some(libc::ENOSYS | libc::EINVAL | libc::E2BIG) => {
                CanonicalNativeDiscoveryRequestErrorV1::SecureResolutionUnavailable(
                    error.to_string(),
                )
            }
            Some(libc::ELOOP) => CanonicalNativeDiscoveryRequestErrorV1::UnsafeLink,
            Some(libc::EXDEV) => CanonicalNativeDiscoveryRequestErrorV1::EscapeOrMount,
            Some(libc::EAGAIN) => race(),
            _ => artifact_io("openat2", error),
        }
    }

    fn root_error(message: &str) -> CanonicalNativeDiscoveryRequestErrorV1 {
        CanonicalNativeDiscoveryRequestErrorV1::CanonicalRootUnavailable(message.to_owned())
    }

    fn artifact_io(
        operation: &str,
        error: std::io::Error,
    ) -> CanonicalNativeDiscoveryRequestErrorV1 {
        CanonicalNativeDiscoveryRequestErrorV1::ArtifactIo(format!("{operation}: {error}"))
    }

    fn race() -> CanonicalNativeDiscoveryRequestErrorV1 {
        CanonicalNativeDiscoveryRequestErrorV1::RaceDetected
    }

    #[cfg(test)]
    mod publisher_tests;
}

#[cfg(target_os = "linux")]
pub use linux::SealedCanonicalRootV1;

#[cfg(target_os = "linux")]
pub(crate) use linux::read_canonical_artifact_exact_v1;

#[cfg(target_os = "linux")]
#[allow(
    unused_imports,
    reason = "the separately reviewed Chunk 2B adapter consumes this crate-private boundary"
)]
pub(crate) use linux::{
    BoundedCanonicalArtifactWriterV1, CanonicalArtifactFinalStateV1,
    CanonicalArtifactPublishDispositionV1, CanonicalArtifactPublishErrorKindV1,
    CanonicalArtifactPublishErrorV1, CanonicalArtifactPublishGateRejectionV1,
    CanonicalArtifactPublishReceiptV1, CanonicalArtifactPublishStateV1,
    CanonicalArtifactTemporaryStateV1, publish_canonical_artifact_create_new_v1,
};

#[cfg(all(test, target_os = "linux"))]
pub(crate) use linux::read_canonical_artifact_exact_with_test_hook_v1;

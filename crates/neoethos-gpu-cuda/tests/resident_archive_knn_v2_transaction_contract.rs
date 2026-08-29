mod resident_archive_knn_v2_transaction_fixture {
    use std::mem::size_of;

    pub const MAX_GENERATION: u64 = 65_535;
    pub const MAX_ARCHIVE_COUNT: u64 = 65_535;
    pub const MAX_EPOCH: u64 = (1_u64 << 31) - 1;
    pub const RANK_IDENTITY: u64 = 0x5345_4d41_4e54_4943;
    pub const RANK_RECEIPT_BASE: u64 = 0x5241_4e4b_0000_0000;
    pub const STAGED_RECEIPT_BASE: u64 = 0x5354_4147_4500_0000;
    pub const TERMINAL_RECEIPT_BASE: u64 = 0x5445_524d_0000_0000;
    pub const TERMINAL_EVENT_ID: u64 = 0x4556_454e_5400_0001;
    pub const COMPACT_TERMINAL_RECEIPT_BYTES: u32 = size_of::<[u64; 4]>() as u32;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CommitFields {
        pub store: u64,
        pub generation: u64,
        pub archive_count: u64,
        pub epoch: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PackError {
        StoreOutOfBounds { value: u64 },
        GenerationOutOfBounds { value: u64 },
        ArchiveCountOutOfBounds { value: u64 },
        EpochOutOfBounds { value: u64 },
    }

    pub fn pack_commit_word(
        store: u64,
        generation: u64,
        archive_count: u64,
        epoch: u64,
    ) -> Result<u64, PackError> {
        if store > 1 {
            return Err(PackError::StoreOutOfBounds { value: store });
        }
        if generation > MAX_GENERATION {
            return Err(PackError::GenerationOutOfBounds { value: generation });
        }
        if archive_count > MAX_ARCHIVE_COUNT {
            return Err(PackError::ArchiveCountOutOfBounds {
                value: archive_count,
            });
        }
        if epoch > MAX_EPOCH {
            return Err(PackError::EpochOutOfBounds { value: epoch });
        }

        Ok(store | (generation << 1) | (archive_count << 17) | (epoch << 33))
    }

    pub const fn decode_commit_word(word: u64) -> CommitFields {
        CommitFields {
            store: word & 1,
            generation: (word >> 1) & 0xffff,
            archive_count: (word >> 17) & 0xffff,
            epoch: (word >> 33) & 0x7fff_ffff,
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RuntimeState {
        GenerationChain,
        RankEnqueued,
        ArchiveStaged,
        TerminalPending,
        Completed,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LedgerPhase {
        ScoreRank,
        StageArchive,
        EvolveDedup,
        AtomicPublish,
        CompactD2h,
        EventRecord,
        EventQuery,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct LedgerEntry {
        pub sequence: u64,
        pub stream: u64,
        pub phase: LedgerPhase,
        pub generation: u64,
        pub receipt_identity: u64,
        pub before_word: u64,
        pub after_word: u64,
        pub d2h_bytes: u32,
        pub event_identity: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ReceiptAxis {
        BoxedReceiptIdentity,
        RunToken,
        Generation,
        SourcePackedWord,
        StoreEpoch,
        ArchiveCountAtStart,
        RankIdentity,
        RankReceiptIdentity,
        SameStreamOrdinal,
        StagedReceiptIdentity,
        StagedDependencyIdentity,
        StagedCount,
        TargetArchiveCount,
        TargetPackedWord,
        TerminalReceiptIdentity,
        TerminalEventIdentity,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TransitionError {
        WrongState {
            expected: RuntimeState,
            actual: RuntimeState,
        },
        AlreadyConsumed {
            receipt_identity: u64,
        },
        ReceiptMismatch {
            axis: ReceiptAxis,
        },
        Pack(PackError),
    }

    impl From<PackError> for TransitionError {
        fn from(error: PackError) -> Self {
            Self::Pack(error)
        }
    }

    #[derive(Debug)]
    struct BoxedRunReceipt {
        run_token: u64,
    }

    #[derive(Debug)]
    struct RunAuthority {
        receipt: Box<BoxedRunReceipt>,
    }

    impl RunAuthority {
        fn new(run_token: u64) -> Self {
            Self {
                receipt: Box::new(BoxedRunReceipt { run_token }),
            }
        }

        fn boxed_receipt_identity(&self) -> usize {
            std::ptr::from_ref(self.receipt.as_ref()) as usize
        }

        fn run_token(&self) -> u64 {
            self.receipt.run_token
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RankReceipt {
        pub receipt_identity: u64,
        pub boxed_receipt_identity: usize,
        pub run_token: u64,
        pub generation: u64,
        pub source_packed_word: u64,
        pub store_epoch: u64,
        pub archive_count_at_start: u64,
        pub rank_identity: u64,
        pub same_stream_ordinal: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct StagedReceipt {
        pub receipt_identity: u64,
        pub boxed_receipt_identity: usize,
        pub run_token: u64,
        pub generation: u64,
        pub source_packed_word: u64,
        pub store_epoch: u64,
        pub archive_count_at_start: u64,
        pub rank_identity: u64,
        pub rank_same_stream_ordinal: u64,
        pub stage_same_stream_ordinal: u64,
        pub staged_dependency_identity: u64,
        pub staged_count: u64,
        pub target_archive_count: u64,
        pub target_packed_word: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TerminalQueryAuthority {
        pub receipt_identity: u64,
        pub boxed_receipt_identity: usize,
        pub run_token: u64,
        pub packed_word: u64,
        pub generation: u64,
        pub event_identity: u64,
    }

    #[derive(Debug)]
    pub struct GenerationChain {
        authority: RunAuthority,
        source_packed_word: u64,
        planned_generation: u64,
        prior_staged_receipt_identity: Option<u64>,
    }

    impl GenerationChain {
        pub fn boxed_receipt_identity(&self) -> usize {
            self.authority.boxed_receipt_identity()
        }

        pub fn source_packed_word(&self) -> u64 {
            self.source_packed_word
        }

        pub fn planned_generation(&self) -> u64 {
            self.planned_generation
        }

        pub fn prior_staged_receipt_identity(&self) -> Option<u64> {
            self.prior_staged_receipt_identity
        }
    }

    #[derive(Debug)]
    pub struct RankEnqueued {
        authority: RunAuthority,
        receipt: RankReceipt,
    }

    impl RankEnqueued {
        pub fn boxed_receipt_identity(&self) -> usize {
            self.authority.boxed_receipt_identity()
        }

        pub fn receipt(&self) -> RankReceipt {
            self.receipt
        }
    }

    #[derive(Debug)]
    pub struct ArchiveStaged {
        authority: RunAuthority,
        receipt: StagedReceipt,
    }

    impl ArchiveStaged {
        pub fn boxed_receipt_identity(&self) -> usize {
            self.authority.boxed_receipt_identity()
        }

        pub fn receipt(&self) -> StagedReceipt {
            self.receipt
        }
    }

    pub struct TerminalPending {
        authority: RunAuthority,
        query_authority: TerminalQueryAuthority,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TerminalProjection {
        pub receipt_identity: u64,
        pub run_token: u64,
        pub packed_word: u64,
        pub store: u64,
        pub generation: u64,
        pub archive_count: u64,
        pub epoch: u64,
        pub d2h_bytes: u32,
        pub event_identity: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Observation {
        pub commit_word: u64,
        pub ledger_len: usize,
        pub enqueue_count: u64,
        pub state: RuntimeState,
        pub score_rank_count: u64,
        pub stage_count: u64,
        pub evolve_count: u64,
        pub atomic_publish_count: u64,
        pub d2h_count: u64,
        pub sync_count: u64,
        pub event_record_count: u64,
        pub event_query_count: u64,
    }

    #[derive(Debug)]
    pub struct ReferenceStream {
        stream_identity: u64,
        run_token: u64,
        boxed_receipt_identity: usize,
        commit_word: u64,
        state: RuntimeState,
        expected_rank: Option<RankReceipt>,
        expected_stage: Option<StagedReceipt>,
        expected_terminal: Option<TerminalQueryAuthority>,
        ledger: Vec<LedgerEntry>,
        enqueue_count: u64,
        score_rank_count: u64,
        stage_count: u64,
        evolve_count: u64,
        atomic_publish_count: u64,
        d2h_count: u64,
        sync_count: u64,
        event_record_count: u64,
        event_query_count: u64,
    }

    impl ReferenceStream {
        pub fn admit(
            stream_identity: u64,
            run_token: u64,
            store: u64,
            generation: u64,
            archive_count: u64,
            epoch: u64,
        ) -> Result<(Self, GenerationChain), PackError> {
            let commit_word = pack_commit_word(store, generation, archive_count, epoch)?;
            let authority = RunAuthority::new(run_token);
            let boxed_receipt_identity = authority.boxed_receipt_identity();
            let stream = Self {
                stream_identity,
                run_token,
                boxed_receipt_identity,
                commit_word,
                state: RuntimeState::GenerationChain,
                expected_rank: None,
                expected_stage: None,
                expected_terminal: None,
                ledger: Vec::new(),
                enqueue_count: 0,
                score_rank_count: 0,
                stage_count: 0,
                evolve_count: 0,
                atomic_publish_count: 0,
                d2h_count: 0,
                sync_count: 0,
                event_record_count: 0,
                event_query_count: 0,
            };
            let chain = GenerationChain {
                authority,
                source_packed_word: commit_word,
                planned_generation: generation,
                prior_staged_receipt_identity: None,
            };
            Ok((stream, chain))
        }

        pub fn commit_word(&self) -> u64 {
            self.commit_word
        }

        pub fn ledger(&self) -> &[LedgerEntry] {
            &self.ledger
        }

        pub fn observation(&self) -> Observation {
            Observation {
                commit_word: self.commit_word,
                ledger_len: self.ledger.len(),
                enqueue_count: self.enqueue_count,
                state: self.state,
                score_rank_count: self.score_rank_count,
                stage_count: self.stage_count,
                evolve_count: self.evolve_count,
                atomic_publish_count: self.atomic_publish_count,
                d2h_count: self.d2h_count,
                sync_count: self.sync_count,
                event_record_count: self.event_record_count,
                event_query_count: self.event_query_count,
            }
        }

        pub fn enqueue_score_and_rank(
            &mut self,
            chain: GenerationChain,
        ) -> Result<RankEnqueued, TransitionError> {
            self.validate_chain(&chain)?;
            let fields = decode_commit_word(self.commit_word);
            let receipt_identity = RANK_RECEIPT_BASE | fields.generation;
            let same_stream_ordinal = self.enqueue_count + 1;
            let receipt = RankReceipt {
                receipt_identity,
                boxed_receipt_identity: chain.authority.boxed_receipt_identity(),
                run_token: chain.authority.run_token(),
                generation: fields.generation,
                source_packed_word: self.commit_word,
                store_epoch: fields.epoch,
                archive_count_at_start: fields.archive_count,
                rank_identity: RANK_IDENTITY,
                same_stream_ordinal,
            };
            self.append_enqueue(
                LedgerPhase::ScoreRank,
                fields.generation,
                receipt_identity,
                self.commit_word,
                self.commit_word,
                0,
                0,
            );
            self.expected_rank = Some(receipt);
            self.state = RuntimeState::RankEnqueued;
            Ok(RankEnqueued {
                authority: chain.authority,
                receipt,
            })
        }

        pub fn enqueue_stage_archive(
            &mut self,
            rank: RankEnqueued,
            staged_count: u64,
        ) -> Result<ArchiveStaged, TransitionError> {
            self.validate_rank_receipt_for_test(rank.receipt)?;
            self.validate_authority(
                &rank.authority,
                rank.receipt.boxed_receipt_identity,
                rank.receipt.run_token,
            )?;

            let source = decode_commit_word(self.commit_word);
            let target_archive_count = source.archive_count.checked_add(staged_count).ok_or(
                PackError::ArchiveCountOutOfBounds {
                    value: staged_count,
                },
            )?;
            let target_packed_word = pack_commit_word(
                source.store ^ 1,
                source.generation + 1,
                target_archive_count,
                source.epoch + 1,
            )?;
            let stage_same_stream_ordinal = self.enqueue_count + 1;
            let receipt = StagedReceipt {
                receipt_identity: STAGED_RECEIPT_BASE | source.generation,
                boxed_receipt_identity: rank.receipt.boxed_receipt_identity,
                run_token: rank.receipt.run_token,
                generation: source.generation,
                source_packed_word: self.commit_word,
                store_epoch: source.epoch,
                archive_count_at_start: source.archive_count,
                rank_identity: rank.receipt.rank_identity,
                rank_same_stream_ordinal: rank.receipt.same_stream_ordinal,
                stage_same_stream_ordinal,
                staged_dependency_identity: rank.receipt.receipt_identity,
                staged_count,
                target_archive_count,
                target_packed_word,
            };
            self.append_enqueue(
                LedgerPhase::StageArchive,
                source.generation,
                rank.receipt.receipt_identity,
                self.commit_word,
                self.commit_word,
                0,
                0,
            );
            self.expected_rank = None;
            self.expected_stage = Some(receipt);
            self.state = RuntimeState::ArchiveStaged;
            Ok(ArchiveStaged {
                authority: rank.authority,
                receipt,
            })
        }

        pub fn enqueue_evolve_and_publish(
            &mut self,
            staged: ArchiveStaged,
        ) -> Result<GenerationChain, TransitionError> {
            self.validate_staged_receipt_for_test(staged.receipt)?;
            self.validate_authority(
                &staged.authority,
                staged.receipt.boxed_receipt_identity,
                staged.receipt.run_token,
            )?;

            let before_word = self.commit_word;
            self.append_enqueue(
                LedgerPhase::EvolveDedup,
                staged.receipt.generation,
                staged.receipt.receipt_identity,
                before_word,
                before_word,
                0,
                0,
            );
            self.append_enqueue(
                LedgerPhase::AtomicPublish,
                staged.receipt.generation,
                staged.receipt.receipt_identity,
                before_word,
                staged.receipt.target_packed_word,
                0,
                0,
            );
            self.commit_word = staged.receipt.target_packed_word;
            self.expected_stage = None;
            self.state = RuntimeState::GenerationChain;
            let next = decode_commit_word(self.commit_word);
            Ok(GenerationChain {
                authority: staged.authority,
                source_packed_word: self.commit_word,
                planned_generation: next.generation,
                prior_staged_receipt_identity: Some(staged.receipt.receipt_identity),
            })
        }

        pub fn enqueue_terminal_seal(
            &mut self,
            chain: GenerationChain,
        ) -> Result<TerminalPending, TransitionError> {
            self.validate_chain(&chain)?;
            let fields = decode_commit_word(self.commit_word);
            let query_authority = TerminalQueryAuthority {
                receipt_identity: TERMINAL_RECEIPT_BASE | fields.generation,
                boxed_receipt_identity: chain.authority.boxed_receipt_identity(),
                run_token: chain.authority.run_token(),
                packed_word: self.commit_word,
                generation: fields.generation,
                event_identity: TERMINAL_EVENT_ID,
            };
            self.append_enqueue(
                LedgerPhase::CompactD2h,
                fields.generation,
                query_authority.receipt_identity,
                self.commit_word,
                self.commit_word,
                COMPACT_TERMINAL_RECEIPT_BYTES,
                0,
            );
            self.append_enqueue(
                LedgerPhase::EventRecord,
                fields.generation,
                query_authority.receipt_identity,
                self.commit_word,
                self.commit_word,
                0,
                TERMINAL_EVENT_ID,
            );
            self.expected_terminal = Some(query_authority);
            self.state = RuntimeState::TerminalPending;
            Ok(TerminalPending {
                authority: chain.authority,
                query_authority,
            })
        }

        pub fn try_complete(
            &mut self,
            pending: TerminalPending,
        ) -> Result<TerminalProjection, TransitionError> {
            self.validate_terminal_query_for_test(pending.query_authority)?;
            self.validate_authority(
                &pending.authority,
                pending.query_authority.boxed_receipt_identity,
                pending.query_authority.run_token,
            )?;
            let fields = decode_commit_word(self.commit_word);
            self.append_query(
                LedgerPhase::EventQuery,
                fields.generation,
                pending.query_authority.receipt_identity,
                self.commit_word,
                TERMINAL_EVENT_ID,
            );
            self.expected_terminal = None;
            self.state = RuntimeState::Completed;
            Ok(TerminalProjection {
                receipt_identity: pending.query_authority.receipt_identity,
                run_token: pending.query_authority.run_token,
                packed_word: self.commit_word,
                store: fields.store,
                generation: fields.generation,
                archive_count: fields.archive_count,
                epoch: fields.epoch,
                d2h_bytes: COMPACT_TERMINAL_RECEIPT_BYTES,
                event_identity: TERMINAL_EVENT_ID,
            })
        }

        pub fn validate_rank_receipt_for_test(
            &self,
            candidate: RankReceipt,
        ) -> Result<(), TransitionError> {
            if self.ledger.iter().any(|entry| {
                entry.phase == LedgerPhase::StageArchive
                    && entry.receipt_identity == candidate.receipt_identity
            }) {
                return Err(TransitionError::AlreadyConsumed {
                    receipt_identity: candidate.receipt_identity,
                });
            }
            if self.state != RuntimeState::RankEnqueued {
                return Err(TransitionError::WrongState {
                    expected: RuntimeState::RankEnqueued,
                    actual: self.state,
                });
            }
            let expected = self.expected_rank.ok_or(TransitionError::WrongState {
                expected: RuntimeState::RankEnqueued,
                actual: self.state,
            })?;
            compare_rank_receipt(expected, candidate)
        }

        pub fn validate_staged_receipt_for_test(
            &self,
            candidate: StagedReceipt,
        ) -> Result<(), TransitionError> {
            if self.ledger.iter().any(|entry| {
                entry.phase == LedgerPhase::EvolveDedup
                    && entry.receipt_identity == candidate.receipt_identity
            }) {
                return Err(TransitionError::AlreadyConsumed {
                    receipt_identity: candidate.receipt_identity,
                });
            }
            if self.state != RuntimeState::ArchiveStaged {
                return Err(TransitionError::WrongState {
                    expected: RuntimeState::ArchiveStaged,
                    actual: self.state,
                });
            }
            let expected = self.expected_stage.ok_or(TransitionError::WrongState {
                expected: RuntimeState::ArchiveStaged,
                actual: self.state,
            })?;
            compare_staged_receipt(expected, candidate)
        }

        pub fn validate_terminal_query_for_test(
            &self,
            candidate: TerminalQueryAuthority,
        ) -> Result<(), TransitionError> {
            if self.ledger.iter().any(|entry| {
                entry.phase == LedgerPhase::EventQuery
                    && entry.receipt_identity == candidate.receipt_identity
            }) {
                return Err(TransitionError::AlreadyConsumed {
                    receipt_identity: candidate.receipt_identity,
                });
            }
            if self.state != RuntimeState::TerminalPending {
                return Err(TransitionError::WrongState {
                    expected: RuntimeState::TerminalPending,
                    actual: self.state,
                });
            }
            let expected = self.expected_terminal.ok_or(TransitionError::WrongState {
                expected: RuntimeState::TerminalPending,
                actual: self.state,
            })?;
            compare_terminal_query(expected, candidate)
        }

        fn validate_chain(&self, chain: &GenerationChain) -> Result<(), TransitionError> {
            if self.state != RuntimeState::GenerationChain {
                return Err(TransitionError::WrongState {
                    expected: RuntimeState::GenerationChain,
                    actual: self.state,
                });
            }
            if chain.authority.boxed_receipt_identity() != self.boxed_receipt_identity {
                return Err(TransitionError::ReceiptMismatch {
                    axis: ReceiptAxis::BoxedReceiptIdentity,
                });
            }
            if chain.authority.run_token() != self.run_token {
                return Err(TransitionError::ReceiptMismatch {
                    axis: ReceiptAxis::RunToken,
                });
            }
            if chain.source_packed_word != self.commit_word {
                return Err(TransitionError::ReceiptMismatch {
                    axis: ReceiptAxis::SourcePackedWord,
                });
            }
            if chain.planned_generation != decode_commit_word(self.commit_word).generation {
                return Err(TransitionError::ReceiptMismatch {
                    axis: ReceiptAxis::Generation,
                });
            }
            Ok(())
        }

        fn validate_authority(
            &self,
            authority: &RunAuthority,
            boxed_receipt_identity: usize,
            run_token: u64,
        ) -> Result<(), TransitionError> {
            if authority.boxed_receipt_identity() != boxed_receipt_identity
                || boxed_receipt_identity != self.boxed_receipt_identity
            {
                return Err(TransitionError::ReceiptMismatch {
                    axis: ReceiptAxis::BoxedReceiptIdentity,
                });
            }
            if authority.run_token() != run_token || run_token != self.run_token {
                return Err(TransitionError::ReceiptMismatch {
                    axis: ReceiptAxis::RunToken,
                });
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn append_enqueue(
            &mut self,
            phase: LedgerPhase,
            generation: u64,
            receipt_identity: u64,
            before_word: u64,
            after_word: u64,
            d2h_bytes: u32,
            event_identity: u64,
        ) {
            self.enqueue_count += 1;
            match phase {
                LedgerPhase::ScoreRank => self.score_rank_count += 1,
                LedgerPhase::StageArchive => self.stage_count += 1,
                LedgerPhase::EvolveDedup => self.evolve_count += 1,
                LedgerPhase::AtomicPublish => self.atomic_publish_count += 1,
                LedgerPhase::CompactD2h => self.d2h_count += 1,
                LedgerPhase::EventRecord => self.event_record_count += 1,
                LedgerPhase::EventQuery => unreachable!("queries are not stream enqueues"),
            }
            self.ledger.push(LedgerEntry {
                sequence: self.ledger.len() as u64 + 1,
                stream: self.stream_identity,
                phase,
                generation,
                receipt_identity,
                before_word,
                after_word,
                d2h_bytes,
                event_identity,
            });
        }

        fn append_query(
            &mut self,
            phase: LedgerPhase,
            generation: u64,
            receipt_identity: u64,
            packed_word: u64,
            event_identity: u64,
        ) {
            debug_assert_eq!(phase, LedgerPhase::EventQuery);
            self.event_query_count += 1;
            self.ledger.push(LedgerEntry {
                sequence: self.ledger.len() as u64 + 1,
                stream: self.stream_identity,
                phase,
                generation,
                receipt_identity,
                before_word: packed_word,
                after_word: packed_word,
                d2h_bytes: 0,
                event_identity,
            });
        }
    }

    fn compare_rank_receipt(
        expected: RankReceipt,
        candidate: RankReceipt,
    ) -> Result<(), TransitionError> {
        if candidate.boxed_receipt_identity != expected.boxed_receipt_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::BoxedReceiptIdentity,
            });
        }
        if candidate.run_token != expected.run_token {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::RunToken,
            });
        }
        if candidate.generation != expected.generation {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::Generation,
            });
        }
        if candidate.source_packed_word != expected.source_packed_word {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::SourcePackedWord,
            });
        }
        if candidate.store_epoch != expected.store_epoch {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::StoreEpoch,
            });
        }
        if candidate.archive_count_at_start != expected.archive_count_at_start {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::ArchiveCountAtStart,
            });
        }
        if candidate.rank_identity != expected.rank_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::RankIdentity,
            });
        }
        if candidate.same_stream_ordinal != expected.same_stream_ordinal {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::SameStreamOrdinal,
            });
        }
        if candidate.receipt_identity != expected.receipt_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::RankReceiptIdentity,
            });
        }
        Ok(())
    }

    fn compare_staged_receipt(
        expected: StagedReceipt,
        candidate: StagedReceipt,
    ) -> Result<(), TransitionError> {
        if candidate.boxed_receipt_identity != expected.boxed_receipt_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::BoxedReceiptIdentity,
            });
        }
        if candidate.run_token != expected.run_token {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::RunToken,
            });
        }
        if candidate.generation != expected.generation {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::Generation,
            });
        }
        if candidate.source_packed_word != expected.source_packed_word {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::SourcePackedWord,
            });
        }
        if candidate.store_epoch != expected.store_epoch {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::StoreEpoch,
            });
        }
        if candidate.archive_count_at_start != expected.archive_count_at_start {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::ArchiveCountAtStart,
            });
        }
        if candidate.rank_identity != expected.rank_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::RankIdentity,
            });
        }
        if candidate.rank_same_stream_ordinal != expected.rank_same_stream_ordinal
            || candidate.stage_same_stream_ordinal != expected.stage_same_stream_ordinal
        {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::SameStreamOrdinal,
            });
        }
        if candidate.receipt_identity != expected.receipt_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::StagedReceiptIdentity,
            });
        }
        if candidate.staged_dependency_identity != expected.staged_dependency_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::StagedDependencyIdentity,
            });
        }
        if candidate.staged_count != expected.staged_count {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::StagedCount,
            });
        }
        if candidate.target_archive_count != expected.target_archive_count {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::TargetArchiveCount,
            });
        }
        if candidate.target_packed_word != expected.target_packed_word {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::TargetPackedWord,
            });
        }
        Ok(())
    }

    fn compare_terminal_query(
        expected: TerminalQueryAuthority,
        candidate: TerminalQueryAuthority,
    ) -> Result<(), TransitionError> {
        if candidate.boxed_receipt_identity != expected.boxed_receipt_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::BoxedReceiptIdentity,
            });
        }
        if candidate.run_token != expected.run_token {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::RunToken,
            });
        }
        if candidate.generation != expected.generation {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::Generation,
            });
        }
        if candidate.packed_word != expected.packed_word {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::SourcePackedWord,
            });
        }
        if candidate.receipt_identity != expected.receipt_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::TerminalReceiptIdentity,
            });
        }
        if candidate.event_identity != expected.event_identity {
            return Err(TransitionError::ReceiptMismatch {
                axis: ReceiptAxis::TerminalEventIdentity,
            });
        }
        Ok(())
    }
}

use resident_archive_knn_v2_transaction_fixture::{
    ArchiveStaged, COMPACT_TERMINAL_RECEIPT_BYTES, CommitFields, GenerationChain, LedgerEntry,
    LedgerPhase, MAX_ARCHIVE_COUNT, MAX_EPOCH, MAX_GENERATION, Observation, PackError,
    RANK_IDENTITY, RANK_RECEIPT_BASE, RankEnqueued, RankReceipt, ReceiptAxis, ReferenceStream,
    RuntimeState, STAGED_RECEIPT_BASE, StagedReceipt, TERMINAL_EVENT_ID, TERMINAL_RECEIPT_BASE,
    TerminalPending, TerminalProjection, TerminalQueryAuthority, TransitionError,
    decode_commit_word, pack_commit_word,
};

macro_rules! assert_not_impl {
    ($type:ty, $trait:path) => {
        const _: fn() = || {
            trait AmbiguousIfImpl<A> {
                fn marker() {}
            }
            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            struct Invalid;
            impl<T: ?Sized + $trait> AmbiguousIfImpl<Invalid> for T {}
            let _ = <$type as AmbiguousIfImpl<_>>::marker;
        };
    };
}

assert_not_impl!(GenerationChain, Clone);
assert_not_impl!(GenerationChain, Copy);
assert_not_impl!(RankEnqueued, Clone);
assert_not_impl!(RankEnqueued, Copy);
assert_not_impl!(ArchiveStaged, Clone);
assert_not_impl!(ArchiveStaged, Copy);
assert_not_impl!(TerminalPending, Clone);
assert_not_impl!(TerminalPending, Copy);

const STREAM_ID: u64 = 0x5354_524d_0000_0001;
const RUN_TOKEN: u64 = 0x5255_4e00_0000_0001;
const SEED_WORD: u64 = 0x0000_0012_0008_000e;
const FIRST_WORD: u64 = 0x0000_0014_000c_0011;
const SECOND_WORD: u64 = 0x0000_0016_000c_0012;
const THIRD_WORD: u64 = 0x0000_0018_0012_0015;

fn admit_seed(run_token: u64) -> (ReferenceStream, GenerationChain) {
    ReferenceStream::admit(STREAM_ID, run_token, 0, 7, 4, 9).unwrap()
}

fn expected_observation(
    commit_word: u64,
    ledger_len: usize,
    enqueue_count: u64,
    state: RuntimeState,
    score_rank_count: u64,
    stage_count: u64,
    evolve_count: u64,
    atomic_publish_count: u64,
    d2h_count: u64,
    event_record_count: u64,
    event_query_count: u64,
) -> Observation {
    Observation {
        commit_word,
        ledger_len,
        enqueue_count,
        state,
        score_rank_count,
        stage_count,
        evolve_count,
        atomic_publish_count,
        d2h_count,
        sync_count: 0,
        event_record_count,
        event_query_count,
    }
}

fn run_three_generations() -> (ReferenceStream, GenerationChain) {
    let (mut stream, initial_chain) = admit_seed(RUN_TOKEN);
    let mut chain = Some(initial_chain);
    for (staged_count, target_word, target_generation, staged_receipt_identity) in [
        (2_u64, FIRST_WORD, 8_u64, 0x5354_4147_4500_0007),
        (0, SECOND_WORD, 9, 0x5354_4147_4500_0008),
        (3, THIRD_WORD, 10, 0x5354_4147_4500_0009),
    ] {
        let rank = stream
            .enqueue_score_and_rank(chain.take().unwrap())
            .unwrap();
        let staged = stream.enqueue_stage_archive(rank, staged_count).unwrap();
        let next_chain = stream.enqueue_evolve_and_publish(staged).unwrap();
        assert_eq!(next_chain.source_packed_word(), target_word);
        assert_eq!(next_chain.planned_generation(), target_generation);
        assert_eq!(
            next_chain.prior_staged_receipt_identity(),
            Some(staged_receipt_identity)
        );
        chain = Some(next_chain);
    }
    (stream, chain.unwrap())
}

#[test]
fn packed_commit_word_has_exact_one_hot_layout_max_round_trip_and_wide_refusals() {
    assert_eq!(pack_commit_word(1, 0, 0, 0), Ok(0x0000_0000_0000_0001));
    assert_eq!(pack_commit_word(0, 1, 0, 0), Ok(0x0000_0000_0000_0002));
    assert_eq!(pack_commit_word(0, 0, 1, 0), Ok(0x0000_0000_0002_0000));
    assert_eq!(pack_commit_word(0, 0, 0, 1), Ok(0x0000_0002_0000_0000));
    assert_eq!(pack_commit_word(0, 7, 4, 9), Ok(SEED_WORD));
    assert_eq!(
        decode_commit_word(SEED_WORD),
        CommitFields {
            store: 0,
            generation: 7,
            archive_count: 4,
            epoch: 9,
        }
    );
    assert_eq!(
        pack_commit_word(1, MAX_GENERATION, MAX_ARCHIVE_COUNT, MAX_EPOCH),
        Ok(u64::MAX)
    );
    assert_eq!(
        decode_commit_word(u64::MAX),
        CommitFields {
            store: 1,
            generation: 65_535,
            archive_count: 65_535,
            epoch: 2_147_483_647,
        }
    );
    assert_eq!(
        pack_commit_word(1, 20_000, 50_000, 2_147_483_647),
        Ok(0xffff_ffff_86a0_9c41)
    );
    assert_eq!(
        pack_commit_word(2, 7, 4, 9),
        Err(PackError::StoreOutOfBounds { value: 2 })
    );
    assert_eq!(
        pack_commit_word(0, 65_536, 4, 9),
        Err(PackError::GenerationOutOfBounds { value: 65_536 })
    );
    assert_eq!(
        pack_commit_word(0, 7, 65_536, 9),
        Err(PackError::ArchiveCountOutOfBounds { value: 65_536 })
    );
    assert_eq!(
        pack_commit_word(0, 7, 4, 2_147_483_648),
        Err(PackError::EpochOutOfBounds {
            value: 2_147_483_648,
        })
    );
}

#[test]
fn rank_and_stage_bind_exact_authority_without_publication_or_terminal_work() {
    let (mut stream, chain) = admit_seed(RUN_TOKEN);
    let boxed_receipt_identity = chain.boxed_receipt_identity();
    assert_ne!(boxed_receipt_identity, 0);
    assert_eq!(chain.source_packed_word(), SEED_WORD);
    assert_eq!(chain.planned_generation(), 7);
    assert_eq!(chain.prior_staged_receipt_identity(), None);

    let rank = stream.enqueue_score_and_rank(chain).unwrap();
    assert_eq!(rank.boxed_receipt_identity(), boxed_receipt_identity);
    assert_eq!(
        rank.receipt(),
        RankReceipt {
            receipt_identity: 0x5241_4e4b_0000_0007,
            boxed_receipt_identity,
            run_token: RUN_TOKEN,
            generation: 7,
            source_packed_word: SEED_WORD,
            store_epoch: 9,
            archive_count_at_start: 4,
            rank_identity: 0x5345_4d41_4e54_4943,
            same_stream_ordinal: 1,
        }
    );
    assert_eq!(stream.commit_word(), SEED_WORD);
    assert_eq!(
        stream.observation(),
        expected_observation(
            SEED_WORD,
            1,
            1,
            RuntimeState::RankEnqueued,
            1,
            0,
            0,
            0,
            0,
            0,
            0
        )
    );

    let staged = stream.enqueue_stage_archive(rank, 2).unwrap();
    assert_eq!(staged.boxed_receipt_identity(), boxed_receipt_identity);
    assert_eq!(
        staged.receipt(),
        StagedReceipt {
            receipt_identity: 0x5354_4147_4500_0007,
            boxed_receipt_identity,
            run_token: RUN_TOKEN,
            generation: 7,
            source_packed_word: SEED_WORD,
            store_epoch: 9,
            archive_count_at_start: 4,
            rank_identity: 0x5345_4d41_4e54_4943,
            rank_same_stream_ordinal: 1,
            stage_same_stream_ordinal: 2,
            staged_dependency_identity: 0x5241_4e4b_0000_0007,
            staged_count: 2,
            target_archive_count: 6,
            target_packed_word: FIRST_WORD,
        }
    );
    assert_eq!(stream.commit_word(), SEED_WORD);
    assert_eq!(
        stream.ledger(),
        [
            LedgerEntry {
                sequence: 1,
                stream: STREAM_ID,
                phase: LedgerPhase::ScoreRank,
                generation: 7,
                receipt_identity: 0x5241_4e4b_0000_0007,
                before_word: SEED_WORD,
                after_word: SEED_WORD,
                d2h_bytes: 0,
                event_identity: 0,
            },
            LedgerEntry {
                sequence: 2,
                stream: STREAM_ID,
                phase: LedgerPhase::StageArchive,
                generation: 7,
                receipt_identity: 0x5241_4e4b_0000_0007,
                before_word: SEED_WORD,
                after_word: SEED_WORD,
                d2h_bytes: 0,
                event_identity: 0,
            },
        ]
    );
    assert_eq!(
        stream.observation(),
        expected_observation(
            SEED_WORD,
            2,
            2,
            RuntimeState::ArchiveStaged,
            1,
            1,
            0,
            0,
            0,
            0,
            0
        )
    );
}

#[test]
fn rank_receipt_axis_refusals_are_inert_before_the_exact_rank_is_consumed() {
    let (mut stream, chain) = admit_seed(RUN_TOKEN);
    let (foreign_stream, foreign_chain) = admit_seed(RUN_TOKEN);
    let foreign_boxed_receipt_identity = foreign_chain.boxed_receipt_identity();
    let rank = stream.enqueue_score_and_rank(chain).unwrap();
    let exact = rank.receipt();
    assert_ne!(exact.boxed_receipt_identity, foreign_boxed_receipt_identity);

    let mut boxed = exact;
    boxed.boxed_receipt_identity = foreign_boxed_receipt_identity;
    let mut receipt_identity = exact;
    receipt_identity.receipt_identity = 0x5241_4e4b_0000_00ff;
    let mut token = exact;
    token.run_token = 0x5255_4e00_0000_0002;
    let mut generation = exact;
    generation.generation = 8;
    let mut source_word = exact;
    source_word.source_packed_word = 0x0000_0014_0008_000f;
    let mut epoch = exact;
    epoch.store_epoch = 10;
    let mut archive = exact;
    archive.archive_count_at_start = 5;
    let mut rank_identity = exact;
    rank_identity.rank_identity = 0x5345_4d41_4e54_4944;
    let mut ordinal = exact;
    ordinal.same_stream_ordinal = 2;

    for (candidate, axis) in [
        (boxed, ReceiptAxis::BoxedReceiptIdentity),
        (receipt_identity, ReceiptAxis::RankReceiptIdentity),
        (token, ReceiptAxis::RunToken),
        (generation, ReceiptAxis::Generation),
        (source_word, ReceiptAxis::SourcePackedWord),
        (epoch, ReceiptAxis::StoreEpoch),
        (archive, ReceiptAxis::ArchiveCountAtStart),
        (rank_identity, ReceiptAxis::RankIdentity),
        (ordinal, ReceiptAxis::SameStreamOrdinal),
    ] {
        assert_eq!(
            stream.validate_rank_receipt_for_test(candidate),
            Err(TransitionError::ReceiptMismatch { axis })
        );
        assert_eq!(
            stream.observation(),
            expected_observation(
                SEED_WORD,
                1,
                1,
                RuntimeState::RankEnqueued,
                1,
                0,
                0,
                0,
                0,
                0,
                0
            )
        );
    }
    assert_eq!(
        foreign_chain.boxed_receipt_identity(),
        foreign_boxed_receipt_identity
    );
    assert_eq!(foreign_stream.commit_word(), SEED_WORD);

    let staged = stream.enqueue_stage_archive(rank, 2).unwrap();
    assert_eq!(staged.receipt().target_packed_word, FIRST_WORD);
    assert_eq!(
        stream.validate_rank_receipt_for_test(exact),
        Err(TransitionError::AlreadyConsumed {
            receipt_identity: 0x5241_4e4b_0000_0007,
        })
    );
    let mut never_issued = exact;
    never_issued.receipt_identity = 0x5241_4e4b_0000_00ff;
    assert_eq!(
        stream.validate_rank_receipt_for_test(never_issued),
        Err(TransitionError::WrongState {
            expected: RuntimeState::RankEnqueued,
            actual: RuntimeState::ArchiveStaged,
        })
    );
}

#[test]
fn staged_receipt_refusals_publish_neither_then_the_exact_authority_publishes_once() {
    let (mut stream, chain) = admit_seed(RUN_TOKEN);
    let rank = stream.enqueue_score_and_rank(chain).unwrap();
    let staged = stream.enqueue_stage_archive(rank, 2).unwrap();
    let exact = staged.receipt();

    let (mut foreign_stream, foreign_chain) = admit_seed(RUN_TOKEN);
    let foreign_rank = foreign_stream
        .enqueue_score_and_rank(foreign_chain)
        .unwrap();
    let foreign_staged = foreign_stream
        .enqueue_stage_archive(foreign_rank, 2)
        .unwrap();
    assert_ne!(
        exact.boxed_receipt_identity,
        foreign_staged.receipt().boxed_receipt_identity
    );
    assert_eq!(
        stream
            .enqueue_evolve_and_publish(foreign_staged)
            .unwrap_err(),
        TransitionError::ReceiptMismatch {
            axis: ReceiptAxis::BoxedReceiptIdentity,
        }
    );
    assert_eq!(
        stream.observation(),
        expected_observation(
            SEED_WORD,
            2,
            2,
            RuntimeState::ArchiveStaged,
            1,
            1,
            0,
            0,
            0,
            0,
            0
        )
    );

    let mut token = exact;
    token.run_token = 0x5255_4e00_0000_0002;
    let mut generation = exact;
    generation.generation = 8;
    let mut source_word = exact;
    source_word.source_packed_word = 0x0000_0014_0008_000f;
    let mut epoch = exact;
    epoch.store_epoch = 10;
    let mut archive = exact;
    archive.archive_count_at_start = 5;
    let mut rank_identity = exact;
    rank_identity.rank_identity = 0x5345_4d41_4e54_4944;
    let mut ordinal = exact;
    ordinal.rank_same_stream_ordinal = 2;
    let mut stage_ordinal = exact;
    stage_ordinal.stage_same_stream_ordinal = 3;
    let mut receipt_identity = exact;
    receipt_identity.receipt_identity = 0x5354_4147_4500_00ff;
    let mut dependency = exact;
    dependency.staged_dependency_identity = 0x5241_4e4b_0000_00ff;
    let mut staged_count = exact;
    staged_count.staged_count = 3;
    let mut target_archive = exact;
    target_archive.target_archive_count = 7;
    let mut target_word = exact;
    target_word.target_packed_word = 0x0000_0014_000e_0011;

    for (candidate, axis) in [
        (token, ReceiptAxis::RunToken),
        (generation, ReceiptAxis::Generation),
        (source_word, ReceiptAxis::SourcePackedWord),
        (epoch, ReceiptAxis::StoreEpoch),
        (archive, ReceiptAxis::ArchiveCountAtStart),
        (rank_identity, ReceiptAxis::RankIdentity),
        (ordinal, ReceiptAxis::SameStreamOrdinal),
        (stage_ordinal, ReceiptAxis::SameStreamOrdinal),
        (receipt_identity, ReceiptAxis::StagedReceiptIdentity),
        (dependency, ReceiptAxis::StagedDependencyIdentity),
        (staged_count, ReceiptAxis::StagedCount),
        (target_archive, ReceiptAxis::TargetArchiveCount),
        (target_word, ReceiptAxis::TargetPackedWord),
    ] {
        assert_eq!(
            stream.validate_staged_receipt_for_test(candidate),
            Err(TransitionError::ReceiptMismatch { axis })
        );
        assert_eq!(
            stream.observation(),
            expected_observation(
                SEED_WORD,
                2,
                2,
                RuntimeState::ArchiveStaged,
                1,
                1,
                0,
                0,
                0,
                0,
                0
            )
        );
    }

    let next_chain = stream.enqueue_evolve_and_publish(staged).unwrap();
    assert_eq!(stream.commit_word(), FIRST_WORD);
    assert_eq!(
        decode_commit_word(stream.commit_word()),
        CommitFields {
            store: 1,
            generation: 8,
            archive_count: 6,
            epoch: 10,
        }
    );
    assert_eq!(next_chain.source_packed_word(), FIRST_WORD);
    assert_eq!(next_chain.planned_generation(), 8);
    assert_eq!(
        next_chain.prior_staged_receipt_identity(),
        Some(0x5354_4147_4500_0007)
    );
    assert_eq!(
        stream.observation(),
        expected_observation(
            FIRST_WORD,
            4,
            4,
            RuntimeState::GenerationChain,
            1,
            1,
            1,
            1,
            0,
            0,
            0
        )
    );
    assert_eq!(
        stream.validate_staged_receipt_for_test(exact),
        Err(TransitionError::AlreadyConsumed {
            receipt_identity: 0x5354_4147_4500_0007,
        })
    );

    let _next_rank = stream.enqueue_score_and_rank(next_chain).unwrap();
    assert_eq!(
        stream.validate_staged_receipt_for_test(exact),
        Err(TransitionError::AlreadyConsumed {
            receipt_identity: 0x5354_4147_4500_0007,
        })
    );
    let mut never_issued = exact;
    never_issued.receipt_identity = 0x5354_4147_4500_00ff;
    assert_eq!(
        stream.validate_staged_receipt_for_test(never_issued),
        Err(TransitionError::WrongState {
            expected: RuntimeState::ArchiveStaged,
            actual: RuntimeState::RankEnqueued,
        })
    );
}

#[test]
fn three_generations_preserve_exact_same_stream_order_with_only_combined_publication() {
    let (stream, final_chain) = run_three_generations();
    assert_eq!(final_chain.source_packed_word(), THIRD_WORD);
    assert_eq!(final_chain.planned_generation(), 10);
    assert_eq!(
        final_chain.prior_staged_receipt_identity(),
        Some(0x5354_4147_4500_0009)
    );
    assert_eq!(
        decode_commit_word(stream.commit_word()),
        CommitFields {
            store: 1,
            generation: 10,
            archive_count: 9,
            epoch: 12,
        }
    );
    assert_eq!(
        stream.observation(),
        expected_observation(
            THIRD_WORD,
            12,
            12,
            RuntimeState::GenerationChain,
            3,
            3,
            3,
            3,
            0,
            0,
            0
        )
    );
    assert_eq!(
        stream.ledger(),
        [
            LedgerEntry {
                sequence: 1,
                stream: STREAM_ID,
                phase: LedgerPhase::ScoreRank,
                generation: 7,
                receipt_identity: 0x5241_4e4b_0000_0007,
                before_word: SEED_WORD,
                after_word: SEED_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 2,
                stream: STREAM_ID,
                phase: LedgerPhase::StageArchive,
                generation: 7,
                receipt_identity: 0x5241_4e4b_0000_0007,
                before_word: SEED_WORD,
                after_word: SEED_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 3,
                stream: STREAM_ID,
                phase: LedgerPhase::EvolveDedup,
                generation: 7,
                receipt_identity: 0x5354_4147_4500_0007,
                before_word: SEED_WORD,
                after_word: SEED_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 4,
                stream: STREAM_ID,
                phase: LedgerPhase::AtomicPublish,
                generation: 7,
                receipt_identity: 0x5354_4147_4500_0007,
                before_word: SEED_WORD,
                after_word: FIRST_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 5,
                stream: STREAM_ID,
                phase: LedgerPhase::ScoreRank,
                generation: 8,
                receipt_identity: 0x5241_4e4b_0000_0008,
                before_word: FIRST_WORD,
                after_word: FIRST_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 6,
                stream: STREAM_ID,
                phase: LedgerPhase::StageArchive,
                generation: 8,
                receipt_identity: 0x5241_4e4b_0000_0008,
                before_word: FIRST_WORD,
                after_word: FIRST_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 7,
                stream: STREAM_ID,
                phase: LedgerPhase::EvolveDedup,
                generation: 8,
                receipt_identity: 0x5354_4147_4500_0008,
                before_word: FIRST_WORD,
                after_word: FIRST_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 8,
                stream: STREAM_ID,
                phase: LedgerPhase::AtomicPublish,
                generation: 8,
                receipt_identity: 0x5354_4147_4500_0008,
                before_word: FIRST_WORD,
                after_word: SECOND_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 9,
                stream: STREAM_ID,
                phase: LedgerPhase::ScoreRank,
                generation: 9,
                receipt_identity: 0x5241_4e4b_0000_0009,
                before_word: SECOND_WORD,
                after_word: SECOND_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 10,
                stream: STREAM_ID,
                phase: LedgerPhase::StageArchive,
                generation: 9,
                receipt_identity: 0x5241_4e4b_0000_0009,
                before_word: SECOND_WORD,
                after_word: SECOND_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 11,
                stream: STREAM_ID,
                phase: LedgerPhase::EvolveDedup,
                generation: 9,
                receipt_identity: 0x5354_4147_4500_0009,
                before_word: SECOND_WORD,
                after_word: SECOND_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
            LedgerEntry {
                sequence: 12,
                stream: STREAM_ID,
                phase: LedgerPhase::AtomicPublish,
                generation: 9,
                receipt_identity: 0x5354_4147_4500_0009,
                before_word: SECOND_WORD,
                after_word: THIRD_WORD,
                d2h_bytes: 0,
                event_identity: 0
            },
        ]
    );
}

#[test]
fn terminal_seal_follows_last_commit_and_only_pending_can_project_the_receipt() {
    let (mut stream, final_chain) = run_three_generations();
    let boxed_receipt_identity = final_chain.boxed_receipt_identity();
    let pending = stream.enqueue_terminal_seal(final_chain).unwrap();
    assert_eq!(stream.commit_word(), THIRD_WORD);
    assert_eq!(
        stream.observation(),
        expected_observation(
            THIRD_WORD,
            14,
            14,
            RuntimeState::TerminalPending,
            3,
            3,
            3,
            3,
            1,
            1,
            0
        )
    );
    assert_eq!(
        &stream.ledger()[12..],
        [
            LedgerEntry {
                sequence: 13,
                stream: STREAM_ID,
                phase: LedgerPhase::CompactD2h,
                generation: 10,
                receipt_identity: 0x5445_524d_0000_000a,
                before_word: THIRD_WORD,
                after_word: THIRD_WORD,
                d2h_bytes: 32,
                event_identity: 0,
            },
            LedgerEntry {
                sequence: 14,
                stream: STREAM_ID,
                phase: LedgerPhase::EventRecord,
                generation: 10,
                receipt_identity: 0x5445_524d_0000_000a,
                before_word: THIRD_WORD,
                after_word: THIRD_WORD,
                d2h_bytes: 0,
                event_identity: 0x4556_454e_5400_0001,
            },
        ]
    );
    assert_eq!(COMPACT_TERMINAL_RECEIPT_BYTES, 32);

    let query_authority = TerminalQueryAuthority {
        receipt_identity: 0x5445_524d_0000_000a,
        boxed_receipt_identity,
        run_token: RUN_TOKEN,
        packed_word: THIRD_WORD,
        generation: 10,
        event_identity: 0x4556_454e_5400_0001,
    };
    let (foreign_stream, foreign_chain) = admit_seed(RUN_TOKEN);
    let mut boxed = query_authority;
    boxed.boxed_receipt_identity = foreign_chain.boxed_receipt_identity();
    let mut token = query_authority;
    token.run_token = 0x5255_4e00_0000_0002;
    let mut packed_word = query_authority;
    packed_word.packed_word = SECOND_WORD;
    let mut generation = query_authority;
    generation.generation = 9;
    let mut receipt_identity = query_authority;
    receipt_identity.receipt_identity = 0x5445_524d_0000_00ff;
    let mut event_identity = query_authority;
    event_identity.event_identity = 0x4556_454e_5400_0002;
    for (candidate, axis) in [
        (boxed, ReceiptAxis::BoxedReceiptIdentity),
        (token, ReceiptAxis::RunToken),
        (packed_word, ReceiptAxis::SourcePackedWord),
        (generation, ReceiptAxis::Generation),
        (receipt_identity, ReceiptAxis::TerminalReceiptIdentity),
        (event_identity, ReceiptAxis::TerminalEventIdentity),
    ] {
        assert_eq!(
            stream.validate_terminal_query_for_test(candidate),
            Err(TransitionError::ReceiptMismatch { axis })
        );
        assert_eq!(
            stream.observation(),
            expected_observation(
                THIRD_WORD,
                14,
                14,
                RuntimeState::TerminalPending,
                3,
                3,
                3,
                3,
                1,
                1,
                0,
            )
        );
    }
    assert_eq!(foreign_stream.commit_word(), SEED_WORD);
    assert_eq!(
        foreign_chain.boxed_receipt_identity(),
        boxed.boxed_receipt_identity
    );
    assert_eq!(
        stream.validate_terminal_query_for_test(query_authority),
        Ok(())
    );
    assert_eq!(
        stream.try_complete(pending).unwrap(),
        TerminalProjection {
            receipt_identity: 0x5445_524d_0000_000a,
            run_token: RUN_TOKEN,
            packed_word: THIRD_WORD,
            store: 1,
            generation: 10,
            archive_count: 9,
            epoch: 12,
            d2h_bytes: 32,
            event_identity: 0x4556_454e_5400_0001,
        }
    );
    assert_eq!(stream.commit_word(), THIRD_WORD);
    assert_eq!(
        stream.observation(),
        expected_observation(
            THIRD_WORD,
            15,
            14,
            RuntimeState::Completed,
            3,
            3,
            3,
            3,
            1,
            1,
            1
        )
    );
    assert_eq!(
        stream.ledger()[14],
        LedgerEntry {
            sequence: 15,
            stream: STREAM_ID,
            phase: LedgerPhase::EventQuery,
            generation: 10,
            receipt_identity: 0x5445_524d_0000_000a,
            before_word: THIRD_WORD,
            after_word: THIRD_WORD,
            d2h_bytes: 0,
            event_identity: 0x4556_454e_5400_0001,
        }
    );
    assert_eq!(
        stream.validate_terminal_query_for_test(query_authority),
        Err(TransitionError::AlreadyConsumed {
            receipt_identity: 0x5445_524d_0000_000a,
        })
    );
    let mut never_issued = query_authority;
    never_issued.receipt_identity = 0x5445_524d_0000_00ff;
    assert_eq!(
        stream.validate_terminal_query_for_test(never_issued),
        Err(TransitionError::WrongState {
            expected: RuntimeState::TerminalPending,
            actual: RuntimeState::Completed,
        })
    );
}

#[test]
fn generation_and_epoch_increment_overflow_refuse_before_stage_evolve_or_publish() {
    let (mut generation_stream, generation_chain) =
        ReferenceStream::admit(STREAM_ID, RUN_TOKEN, 0, 65_535, 4, 9).unwrap();
    let generation_rank = generation_stream
        .enqueue_score_and_rank(generation_chain)
        .unwrap();
    assert_eq!(
        generation_stream
            .enqueue_stage_archive(generation_rank, 1)
            .unwrap_err(),
        TransitionError::Pack(PackError::GenerationOutOfBounds { value: 65_536 })
    );
    assert_eq!(
        generation_stream.observation(),
        expected_observation(
            0x0000_0012_0009_fffe,
            1,
            1,
            RuntimeState::RankEnqueued,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    );

    let (mut epoch_stream, epoch_chain) =
        ReferenceStream::admit(STREAM_ID, RUN_TOKEN, 0, 7, 4, 2_147_483_647).unwrap();
    let epoch_rank = epoch_stream.enqueue_score_and_rank(epoch_chain).unwrap();
    assert_eq!(
        epoch_stream
            .enqueue_stage_archive(epoch_rank, 1)
            .unwrap_err(),
        TransitionError::Pack(PackError::EpochOutOfBounds {
            value: 2_147_483_648,
        })
    );
    assert_eq!(
        epoch_stream.observation(),
        expected_observation(
            0xffff_fffe_0008_000e,
            1,
            1,
            RuntimeState::RankEnqueued,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    );

    let (mut archive_stream, archive_chain) = admit_seed(RUN_TOKEN);
    let archive_rank = archive_stream
        .enqueue_score_and_rank(archive_chain)
        .unwrap();
    assert_eq!(
        archive_stream
            .enqueue_stage_archive(archive_rank, u64::MAX)
            .unwrap_err(),
        TransitionError::Pack(PackError::ArchiveCountOutOfBounds { value: u64::MAX })
    );
    assert_eq!(
        archive_stream.observation(),
        expected_observation(
            SEED_WORD,
            1,
            1,
            RuntimeState::RankEnqueued,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    );
}

#[test]
fn receipt_identity_constants_do_not_overlap_the_bound_generations() {
    assert_eq!(RANK_RECEIPT_BASE | 7, 0x5241_4e4b_0000_0007);
    assert_eq!(STAGED_RECEIPT_BASE | 9, 0x5354_4147_4500_0009);
    assert_eq!(TERMINAL_RECEIPT_BASE | 10, 0x5445_524d_0000_000a);
    assert_eq!(RANK_IDENTITY, 0x5345_4d41_4e54_4943);
    assert_eq!(TERMINAL_EVENT_ID, 0x4556_454e_5400_0001);
}

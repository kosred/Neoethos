//! Background-job manager for the TUI.
//!
//! When the user presses Enter on a launch button (Discover/Train), we
//! spawn a child `neoethos-cli` subprocess and stream its stderr lines into
//! a ring buffer. The TUI reads from the ring buffer on each frame to
//! render a live log panel.
//!
//! Each job runs in its own OS process — the TUI never blocks on it —
//! and a reader thread drains stdout+stderr into a channel. The TUI
//! drains the channel on every tick into the per-job ring buffer.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Instant;

const RING_BUFFER_LINES: usize = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    Completed,
    Failed,
}

pub struct Job {
    pub label: String,
    pub command_summary: String,
    pub status: JobStatus,
    pub started_at: Instant,
    pub log: VecDeque<String>,
    rx: Receiver<LogLine>,
}

enum LogLine {
    Out(String),
    Err(String),
    /// Sent by the watcher thread when both stdout and stderr have
    /// been fully drained — followed by either ExitOk or ExitFail.
    ExitOk,
    ExitFail(i32),
}

impl Job {
    fn drain(&mut self) {
        while let Ok(line) = self.rx.try_recv() {
            match line {
                LogLine::Out(s) | LogLine::Err(s) => {
                    if self.log.len() == RING_BUFFER_LINES {
                        self.log.pop_front();
                    }
                    self.log.push_back(s);
                }
                LogLine::ExitOk => self.status = JobStatus::Completed,
                LogLine::ExitFail(_) => self.status = JobStatus::Failed,
            }
        }
    }

    pub fn elapsed_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    pub fn tail(&self, n: usize) -> impl Iterator<Item = &String> {
        let start = self.log.len().saturating_sub(n);
        self.log.iter().skip(start)
    }
}

pub struct JobManager {
    jobs: Vec<Job>,
}

impl Default for JobManager {
    fn default() -> Self {
        Self::new()
    }
}

impl JobManager {
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    /// Drain message queues for every running job. Call once per render tick.
    pub fn tick(&mut self) {
        for job in &mut self.jobs {
            job.drain();
        }
    }

    pub fn jobs(&self) -> &[Job] {
        &self.jobs
    }

    /// Return the most recent job whose label starts with `prefix`, if
    /// any. Used by pages to find their own job (e.g. Discover page
    /// looks for "discover").
    pub fn latest_for(&self, prefix: &str) -> Option<&Job> {
        self.jobs
            .iter()
            .rev()
            .find(|j| j.label.to_lowercase().starts_with(&prefix.to_lowercase()))
    }

    pub fn has_running(&self, prefix: &str) -> bool {
        self.jobs.iter().any(|j| {
            j.status == JobStatus::Running
                && j.label.to_lowercase().starts_with(&prefix.to_lowercase())
        })
    }

    /// Spawn `neoethos-cli <args>`. Returns the new job's index in
    /// `self.jobs`. If the subprocess cannot be launched, the error is
    /// captured as the job's first log line and the job is marked Failed
    /// — never bubbled up, so the TUI keeps running.
    pub fn spawn(&mut self, label: impl Into<String>, args: Vec<String>) -> usize {
        self.spawn_with_env(label, args, Vec::new())
    }

    /// Same as [`spawn`] but also injects env vars on the child process
    /// only. This is the **safe** way to hand a value to a subprocess: it
    /// uses `Command::env` and never touches the parent (TUI) process's
    /// environment, so it cannot race with rayon/tokio threads inside the
    /// TUI that read env (e.g. `ToSocketAddrs` DNS lookups).
    pub fn spawn_with_env(
        &mut self,
        label: impl Into<String>,
        args: Vec<String>,
        envs: Vec<(String, String)>,
    ) -> usize {
        let label = label.into();
        let command_summary = format!("neoethos-cli {}", args.join(" "));

        // Resolve the current binary so the spawned process is the same
        // build as the running TUI (otherwise pressing the button would
        // run an older `~/bin/neoethos-cli.exe` and confuse everyone).
        let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("neoethos-cli"));

        let (tx, rx) = channel::<LogLine>();

        let mut cmd = Command::new(exe);
        cmd.args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in &envs {
            cmd.env(k, v);
        }

        match cmd.spawn() {
            Ok(mut child) => {
                let stdout = child.stdout.take();
                let stderr = child.stderr.take();
                if let Some(stdout) = stdout {
                    let tx = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stdout);
                        for line in reader.lines().map_while(Result::ok) {
                            if tx.send(LogLine::Out(line)).is_err() {
                                break;
                            }
                        }
                    });
                }
                if let Some(stderr) = stderr {
                    let tx = tx.clone();
                    thread::spawn(move || {
                        let reader = BufReader::new(stderr);
                        for line in reader.lines().map_while(Result::ok) {
                            if tx.send(LogLine::Err(line)).is_err() {
                                break;
                            }
                        }
                    });
                }

                // Watcher: wait for the child, send ExitOk/ExitFail.
                // The child handle moves into this thread so the main
                // TUI loop never blocks on it.
                let waiter_tx = tx.clone();
                let mut child = child;
                thread::spawn(move || match child.wait() {
                    Ok(status) => {
                        let _ = if status.success() {
                            waiter_tx.send(LogLine::ExitOk)
                        } else {
                            waiter_tx.send(LogLine::ExitFail(status.code().unwrap_or(-1)))
                        };
                    }
                    Err(_) => {
                        let _ = waiter_tx.send(LogLine::ExitFail(-1));
                    }
                });

                let job = Job {
                    label: label.clone(),
                    command_summary,
                    status: JobStatus::Running,
                    started_at: Instant::now(),
                    log: VecDeque::with_capacity(RING_BUFFER_LINES),
                    rx,
                };
                self.jobs.push(job);
                self.jobs.len() - 1
            }
            Err(err) => {
                let mut log = VecDeque::with_capacity(RING_BUFFER_LINES);
                log.push_back(format!("failed to spawn: {}", err));
                self.jobs.push(Job {
                    label,
                    command_summary,
                    status: JobStatus::Failed,
                    started_at: Instant::now(),
                    log,
                    rx,
                });
                self.jobs.len() - 1
            }
        }
    }
}

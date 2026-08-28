use std::error::Error;
use std::fmt;

pub const X86_64_V3_REQUIREMENTS_V1: [X8664V3RequirementV1; 22] = [
    X8664V3RequirementV1::Avx,
    X8664V3RequirementV1::Avx2,
    X8664V3RequirementV1::Bmi1,
    X8664V3RequirementV1::Bmi2,
    X8664V3RequirementV1::Cmpxchg16b,
    X8664V3RequirementV1::F16c,
    X8664V3RequirementV1::Fma,
    X8664V3RequirementV1::Fxsr,
    X8664V3RequirementV1::LahfSahf,
    X8664V3RequirementV1::Lzcnt,
    X8664V3RequirementV1::Movbe,
    X8664V3RequirementV1::Popcnt,
    X8664V3RequirementV1::Sse,
    X8664V3RequirementV1::Sse2,
    X8664V3RequirementV1::Sse3,
    X8664V3RequirementV1::Sse41,
    X8664V3RequirementV1::Sse42,
    X8664V3RequirementV1::Ssse3,
    X8664V3RequirementV1::X87,
    X8664V3RequirementV1::Xsave,
    X8664V3RequirementV1::Xcr0Xmm,
    X8664V3RequirementV1::Xcr0Ymm,
];

const ALL_REQUIREMENTS_MASK_V1: u32 = (1_u32 << X86_64_V3_REQUIREMENTS_V1.len()) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum X8664V3RequirementV1 {
    Avx,
    Avx2,
    Bmi1,
    Bmi2,
    Cmpxchg16b,
    F16c,
    Fma,
    Fxsr,
    LahfSahf,
    Lzcnt,
    Movbe,
    Popcnt,
    Sse,
    Sse2,
    Sse3,
    Sse41,
    Sse42,
    Ssse3,
    X87,
    Xsave,
    Xcr0Xmm,
    Xcr0Ymm,
}

impl X8664V3RequirementV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Avx => "avx",
            Self::Avx2 => "avx2",
            Self::Bmi1 => "bmi1",
            Self::Bmi2 => "bmi2",
            Self::Cmpxchg16b => "cmpxchg16b",
            Self::F16c => "f16c",
            Self::Fma => "fma",
            Self::Fxsr => "fxsr",
            Self::LahfSahf => "lahfsahf",
            Self::Lzcnt => "lzcnt",
            Self::Movbe => "movbe",
            Self::Popcnt => "popcnt",
            Self::Sse => "sse",
            Self::Sse2 => "sse2",
            Self::Sse3 => "sse3",
            Self::Sse41 => "sse4.1",
            Self::Sse42 => "sse4.2",
            Self::Ssse3 => "ssse3",
            Self::X87 => "x87",
            Self::Xsave => "xsave",
            Self::Xcr0Xmm => "xcr0_xmm",
            Self::Xcr0Ymm => "xcr0_ymm",
        }
    }

    const fn bit(self) -> u32 {
        1_u32 << self as u8
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct X8664V3FeatureSetV1(u32);

impl X8664V3FeatureSetV1 {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn all_required() -> Self {
        Self(ALL_REQUIREMENTS_MASK_V1)
    }

    pub const fn contains(self, requirement: X8664V3RequirementV1) -> bool {
        self.0 & requirement.bit() != 0
    }

    pub const fn with(self, requirement: X8664V3RequirementV1) -> Self {
        Self(self.0 | requirement.bit())
    }

    pub const fn without(self, requirement: X8664V3RequirementV1) -> Self {
        Self(self.0 & !requirement.bit())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectedCpuArchitectureV1 {
    X8664,
    Other,
}

impl DetectedCpuArchitectureV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::X8664 => "x86_64",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct X8664V3SnapshotV1 {
    architecture: DetectedCpuArchitectureV1,
    available: X8664V3FeatureSetV1,
}

impl X8664V3SnapshotV1 {
    pub const fn new(
        architecture: DetectedCpuArchitectureV1,
        available: X8664V3FeatureSetV1,
    ) -> Self {
        Self {
            architecture,
            available,
        }
    }

    pub const fn architecture(self) -> DetectedCpuArchitectureV1 {
        self.architecture
    }

    pub const fn available(self) -> X8664V3FeatureSetV1 {
        self.available
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum X8664V3PreflightErrorCodeV1 {
    UnsupportedArchitecture,
    MissingRequirements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct X8664V3PreflightErrorV1 {
    code: X8664V3PreflightErrorCodeV1,
    architecture: DetectedCpuArchitectureV1,
    missing_requirements: Vec<X8664V3RequirementV1>,
}

impl X8664V3PreflightErrorV1 {
    pub const fn code(&self) -> X8664V3PreflightErrorCodeV1 {
        self.code
    }

    pub const fn architecture(&self) -> DetectedCpuArchitectureV1 {
        self.architecture
    }

    pub fn missing_requirements(&self) -> &[X8664V3RequirementV1] {
        &self.missing_requirements
    }
}

impl fmt::Display for X8664V3PreflightErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "NEOETHOS_CPU_PREFLIGHT_V1 status=refused required=x86-64-v3 architecture={} missing=",
            self.architecture.as_str()
        )?;
        if self.code == X8664V3PreflightErrorCodeV1::UnsupportedArchitecture {
            formatter.write_str("architecture_x86_64")?;
        } else {
            for (index, requirement) in self.missing_requirements.iter().enumerate() {
                if index != 0 {
                    formatter.write_str(",")?;
                }
                formatter.write_str(requirement.as_str())?;
            }
        }
        formatter.write_str(" action=install_or_run_on_x86-64-v3")
    }
}

impl Error for X8664V3PreflightErrorV1 {}

pub fn evaluate_x86_64_v3_snapshot_v1(
    snapshot: X8664V3SnapshotV1,
) -> Result<(), X8664V3PreflightErrorV1> {
    if snapshot.architecture != DetectedCpuArchitectureV1::X8664 {
        return Err(X8664V3PreflightErrorV1 {
            code: X8664V3PreflightErrorCodeV1::UnsupportedArchitecture,
            architecture: snapshot.architecture,
            missing_requirements: Vec::new(),
        });
    }

    let missing_requirements = X86_64_V3_REQUIREMENTS_V1
        .iter()
        .copied()
        .filter(|requirement| !snapshot.available.contains(*requirement))
        .collect::<Vec<_>>();
    if missing_requirements.is_empty() {
        Ok(())
    } else {
        Err(X8664V3PreflightErrorV1 {
            code: X8664V3PreflightErrorCodeV1::MissingRequirements,
            architecture: snapshot.architecture,
            missing_requirements,
        })
    }
}

pub fn require_current_x86_64_v3_v1() -> Result<(), X8664V3PreflightErrorV1> {
    evaluate_x86_64_v3_snapshot_v1(detect_current_x86_64_v3_snapshot_v1())
}

#[cfg(target_arch = "x86_64")]
pub fn detect_current_x86_64_v3_snapshot_v1() -> X8664V3SnapshotV1 {
    use X8664V3RequirementV1 as Requirement;

    let mut available = X8664V3FeatureSetV1::empty()
        .with(Requirement::Fxsr)
        .with(Requirement::Sse)
        .with(Requirement::Sse2)
        .with(Requirement::X87);

    if std::arch::is_x86_feature_detected!("avx2") {
        available = available.with(Requirement::Avx2);
    }
    if std::arch::is_x86_feature_detected!("bmi1") {
        available = available.with(Requirement::Bmi1);
    }
    if std::arch::is_x86_feature_detected!("bmi2") {
        available = available.with(Requirement::Bmi2);
    }
    if std::arch::is_x86_feature_detected!("cmpxchg16b") {
        available = available.with(Requirement::Cmpxchg16b);
    }
    if std::arch::is_x86_feature_detected!("f16c") {
        available = available.with(Requirement::F16c);
    }
    if std::arch::is_x86_feature_detected!("fma") {
        available = available.with(Requirement::Fma);
    }
    if std::arch::is_x86_feature_detected!("lzcnt") {
        available = available.with(Requirement::Lzcnt);
    }
    if std::arch::is_x86_feature_detected!("movbe") {
        available = available.with(Requirement::Movbe);
    }
    if std::arch::is_x86_feature_detected!("popcnt") {
        available = available.with(Requirement::Popcnt);
    }
    if std::arch::is_x86_feature_detected!("sse3") {
        available = available.with(Requirement::Sse3);
    }
    if std::arch::is_x86_feature_detected!("sse4.1") {
        available = available.with(Requirement::Sse41);
    }
    if std::arch::is_x86_feature_detected!("sse4.2") {
        available = available.with(Requirement::Sse42);
    }
    if std::arch::is_x86_feature_detected!("ssse3") {
        available = available.with(Requirement::Ssse3);
    }
    if std::arch::is_x86_feature_detected!("xsave") {
        available = available.with(Requirement::Xsave);
    }

    if raw_cpuid::CpuId::new()
        .get_extended_processor_and_feature_identifiers()
        .is_some_and(|features| features.has_lahf_sahf())
    {
        available = available.with(Requirement::LahfSahf);
    }

    // The standard detector reports AVX only when the CPU feature and the OS
    // XMM/YMM extended-state support needed to execute AVX are all usable.
    if std::arch::is_x86_feature_detected!("avx") {
        available = available
            .with(Requirement::Avx)
            .with(Requirement::Xcr0Xmm)
            .with(Requirement::Xcr0Ymm);
    }

    X8664V3SnapshotV1::new(DetectedCpuArchitectureV1::X8664, available)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn detect_current_x86_64_v3_snapshot_v1() -> X8664V3SnapshotV1 {
    X8664V3SnapshotV1::new(
        DetectedCpuArchitectureV1::Other,
        X8664V3FeatureSetV1::empty(),
    )
}

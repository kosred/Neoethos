//! Resolve the current host once before a build starts.

use std::collections::BTreeSet;
use std::io::ErrorKind;
use std::process::Command;

#[derive(Debug, PartialEq, Eq)]
struct NvidiaDevice {
    uuid: String,
    pci_bus_id: String,
    name: String,
    compute_capability: u32,
}

fn main() {
    let available = std::thread::available_parallelism()
        .expect("std::thread::available_parallelism must resolve for a host build")
        .get();
    let worker_limit = available.saturating_sub(2).max(1);

    let devices = query_nvidia_devices()
        .and_then(canonicalize_nvidia_devices)
        .unwrap_or_else(|error| panic!("{error}"));
    let accelerator_mode = if devices.is_empty() {
        "cpu_only"
    } else {
        "nvidia"
    };
    let architectures = if devices.is_empty() {
        "none".to_owned()
    } else {
        devices
            .iter()
            .map(|device| device.compute_capability)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|capability| capability.to_string())
            .collect::<Vec<_>>()
            .join(";")
    };

    println!("available_parallelism={available}");
    println!("automatic_worker_limit={worker_limit}");
    println!("accelerator_mode={accelerator_mode}");
    println!("cuda_architectures={architectures}");
    for device in devices {
        println!(
            "gpu_device=uuid:{}|pci_bus_id:{}|name:{}|compute_capability:{}",
            device.uuid, device.pci_bus_id, device.name, device.compute_capability
        );
    }
}

fn query_nvidia_devices() -> Result<Vec<NvidiaDevice>, String> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=uuid,pci.bus_id,name,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("could not execute nvidia-smi: {error}")),
    };
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi exists but its device query failed; refusing a silent CPU fallback: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("nvidia-smi device query returned non-UTF-8 output: {error}"))?;
    if stdout.trim().is_empty() {
        Ok(Vec::new())
    } else {
        parse_nvidia_smi_devices(&stdout)
    }
}

fn canonicalize_nvidia_devices(
    mut devices: Vec<NvidiaDevice>,
) -> Result<Vec<NvidiaDevice>, String> {
    let mut uuids = BTreeSet::new();
    let mut pci_bus_ids = BTreeSet::new();
    for device in &devices {
        if !uuids.insert(device.uuid.as_str()) {
            return Err(format!(
                "nvidia-smi reported duplicate GPU UUID {:?}",
                device.uuid
            ));
        }
        if !pci_bus_ids.insert(device.pci_bus_id.as_str()) {
            return Err(format!(
                "nvidia-smi reported duplicate PCI bus ID {:?}",
                device.pci_bus_id
            ));
        }
    }
    devices.sort_by(|left, right| {
        left.uuid
            .cmp(&right.uuid)
            .then_with(|| left.pci_bus_id.cmp(&right.pci_bus_id))
    });
    Ok(devices)
}

fn parse_nvidia_smi_devices(stdout: &str) -> Result<Vec<NvidiaDevice>, String> {
    let mut devices = Vec::new();
    for (row_index, row) in stdout.lines().enumerate() {
        let fields = row.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() < 4 {
            return Err(format!(
                "nvidia-smi row {} has {} fields, expected uuid,pci.bus_id,name,compute_cap: {row:?}",
                row_index + 1,
                fields.len()
            ));
        }
        let uuid = fields[0];
        let pci_bus_id = fields[1];
        let compute_capability_text = fields[fields.len() - 1];
        let name = fields[2..fields.len() - 1].join(",").trim().to_owned();
        if uuid.is_empty() || pci_bus_id.is_empty() || name.is_empty() {
            return Err(format!(
                "nvidia-smi row {} has an empty UUID, PCI bus ID, or name: {row:?}",
                row_index + 1
            ));
        }
        devices.push(NvidiaDevice {
            uuid: uuid.to_owned(),
            pci_bus_id: pci_bus_id.to_owned(),
            name,
            compute_capability: parse_compute_capability(compute_capability_text)
                .map_err(|error| format!("nvidia-smi row {}: {error}", row_index + 1))?,
        });
    }
    if devices.is_empty() {
        return Err("nvidia-smi reported no visible NVIDIA devices".to_owned());
    }
    Ok(devices)
}

fn parse_compute_capability(value: &str) -> Result<u32, String> {
    let (major, minor) = value
        .trim()
        .split_once('.')
        .ok_or_else(|| format!("invalid compute capability {value:?}; expected major.minor"))?;
    if major.is_empty()
        || minor.len() != 1
        || !major.bytes().all(|byte| byte.is_ascii_digit())
        || !minor.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!(
            "invalid compute capability {value:?}; expected numeric major.minor"
        ));
    }
    let major = major
        .parse::<u32>()
        .map_err(|_| format!("compute capability major is out of range: {value:?}"))?;
    let minor = minor
        .parse::<u32>()
        .map_err(|_| format!("compute capability minor is out of range: {value:?}"))?;
    major
        .checked_mul(10)
        .and_then(|base| base.checked_add(minor))
        .filter(|capability| *capability > 0)
        .ok_or_else(|| format!("compute capability is out of range: {value:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_current_and_next_generation_capabilities() {
        assert_eq!(parse_compute_capability("8.6").unwrap(), 86);
        assert_eq!(parse_compute_capability("8.9").unwrap(), 89);
        assert_eq!(parse_compute_capability("10.0").unwrap(), 100);
        assert_eq!(parse_compute_capability("12.0").unwrap(), 120);
    }

    #[test]
    fn rejects_ambiguous_or_host_magic_capabilities() {
        for invalid in ["", "89", "8.90", "sm_89", "native", "0.0"] {
            assert!(
                parse_compute_capability(invalid).is_err(),
                "{invalid:?} unexpectedly passed"
            );
        }
    }

    #[test]
    fn canonicalizes_every_visible_device_without_collapsing_same_arch_cards() {
        let devices = canonicalize_nvidia_devices(
            parse_nvidia_smi_devices(
                "GPU-c, 00000000:03:00.0, NVIDIA GeForce RTX 3090, 8.6\n\
                 GPU-a, 00000000:01:00.0, NVIDIA GeForce RTX 4090, 8.9\n\
                 GPU-b, 00000000:02:00.0, NVIDIA L40S, 8.9\n",
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[0].uuid, "GPU-a");
        assert_eq!(devices[0].pci_bus_id, "00000000:01:00.0");
        assert_eq!(devices[0].compute_capability, 89);
        assert_eq!(devices[1].uuid, "GPU-b");
        assert_eq!(devices[1].compute_capability, 89);
        assert_eq!(devices[2].uuid, "GPU-c");
        assert_eq!(devices[2].compute_capability, 86);
    }

    #[test]
    fn rejects_duplicate_stable_gpu_identities() {
        let duplicate_uuid = parse_nvidia_smi_devices(
            "GPU-a, 00000000:01:00.0, NVIDIA GeForce RTX 4090, 8.9\n\
             GPU-a, 00000000:02:00.0, NVIDIA GeForce RTX 4090, 8.9\n",
        )
        .unwrap();
        assert!(canonicalize_nvidia_devices(duplicate_uuid).is_err());

        let duplicate_pci = parse_nvidia_smi_devices(
            "GPU-a, 00000000:01:00.0, NVIDIA GeForce RTX 4090, 8.9\n\
             GPU-b, 00000000:01:00.0, NVIDIA GeForce RTX 3090, 8.6\n",
        )
        .unwrap();
        assert!(canonicalize_nvidia_devices(duplicate_pci).is_err());
    }
}

#![deny(warnings)]

//! Standalone S2 Philox oracle.
//!
//! This file intentionally has no Cargo or NeoEthos imports. It is compiled
//! directly with `rustc --test` so it cannot reuse production Rust/CUDA code.

const UPSTREAM_KAT: &str = include_str!("upstream_philox4x32_10_kat.txt");
const BOUNDARY_GOLDENS: &str = include_str!("philox4x32_10_boundary_v1.tsv");
const ADDRESS_GOLDENS: &str = include_str!("address_mapping_candidate_v1.tsv");

fn parse_hex_u32(token: &str) -> u32 {
    u32::from_str_radix(token, 16).expect("valid eight-digit hexadecimal word")
}

fn parse_hex_u64(token: &str) -> u64 {
    u64::from_str_radix(token, 16).expect("valid sixteen-digit hexadecimal word")
}

fn parse_run_identity(token: &str) -> [u8; 32] {
    assert_eq!(token.len(), 64, "run identity must contain 32 bytes");
    let mut identity = [0_u8; 32];
    for (index, byte) in identity.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&token[index * 2..index * 2 + 2], 16)
            .expect("valid hexadecimal run-identity byte");
    }
    identity
}

fn independent_philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    const MULTIPLIER_0: u32 = 0xd251_1f53;
    const MULTIPLIER_1: u32 = 0xcd9e_8d57;
    const WEYL_0: u32 = 0x9e37_79b9;
    const WEYL_1: u32 = 0xbb67_ae85;

    for _ in 0..10 {
        let product_0 = u64::from(MULTIPLIER_0) * u64::from(counter[0]);
        let product_1 = u64::from(MULTIPLIER_1) * u64::from(counter[2]);
        counter = [
            (product_1 >> 32) as u32 ^ counter[1] ^ key[0],
            product_1 as u32,
            (product_0 >> 32) as u32 ^ counter[3] ^ key[1],
            product_0 as u32,
        ];
        key[0] = key[0].wrapping_add(WEYL_0);
        key[1] = key[1].wrapping_add(WEYL_1);
    }
    counter
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DrawAddress {
    counter: [u32; 4],
    key: [u32; 2],
}

#[derive(Clone, Copy)]
struct AddressInput {
    search_seed: u64,
    run_identity: [u8; 32],
    generation: u32,
    candidate_identity: u64,
    operator_identity: u32,
    decision_slot: u32,
    rejection_attempt: u32,
}

fn candidate_address_mapping_v1(input: AddressInput) -> DrawAddress {
    let run_word_0 = u32::from_le_bytes(input.run_identity[0..4].try_into().unwrap());
    let run_word_1 = u32::from_le_bytes(input.run_identity[4..8].try_into().unwrap());
    let draw_index =
        (u64::from(input.decision_slot) << 32) | u64::from(input.rejection_attempt);
    DrawAddress {
        counter: [
            input.candidate_identity as u32,
            (input.candidate_identity >> 32) as u32,
            input.generation,
            draw_index as u32,
        ],
        key: [
            input.search_seed as u32 ^ run_word_0 ^ input.operator_identity,
            (input.search_seed >> 32) as u32 ^ run_word_1 ^ (draw_index >> 32) as u32,
        ],
    }
}

#[test]
fn official_random123_known_answer_vectors_match() {
    let mut checked = 0;
    for line in UPSTREAM_KAT.lines().filter(|line| !line.trim().is_empty()) {
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        assert_eq!(fields.len(), 12, "unexpected upstream KAT row: {line}");
        assert_eq!(fields[0], "philox4x32");
        assert_eq!(fields[1], "10");
        let counter = [
            parse_hex_u32(fields[2]),
            parse_hex_u32(fields[3]),
            parse_hex_u32(fields[4]),
            parse_hex_u32(fields[5]),
        ];
        let key = [parse_hex_u32(fields[6]), parse_hex_u32(fields[7])];
        let expected = [
            parse_hex_u32(fields[8]),
            parse_hex_u32(fields[9]),
            parse_hex_u32(fields[10]),
            parse_hex_u32(fields[11]),
        ];
        assert_eq!(independent_philox4x32_10(counter, key), expected);
        checked += 1;
    }
    assert_eq!(checked, 3, "expected the three official Philox4x32-10 KATs");
}

#[test]
fn upstream_generated_boundary_vectors_match() {
    let mut checked = 0;
    for line in BOUNDARY_GOLDENS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        assert_eq!(fields.len(), 11, "unexpected boundary row: {line}");
        let counter = [
            parse_hex_u32(fields[1]),
            parse_hex_u32(fields[2]),
            parse_hex_u32(fields[3]),
            parse_hex_u32(fields[4]),
        ];
        let key = [parse_hex_u32(fields[5]), parse_hex_u32(fields[6])];
        let expected = [
            parse_hex_u32(fields[7]),
            parse_hex_u32(fields[8]),
            parse_hex_u32(fields[9]),
            parse_hex_u32(fields[10]),
        ];
        assert_eq!(
            independent_philox4x32_10(counter, key),
            expected,
            "boundary case {}",
            fields[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 5, "expected five independent boundary vectors");
}

#[test]
fn candidate_address_mapping_and_draws_match_upstream_goldens() {
    let mut checked = 0;
    for line in ADDRESS_GOLDENS
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let fields: Vec<_> = line.split_ascii_whitespace().collect();
        assert_eq!(fields.len(), 18, "unexpected address row: {line}");
        let input = AddressInput {
            search_seed: parse_hex_u64(fields[1]),
            run_identity: parse_run_identity(fields[2]),
            generation: parse_hex_u32(fields[3]),
            candidate_identity: parse_hex_u64(fields[4]),
            operator_identity: parse_hex_u32(fields[5]),
            decision_slot: parse_hex_u32(fields[6]),
            rejection_attempt: parse_hex_u32(fields[7]),
        };
        let expected_address = DrawAddress {
            counter: [
                parse_hex_u32(fields[8]),
                parse_hex_u32(fields[9]),
                parse_hex_u32(fields[10]),
                parse_hex_u32(fields[11]),
            ],
            key: [parse_hex_u32(fields[12]), parse_hex_u32(fields[13])],
        };
        let expected_output = [
            parse_hex_u32(fields[14]),
            parse_hex_u32(fields[15]),
            parse_hex_u32(fields[16]),
            parse_hex_u32(fields[17]),
        ];
        let address = candidate_address_mapping_v1(input);
        assert_eq!(address, expected_address, "address case {}", fields[0]);
        assert_eq!(
            independent_philox4x32_10(address.counter, address.key),
            expected_output,
            "draw case {}",
            fields[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 10, "expected ten address-mapping vectors");
}

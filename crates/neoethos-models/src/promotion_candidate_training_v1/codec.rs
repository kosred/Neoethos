//! Enforce the wire cap while serializing, before allocating an oversized payload.
use super::*;
use std::io::{self, Write};

struct BoundedWriter {
    bytes: Option<Vec<u8>>,
    written: usize,
    exceeded: bool,
    limit: usize,
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.written) {
            self.exceeded = true;
            return Err(io::Error::other("candidate handoff byte cap exceeded"));
        }
        if let Some(output) = &mut self.bytes {
            output.extend_from_slice(bytes);
        }
        self.written += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialize<T: Serialize + ?Sized>(
    value: &T,
    retain: bool,
    limit: usize,
) -> Result<Option<Vec<u8>>, PromotionCandidateTrainingRefusalV1> {
    let mut writer = BoundedWriter {
        bytes: retain.then(|| Vec::with_capacity(16 * 1024)),
        written: 0,
        exceeded: false,
        limit,
    };
    let result = serde_json::to_writer(&mut writer, value);
    if writer.exceeded {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge,
            if limit == MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 {
                "candidate JSON exceeds the 8 MiB wire cap".to_owned()
            } else {
                format!("candidate JSON exceeds its {limit}-byte expansion cap")
            },
        ));
    }
    result.map_err(|error| {
        refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
            format!("encode candidate JSON: {error}"),
        )
    })?;
    Ok(writer.bytes)
}

pub(super) fn encode_bounded<T: Serialize + ?Sized>(
    value: &T,
) -> Result<Vec<u8>, PromotionCandidateTrainingRefusalV1> {
    Ok(
        serialize(value, true, MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1)?
            .expect("retaining writer has bytes"),
    )
}

pub(super) fn check_bounded<T: Serialize + ?Sized>(
    value: &T,
) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    serialize(value, false, MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1).map(|_| ())
}

/// Independent expansion bound, not an increase to the complete 8 MiB wire
/// cap. A full binary feature plan serialized as JSON can exceed the wire cap
/// even after all duplicate receipts have been removed.
const MAX_RECEIPT_JSON_BYTES: usize = 64 * 1024 * 1024;
const ZLIB_RECEIPT_CODEC: &str = "zlib-json-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompressedReceipt {
    codec: String,
    json_bytes: usize,
    json_sha256: String,
    #[serde(deserialize_with = "deserialize_compressed_bytes")]
    bytes: Vec<u8>,
}

// Bound allocation even when callers use Deserialize directly rather than the
// file reader's complete-wire cap. Do not reserve an untrusted size_hint.
fn deserialize_compressed_bytes<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct BytesVisitor;
    impl<'de> serde::de::Visitor<'de> for BytesVisitor {
        type Value = Vec<u8>;
        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a bounded zlib byte array")
        }
        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut bytes = Vec::new();
            while let Some(byte) = sequence.next_element::<u8>()? {
                if bytes.len() == MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 {
                    return Err(serde::de::Error::custom(
                        "compressed receipt exceeds wire cap",
                    ));
                }
                bytes.push(byte);
            }
            Ok(bytes)
        }
    }
    deserializer.deserialize_seq(BytesVisitor)
}

impl CompressedReceipt {
    pub(super) fn new(
        receipt: &CanonicalSearchInputReceiptV2,
    ) -> Result<Self, PromotionCandidateTrainingRefusalV1> {
        let json =
            serialize(receipt, true, MAX_RECEIPT_JSON_BYTES)?.expect("retaining writer has bytes");
        Self::from_json(&json)
    }

    fn from_json(json: &[u8]) -> Result<Self, PromotionCandidateTrainingRefusalV1> {
        if json.len() > MAX_RECEIPT_JSON_BYTES {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge,
                "expanded receipt exceeds the 64 MiB JSON cap",
            ));
        }
        let writer = BoundedWriter {
            bytes: Some(Vec::new()),
            written: 0,
            exceeded: false,
            limit: MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1,
        };
        let mut encoder = flate2::write::ZlibEncoder::new(writer, flate2::Compression::new(6));
        let result = encoder.write_all(json).and_then(|()| encoder.try_finish());
        if encoder.get_ref().exceeded {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge,
                "compressed receipt exceeds the 8 MiB wire cap",
            ));
        }
        result.map_err(compression_error)?;
        let writer = encoder.finish().map_err(compression_error)?;
        Ok(Self {
            codec: ZLIB_RECEIPT_CODEC.to_owned(),
            json_bytes: json.len(),
            json_sha256: format!("{:x}", Sha256::digest(json)),
            bytes: writer.bytes.expect("retaining writer has bytes"),
        })
    }

    fn decode_json(&self, limit: usize) -> Result<Vec<u8>, PromotionCandidateTrainingRefusalV1> {
        if self.json_bytes > limit {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge,
                "expanded receipt exceeds its independent JSON cap",
            ));
        }
        if self.codec != ZLIB_RECEIPT_CODEC
            || self.bytes.is_empty()
            || self.bytes.len() > MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1
            || self.json_bytes == 0
            || !is_sha256_v1(&self.json_sha256)
        {
            return Err(compression_error("invalid compressed receipt envelope"));
        }
        let mut decoder = flate2::Decompress::new(true);
        let mut output = Vec::new();
        let mut chunk = [0u8; 64 * 1024];
        loop {
            let input_before = decoder.total_in();
            let output_before = decoder.total_out();
            let status = decoder
                .decompress(
                    &self.bytes[input_before as usize..],
                    &mut chunk,
                    flate2::FlushDecompress::None,
                )
                .map_err(compression_error)?;
            let produced = (decoder.total_out() - output_before) as usize;
            // Check both the independent cap and the envelope's claimed length
            // before retaining each chunk. A forged small length cannot bypass
            // expansion admission.
            if produced > limit.saturating_sub(output.len())
                || produced > self.json_bytes.saturating_sub(output.len())
            {
                return Err(refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge,
                    "compressed receipt expands beyond its admitted length",
                ));
            }
            output.extend_from_slice(&chunk[..produced]);
            if status == flate2::Status::StreamEnd {
                // StreamEnd verifies the zlib checksum; EOF alone does not.
                // Reject trailing bytes and concatenated streams as well.
                if decoder.total_in() != self.bytes.len() as u64
                    || output.len() != self.json_bytes
                    || format!("{:x}", Sha256::digest(&output)) != self.json_sha256
                {
                    return Err(compression_error(
                        "compressed receipt length/hash/trailing bytes mismatch",
                    ));
                }
                return Ok(output);
            }
            if decoder.total_in() == input_before && produced == 0 {
                return Err(compression_error("truncated compressed receipt"));
            }
        }
    }

    pub(super) fn decode(
        &self,
    ) -> Result<CanonicalSearchInputReceiptV2, PromotionCandidateTrainingRefusalV1> {
        let json = self.decode_json(MAX_RECEIPT_JSON_BYTES)?;
        CanonicalSearchInputReceiptV2::from_json_bytes(&json).map_err(compression_error)
    }

    pub(super) fn validate_against(
        &self,
        receipt: &CanonicalSearchInputReceiptV2,
    ) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        if &self.decode()? != receipt {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
                "compressed receipt differs from the reconstructed handoff receipt",
            ));
        }
        Ok(())
    }
}

fn compression_error(error: impl fmt::Display) -> PromotionCandidateTrainingRefusalV1 {
    refusal_v1(
        PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
        format!("compressed receipt: {error}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_writer_refuses_before_retaining_oversize_chunk() {
        let mut writer = BoundedWriter {
            bytes: Some(Vec::new()),
            written: MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 - 1,
            exceeded: false,
            limit: MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1,
        };
        assert!(writer.write_all(b"xx").is_err());
        assert!(writer.bytes.unwrap().is_empty());
        assert!(writer.exceeded);
    }
    #[test]
    fn count_only_and_retained_json_have_identical_limits() {
        let small = vec!["escaped\"", "Greek: α"];
        assert_eq!(
            encode_bounded(&small).unwrap(),
            serde_json::to_vec(&small).unwrap()
        );
        check_bounded(&small).unwrap();
        let large = "x".repeat(MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1);
        assert_eq!(
            encode_bounded(&large).unwrap_err().code(),
            PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge
        );
        assert_eq!(
            check_bounded(&large).unwrap_err().code(),
            PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge
        );
    }

    #[test]
    fn compressed_receipt_rejects_corruption_truncation_and_trailing_streams() {
        let original =
            CompressedReceipt::new(&crate::promotion_candidate_training_v1_tests::exact_receipt())
                .unwrap();
        let json = original.decode_json(MAX_RECEIPT_JSON_BYTES).unwrap();
        for removed in [1, 4, original.bytes.len() / 2] {
            let mut changed = original.clone();
            changed.bytes.truncate(changed.bytes.len() - removed);
            assert!(changed.decode_json(MAX_RECEIPT_JSON_BYTES).is_err());
        }
        let mut changed = original.clone();
        let last = changed.bytes.len() - 1;
        changed.bytes[last] ^= 1;
        assert!(changed.decode().is_err(), "zlib checksum corruption");
        let mut changed = original.clone();
        changed.bytes.extend_from_slice(&original.bytes);
        assert!(changed.decode().is_err(), "concatenated stream");
        let mut changed = original.clone();
        changed.bytes.push(0);
        assert!(changed.decode().is_err(), "trailing garbage");
        let mut changed = original.clone();
        changed.json_sha256 = "0".repeat(64);
        assert!(changed.decode().is_err(), "claimed hash");
        let mut changed = original.clone();
        changed.codec = "gzip".to_owned();
        assert!(changed.decode().is_err(), "unknown codec");
        assert_eq!(original.decode_json(MAX_RECEIPT_JSON_BYTES).unwrap(), json);
    }

    #[test]
    fn compressed_receipt_expansion_is_bounded_before_retention() {
        // Exercise the same decoder with a small injected bound, not a giant
        // decompression-bomb fixture. Both truthful and forged lengths refuse.
        let original = CompressedReceipt::from_json(&vec![b'x'; 4096]).unwrap();
        assert_eq!(
            original.decode_json(128).unwrap_err().code(),
            PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge
        );
        let mut forged = original.clone();
        forged.json_bytes = 128;
        assert_eq!(
            forged.decode_json(128).unwrap_err().code(),
            PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge
        );
        forged.json_bytes = MAX_RECEIPT_JSON_BYTES + 1;
        assert_eq!(
            forged.decode().unwrap_err().code(),
            PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge
        );
        let mut oversized_writer = BoundedWriter {
            bytes: Some(Vec::new()),
            written: 0,
            exceeded: false,
            limit: 128,
        };
        assert!(oversized_writer.write_all(&[0; 129]).is_err());
        assert!(oversized_writer.bytes.unwrap().is_empty());
    }
}

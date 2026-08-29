//! Allocation limits shared by every source-format reader.

use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportLimitKind {
    SourceBytes,
    StagingBytes,
    CandidateBytes,
    CsvRecordBytes,
    CsvFieldBytes,
    CsvFieldCount,
    JsonNesting,
    JsonStringBytes,
    JsonObjectBytes,
    ArrowMetadataBytes,
    ArrowMessageBytes,
    ArrowBodyBytes,
    ArrowBatchRows,
    ArrowColumns,
    ArrowBatchBytes,
    ParquetFooterBytes,
    ParquetPageHeaderBytes,
    ParquetRowGroupBytes,
    ParquetPageBytes,
    ParquetDictionaryBytes,
    ParquetDecompressedBytes,
    ParquetCompressionRatio,
    VortexMetadataBytes,
    VortexChunkBytes,
    TotalRows,
    CheckedArithmetic,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportLimitError {
    kind: ImportLimitKind,
    actual: u128,
    maximum: u128,
}

impl ImportLimitError {
    pub const fn kind(&self) -> ImportLimitKind {
        self.kind
    }

    pub const fn actual(&self) -> u128 {
        self.actual
    }

    pub const fn maximum(&self) -> u128 {
        self.maximum
    }
}

impl fmt::Display for ImportLimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "import limit {:?} exceeded: actual {}, maximum {}",
            self.kind, self.actual, self.maximum
        )
    }
}

impl Error for ImportLimitError {}

#[derive(Clone, Debug)]
pub struct ImportLimits {
    max_source_bytes: u64,
    max_staging_bytes: u64,
    max_candidate_bytes: u64,
    required_free_disk_bytes: u64,
    max_csv_record_bytes: usize,
    max_csv_field_bytes: usize,
    max_csv_field_count: usize,
    max_json_nesting: usize,
    max_json_string_bytes: usize,
    max_json_object_bytes: usize,
    max_arrow_metadata_bytes: usize,
    max_arrow_message_bytes: usize,
    max_arrow_body_bytes: usize,
    max_arrow_batch_rows: usize,
    max_arrow_columns: usize,
    max_arrow_batch_bytes: usize,
    max_parquet_footer_bytes: usize,
    max_parquet_page_header_bytes: usize,
    max_parquet_row_group_bytes: u64,
    max_parquet_page_bytes: u64,
    max_parquet_dictionary_bytes: u64,
    max_parquet_decompressed_bytes: u64,
    max_parquet_compression_ratio: u64,
    max_vortex_metadata_bytes: usize,
    max_vortex_chunk_bytes: usize,
    max_total_rows: u64,
}

impl Default for ImportLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 64 * 1024 * 1024 * 1024,
            max_staging_bytes: 64 * 1024 * 1024 * 1024,
            max_candidate_bytes: 64 * 1024 * 1024 * 1024,
            required_free_disk_bytes: 2 * 1024 * 1024 * 1024,
            max_csv_record_bytes: 1024 * 1024,
            max_csv_field_bytes: 256 * 1024,
            max_csv_field_count: 256,
            max_json_nesting: 64,
            max_json_string_bytes: 1024 * 1024,
            max_json_object_bytes: 2 * 1024 * 1024,
            max_arrow_metadata_bytes: 16 * 1024 * 1024,
            max_arrow_message_bytes: 64 * 1024 * 1024,
            max_arrow_body_bytes: 256 * 1024 * 1024,
            max_arrow_batch_rows: 64 * 1024,
            max_arrow_columns: 256,
            max_arrow_batch_bytes: 128 * 1024 * 1024,
            max_parquet_footer_bytes: 16 * 1024 * 1024,
            max_parquet_page_header_bytes: 64 * 1024,
            max_parquet_row_group_bytes: 128 * 1024 * 1024,
            max_parquet_page_bytes: 128 * 1024 * 1024,
            max_parquet_dictionary_bytes: 256 * 1024 * 1024,
            max_parquet_decompressed_bytes: 64 * 1024 * 1024 * 1024,
            max_parquet_compression_ratio: 1_000,
            max_vortex_metadata_bytes: 16 * 1024 * 1024,
            max_vortex_chunk_bytes: 256 * 1024 * 1024,
            max_total_rows: 250_000_000,
        }
    }
}

impl ImportLimits {
    pub fn conservative_for_tests() -> Self {
        Self {
            max_source_bytes: 32 * 1024 * 1024,
            max_staging_bytes: 32 * 1024 * 1024,
            max_candidate_bytes: 32 * 1024 * 1024,
            required_free_disk_bytes: 16 * 1024 * 1024,
            max_csv_record_bytes: 64 * 1024,
            max_csv_field_bytes: 16 * 1024,
            max_csv_field_count: 64,
            max_json_nesting: 32,
            max_json_string_bytes: 64 * 1024,
            max_json_object_bytes: 128 * 1024,
            max_arrow_metadata_bytes: 1024 * 1024,
            max_arrow_message_bytes: 4 * 1024 * 1024,
            max_arrow_body_bytes: 8 * 1024 * 1024,
            max_arrow_batch_rows: 4 * 1024,
            max_arrow_columns: 64,
            max_arrow_batch_bytes: 8 * 1024 * 1024,
            max_parquet_footer_bytes: 1024 * 1024,
            max_parquet_page_header_bytes: 16 * 1024,
            max_parquet_row_group_bytes: 4 * 1024 * 1024,
            max_parquet_page_bytes: 4 * 1024 * 1024,
            max_parquet_dictionary_bytes: 4 * 1024 * 1024,
            max_parquet_decompressed_bytes: 64 * 1024 * 1024,
            max_parquet_compression_ratio: 100,
            max_vortex_metadata_bytes: 1024 * 1024,
            max_vortex_chunk_bytes: 8 * 1024 * 1024,
            max_total_rows: 1_000_000,
        }
    }

    pub fn with_parquet_decode_bounds(
        mut self,
        max_row_group_bytes: u64,
        max_page_bytes: u64,
        max_dictionary_bytes: u64,
        max_decompressed_bytes: u64,
        max_compression_ratio: u64,
    ) -> Self {
        self.max_parquet_row_group_bytes = max_row_group_bytes;
        self.max_parquet_page_bytes = max_page_bytes;
        self.max_parquet_dictionary_bytes = max_dictionary_bytes;
        self.max_parquet_decompressed_bytes = max_decompressed_bytes;
        self.max_parquet_compression_ratio = max_compression_ratio;
        self
    }

    pub fn with_storage_bounds(
        mut self,
        max_source_bytes: u64,
        max_staging_bytes: u64,
        max_candidate_bytes: u64,
        required_free_disk_bytes: u64,
    ) -> Self {
        self.max_source_bytes = max_source_bytes;
        self.max_staging_bytes = max_staging_bytes;
        self.max_candidate_bytes = max_candidate_bytes;
        self.required_free_disk_bytes = required_free_disk_bytes;
        self
    }

    pub const fn max_source_bytes(&self) -> u64 {
        self.max_source_bytes
    }
    pub const fn max_staging_bytes(&self) -> u64 {
        self.max_staging_bytes
    }
    pub const fn max_candidate_bytes(&self) -> u64 {
        self.max_candidate_bytes
    }
    pub const fn required_free_disk_bytes(&self) -> u64 {
        self.required_free_disk_bytes
    }
    pub const fn max_csv_record_bytes(&self) -> usize {
        self.max_csv_record_bytes
    }
    pub const fn max_csv_field_bytes(&self) -> usize {
        self.max_csv_field_bytes
    }
    pub const fn max_csv_field_count(&self) -> usize {
        self.max_csv_field_count
    }
    pub const fn max_json_nesting(&self) -> usize {
        self.max_json_nesting
    }
    pub const fn max_json_string_bytes(&self) -> usize {
        self.max_json_string_bytes
    }
    pub const fn max_json_object_bytes(&self) -> usize {
        self.max_json_object_bytes
    }
    pub const fn max_arrow_metadata_bytes(&self) -> usize {
        self.max_arrow_metadata_bytes
    }
    pub const fn max_arrow_message_bytes(&self) -> usize {
        self.max_arrow_message_bytes
    }
    pub const fn max_arrow_body_bytes(&self) -> usize {
        self.max_arrow_body_bytes
    }
    pub const fn max_arrow_batch_rows(&self) -> usize {
        self.max_arrow_batch_rows
    }
    pub const fn max_arrow_columns(&self) -> usize {
        self.max_arrow_columns
    }
    pub const fn max_arrow_batch_bytes(&self) -> usize {
        self.max_arrow_batch_bytes
    }
    pub const fn max_parquet_footer_bytes(&self) -> usize {
        self.max_parquet_footer_bytes
    }
    pub const fn max_parquet_page_header_bytes(&self) -> usize {
        self.max_parquet_page_header_bytes
    }
    pub const fn max_parquet_row_group_bytes(&self) -> u64 {
        self.max_parquet_row_group_bytes
    }
    pub const fn max_parquet_page_bytes(&self) -> u64 {
        self.max_parquet_page_bytes
    }
    pub const fn max_parquet_dictionary_bytes(&self) -> u64 {
        self.max_parquet_dictionary_bytes
    }
    pub const fn max_parquet_decompressed_bytes(&self) -> u64 {
        self.max_parquet_decompressed_bytes
    }
    pub const fn max_parquet_compression_ratio(&self) -> u64 {
        self.max_parquet_compression_ratio
    }
    pub const fn max_vortex_metadata_bytes(&self) -> usize {
        self.max_vortex_metadata_bytes
    }
    pub const fn max_vortex_chunk_bytes(&self) -> usize {
        self.max_vortex_chunk_bytes
    }
    pub const fn max_total_rows(&self) -> u64 {
        self.max_total_rows
    }

    pub fn required_peak_disk_bytes(&self, source_bytes: u64) -> Result<u64, ImportLimitError> {
        let total = u128::from(source_bytes)
            + u128::from(self.max_candidate_bytes)
            + u128::from(self.required_free_disk_bytes);
        self.check(
            ImportLimitKind::CheckedArithmetic,
            total,
            u128::from(u64::MAX),
        )?;
        Ok(total as u64)
    }

    fn check(
        &self,
        kind: ImportLimitKind,
        actual: u128,
        maximum: u128,
    ) -> Result<(), ImportLimitError> {
        if actual > maximum {
            return Err(ImportLimitError {
                kind,
                actual,
                maximum,
            });
        }
        Ok(())
    }

    pub fn check_source_bytes(&self, actual: u64) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::SourceBytes,
            actual.into(),
            self.max_source_bytes.into(),
        )
    }
    pub fn check_staging_bytes(&self, actual: u64) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::StagingBytes,
            actual.into(),
            self.max_staging_bytes.into(),
        )
    }

    pub fn check_candidate_bytes(&self, actual: u64) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::CandidateBytes,
            u128::from(actual),
            u128::from(self.max_candidate_bytes),
        )
    }
    pub fn check_csv_record_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::CsvRecordBytes,
            actual as u128,
            self.max_csv_record_bytes as u128,
        )
    }
    pub fn check_csv_field_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::CsvFieldBytes,
            actual as u128,
            self.max_csv_field_bytes as u128,
        )
    }
    pub fn check_csv_field_count(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::CsvFieldCount,
            actual as u128,
            self.max_csv_field_count as u128,
        )
    }
    pub fn check_json_nesting(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::JsonNesting,
            actual as u128,
            self.max_json_nesting as u128,
        )
    }
    pub fn check_json_string_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::JsonStringBytes,
            actual as u128,
            self.max_json_string_bytes as u128,
        )
    }
    pub fn check_json_object_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::JsonObjectBytes,
            actual as u128,
            self.max_json_object_bytes as u128,
        )
    }
    pub fn check_arrow_metadata_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ArrowMetadataBytes,
            actual as u128,
            self.max_arrow_metadata_bytes as u128,
        )
    }
    pub fn check_arrow_message_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ArrowMessageBytes,
            actual as u128,
            self.max_arrow_message_bytes as u128,
        )
    }
    pub fn check_arrow_body_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ArrowBodyBytes,
            actual as u128,
            self.max_arrow_body_bytes as u128,
        )
    }
    pub fn check_arrow_batch_rows(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ArrowBatchRows,
            actual as u128,
            self.max_arrow_batch_rows as u128,
        )
    }
    pub fn check_arrow_columns(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ArrowColumns,
            actual as u128,
            self.max_arrow_columns as u128,
        )
    }
    pub fn check_arrow_batch_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ArrowBatchBytes,
            actual as u128,
            self.max_arrow_batch_bytes as u128,
        )
    }
    pub fn check_parquet_footer_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ParquetFooterBytes,
            actual as u128,
            self.max_parquet_footer_bytes as u128,
        )
    }
    pub fn check_parquet_page_header_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ParquetPageHeaderBytes,
            actual as u128,
            self.max_parquet_page_header_bytes as u128,
        )
    }
    pub fn check_parquet_row_group_bytes(&self, actual: u64) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ParquetRowGroupBytes,
            actual.into(),
            self.max_parquet_row_group_bytes.into(),
        )
    }
    pub fn check_parquet_page_bytes(&self, actual: u64) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ParquetPageBytes,
            actual.into(),
            self.max_parquet_page_bytes.into(),
        )
    }
    pub fn check_parquet_dictionary_bytes(&self, actual: u64) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ParquetDictionaryBytes,
            actual.into(),
            self.max_parquet_dictionary_bytes.into(),
        )
    }
    pub fn check_parquet_decompressed_bytes(&self, actual: u64) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::ParquetDecompressedBytes,
            actual.into(),
            self.max_parquet_decompressed_bytes.into(),
        )
    }
    pub fn check_parquet_compression_ratio(
        &self,
        compressed: u64,
        uncompressed: u64,
    ) -> Result<(), ImportLimitError> {
        let ratio = if compressed == 0 {
            u128::from(uncompressed)
        } else {
            u128::from(uncompressed).div_ceil(u128::from(compressed))
        };
        self.check(
            ImportLimitKind::ParquetCompressionRatio,
            ratio,
            self.max_parquet_compression_ratio.into(),
        )
    }
    pub fn check_vortex_metadata_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::VortexMetadataBytes,
            actual as u128,
            self.max_vortex_metadata_bytes as u128,
        )
    }
    pub fn check_vortex_chunk_bytes(&self, actual: usize) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::VortexChunkBytes,
            actual as u128,
            self.max_vortex_chunk_bytes as u128,
        )
    }
    pub fn check_total_rows(&self, actual: u64) -> Result<(), ImportLimitError> {
        self.check(
            ImportLimitKind::TotalRows,
            actual.into(),
            self.max_total_rows.into(),
        )
    }

    pub fn checked_layout_bytes(
        &self,
        rows: usize,
        columns: usize,
        element_bytes: usize,
    ) -> Result<usize, ImportLimitError> {
        let value = rows
            .checked_mul(columns)
            .and_then(|value| value.checked_mul(element_bytes))
            .ok_or(ImportLimitError {
                kind: ImportLimitKind::CheckedArithmetic,
                actual: u128::MAX,
                maximum: self.max_arrow_batch_bytes as u128,
            })?;
        self.check_arrow_batch_bytes(value)?;
        Ok(value)
    }
}

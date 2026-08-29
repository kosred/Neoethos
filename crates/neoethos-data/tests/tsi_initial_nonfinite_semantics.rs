use vector_ta::indicators::tsi::{tsi_row_scalar_into, tsi_scalar};

#[test]
fn default_scalar_resumes_after_an_initial_nonfinite_bar_like_the_general_path() {
    let mut data = (0..96)
        .map(|i| 100.0 + i as f64 * 0.125 + ((i % 7) as f64 - 3.0) * 0.03125)
        .collect::<Vec<_>>();
    data[1] = f64::NAN;

    let mut general = vec![f64::NAN; data.len()];
    let default_api = unsafe {
        tsi_row_scalar_into(&data, 25, 13, 0, &mut general).unwrap();
        tsi_scalar(&data, 25, 13, 0).unwrap().values
    };

    assert!(
        general.iter().skip(38).any(|value| value.is_finite()),
        "the general TSI path must recover after the invalid input row"
    );
    for (index, (default_value, general_value)) in default_api.iter().zip(&general).enumerate() {
        assert_eq!(
            default_value.to_bits(),
            general_value.to_bits(),
            "default API and general TSI validity/value diverged at {index}"
        );
    }
}

//! Hybrid base encoding utility functions.

use std::iter::Product;
use super::algs::{AlgorithmError, Result}; // 引入你的错误枚举（路径需根据实际项目调整）

pub trait IntoF32 {
    /// Convert to f32, returning a Result to handle possible errors.
    fn into_f32(self) -> Result<f32>;
}

impl IntoF32 for u16 {
    fn into_f32(self) -> Result<f32> {
        self.try_into().map_err(|_| AlgorithmError::InvalidParameters("u16 转换为 f32 超出范围".to_string()))
    }
}

/// Calculate the total number of actions in a mixed radix encoding.
/// 
/// For example, [2, 3, 4] represents 2×3×4=24 actions.
pub(crate) fn dim_product<T>(uppers: &[T]) -> Result<usize>
where
    T: Copy + Clone + Product<T> + TryInto<usize>,
{
    uppers
        .iter()
        .copied()
        .product::<T>()
        .try_into()
        .map_err(|_| AlgorithmError::InvalidParameters("乘积超出 usize".into()))
}

/// Hybrid base encoding: Convert multi-dimensional action values to a single index.
/// 
/// # Error Scenarios
/// 1. Action value dimension does not match action space dimension (e.g., value.len()=2 but uppers.len()=3)
/// 2. Action value exceeds the upper limit of its corresponding dimension (e.g., value=5 but uppers=4, allowed range 0~3)
pub(crate) fn encode_mixed_radix<T>(value: &[T], uppers: &[T]) -> Result<usize>
where
    T: Copy + TryInto<usize>,
{
    if value.len() != uppers.len() {
        return Err(AlgorithmError::InvalidParameters(
            format!("维度不匹配 {} vs {}", value.len(), uppers.len()).into(),
        ));
    }
    let mut idx = 0;
    let mut base = 1;
    // 从最低维往最高维走
    for (&v, &u) in value.iter().zip(uppers).rev() {
        let v: usize = v.try_into().map_err(|_| AlgorithmError::InvalidParameters("v 越界".into()))?;
        let u: usize = u.try_into().map_err(|_| AlgorithmError::InvalidParameters("u 越界".into()))?;
        if v >= u {
            return Err(AlgorithmError::InvalidParameters(
                format!("v={} >= u={}", v, u).into(),
            ));
        }
        idx += v * base;
        base *= u;
    }
    Ok(idx)
}

/// Hybrid base decoding: Convert a single index back to multi-dimensional action values.
/// 
/// # Error Scenarios
/// 1. Index exceeds the total number of actions (e.g., index=24 when uppers=[2, 3, 4])
pub(crate) fn decode_mixed_radix<T>(mut index: usize, uppers: &[T]) -> Result<Vec<T>>
where
    T: Copy + TryFrom<usize> + TryInto<usize>,
{
    let mut vec = Vec::with_capacity(uppers.len());
    // 同样从最低维往最高维走
    for &u in uppers.iter().rev() {
        let u: usize = u.try_into().map_err(|_| AlgorithmError::InvalidParameters("u 越界".into()))?;
        let rem = index % u;
        vec.push(T::try_from(rem).map_err(|_| AlgorithmError::InvalidParameters("rem 越界".into()))?);
        index /= u;
    }
    vec.reverse(); // 因为我们是倒着 push 的
    Ok(vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_product() {
        assert_eq!(dim_product(&[2u16, 3, 4]), Ok(24));
        assert_eq!(dim_product(&[5u8, 6, 7]), Ok(210));
    }

    #[test]
    fn test_round_trip() {
        let uppers = vec![2u8, 3, 4];
        for idx in 0..24 {
            let dec = decode_mixed_radix(idx, &uppers).unwrap();
            let enc = encode_mixed_radix(&dec, &uppers).unwrap();
            assert_eq!(enc, idx, "round trip failed at {}", idx);
        }
    }

    #[test]
    fn test_encode_edge() {
        let uppers = vec![2u16, 3, 4];
        // 最低维最大合法值
        assert_eq!(encode_mixed_radix(&[0, 0, 3], &uppers), Ok(3));
        // 最高维最大合法值
        assert_eq!(encode_mixed_radix(&[1, 2, 3], &uppers), Ok(1*12 + 2*4 + 3));
    }

    #[test]
    fn test_decode_edge() {
        let uppers = vec![2u8, 3, 4];
        assert_eq!(decode_mixed_radix(23, &uppers), Ok(vec![1, 2, 3]));
        assert_eq!(decode_mixed_radix(0, &uppers), Ok(vec![0, 0, 0]));
    }
}
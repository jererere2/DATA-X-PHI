#![allow(dead_code)]

// src/tensor_utils.rs
// Carga de tensores usando `from_raw_parts` (opción 3b).
// Esta función es *unsafe* porque usa punteros crudos, pero el resto del
// código garantiza que los datos provienen de un mmap alineado y no se
// modifican mientras el tensor está en uso.

use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Tensor;
use std::mem;

/// Carga un tensor de tipo f32 desde un slice de bytes usando
/// `std::slice::from_raw_parts`. El shape se especifica como [usize; 2]
/// y se retorna un tensor 2D de Burn sin copias innecesarias.
pub fn load_tensor_raw_parts(
    raw_bytes: &[u8],
    shape: [usize; 2],
    device: &NdArrayDevice,
) -> Tensor<NdArray<f32>, 2> {
    let len_f32 = raw_bytes.len() / mem::size_of::<f32>();
    assert_eq!(
        len_f32,
        shape[0] * shape[1],
        "Tamaño del buffer no coincide con el shape"
    );
    // SAFETY: los bytes provienen de un mmap alineado y no se modifican
    // mientras el slice está en uso.
    let f32_slice: &[f32] = unsafe { std::slice::from_raw_parts(raw_bytes.as_ptr() as *const f32, len_f32) };
    Tensor::<NdArray<f32>, 1>::from_data(f32_slice, device).reshape(shape)
}

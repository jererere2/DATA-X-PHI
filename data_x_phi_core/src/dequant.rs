// src/dequant.rs
// ==============================================================================
// MOTOR DE DESCUANTIZACIÓN NATIVA BARE-METAL: Q4_K_M & Q6_K
// CÁLCULO VECTORIAL PARALELO EN AMD RYZEN 9 3900X (24 HILOS CON RAYON)
// LORD AZ12 SOVEREIGN CORE // NOUSRESEARCH HERMES-4.3-36B
// ==============================================================================

#![allow(dead_code)]

use rayon::prelude::*;

pub const QK_K: usize = 256;
pub const BLOCK_Q4_K_SIZE: usize = 144;
pub const BLOCK_Q6_K_SIZE: usize = 210;

/// Conversión IEEE 754 de half precision (f16 / u16) a single precision (f32).
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut m = mant;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3ff;
            let exp_f32 = (127 - 15 + 1 - e) as u32;
            f32::from_bits((sign << 31) | (exp_f32 << 23) | (m << 13))
        }
    } else if exp == 31 {
        let exp_f32 = 255u32;
        f32::from_bits((sign << 31) | (exp_f32 << 23) | (mant << 13))
    } else {
        let exp_f32 = (exp + 127 - 15) as u32;
        f32::from_bits((sign << 31) | (exp_f32 << 23) | (mant << 13))
    }
}

/// Extrae las escalas de 6 bits y mínimos de 6 bits de un bloque Q4_K.
#[inline(always)]
pub fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Descuantiza un super-bloque Q4_K (144 bytes) hacia un slice de 256 valores f32.
pub fn dequantize_block_q4_k(src: &[u8], dst: &mut [f32]) {
    debug_assert!(src.len() >= BLOCK_Q4_K_SIZE);
    debug_assert!(dst.len() >= QK_K);

    let d_raw = u16::from_le_bytes([src[0], src[1]]);
    let dmin_raw = u16::from_le_bytes([src[2], src[3]]);
    let d = f16_to_f32(d_raw);
    let dmin = f16_to_f32(dmin_raw);
    let scales = &src[4..16];
    let qs = &src[16..144];

    for j in 0..8 {
        let (sc, m) = get_scale_min_k4(j, scales);
        let dl = d * (sc as f32);
        let ml = dmin * (m as f32);

        let sub_offset = (j / 2) * 32;
        let dst_offset = j * 32;

        if j % 2 == 0 {
            for l in 0..32 {
                let q = qs[sub_offset + l] & 0x0F;
                dst[dst_offset + l] = dl * (q as f32) - ml;
            }
        } else {
            for l in 0..32 {
                let q = qs[sub_offset + l] >> 4;
                dst[dst_offset + l] = dl * (q as f32) - ml;
            }
        }
    }
}

/// Producto punto fusionado (Fused Vec-Dot) entre una fila Q4_K y un vector x (f32).
/// No expande la matriz a memoria: multiplica los pesos en el vuelo dentro de la caché L1.
pub fn vec_dot_q4_k(row_bytes: &[u8], x: &[f32]) -> f32 {
    let num_blocks = x.len() / QK_K;
    let mut total_sum = 0.0f32;

    for b in 0..num_blocks {
        let b_offset = b * BLOCK_Q4_K_SIZE;
        let x_offset = b * QK_K;
        let src = &row_bytes[b_offset..b_offset + BLOCK_Q4_K_SIZE];
        let x_sub = &x[x_offset..x_offset + QK_K];

        let d = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([src[2], src[3]]));
        let scales = &src[4..16];
        let qs = &src[16..144];

        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, scales);
            let dl = d * (sc as f32);
            let ml = dmin * (m as f32);

            let sub_offset = (j / 2) * 32;
            let dst_offset = j * 32;

            let mut block_sum = 0.0f32;
            if j % 2 == 0 {
                for l in 0..32 {
                    let q = qs[sub_offset + l] & 0x0F;
                    let w = dl * (q as f32) - ml;
                    block_sum += w * x_sub[dst_offset + l];
                }
            } else {
                for l in 0..32 {
                    let q = qs[sub_offset + l] >> 4;
                    let w = dl * (q as f32) - ml;
                    block_sum += w * x_sub[dst_offset + l];
                }
            }
            total_sum += block_sum;
        }
    }

    total_sum
}

/// Multiplicación Matriz-Vector paralela (y = W * x) para tensores Q4_K en 24 hilos.
/// W tiene dimensiones [num_rows, in_dim].
pub fn mat_vec_mul_q4_k(matrix_bytes: &[u8], num_rows: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(in_dim % QK_K, 0);
    debug_assert_eq!(y.len(), num_rows);

    let row_stride = (in_dim / QK_K) * BLOCK_Q4_K_SIZE;
    debug_assert!(matrix_bytes.len() >= num_rows * row_stride);

    y.par_iter_mut().enumerate().for_each(|(r, out_val)| {
        let row_start = r * row_stride;
        let row_slice = &matrix_bytes[row_start..row_start + row_stride];
        *out_val = vec_dot_q4_k(row_slice, x);
    });
}

/// Producto punto fusionado (Fused Vec-Dot) entre una fila Q6_K y un vector x (f32).
pub fn vec_dot_q6_k(row_bytes: &[u8], x: &[f32]) -> f32 {
    let num_blocks = x.len() / QK_K;
    let mut total_sum = 0.0f32;

    for b in 0..num_blocks {
        let b_offset = b * BLOCK_Q6_K_SIZE;
        let x_offset = b * QK_K;
        let src = &row_bytes[b_offset..b_offset + BLOCK_Q6_K_SIZE];
        let x_sub = &x[x_offset..x_offset + QK_K];

        let ql = &src[0..128];
        let qh = &src[128..192];
        let sc = unsafe { std::slice::from_raw_parts(src[192..208].as_ptr() as *const i8, 16) };
        let d = f16_to_f32(u16::from_le_bytes([src[208], src[209]]));

        let mut block_sum = 0.0f32;

        for half in 0..2 {
            let ql_half = &ql[half * 64..(half + 1) * 64];
            let qh_half = &qh[half * 32..(half + 1) * 32];
            let sc_half = &sc[half * 8..(half + 1) * 8];
            let y_offset = half * 128;

            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql_half[l + 0] & 0xF) | (((qh_half[l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 = ((ql_half[l + 32] & 0xF) | (((qh_half[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql_half[l + 0] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) as i8 - 32;

                let w1 = d * (sc_half[is + 0] as f32) * (q1 as f32);
                let w2 = d * (sc_half[is + 2] as f32) * (q2 as f32);
                let w3 = d * (sc_half[is + 4] as f32) * (q3 as f32);
                let w4 = d * (sc_half[is + 6] as f32) * (q4 as f32);

                block_sum += w1 * x_sub[y_offset + l + 0];
                block_sum += w2 * x_sub[y_offset + l + 32];
                block_sum += w3 * x_sub[y_offset + l + 64];
                block_sum += w4 * x_sub[y_offset + l + 96];
            }
        }
        total_sum += block_sum;
    }

    total_sum
}

/// Multiplicación Matriz-Vector paralela (y = W * x) para tensores Q6_K en 24 hilos.
pub fn mat_vec_mul_q6_k(matrix_bytes: &[u8], num_rows: usize, in_dim: usize, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(in_dim % QK_K, 0);
    debug_assert_eq!(y.len(), num_rows);

    let row_stride = (in_dim / QK_K) * BLOCK_Q6_K_SIZE;
    debug_assert!(matrix_bytes.len() >= num_rows * row_stride);

    y.par_iter_mut().enumerate().for_each(|(r, out_val)| {
        let row_start = r * row_stride;
        let row_slice = &matrix_bytes[row_start..row_start + row_stride];
        *out_val = vec_dot_q6_k(row_slice, x);
    });
}


/// Normalización Root-Mean-Square (RMSNorm) clásica de Transformers (Gemma / Llama / Hermes).
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v * inv_rms) * weight[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        assert_eq!(f16_to_f32(0), 0.0);
        let one = f16_to_f32(0x3C00);
        assert!((one - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_real_q4_k_matrix_multiply() {
        let gguf_path = "/run/media/administrator/DATA-X-PHI/hermes-4_3_36b-Q4_K_M.gguf";
        if let Ok(file) = std::fs::File::open(gguf_path) {
            let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
            let data_start = 6290368;
            let offset = 1098387456; // blk.0.attn_k.weight (dims: [1024, 5120])
            let matrix_bytes = &mmap[data_start + offset..];

            let x = vec![1.0f32; 5120];
            let mut y = vec![0.0f32; 1024];

            let t0 = std::time::Instant::now();
            mat_vec_mul_q4_k(matrix_bytes, 1024, 5120, &x, &mut y);
            let elapsed = t0.elapsed();

            println!("⚡ [TEST] Multiplicación Q4_K [1024, 5120] en 24 hilos completada en: {:?}", elapsed);
            println!("Muestra salida y[0..5]: {:?}", &y[..5]);

            assert!(y[0].is_finite());
            assert!(!y[0].is_nan());
        }
    }
}


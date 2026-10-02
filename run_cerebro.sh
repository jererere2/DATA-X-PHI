#!/usr/bin/env bash
# ==============================================================================
# SCRIPT DE LANZAMIENTO: CEREBRO CUÁNTICO SODAX / ARCHON-G - DATA (V1.7)
# MODELO: NousResearch/Hermes-4.3-36B-GGUF (21.7 GB)
# ARQUITECTURA: 128,256 Nodos Conceptuales (Zero-Token Bare-Metal Rust)
# HARDWARE: AMD Ryzen 9 3900X (24 Hilos) | AMD Radeon RX Vega 64 (HBM2)
# ==============================================================================

set -e

echo "🚀 Iniciando Cerebro Cuántico SODAX (Hermes-4.3-36B Bare-Metal OOC)..."

export OPENBLAS_NUM_THREADS=24
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export RAYON_NUM_THREADS=24

BINARY_DIR="/run/media/administrator/DATA-X-PHI/data_x_phi_core"
RELEASE_DIR="$BINARY_DIR/target/release"
BINARY="$RELEASE_DIR/data_x_phi_core"

export LD_LIBRARY_PATH="$RELEASE_DIR:$RELEASE_DIR/build/openblas-src-3c0f64fe07461f49/out/OpenBLAS-0.3.32:${LD_LIBRARY_PATH}"

echo "⚡ Ejecutando núcleo nativo en puerto 3030..."
cd "$BINARY_DIR"
exec "$BINARY"

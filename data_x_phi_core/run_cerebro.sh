#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
echo "🔥 Ensamblando el núcleo en el procesador Ryzen..."
nix-shell --run "cargo build --release && ./target/release/data_x_phi_core"

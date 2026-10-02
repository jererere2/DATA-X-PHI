{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "data-x-phi-rust-env";

  buildInputs = with pkgs; [
    rustc cargo gcc pkg-config openblas openssl
  ];

  shellHook = ''
    export RUSTFLAGS="-C target-cpu=native"
    export OPENBLAS_NUM_THREADS="24"
    export OMP_NUM_THREADS="24"
  '';
}

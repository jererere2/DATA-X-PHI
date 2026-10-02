// src/main.rs
// ==============================================================================
// CEREBRO CUÁNTICO SODAX / ARCHON-G - NÚCLEO DATA (V1.7)
// MODELO SOBERANO: NousResearch/Hermes-4.3-36B-GGUF (21.7 GB)
// TOPOLOGÍA: 128,256 Nodos Conceptuales (Sin Tokenizadores Clásicos)
// HARDWARE: AMD Ryzen 9 3900X (24 Hilos) | AMD Radeon RX Vega 64 (HBM2)
// LORD AZ12 SOVEREIGN CONTROLLER
// ==============================================================================

use std::sync::{Arc, Mutex};
mod server;
mod forward_pass;
mod tensor_utils;
pub mod dequant;

#[tokio::main]
async fn main() {
    println!("\n==================================================================");
    println!("  CEREBRO CUÁNTICO SODAX / ARCHON-G - NÚCLEO DATA (V1.7)");
    println!("  Modelo Soberano: NousResearch/Hermes-4.3-36B-GGUF (21.7 GB)");
    println!("  Topología: 128,256 Nodos Conceptuales (Cero Tokenizadores Clásicos)");
    println!("  Hardware: AMD Ryzen 9 3900X (24 Hilos) | NVMe LUKS2/XFS");
    println!("  GPU Acelerador: AMD Radeon RX Vega 64 (56 CUs / Vulkan Wave64)");
    println!("  Lord AZ12 System Controller");
    println!("==================================================================");

    // Inicializar el motor Bare-Metal Hermes 36B Out-Of-Core
    println!("⚡ Inicializando proyección Out-Of-Core de Hermes 36B GGUF...");
    let engine = match forward_pass::HermesOocEngine::new() {
        Ok(eng) => {
            println!("✅ Motor Hermes 36B OOC cargado exitosamente (21.7 GB mapeados en memoria zero-copy).");
            Arc::new(eng)
        }
        Err(e) => {
            eprintln!("❌ Error crítico cargando Hermes 36B GGUF: {}", e);
            std::process::exit(1);
        }
    };

    // Cargar estado de memoria persistente si existe
    let state_file = "/run/media/administrator/DATA-X-PHI/state.json";
    let initial_state = if let Ok(content) = std::fs::read_to_string(state_file) {
        match serde_json::from_str::<server::State>(&content) {
            Ok(mut st) => {
                st.model_id = "NousResearch/Hermes-4.3-36B-GGUF (Bare-Metal OOC / Vulkan Wave64)".into();
                println!("🧠 Memoria persistente recuperada: {} interacciones registradas.", st.history.len());
                st
            }
            Err(_) => server::State::default(),
        }
    } else {
        println!("✨ Espacio sináptico inicializado para DATA y Lord AZ12.");
        server::State::default()
    };

    let shared_state = Arc::new(Mutex::new(initial_state));

    let ctx = server::AppContext {
        state: shared_state,
        engine,
    };

    println!("🌐 Consola Web lista en: http://127.0.0.1:3030");
    println!("⚡ Servidor DATA activo en http://127.0.0.1:3030");
    println!("Presiona Ctrl+C en cualquier momento para detener.\n");

    server::run(ctx).await;
}

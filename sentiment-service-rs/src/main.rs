//! Slim native-TensorRT sentiment service (Rust). Drop-in HTTP-compatible replacement
//! for the Python service's native-TensorRT serving path, with no PyTorch/transformers
//! in the runtime image. See README.md.

mod config;
mod engine;
mod inference;
mod routes;
mod schemas;
#[cfg(test)]
mod test_support;
mod tokenizer;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};

use config::Config;
use inference::Engine;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();
    tracing::info!(
        engine = %config.engine_path,
        tokenizer = %config.tokenizer_path,
        port = config.port,
        "starting sentiment-service-rs"
    );

    let tokenizer = tokenizer::load_or_fail(
        &config.tokenizer_path,
        config.max_length,
        config.pad_to_multiple_of,
    )?;

    let backend = engine::build(&config.engine_path, config.force_stub)
        .context("failed to initialize inference backend")?;
    tracing::info!(backend = backend.name(), "inference backend ready");

    let port = config.port;
    let engine = Arc::new(Engine::new(config, tokenizer, backend));
    let app = routes::router(engine);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    tracing::info!(%addr, "listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server error")?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received");
}

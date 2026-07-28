## Multi-stage build — compiles cs-stratum-bridge (Rust) and ships a slim
## runtime image. Mirrors the same "old glibc floor" reasoning cs-miner's
## hive/release.sh uses (see that file's header comment): building against
## an older Debian base keeps the binary loadable on whatever runtime image
## it ends up in, even if that drifts from the build image over time.
FROM rust:1-bookworm AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# sqlx::migrate!("./migrations") in src/main.rs is a COMPILE-TIME macro —
# it reads this directory while building, not just at runtime.
COPY migrations ./migrations

RUN cargo build --release --bin cs-stratum-bridge

## Runtime image. debian:bookworm-slim (not distroless/alpine): the
## postgresql_embedded crate downloads and RUNS a real Postgres server
## binary at startup (see config.rs's EMBEDDED_DB_DATA_DIR doc comment) —
## it needs a normal glibc userland, not a static/musl one. ca-certificates
## is required for every outbound HTTPS call this service makes (GGUF/
## tokenizer downloads it may proxy, ADMIN_REPORT_URL POSTs, etc.).
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/cs-stratum-bridge /usr/local/bin/cs-stratum-bridge

# EMBEDDED_DB_DATA_DIR / AUDITOR_CACHE_DIR default to relative paths
# (./bridge_pgdata, ./auditor_cache) — anchor them under /app so the
# docker-compose volume mounts below land in a predictable place.
ENV EMBEDDED_DB_DATA_DIR=/app/bridge_pgdata
ENV AUDITOR_CACHE_DIR=/app/auditor_cache

EXPOSE 3532 3533

ENTRYPOINT ["/usr/local/bin/cs-stratum-bridge"]

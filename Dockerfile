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

## libgssapi-krb5-2/libxml2 aren't linked by cs-stratum-bridge itself — they're
## runtime deps of the prebuilt embedded-Postgres binaries this crate
## downloads to /root/.theseus at startup (initdb/postgres/libpq.so.5 all
## link libgssapi_krb5.so.2; postgres also links libxml2.so.2). Found via
## `ldd` against the actual downloaded binaries after a plain libssl3 image
## failed with "libgssapi_krb5.so.2: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libgssapi-krb5-2 \
        libxml2 \
    && rm -rf /var/lib/apt/lists/*

## The embedded-Postgres binaries this crate manages (initdb/postgres) hard-
## refuse to run as root ("cannot be run as root") — the whole container
## needs to run as a non-root user. `setpriv` (in docker-entrypoint.sh)
## drops from root to this user right before exec'ing the real binary, but
## does NOT touch $HOME on its own — HOME must be set explicitly too, or
## the crate keeps resolving its download cache to /root/.theseus (owned by
## root, unwritable by `bridge`) instead of a path this user can write to.
RUN groupadd --system --gid 10001 bridge \
    && useradd --system --uid 10001 --gid bridge --home-dir /app --no-create-home bridge
ENV HOME=/app

WORKDIR /app
COPY --from=builder /build/target/release/cs-stratum-bridge /usr/local/bin/cs-stratum-bridge
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && chown -R bridge:bridge /app

# EMBEDDED_DB_DATA_DIR / AUDITOR_CACHE_DIR default to relative paths
# (./bridge_pgdata, ./auditor_cache) — anchor them under /app so the
# docker-compose volume mounts below land in a predictable place.
ENV EMBEDDED_DB_DATA_DIR=/app/bridge_pgdata
ENV AUDITOR_CACHE_DIR=/app/auditor_cache

EXPOSE 3532 3533

# Stays root at container start (needed to chown bind-mounted volumes) —
# docker-entrypoint.sh drops to the `bridge` user before ever exec'ing the
# real binary.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

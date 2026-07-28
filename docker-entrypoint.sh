#!/bin/sh
set -e

# The embedded-Postgres binaries this crate downloads at startup (see
# postgresql_embedded in main.rs) hard-refuse to run as root ("initdb:
# cannot be run as root") — the whole container must run as a non-root
# user for the embedded DB to ever start. Bind-mounted host dirs
# (EMBEDDED_DB_DATA_DIR/AUDITOR_CACHE_DIR) may still be root-owned from a
# fresh `docker compose up` on the host, so fix ownership here (as root,
# before dropping privileges) instead of requiring an operator to `chown`
# the host paths by hand before every deploy.
chown -R bridge:bridge "${EMBEDDED_DB_DATA_DIR:-/app/bridge_pgdata}" "${AUDITOR_CACHE_DIR:-/app/auditor_cache}" 2>/dev/null || true

exec setpriv --reuid bridge --regid bridge --clear-groups /usr/local/bin/cs-stratum-bridge "$@"

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

# A container killed/restarted without a clean shutdown leaves a stale
# postmaster.pid in the persisted data dir. In a fresh container PID
# namespace the recorded PID often belongs to an unrelated live process
# (even the bridge itself), so pg_ctl refuses: "another server might be
# running ... could not start server" -> crash loop forever. This container
# is the sole owner of the data dir, so a leftover pid/lock file is always
# stale at this point; remove it.
rm -f "${EMBEDDED_DB_DATA_DIR:-/app/bridge_pgdata}/postmaster.pid"

# The embedded-Postgres release currently downloaded (18.6.1) ships an
# incomplete share/timezone: UTC, GMT and Etc/UTC are missing (links dropped
# on extraction), but initdb wrote "Etc/UTC" into postgresql.conf, so
# Postgres FATALs ("invalid value for parameter TimeZone") and never starts.
# GMT+0 is present in that tree and is equivalent (UTC offset 0).
PGCONF="${EMBEDDED_DB_DATA_DIR:-/app/bridge_pgdata}/postgresql.conf"
if [ -f "$PGCONF" ]; then
  sed -i -E "s#^([[:space:]]*(log_)?timezone[[:space:]]*=[[:space:]]*)'?Etc/UTC'?#\\1'GMT+0'#" "$PGCONF"
fi

exec setpriv --reuid bridge --regid bridge --clear-groups /usr/local/bin/cs-stratum-bridge "$@"

# cs-stratum-bridge

Bridge/adapter between stratum-speaking `cs-miner` clients and the `csd`
daemon — same pattern as
[kaspa-stratum-bridge](https://github.com/onemorebsmith/kaspa-stratum-bridge).

`cs-miner` connects only to this bridge. PoW `mining.*` traffic is relayed
through, unchanged, to the real upstream pool (`back-pool`). OPoI traffic
(`opoi.assign` / `opoi.submit_result`) is intercepted and owned entirely by
this service, which talks directly to `csd`'s JSON-RPC to drive the
COMMIT → REVEAL → publish → payout lifecycle on-chain.

See `.env.example` for configuration. Copy to `.env` (never commit it) and
fill in real values — the OPoI stake/payout address's private key must
already be imported into the `csd` node's own wallet; this service never
holds key material itself.

## Status

v1 in progress — see the implementation plan for the build/verify sequence.

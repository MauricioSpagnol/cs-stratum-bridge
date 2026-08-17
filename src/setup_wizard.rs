//! First-run interactive setup wizard.
//!
//! `Config::load()` (`config.rs`) requires several env vars with no
//! default — today, a missing one makes `clap` abort with a generic CLI
//! usage message. Separately, even with every `clap`-required var set, the
//! process can still die minutes later with "no OPoI stake found ... and no
//! OPOI_COLLATERAL_TXID configured" (the daemon-side stake collateral isn't
//! `clap`-required, but is required at runtime). `ensure_configured()` runs
//! before `Config::load()` and closes both gaps at once: if everything
//! needed is already set (via `.env` or the process environment), it's a
//! silent no-op; otherwise it offers a manual path (exit so the operator
//! edits `.env` by hand) or an automatic one (prompt for each value, test
//! the csd RPC connection, and offer to pick a UTXO for the stake
//! collateral from `listunspent`).

use std::io::{self, BufRead, Read, Write};
use std::path::Path;

use crate::rpc::CsdRpcClient;

struct RequiredVar {
    key: &'static str,
    prompt: &'static str,
    suggested_default: Option<&'static str>,
}

const REQUIRED_VARS: &[RequiredVar] = &[
    RequiredVar {
        key: "UPSTREAM_POOL_ADDR",
        prompt: "Address (host:port) of the real back-pool — all mining.* traffic is relayed there",
        suggested_default: None,
    },
    RequiredVar {
        key: "CSD_RPC_URL",
        prompt: "csd RPC URL",
        suggested_default: Some("http://127.0.0.1:26124"),
    },
    RequiredVar {
        key: "CSD_RPC_USER",
        prompt: "csd RPC username (rpcuser in cs.conf)",
        suggested_default: None,
    },
    RequiredVar {
        key: "CSD_RPC_PASS",
        prompt: "csd RPC password (rpcpassword in cs.conf)",
        suggested_default: None,
    },
    RequiredVar {
        key: "OPOI_ADDRESSES",
        prompt: "CS address that will hold the OPoI stake (its private key must already be imported into the csd wallet)",
        suggested_default: None,
    },
    RequiredVar {
        key: "OPOI_REQUESTER_API_KEY",
        prompt: "API key for whoever will submit prompts (x-opoi-api-key header)",
        suggested_default: None,
    },
];

const ENV_PATH: &str = ".env";
const ENV_EXAMPLE_PATH: &str = ".env.example";

/// Entry point, called from `main()` before `Config::load()`. A silent
/// no-op when every required var is already set — the happy path for an
/// operator who already has a working `.env` is unchanged.
pub async fn ensure_configured() {
    dotenvy::dotenv().ok();

    let missing: Vec<&RequiredVar> = REQUIRED_VARS.iter().filter(|v| !is_set(v.key)).collect();
    if missing.is_empty() {
        // Every clap-required var is set, but that alone doesn't guarantee
        // the bridge can actually START — see ensure_stake_configured's doc.
        ensure_stake_configured().await;
        return;
    }

    println!();
    println!("cs-stratum-bridge: incomplete configuration — missing {} variable(s):", missing.len());
    for v in &missing {
        println!("  - {}", v.key);
    }
    println!();
    println!("How would you like to configure it?");
    println!("  1) Manual — I'll exit here, you edit {ENV_PATH} and run again");
    println!("  2) Automatic — I'll ask for each value, test the connection to csd, and show the available UTXOs for the stake");
    print!("> ");
    io::stdout().flush().ok();

    match read_line().trim() {
        "2" => run_automatic().await,
        _ => run_manual(),
    }
}

fn is_set(key: &str) -> bool {
    std::env::var(key).map(|v| !v.trim().is_empty()).unwrap_or(false)
}

fn read_line() -> String {
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).ok();
    line.trim_end_matches(['\n', '\r']).to_string()
}

fn run_manual() {
    println!();
    let env_exists = Path::new(ENV_PATH).exists();
    if !env_exists && Path::new(ENV_EXAMPLE_PATH).exists() {
        match std::fs::copy(ENV_EXAMPLE_PATH, ENV_PATH) {
            Ok(_) => println!("Created {ENV_PATH} from {ENV_EXAMPLE_PATH} — fill in the real values and run again."),
            Err(e) => println!("Couldn't copy {ENV_EXAMPLE_PATH} to {ENV_PATH} ({e}) — create the file by hand and run again."),
        }
    } else if env_exists {
        println!("{ENV_PATH} already exists but is missing the variable(s) above — edit it and run again.");
    } else {
        println!("Create a {ENV_PATH} file in this folder with the variable(s) above and run again.");
    }
    std::process::exit(1);
}

async fn run_automatic() {
    let mut collected: Vec<(String, String)> = Vec::new();

    for v in REQUIRED_VARS {
        if is_set(v.key) {
            continue; // already set via process env — don't re-ask
        }
        let value = if v.key == "OPOI_REQUESTER_API_KEY" { prompt_api_key(v) } else { prompt_value(v) };
        set_collected(&mut collected, v.key, value);
    }

    let mut connected = false;
    loop {
        let rpc_url = std::env::var("CSD_RPC_URL").unwrap_or_default();
        let rpc_user = std::env::var("CSD_RPC_USER").unwrap_or_default();
        let rpc_pass = std::env::var("CSD_RPC_PASS").unwrap_or_default();
        let client = CsdRpcClient::new(rpc_url.clone(), rpc_user, rpc_pass);

        println!();
        println!("Testing connection to csd at {rpc_url}...");
        match client.get_chain_height().await {
            Ok(height) => {
                println!("OK — csd responded, current height: {height}");
                connected = true;
                break;
            }
            Err(e) => {
                println!("Connection failed: {e}");
                print!("Try again with different CSD_RPC_URL/USER/PASS values? (y/n) ");
                io::stdout().flush().ok();
                if read_line().trim().eq_ignore_ascii_case("y") {
                    for key in ["CSD_RPC_URL", "CSD_RPC_USER", "CSD_RPC_PASS"] {
                        let value = prompt_value(var_by_key(key));
                        set_collected(&mut collected, key, value);
                    }
                } else {
                    println!(
                        "Continuing without validating the connection — configure OPOI_COLLATERAL_TXID/OPOI_COLLATERAL_VOUT in {ENV_PATH} by hand later, if needed."
                    );
                    break;
                }
            }
        }
    }

    if connected {
        let rpc_url = std::env::var("CSD_RPC_URL").unwrap_or_default();
        let rpc_user = std::env::var("CSD_RPC_USER").unwrap_or_default();
        let rpc_pass = std::env::var("CSD_RPC_PASS").unwrap_or_default();
        let client = CsdRpcClient::new(rpc_url, rpc_user, rpc_pass);
        let primary = primary_opoi_address();
        maybe_offer_stake_picker(&client, &primary, &mut collected).await;
    }

    write_env_file(&collected);
    println!();
    println!("{ENV_PATH} saved. Continuing...");
    println!();
}

fn var_by_key(key: &str) -> &'static RequiredVar {
    REQUIRED_VARS.iter().find(|v| v.key == key).expect("unknown required var key")
}

/// First (primary) address in the comma-separated OPOI_ADDRESSES — the one
/// `opoi/engine.rs::ensure_stake` auto-bootstraps from
/// OPOI_COLLATERAL_TXID/_VOUT. Empty if OPOI_ADDRESSES itself is unset.
fn primary_opoi_address() -> String {
    std::env::var("OPOI_ADDRESSES")
        .unwrap_or_default()
        .split(',')
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Checks whether `primary` already has an on-chain OPoI stake before
/// bothering the operator with the UTXO picker — used by `run_automatic`,
/// right after a freshly-collected OPOI_ADDRESSES, where a brand-new
/// address almost always needs one but might occasionally already have
/// been staked out-of-band.
async fn maybe_offer_stake_picker(client: &CsdRpcClient, primary: &str, collected: &mut Vec<(String, String)>) {
    if primary.is_empty() {
        return;
    }
    match client.get_opoi_stake(primary).await {
        Ok(Some(_)) => {
            println!();
            println!("Address {primary} already has an active OPoI stake — no new collateral needed.");
        }
        Ok(None) => offer_stake_utxo_picker(client, collected).await,
        Err(e) => {
            println!();
            println!("Couldn't check whether {primary} already has a stake ({e}) — showing the available UTXOs anyway, in case you want to configure collateral.");
            offer_stake_utxo_picker(client, collected).await;
        }
    }
}

/// Second gap this wizard closes: even with every `clap`-required var set
/// (so `ensure_configured`'s main check finds nothing missing), the bridge
/// still can't actually START without either an ACTIVE on-chain OPoI stake
/// for its primary address, or OPOI_COLLATERAL_TXID/_VOUT to create one —
/// `opoi/engine.rs::ensure_stake` bails with "no OPoI stake found ... and no
/// OPOI_COLLATERAL_TXID configured" otherwise (confirmed live: an operator
/// with a fully clap-valid `.env` still hit exactly this on startup). Runs
/// once at the end of `ensure_configured` whenever the required-var check
/// alone found nothing to do, since that's exactly the case this can't see.
async fn ensure_stake_configured() {
    if is_set("OPOI_COLLATERAL_TXID") {
        return; // already configured — ensure_stake will use it, or the stake already exists
    }
    let primary = primary_opoi_address();
    if primary.is_empty() {
        return; // OPOI_ADDRESSES itself is unset — Config::load() will already fail on this
    }

    let rpc_url = std::env::var("CSD_RPC_URL").unwrap_or_default();
    let rpc_user = std::env::var("CSD_RPC_USER").unwrap_or_default();
    let rpc_pass = std::env::var("CSD_RPC_PASS").unwrap_or_default();
    let client = CsdRpcClient::new(rpc_url, rpc_user, rpc_pass);

    let stake = match client.get_opoi_stake(&primary).await {
        Ok(s) => s,
        Err(_) => return, // can't reach csd right now — let normal startup surface the real error
    };
    if stake.is_some() {
        return; // already staked — nothing to do
    }

    println!();
    println!(
        "OPoI address {primary} doesn't have an active stake yet, and OPOI_COLLATERAL_TXID isn't set in {ENV_PATH} — without it, cs-stratum-bridge can't start."
    );
    print!("Pick a UTXO now as the stake collateral? (y/n) ");
    io::stdout().flush().ok();
    if !read_line().trim().eq_ignore_ascii_case("y") {
        return;
    }

    let mut collected: Vec<(String, String)> = Vec::new();
    offer_stake_utxo_picker(&client, &mut collected).await;
    if !collected.is_empty() {
        patch_env_file(&collected);
    }
}

fn set_collected(collected: &mut Vec<(String, String)>, key: &str, value: String) {
    std::env::set_var(key, &value);
    if let Some(entry) = collected.iter_mut().find(|(k, _)| k == key) {
        entry.1 = value;
    } else {
        collected.push((key.to_string(), value));
    }
}

fn prompt_value(v: &RequiredVar) -> String {
    loop {
        println!();
        println!("{}: {}", v.key, v.prompt);
        match v.suggested_default {
            Some(default) => print!("[{default}] > "),
            None => print!("> "),
        }
        io::stdout().flush().ok();

        let input = read_line();
        let value = if input.trim().is_empty() {
            v.suggested_default.unwrap_or_default().to_string()
        } else {
            input.trim().to_string()
        };
        if !value.is_empty() {
            return value;
        }
        println!("This value is required, try again.");
    }
}

fn prompt_api_key(v: &RequiredVar) -> String {
    println!();
    println!("{}: {}", v.key, v.prompt);
    print!("[Press Enter to generate a random key] > ");
    io::stdout().flush().ok();

    let input = read_line();
    if input.trim().is_empty() {
        let key = generate_random_hex(32);
        println!("Generated: {key}");
        key
    } else {
        input.trim().to_string()
    }
}

/// Reads `num_bytes` from `/dev/urandom` and hex-encodes them — Linux-only,
/// same assumption the rest of this codebase already makes (HiveOS rigs,
/// systemd units). Falls back to a low-quality timestamp-derived value in
/// the (practically impossible on Linux) case `/dev/urandom` can't be
/// opened, rather than panicking the wizard over a non-critical value the
/// operator can always overwrite by hand later.
fn generate_random_hex(num_bytes: usize) -> String {
    let mut buf = vec![0u8; num_bytes];
    let ok = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut buf)).is_ok();
    if !ok {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((nanos >> ((i % 8) * 8)) & 0xff) as u8;
        }
    }
    hex::encode(buf)
}

async fn offer_stake_utxo_picker(client: &CsdRpcClient, collected: &mut Vec<(String, String)>) {
    println!();
    println!("Fetching UTXOs from the csd wallet to use as stake collateral...");
    match client.list_unspent().await {
        Ok(utxos) if !utxos.is_empty() => {
            println!();
            println!("{:<4} {:<66} {:<5} {:>14} {:>6}  {}", "#", "txid", "vout", "amount (CS)", "conf", "address");
            for (i, u) in utxos.iter().enumerate() {
                println!(
                    "{:<4} {:<66} {:<5} {:>14.8} {:>6}  {}",
                    i + 1,
                    u.txid,
                    u.vout,
                    u.amount,
                    u.confirmations,
                    u.address
                );
            }
            println!();
            print!("UTXO number to use as stake collateral (Enter to skip): ");
            io::stdout().flush().ok();

            let choice = read_line();
            match choice.trim().parse::<usize>() {
                Ok(idx) if idx >= 1 && idx <= utxos.len() => {
                    let picked = &utxos[idx - 1];
                    set_collected(collected, "OPOI_COLLATERAL_TXID", picked.txid.clone());
                    set_collected(collected, "OPOI_COLLATERAL_VOUT", picked.vout.to_string());
                    println!("Using {}:{} as stake collateral.", picked.txid, picked.vout);
                }
                _ => {
                    println!(
                        "Skipped — without collateral the OPoI stake won't activate on its own. Set OPOI_COLLATERAL_TXID/OPOI_COLLATERAL_VOUT in {ENV_PATH} once you have a UTXO available."
                    );
                }
            }
        }
        Ok(_) => {
            println!(
                "No UTXOs available yet in this wallet — the OPoI stake won't activate on its own until this wallet has a balance. Set OPOI_COLLATERAL_TXID/OPOI_COLLATERAL_VOUT in {ENV_PATH} once you have one."
            );
        }
        Err(e) => {
            println!("Couldn't list UTXOs ({e}) — set OPOI_COLLATERAL_TXID/OPOI_COLLATERAL_VOUT in {ENV_PATH} by hand later.");
        }
    }
}

/// Writes `.env` by copying `.env.example` line by line and substituting
/// only the `KEY=...` lines whose key was collected this run — every
/// other line (comments, untouched defaults) survives unchanged. Falls
/// back to a bare `KEY=value` dump if `.env.example` isn't present.
fn write_env_file(collected: &[(String, String)]) {
    let contents = match std::fs::read_to_string(ENV_EXAMPLE_PATH) {
        Ok(example) => {
            let mut out = String::with_capacity(example.len());
            for line in example.lines() {
                let replaced = line.find('=').and_then(|eq| {
                    let key = &line[..eq];
                    collected.iter().find(|(k, _)| k == key).map(|(k, v)| format!("{k}={v}"))
                });
                out.push_str(&replaced.unwrap_or_else(|| line.to_string()));
                out.push('\n');
            }
            out
        }
        Err(_) => collected.iter().map(|(k, v)| format!("{k}={v}\n")).collect(),
    };

    if let Err(e) = std::fs::write(ENV_PATH, contents) {
        println!("Couldn't save {ENV_PATH} ({e}) — the collected variables only exist in this process's environment.");
    }
}

/// Updates specific keys in an ALREADY-EXISTING `.env` in place — unlike
/// `write_env_file` (which regenerates the whole file from
/// `.env.example`), this preserves every other line exactly as the
/// operator left it, including any custom values on optional fields.
/// Used by `ensure_stake_configured`, where `.env` is already a real,
/// working config the wizard must not otherwise disturb. Appends any key
/// not already present as a fallback (shouldn't happen for
/// OPOI_COLLATERAL_TXID/_VOUT, which `.env.example` always lists).
fn patch_env_file(pairs: &[(String, String)]) {
    let existing = match std::fs::read_to_string(ENV_PATH) {
        Ok(s) => s,
        Err(e) => {
            println!("Couldn't read {ENV_PATH} ({e}) — the collected variables only exist in this process's environment.");
            return;
        }
    };

    let mut applied = vec![false; pairs.len()];
    let mut out = String::with_capacity(existing.len() + 64);
    for line in existing.lines() {
        let replacement = line.find('=').and_then(|eq| {
            let key = &line[..eq];
            pairs.iter().position(|(k, _)| k == key)
        });
        match replacement {
            Some(idx) => {
                out.push_str(&format!("{}={}", pairs[idx].0, pairs[idx].1));
                applied[idx] = true;
            }
            None => out.push_str(line),
        }
        out.push('\n');
    }
    for (idx, (k, v)) in pairs.iter().enumerate() {
        if !applied[idx] {
            out.push_str(&format!("{k}={v}\n"));
        }
    }

    match std::fs::write(ENV_PATH, out) {
        Ok(_) => println!("{ENV_PATH} updated."),
        Err(e) => println!("Couldn't save {ENV_PATH} ({e}) — the collected variables only exist in this process's environment."),
    }
}

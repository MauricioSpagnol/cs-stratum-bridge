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
        prompt: "Endereço (host:port) do back-pool real — todo tráfego mining.* é repassado pra lá",
        suggested_default: None,
    },
    RequiredVar {
        key: "CSD_RPC_URL",
        prompt: "URL do RPC do csd",
        suggested_default: Some("http://127.0.0.1:26124"),
    },
    RequiredVar {
        key: "CSD_RPC_USER",
        prompt: "Usuário do RPC do csd (rpcuser no cs.conf)",
        suggested_default: None,
    },
    RequiredVar {
        key: "CSD_RPC_PASS",
        prompt: "Senha do RPC do csd (rpcpassword no cs.conf)",
        suggested_default: None,
    },
    RequiredVar {
        key: "OPOI_ADDRESSES",
        prompt: "Endereço CS que vai fazer a stake OPoI (a chave privada já precisa estar importada na wallet do csd)",
        suggested_default: None,
    },
    RequiredVar {
        key: "OPOI_REQUESTER_API_KEY",
        prompt: "Chave de API pra quem for enviar prompts usar (header x-opoi-api-key)",
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
    println!("cs-stratum-bridge: configuração incompleta — falta(m) {} variável(is):", missing.len());
    for v in &missing {
        println!("  - {}", v.key);
    }
    println!();
    println!("Como prefere configurar?");
    println!("  1) Manual — fecho aqui, você edita o {ENV_PATH} e roda de novo");
    println!("  2) Automático — eu pergunto cada valor, testo a conexão com o csd, e mostro as UTXOs disponíveis pra stake");
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
            Ok(_) => println!("Criei {ENV_PATH} a partir de {ENV_EXAMPLE_PATH} — preencha os valores reais e rode de novo."),
            Err(e) => println!("Não consegui copiar {ENV_EXAMPLE_PATH} pra {ENV_PATH} ({e}) — crie o arquivo manualmente e rode de novo."),
        }
    } else if env_exists {
        println!("{ENV_PATH} já existe mas está faltando a(s) variável(is) acima — edite-o e rode de novo.");
    } else {
        println!("Crie um arquivo {ENV_PATH} nesta pasta com a(s) variável(is) acima e rode de novo.");
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
        println!("Testando conexão com o csd em {rpc_url}...");
        match client.get_chain_height().await {
            Ok(height) => {
                println!("OK — csd respondeu, altura atual: {height}");
                connected = true;
                break;
            }
            Err(e) => {
                println!("Falha ao conectar: {e}");
                print!("Tentar de novo com outros dados de CSD_RPC_URL/USER/PASS? (s/n) ");
                io::stdout().flush().ok();
                if read_line().trim().eq_ignore_ascii_case("s") {
                    for key in ["CSD_RPC_URL", "CSD_RPC_USER", "CSD_RPC_PASS"] {
                        let value = prompt_value(var_by_key(key));
                        set_collected(&mut collected, key, value);
                    }
                } else {
                    println!(
                        "Seguindo sem validar a conexão — configure OPOI_COLLATERAL_TXID/OPOI_COLLATERAL_VOUT no {ENV_PATH} manualmente depois, se precisar."
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
    println!("{ENV_PATH} salvo. Continuando...");
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
            println!("Endereço {primary} já tem stake OPoI ativa — não precisa de colateral nova.");
        }
        Ok(None) => offer_stake_utxo_picker(client, collected).await,
        Err(e) => {
            println!();
            println!("Não consegui checar se {primary} já tem stake ({e}) — mostrando as UTXOs disponíveis mesmo assim, caso queira configurar uma colateral.");
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
        "Endereço OPoI {primary} ainda não tem stake ativa, e OPOI_COLLATERAL_TXID não está configurado no {ENV_PATH} — sem isso o cs-stratum-bridge não consegue subir."
    );
    print!("Quer escolher uma UTXO agora como colateral da stake? (s/n) ");
    io::stdout().flush().ok();
    if !read_line().trim().eq_ignore_ascii_case("s") {
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
        println!("Valor obrigatório, tente de novo.");
    }
}

fn prompt_api_key(v: &RequiredVar) -> String {
    println!();
    println!("{}: {}", v.key, v.prompt);
    print!("[Enter para gerar uma chave aleatória] > ");
    io::stdout().flush().ok();

    let input = read_line();
    if input.trim().is_empty() {
        let key = generate_random_hex(32);
        println!("Gerado: {key}");
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
    println!("Buscando UTXOs da wallet do csd pra usar como colateral da stake...");
    match client.list_unspent().await {
        Ok(utxos) if !utxos.is_empty() => {
            println!();
            println!("{:<4} {:<66} {:<5} {:>14} {:>6}  {}", "#", "txid", "vout", "valor (CS)", "conf", "endereço");
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
            print!("Número da UTXO pra usar como colateral da stake (Enter pra pular): ");
            io::stdout().flush().ok();

            let choice = read_line();
            match choice.trim().parse::<usize>() {
                Ok(idx) if idx >= 1 && idx <= utxos.len() => {
                    let picked = &utxos[idx - 1];
                    set_collected(collected, "OPOI_COLLATERAL_TXID", picked.txid.clone());
                    set_collected(collected, "OPOI_COLLATERAL_VOUT", picked.vout.to_string());
                    println!("Usando {}:{} como colateral da stake.", picked.txid, picked.vout);
                }
                _ => {
                    println!(
                        "Pulado — sem colateral configurada a stake OPoI não ativa sozinha. Configure OPOI_COLLATERAL_TXID/OPOI_COLLATERAL_VOUT no {ENV_PATH} quando tiver uma UTXO disponível."
                    );
                }
            }
        }
        Ok(_) => {
            println!(
                "Nenhuma UTXO disponível ainda nessa wallet — a stake OPoI não ativa sozinha até essa carteira ter saldo. Configure OPOI_COLLATERAL_TXID/OPOI_COLLATERAL_VOUT no {ENV_PATH} quando tiver."
            );
        }
        Err(e) => {
            println!("Não consegui listar UTXOs ({e}) — configure OPOI_COLLATERAL_TXID/OPOI_COLLATERAL_VOUT no {ENV_PATH} manualmente depois.");
        }
    }
}

/// Writes `.env` by copying `.env.example` line by line and substituting
/// only the `CHAVE=...` lines whose key was collected this run — every
/// other line (comments, untouched defaults) survives unchanged. Falls
/// back to a bare `CHAVE=valor` dump if `.env.example` isn't present.
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
        println!("Não consegui salvar {ENV_PATH} ({e}) — as variáveis coletadas ficaram só no ambiente deste processo.");
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
            println!("Não consegui ler {ENV_PATH} ({e}) — as variáveis coletadas ficaram só no ambiente deste processo.");
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
        Ok(_) => println!("{ENV_PATH} atualizado."),
        Err(e) => println!("Não consegui salvar {ENV_PATH} ({e}) — as variáveis coletadas ficaram só no ambiente deste processo."),
    }
}

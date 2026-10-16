#!/usr/bin/env bash
# ─── cs-stratum-bridge release ───────────────────────────────────────────────
# Builda e publica a imagem no registry.
# Usage: ./release.sh
#
# Pré-requisitos: docker autenticado no registry (docker login gru.ocir.io)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="gru.ocir.io/grroyghnc4ad"
IMAGE="$REGISTRY/cs-stratum-bridge:latest"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'
info() { echo -e "${CYAN}[release]${RESET} $*"; }
ok()   { echo -e "${GREEN}[release]${RESET} $*"; }
die()  { echo -e "${RED}[release] ERRO:${RESET} $*" >&2; exit 1; }

info "Buildando ${BOLD}$IMAGE${RESET} (contexto: $SCRIPT_DIR) …"
docker build "$SCRIPT_DIR" -t "$IMAGE" || die "Build falhou"
ok "Build concluído."

info "Publicando ${BOLD}$IMAGE${RESET} …"
docker push "$IMAGE" || die "Push falhou"
ok "Push concluído."

ok "Release completo: $IMAGE"

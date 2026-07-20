#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  script_source="${BASH_SOURCE[0]:-}"
  if [[ -n "${script_source}" && -r "${script_source}" ]]; then
    exec sudo -E bash "${script_source}" "$@"
  fi
  echo "This setup script needs root privileges. Re-run it with sudo." >&2
  exit 1
fi

LOG_DIR="/var/log/devin-outposts-launchable"
STATE_DIR="/var/lib/devin-outposts-launchable"
CONFIG_DIR="/etc/devin-outposts"
ENV_FILE="${CONFIG_DIR}/worker.env"
SERVICE_NAME="devin-outposts-worker.service"
DEVIN_API_URL_DEFAULT="https://api.devin.ai"
REMOTE_BASE_URL="https://static.devin.ai/devin-rs/remote"

DEVIN_TOKEN="${DEVIN_WORKER_TOKEN:-${DEVIN_OUTPOSTS_TOKEN:-}}"
OUTPOST_REFERENCE="${DEVIN_OUTPOST_NAME:-${DEVIN_OUTPOST_ID:-}}"
DEVIN_API_URL_VALUE="${DEVIN_API_URL:-${DEVIN_API_URL_DEFAULT}}"
REPO_URL_VALUE="${REPO_URL:-}"
unset DEVIN_WORKER_TOKEN DEVIN_OUTPOSTS_TOKEN

mkdir -p "${LOG_DIR}" "${STATE_DIR}" "${CONFIG_DIR}"
touch "${LOG_DIR}/setup.log"
chmod 0700 "${STATE_DIR}" "${CONFIG_DIR}"
chmod 0600 "${LOG_DIR}/setup.log"
rm -f "${STATE_DIR}/ready" "${STATE_DIR}/worker-loop-ready"
exec > >(tee -a "${LOG_DIR}/setup.log") 2>&1

die() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "== Devin Outposts on Brev setup =="

[[ -n "${DEVIN_TOKEN}" ]] || die "DEVIN_WORKER_TOKEN is required. Paste the cog_... token for a Devin service user with Use outpost machine permission."
[[ -n "${OUTPOST_REFERENCE}" ]] || die "DEVIN_OUTPOST_NAME is required. Enter the exact name shown in Devin's Outposts settings."

for value_name in DEVIN_TOKEN OUTPOST_REFERENCE DEVIN_API_URL_VALUE REPO_URL_VALUE; do
  value="${!value_name}"
  if [[ "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
    die "${value_name} must be a single line."
  fi
done

OUTPOST_REFERENCE_PATTERN='^[A-Za-z0-9._: -]+$'
[[ "${OUTPOST_REFERENCE}" =~ ${OUTPOST_REFERENCE_PATTERN} ]] || die "DEVIN_OUTPOST_NAME contains unsupported characters."

while [[ "${DEVIN_API_URL_VALUE}" == */ ]]; do
  DEVIN_API_URL_VALUE="${DEVIN_API_URL_VALUE%/}"
done
[[ "${DEVIN_API_URL_VALUE}" == https://* ]] || die "DEVIN_API_URL must use HTTPS."
[[ "${DEVIN_API_URL_VALUE}" != *"@"* ]] || die "DEVIN_API_URL must not contain embedded credentials."
[[ "${DEVIN_API_URL_VALUE}" != *"?"* && "${DEVIN_API_URL_VALUE}" != *"#"* ]] || die "DEVIN_API_URL must not contain a query or fragment."
[[ "${DEVIN_API_URL_VALUE}" != *" "* && "${DEVIN_API_URL_VALUE}" != *$'\t'* ]] || die "DEVIN_API_URL must not contain whitespace."

USER_NAME="${DEVIN_LAUNCHABLE_USER:-ubuntu}"
if ! id "${USER_NAME}" >/dev/null 2>&1; then
  USER_NAME="$(getent passwd 1000 | cut -d: -f1 || true)"
fi
if [[ -z "${USER_NAME}" ]] || ! id "${USER_NAME}" >/dev/null 2>&1; then
  die "Could not find the target user. Set DEVIN_LAUNCHABLE_USER and run the setup again."
fi

USER_GROUP="$(id -gn "${USER_NAME}")"
USER_HOME="$(getent passwd "${USER_NAME}" | cut -d: -f6)"
WORKSPACE_DIR="${USER_HOME}/devin-workspace"

echo "Target user: ${USER_NAME}"
echo "Workspace: ${WORKSPACE_DIR}"
echo "Devin API: ${DEVIN_API_URL_VALUE}"
echo "Outpost name: ${OUTPOST_REFERENCE}"
echo "Devin token: provided (value hidden)"

echo "Checking NVIDIA GPU access..."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is missing. Choose an NVIDIA GPU instance and relaunch."
nvidia-smi -L >/dev/null 2>&1 || die "No working NVIDIA GPU was detected. Choose an NVIDIA GPU instance and relaunch."
GPU_SUMMARY="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 1)"
echo "GPU: ${GPU_SUMMARY}"
if ! ldconfig -p 2>/dev/null | grep 'libcuda\.so\.1' >/dev/null; then
  die "The NVIDIA driver library libcuda.so.1 is unavailable. Relaunch with Brev's CUDA-ready GPU image."
fi
if command -v nvcc >/dev/null 2>&1; then
  echo "CUDA compiler: $(nvcc --version | tail -n 1)"
else
  echo "Note: the GPU driver is ready, but nvcc is not on PATH. Install a task-specific CUDA toolkit on the VM if the workload needs nvcc."
fi

export DEBIAN_FRONTEND=noninteractive
echo "Installing base packages..."
apt-get update
apt-get install -y ca-certificates curl git jq util-linux

run_as_user() {
  sudo -u "${USER_NAME}" -H env \
    HOME="${USER_HOME}" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "$@"
}

prepare_workspace() {
  local normalized_repo repo_name repo_dir existing_origin clone_dir repos_dir

  mkdir -p "${WORKSPACE_DIR}"
  chown "${USER_NAME}:${USER_GROUP}" "${WORKSPACE_DIR}"
  repos_dir="${WORKSPACE_DIR}/repos"
  mkdir -p "${repos_dir}"
  chown "${USER_NAME}:${USER_GROUP}" "${repos_dir}"

  if [[ -z "${REPO_URL_VALUE}" ]]; then
    echo "No REPO_URL supplied; using an empty workspace."
    return
  fi

  [[ "${REPO_URL_VALUE}" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?/?$ ]] \
    || die "REPO_URL must be a public GitHub HTTPS URL, for example https://github.com/owner/repo."

  normalized_repo="${REPO_URL_VALUE%/}"
  normalized_repo="${normalized_repo%.git}.git"
  repo_name="${normalized_repo##*/}"
  repo_name="${repo_name%.git}"
  repo_dir="${repos_dir}/${repo_name}"
  echo "Checking public repository access..."
  run_as_user env GIT_TERMINAL_PROMPT=0 git ls-remote --exit-code "${normalized_repo}" HEAD >/dev/null \
    || die "REPO_URL is not publicly cloneable. Private repositories are not supported in this first version."

  if [[ -d "${repo_dir}/.git" ]]; then
    existing_origin="$(run_as_user git -C "${repo_dir}" remote get-url origin 2>/dev/null || true)"
    existing_origin="${existing_origin%/}"
    existing_origin="${existing_origin%.git}.git"
    [[ "${existing_origin}" == "${normalized_repo}" ]] \
      || die "The workspace already contains a different repository. Use a new Brev instance."
    echo "Requested repository is already present; preserving the working tree."
    return
  fi

  if find "${repos_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "The repository directory is non-empty but does not contain the requested Git repository. Use a new Brev instance."
  fi

  clone_dir="$(mktemp -d /var/tmp/devin-repo.XXXXXX)"
  chown "${USER_NAME}:${USER_GROUP}" "${clone_dir}"
  run_as_user env GIT_TERMINAL_PROMPT=0 git clone --single-branch --no-tags "${normalized_repo}" "${clone_dir}/repo"
  mv "${clone_dir}/repo" "${repo_dir}"
  rmdir "${clone_dir}"
  chown -R "${USER_NAME}:${USER_GROUP}" "${repo_dir}"
  echo "Repository cloned into ${repo_dir}. Submodules were not initialized and repository content was not executed."
}

resolve_outpost_access() {
  local available count escaped_token http_status matches response_file

  echo "Validating the Devin token and resolving the Outpost..."
  escaped_token="${DEVIN_TOKEN//\\/\\\\}"
  escaped_token="${escaped_token//\"/\\\"}"
  response_file="$(mktemp)"
  http_status="$(
    printf 'silent\nshow-error\nheader = "Authorization: Bearer %s"\n' "${escaped_token}" \
      | curl --config - \
        --connect-timeout 20 \
        --max-time 60 \
        --output "${response_file}" \
        --write-out '%{http_code}' \
        "${DEVIN_API_URL_VALUE}/opbeta/outposts?first=200" \
      || true
  )"

  if [[ "${http_status}" != "200" ]]; then
    rm -f "${response_file}"
    die "Devin rejected access to the Outposts API (HTTP ${http_status:-request-failed}). Confirm the token has UseOutpostsMachine permission and check DEVIN_API_URL."
  fi

  matches="$(
    jq -c --arg reference "${OUTPOST_REFERENCE}" \
      '[.items[] | select(.metadata.outpost_id == $reference or .spec.name == $reference)]' \
      "${response_file}"
  )" || {
    rm -f "${response_file}"
    die "Devin returned an invalid Outposts API response."
  }
  count="$(jq -r 'length' <<<"${matches}")"
  if [[ "${count}" != "1" ]]; then
    available="$(jq -r '[.items[].spec.name] | join(", ")' "${response_file}" 2>/dev/null || true)"
    rm -f "${response_file}"
    if [[ "${count}" == "0" ]]; then
      die "No Outpost matched '${OUTPOST_REFERENCE}'. Enter the exact Outpost name shown in Devin. Available Outposts: ${available:-none}."
    fi
    die "More than one Outpost is named '${OUTPOST_REFERENCE}'. Enter its outpost_env-... ID instead."
  fi

  OUTPOST_ID="$(jq -r '.[0].metadata.outpost_id' <<<"${matches}")"
  OUTPOST_NAME="$(jq -r '.[0].spec.name' <<<"${matches}")"
  rm -f "${response_file}"
  [[ "${OUTPOST_ID}" =~ ^outpost_env-[A-Za-z0-9_-]+$ ]] || die "Devin returned an invalid Outpost ID."
  echo "Resolved Outpost: ${OUTPOST_NAME} (${OUTPOST_ID})"
}

write_worker_files() {
  local escaped_api escaped_group escaped_home escaped_id escaped_state escaped_token escaped_user escaped_workspace

  escaped_token="${DEVIN_TOKEN//\\/\\\\}"
  escaped_token="${escaped_token//\"/\\\"}"
  escaped_api="${DEVIN_API_URL_VALUE//\\/\\\\}"
  escaped_api="${escaped_api//\"/\\\"}"
  escaped_id="${OUTPOST_ID//\\/\\\\}"
  escaped_id="${escaped_id//\"/\\\"}"
  escaped_workspace="${WORKSPACE_DIR//\\/\\\\}"
  escaped_workspace="${escaped_workspace//\"/\\\"}"
  escaped_state="${STATE_DIR//\\/\\\\}"
  escaped_state="${escaped_state//\"/\\\"}"
  escaped_home="${USER_HOME//\\/\\\\}"
  escaped_home="${escaped_home//\"/\\\"}"
  escaped_user="${USER_NAME//\\/\\\\}"
  escaped_user="${escaped_user//\"/\\\"}"
  escaped_group="${USER_GROUP//\\/\\\\}"
  escaped_group="${escaped_group//\"/\\\"}"

  mkdir -p /var/log/devin
  chown "${USER_NAME}:${USER_GROUP}" /var/log/devin
  chmod 0700 /var/log/devin

  umask 077
  cat > "${ENV_FILE}.tmp" <<EOF
DEVIN_OUTPOSTS_TOKEN="${escaped_token}"
DEVIN_API_URL="${escaped_api}"
DEVIN_OUTPOST_ID="${escaped_id}"
DEVIN_WORKSPACE="${escaped_workspace}"
DEVIN_STATE_DIR="${escaped_state}"
DEVIN_USER_HOME="${escaped_home}"
DEVIN_USER_NAME="${escaped_user}"
DEVIN_USER_GROUP="${escaped_group}"
EOF
  chown root:root "${ENV_FILE}.tmp"
  chmod 0600 "${ENV_FILE}.tmp"
  mv "${ENV_FILE}.tmp" "${ENV_FILE}"

  cat > /usr/local/bin/devin-outposts-worker <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

: "${DEVIN_OUTPOSTS_TOKEN:?DEVIN_OUTPOSTS_TOKEN is required}"
: "${DEVIN_API_URL:?DEVIN_API_URL is required}"
: "${DEVIN_OUTPOST_ID:?DEVIN_OUTPOST_ID is required}"
: "${DEVIN_WORKSPACE:?DEVIN_WORKSPACE is required}"
: "${DEVIN_STATE_DIR:?DEVIN_STATE_DIR is required}"
: "${DEVIN_USER_HOME:?DEVIN_USER_HOME is required}"
: "${DEVIN_USER_NAME:?DEVIN_USER_NAME is required}"
: "${DEVIN_USER_GROUP:?DEVIN_USER_GROUP is required}"

REMOTE_BASE_URL="${DEVIN_REMOTE_BASE_URL_INTERNAL:-https://static.devin.ai/devin-rs/remote}"
POLL_SECONDS="${DEVIN_POLL_SECONDS_INTERNAL:-5}"
WORKER_ONCE="${DEVIN_WORKER_ONCE_INTERNAL:-false}"
BINARY_DIR="${DEVIN_STATE_DIR}/binaries"
READY_FILE="${DEVIN_STATE_DIR}/worker-loop-ready"
SESSION_STATE_ROOT="${DEVIN_USER_HOME}/.devin/worker/sessions"
ACCEPTOR_FILE="${DEVIN_STATE_DIR}/acceptor_id"
LOCK_FILE="${DEVIN_STATE_DIR}/worker.lock"
WORKER_HOST="${HOSTNAME:-$(hostname)}"
WORKER_HOST="${WORKER_HOST//[^A-Za-z0-9_.-]/-}"
USER_UID="$(id -u "${DEVIN_USER_NAME}")"
USER_GID="$(id -g "${DEVIN_USER_NAME}")"
ACTIVE_SESSION=""
ACTIVE_ACCEPTOR=""
ACTIVE_SESSION_STATE=""
REMOTE_PID=""
LAST_SESSION_STATUS=""

[[ "${POLL_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "DEVIN_POLL_SECONDS_INTERNAL must be a non-negative number." >&2
  exit 1
}
[[ "${WORKER_ONCE}" == "true" || "${WORKER_ONCE}" == "false" ]] || {
  echo "DEVIN_WORKER_ONCE_INTERNAL must be true or false." >&2
  exit 1
}
mkdir -p "${BINARY_DIR}" "${SESSION_STATE_ROOT}" "${DEVIN_WORKSPACE}/repos"
chmod 0700 "${BINARY_DIR}"
chown root:"${DEVIN_USER_GROUP}" \
  "${DEVIN_USER_HOME}/.devin" \
  "${DEVIN_USER_HOME}/.devin/worker" \
  "${SESSION_STATE_ROOT}"
chmod 0710 \
  "${DEVIN_USER_HOME}/.devin" \
  "${DEVIN_USER_HOME}/.devin/worker" \
  "${SESSION_STATE_ROOT}"
chown "${DEVIN_USER_NAME}:${DEVIN_USER_GROUP}" "${DEVIN_WORKSPACE}" "${DEVIN_WORKSPACE}/repos"
exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "Another Devin Outposts worker already holds ${LOCK_FILE}." >&2
  exit 1
}
if [[ ! -s "${ACCEPTOR_FILE}" ]]; then
  printf 'brev-%s-%s\n' "${WORKER_HOST}" "$(cat /proc/sys/kernel/random/uuid)" > "${ACCEPTOR_FILE}.tmp"
  chmod 0600 "${ACCEPTOR_FILE}.tmp"
  mv -f "${ACCEPTOR_FILE}.tmp" "${ACCEPTOR_FILE}"
fi
ACCEPTOR_ID="$(<"${ACCEPTOR_FILE}")"
[[ "${ACCEPTOR_ID}" =~ ^brev-[A-Za-z0-9_.-]+$ ]] || {
  echo "The persisted Outposts acceptor ID is invalid." >&2
  exit 1
}

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

curl_api() {
  local escaped_token
  escaped_token="${DEVIN_OUTPOSTS_TOKEN//\\/\\\\}"
  escaped_token="${escaped_token//\"/\\\"}"
  printf 'silent\nshow-error\nheader = "Authorization: Bearer %s"\n' "${escaped_token}" \
    | curl --config - --connect-timeout 10 --max-time 30 "$@"
}

download_remote() {
  local actual_sha binary_path expected_sha requested_sha sha temp_binary

  requested_sha="${1:-}"
  sha="${requested_sha}"
  if [[ -z "${sha}" || "${sha}" == "null" ]]; then
    sha="$(curl -fsSL --retry 2 --retry-max-time 60 --connect-timeout 10 --max-time 60 "${REMOTE_BASE_URL}/latest_linux_x64")"
  fi
  sha="${sha//$'\r'/}"
  sha="${sha//$'\n'/}"
  [[ "${sha}" =~ ^[0-9a-fA-F]{7,64}$ ]] || {
    log "Refusing invalid devin-remote SHA '${sha}'."
    return 1
  }

  expected_sha="$(curl -fsSL --retry 2 --retry-max-time 60 --connect-timeout 10 --max-time 60 "${REMOTE_BASE_URL}/devin-remote_${sha}_linux_x64.sha256")"
  expected_sha="${expected_sha%%[[:space:]]*}"
  [[ "${expected_sha}" =~ ^[0-9a-fA-F]{64}$ ]] || {
    log "The published checksum for devin-remote ${sha} is invalid."
    return 1
  }

  binary_path="${BINARY_DIR}/devin-remote_${sha}_linux_x64"
  if [[ -x "${binary_path}" ]]; then
    actual_sha="$(sha256sum "${binary_path}" | cut -d' ' -f1)"
    if [[ "${actual_sha}" == "${expected_sha}" ]]; then
      printf '%s\n' "${binary_path}"
      return 0
    fi
    rm -f "${binary_path}"
  fi

  temp_binary="$(mktemp "${BINARY_DIR}/.devin-remote.XXXXXX")"
  if ! curl -fsSL --retry 2 --retry-max-time 60 --connect-timeout 10 --max-time 60 \
    "${REMOTE_BASE_URL}/devin-remote_${sha}_linux_x64" -o "${temp_binary}"; then
    rm -f "${temp_binary}"
    return 1
  fi
  actual_sha="$(sha256sum "${temp_binary}" | cut -d' ' -f1)"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    rm -f "${temp_binary}"
    log "Checksum verification failed for devin-remote ${sha}."
    return 1
  fi
  chmod 0755 "${temp_binary}"
  mv -f "${temp_binary}" "${binary_path}"
  log "Installed verified devin-remote ${sha}."
  printf '%s\n' "${binary_path}"
}

release_claim() {
  local body http_status

  [[ -n "${ACTIVE_SESSION}" && -n "${ACTIVE_ACCEPTOR}" ]] || return 0
  body="$(printf '{\"acceptor_id\":\"%s\"}' "${ACTIVE_ACCEPTOR}")"
  http_status="$(
    curl_api \
      --request POST \
      --connect-timeout 5 \
      --max-time 15 \
      --header 'Content-Type: application/json' \
      --data "${body}" \
      --output /dev/null \
      --write-out '%{http_code}' \
      "${DEVIN_API_URL}/opbeta/outposts/devins/${ACTIVE_SESSION}/release" \
      || true
  )"
  if [[ "${http_status}" == "200" || "${http_status}" == "204" || "${http_status}" == "404" ]]; then
    log "Released session ${ACTIVE_SESSION}."
  else
    log "Release for session ${ACTIVE_SESSION} returned HTTP ${http_status:-request-failed}; its claim will expire automatically."
  fi
  ACTIVE_SESSION=""
  ACTIVE_ACCEPTOR=""
}

reconcile_owned_claims() {
  local claimed encoded_acceptor session_id
  local -a sessions=()

  encoded_acceptor="$(jq -rn --arg value "${ACCEPTOR_ID}" '$value | @uri')"
  if ! claimed="$(
    curl_api --fail \
      "${DEVIN_API_URL}/opbeta/outposts/devins?phase=claimed&acceptor_id=${encoded_acceptor}&first=50"
  )"; then
    log "Could not reconcile claims owned by this worker."
    return 1
  fi
  jq -e '.items | type == "array"' <<<"${claimed}" >/dev/null 2>&1 || {
    log "Devin returned an invalid claimed-session response."
    return 1
  }
  mapfile -t sessions < <(
    jq -r --arg outpost "${DEVIN_OUTPOST_ID}" \
      '.items[] | select(.metadata.outpost_id == $outpost) | .metadata.session_id' \
      <<<"${claimed}"
  )
  for session_id in "${sessions[@]}"; do
    [[ "${session_id}" =~ ^devin-[A-Za-z0-9-]+$ ]] || continue
    ACTIVE_SESSION="${session_id}"
    ACTIVE_ACCEPTOR="${ACCEPTOR_ID}"
    log "Releasing a stale claim from an earlier worker process for ${session_id}."
    release_claim
  done
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${REMOTE_PID}" ]] && kill -0 "${REMOTE_PID}" 2>/dev/null; then
    kill "${REMOTE_PID}" 2>/dev/null || true
    wait "${REMOTE_PID}" 2>/dev/null || true
  fi
  release_claim
  if [[ -n "${ACTIVE_SESSION_STATE}" && -d "${ACTIVE_SESSION_STATE}" ]]; then
    chown -R root:root "${ACTIVE_SESSION_STATE}" 2>/dev/null || true
    chmod 0700 "${ACTIVE_SESSION_STATE}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_remote() (
  local binary_path="$1"
  local gateway_url="$2"
  local connect_token="$3"
  local session_id="$4"
  local session_state="$5"
  local runtime_gid="${USER_GID}"
  local runtime_home="${DEVIN_USER_HOME}"
  local runtime_uid="${USER_UID}"
  local runtime_user="${DEVIN_USER_NAME}"
  local runtime_workspace="${DEVIN_WORKSPACE}"
  local env_name

  for env_name in $(compgen -e); do
    unset "${env_name}" 2>/dev/null || true
  done
  export HOME="${runtime_home}"
  export USER="${runtime_user}"
  export LOGNAME="${USER}"
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin"
  export TMPDIR="/tmp"
  export LANG="C.UTF-8"
  export TZ="UTC"
  export DEVIN_OUTPOST_GATEWAY_URL="${gateway_url}"
  export DEVIN_OUTPOST_CONNECT_TOKEN="${connect_token}"
  export DEVIN_OUTPOST_SESSION_ID="${session_id}"
  export DEVIN_REMOTE_STATE_DIR="${session_state}"
  export DEVIN_OUTPOST_DESKTOP="false"
  cd "${runtime_workspace}"
  exec /usr/bin/setpriv \
    --reuid="${runtime_uid}" \
    --regid="${runtime_gid}" \
    --init-groups \
    "${binary_path}" serve
)

get_session_status() {
  local http_status response_file session_status

  response_file="$(mktemp "${DEVIN_STATE_DIR}/.status.XXXXXX")"
  http_status="$(
    curl_api \
      --output "${response_file}" \
      --write-out '%{http_code}' \
      "${DEVIN_API_URL}/opbeta/outposts/devins/${ACTIVE_SESSION}" \
      2>/dev/null \
      || true
  )"
  if [[ "${http_status}" == "404" ]]; then
    rm -f "${response_file}"
    printf 'missing\n'
    return 0
  fi
  if [[ "${http_status}" != "200" ]]; then
    rm -f "${response_file}"
    return 1
  fi
  session_status="$(jq -er '.status.session_status // empty' "${response_file}" 2>/dev/null)" || {
    rm -f "${response_file}"
    return 1
  }
  rm -f "${response_file}"
  printf '%s\n' "${session_status}"
}

wait_for_terminal_status() {
  local session_status

  for _ in $(seq 1 5); do
    if session_status="$(get_session_status)"; then
      LAST_SESSION_STATUS="${session_status}"
      if [[ "${session_status}" == "suspended" || "${session_status}" == "terminated" || "${session_status}" == "missing" ]]; then
        return 0
      fi
    fi
    sleep 2
  done
}

latest_binary="$(download_remote "")"
log "Direct Outposts worker ready with $(${latest_binary} --version 2>/dev/null || echo devin-remote)."
reconcile_owned_claims

while true; do
  if ! queue="$(
    curl_api --fail \
      "${DEVIN_API_URL}/opbeta/outposts/devins?outpost=${DEVIN_OUTPOST_ID}&phase=pending&first=50"
  )"; then
    log "Could not poll the Outpost queue; retrying."
    sleep "${POLL_SECONDS}"
    continue
  fi

  if ! jq -e '.items | type == "array"' <<<"${queue}" >/dev/null 2>&1; then
    log "Devin returned an invalid queue response; retrying."
    sleep "${POLL_SECONDS}"
    continue
  fi
  if ! entry="$(jq -c '.items | sort_by(.metadata.created_at) | .[0] // empty' <<<"${queue}")"; then
    log "Devin returned an invalid queue response; retrying."
    sleep "${POLL_SECONDS}"
    continue
  fi
  if [[ ! -f "${READY_FILE}" ]]; then
    printf 'ready_at=%s\noutpost_id=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${DEVIN_OUTPOST_ID}" \
      > "${READY_FILE}.tmp"
    chmod 0600 "${READY_FILE}.tmp"
    mv -f "${READY_FILE}.tmp" "${READY_FILE}"
  fi
  if [[ -z "${entry}" ]]; then
    sleep "${POLL_SECONDS}"
    continue
  fi

  session_id="$(jq -r '.metadata.session_id // empty' <<<"${entry}")"
  platform="$(jq -r '.spec.platform // empty' <<<"${entry}")"
  remote_sha="$(jq -r '.spec.remote_binary_sha // empty' <<<"${entry}")"
  if [[ ! "${session_id}" =~ ^devin-[A-Za-z0-9-]+$ || "${platform}" != "linux" ]]; then
    log "Skipping an invalid or non-Linux queue entry."
    sleep "${POLL_SECONDS}"
    continue
  fi

  if ! binary_path="$(download_remote "${remote_sha}")"; then
    log "Could not prepare the runtime for session ${session_id}; retrying."
    sleep "${POLL_SECONDS}"
    continue
  fi

  claim_file="$(mktemp "${DEVIN_STATE_DIR}/.claim.XXXXXX")"
  claim_body="$(printf '{\"acceptor_id\":\"%s\"}' "${ACCEPTOR_ID}")"
  claim_status="$(
    curl_api \
      --request POST \
      --header 'Content-Type: application/json' \
      --data "${claim_body}" \
      --output "${claim_file}" \
      --write-out '%{http_code}' \
      "${DEVIN_API_URL}/opbeta/outposts/devins/${session_id}/claim" \
      || true
  )"
  if [[ "${claim_status}" == "409" ]]; then
    rm -f "${claim_file}"
    sleep 1
    continue
  fi
  if [[ "${claim_status}" != "200" && "${claim_status}" != "201" ]]; then
    rm -f "${claim_file}"
    log "Claim for session ${session_id} returned HTTP ${claim_status:-request-failed}; retrying."
    reconcile_owned_claims || true
    sleep "${POLL_SECONDS}"
    continue
  fi

  ACTIVE_SESSION="${session_id}"
  ACTIVE_ACCEPTOR="${ACCEPTOR_ID}"
  gateway_url="$(jq -r '.status.gateway_url // empty' "${claim_file}" 2>/dev/null || true)"
  connect_token="$(jq -r '.status.connect_token // empty' "${claim_file}" 2>/dev/null || true)"
  claimed_session="$(jq -r '.metadata.session_id // empty' "${claim_file}" 2>/dev/null || true)"
  claimed_outpost="$(jq -r '.metadata.outpost_id // empty' "${claim_file}" 2>/dev/null || true)"
  claimed_acceptor="$(jq -r '.status.acceptor_id // empty' "${claim_file}" 2>/dev/null || true)"
  claimed_remote_sha="$(jq -r '.spec.remote_binary_sha // empty' "${claim_file}" 2>/dev/null || true)"
  claim_deadline="$(jq -r '.status.claim_deadline // empty' "${claim_file}" 2>/dev/null || true)"
  rm -f "${claim_file}"
  if [[ "${claimed_session}" != "${session_id}" \
    || "${claimed_outpost}" != "${DEVIN_OUTPOST_ID}" \
    || "${claimed_acceptor}" != "${ACCEPTOR_ID}" \
    || "${gateway_url}" != wss://* \
    || -z "${connect_token}" \
    || -z "${claim_deadline}" ]]; then
    log "The claim response for session ${session_id} was incomplete; releasing it."
    release_claim
    sleep "${POLL_SECONDS}"
    continue
  fi
  if [[ -n "${claimed_remote_sha}" && "${claimed_remote_sha}" != "${remote_sha}" ]]; then
    if ! binary_path="$(download_remote "${claimed_remote_sha}")"; then
      log "Could not prepare the runtime pinned by the claim for ${session_id}; releasing it."
      release_claim
      sleep "${POLL_SECONDS}"
      continue
    fi
  fi

  session_state="${SESSION_STATE_ROOT}/${session_id}"
  mkdir -p "${session_state}"
  chmod 0700 "${session_state}"
  chown -R "${DEVIN_USER_NAME}:${DEVIN_USER_GROUP}" "${session_state}"
  ACTIVE_SESSION_STATE="${session_state}"
  log "Serving session ${session_id} with a verified direct runtime."
  set +e
  run_remote "${binary_path}" "${gateway_url}" "${connect_token}" "${session_id}" "${session_state}" &
  REMOTE_PID=$!
  while kill -0 "${REMOTE_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
    current_status="$(get_session_status 2>/dev/null || true)"
    if [[ "${current_status}" == "terminated" ]]; then
      log "Session ${session_id} terminated; stopping its runtime."
      kill "${REMOTE_PID}" 2>/dev/null || true
      break
    fi
  done
  wait "${REMOTE_PID}"
  remote_exit=$?
  REMOTE_PID=""
  set -e
  log "Runtime for session ${session_id} exited with status ${remote_exit}."
  wait_for_terminal_status
  release_claim
  if [[ "${LAST_SESSION_STATUS}" == "terminated" || "${LAST_SESSION_STATUS}" == "missing" ]]; then
    rm -rf "${session_state}"
  else
    chown -R root:root "${session_state}"
    chmod 0700 "${session_state}"
  fi
  ACTIVE_SESSION_STATE=""
  LAST_SESSION_STATUS=""
  if [[ "${WORKER_ONCE}" == "true" ]]; then
    exit 0
  fi
  sleep 1
done
WORKER
  chmod 0755 /usr/local/bin/devin-outposts-worker

  cat > "/etc/systemd/system/${SERVICE_NAME}" <<EOF
[Unit]
Description=Devin Outposts worker on Brev
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${WORKSPACE_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=/usr/local/bin/devin-outposts-worker
Restart=always
RestartSec=5
KillMode=control-group
TimeoutStopSec=30
UMask=0077

[Install]
WantedBy=multi-user.target
EOF
  chmod 0644 "/etc/systemd/system/${SERVICE_NAME}"

  chown -R root:root "${STATE_DIR}"
  chmod 0700 "${STATE_DIR}"
}

prepare_workspace
resolve_outpost_access
write_worker_files

echo "Starting the Devin Outposts worker..."
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}" >/dev/null
systemctl restart "${SERVICE_NAME}"

for _ in $(seq 1 180); do
  sleep 1
  systemctl is-active --quiet "${SERVICE_NAME}" || die "The Devin worker stopped during startup. Inspect it with: sudo journalctl -u ${SERVICE_NAME}"
  [[ -f "${STATE_DIR}/worker-loop-ready" ]] && break
done
[[ -f "${STATE_DIR}/worker-loop-ready" ]] \
  || die "The worker could not verify the direct runtime and poll the Outpost queue. Inspect it with: sudo journalctl -u ${SERVICE_NAME}"

cat > "${STATE_DIR}/ready" <<EOF
ready_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
worker_mode=direct-devin-remote
outpost_name=${OUTPOST_NAME}
outpost_id=${OUTPOST_ID}
api_url=${DEVIN_API_URL_VALUE}
gpu=${GPU_SUMMARY}
workspace=${WORKSPACE_DIR}
EOF
chmod 0600 "${STATE_DIR}/ready"

echo "== Devin Outposts is ready =="
echo "The direct runtime worker is polling '${OUTPOST_NAME}' from this Brev GPU VM. Start a session on that Outpost in Devin."
echo "Status: sudo systemctl status ${SERVICE_NAME}"
echo "GPU: nvidia-smi"

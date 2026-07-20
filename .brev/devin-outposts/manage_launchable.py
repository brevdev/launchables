#!/usr/bin/env python3
"""Render, create, edit, or verify the Devin Outposts Brev Launchable."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_BASE = "https://brevapi.us-west-2-prod.control-plane.brev.dev/api"
DEFAULT_ORG_ID = "org-34RfgPNxZ6aPsVmF6qz91Z25Sxq"
DEFAULT_LAUNCHABLE_ID = "env-3Ge1ZXazlZQuJHfQed2od9IT8R5"
SCRIPT_LIMIT = 16 * 1024
HERE = Path(__file__).resolve().parent
SETUP_REVISION = "e2ddafb659c9ceb18fb6db0241ddedf52527166a"
SETUP_SHA256 = "46ffaf606d0a8e9f6efeedcdde1f561ec24a629eb666ccbe9db28ae347821cb7"
SETUP_PATH = ".brev/devin-outposts/launchable-setup.sh"
SETUP_URL = (
    "https://raw.githubusercontent.com/brevdev/launchables/"
    f"{SETUP_REVISION}/{SETUP_PATH}"
)
EXPECTED_PARAMETERS = [
    "DEVIN_WORKER_TOKEN",
    "DEVIN_OUTPOST_NAME",
    "DEVIN_API_URL",
    "REPO_URL",
]
EXPECTED_PARAMETER_REQUIREMENTS = {
    "DEVIN_WORKER_TOKEN": True,
    "DEVIN_OUTPOST_NAME": True,
    "DEVIN_API_URL": False,
    "REPO_URL": False,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_access_token() -> str:
    credentials_path = Path.home() / ".brev" / "credentials.json"
    if not credentials_path.exists():
        raise SystemExit("Brev credentials are missing. Run `brev login` and retry.")
    credentials = load_json(credentials_path)
    try:
        return credentials["access_token"]
    except KeyError as exc:
        raise SystemExit("Brev access token is missing. Run `brev login` and retry.") from exc


def token_seconds_left(token: str) -> int | None:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload))
        return int(decoded.get("exp", 0)) - int(time.time())
    except Exception:
        return None


def request(
    method: str,
    path: str,
    token: str | None = None,
    body: dict | None = None,
    not_found_retries: int = 0,
) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(not_found_retries + 1):
        req = urllib.request.Request(API_BASE + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and attempt < not_found_retries:
                exc.close()
                time.sleep(2**attempt)
                continue
            detail = exc.read().decode(errors="replace")
            raise SystemExit(f"{method} {path} failed: HTTP {exc.code}: {detail}") from exc
    raise AssertionError("unreachable")


def build_bootstrap() -> str:
    setup_sha256 = hashlib.sha256((HERE / "launchable-setup.sh").read_bytes()).hexdigest()
    if setup_sha256 != SETUP_SHA256:
        raise SystemExit(
            "The local setup script differs from the pinned public installer. "
            "Publish it and update SETUP_REVISION and SETUP_SHA256 first."
        )

    bootstrap = f'''#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly SETUP_REVISION="{SETUP_REVISION}"
readonly SETUP_SHA256="{SETUP_SHA256}"
readonly SETUP_URL="{SETUP_URL}"

if ! command -v curl >/dev/null 2>&1; then
  if [[ "${{EUID}}" -eq 0 ]]; then
    apt-get update
    apt-get install -y ca-certificates curl
  else
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
  fi
fi
command -v sha256sum >/dev/null 2>&1 || {{
  echo "sha256sum is required to verify the Devin Outposts installer." >&2
  exit 1
}}

setup_script="$(mktemp /tmp/devin-outposts-setup.XXXXXX)"
trap 'rm -f "$setup_script"' EXIT

env -u DEVIN_WORKER_TOKEN -u DEVIN_OUTPOSTS_TOKEN \
  curl --disable --fail --silent --show-error --location \
    --retry 3 --retry-delay 2 --retry-all-errors \
    --connect-timeout 20 --max-time 120 \
    --proto '=https' --proto-redir '=https' --tlsv1.2 \
    "${{SETUP_URL}}" --output "${{setup_script}}"

printf '%s  %s\n' "${{SETUP_SHA256}}" "${{setup_script}}" \
  | sha256sum --check --status || {{
    echo "Downloaded Devin Outposts installer failed checksum verification." >&2
    exit 1
  }}

chmod 0700 "${{setup_script}}"
bash "${{setup_script}}"
'''
    byte_len = len(bootstrap.encode())
    if byte_len > SCRIPT_LIMIT:
        raise SystemExit(f"Lifecycle bootstrap is {byte_len} bytes, over the {SCRIPT_LIMIT}-byte limit.")
    print(f"lifecycle_bootstrap_bytes={byte_len}")
    return bootstrap


def build_payload() -> dict:
    payload = load_json(HERE / "launchable-create.payload.json")
    payload["buildRequest"]["vmBuild"]["lifeCycleScriptAttr"]["script"] = build_bootstrap()
    return payload


def public_summary(launchable: dict, lifecycle_script: str | None = None) -> dict:
    build = launchable.get("buildRequest", {})
    workspace = launchable.get("createWorkspaceRequest", {})
    lifecycle_attr = build.get("vmBuild", {}).get("lifeCycleScriptAttr", {})
    lifecycle = lifecycle_attr.get("script", "") if lifecycle_script is None else lifecycle_script
    return {
        "id": launchable.get("id"),
        "name": launchable.get("name"),
        "viewAccess": launchable.get("viewAccess"),
        "instanceType": workspace.get("instanceType"),
        "storage": workspace.get("storage"),
        "parameterNames": [item.get("name") for item in build.get("parameters", [])],
        "parameterRequirements": {
            item.get("name"): item.get("required") for item in build.get("parameters", [])
        },
        "ports": build.get("ports"),
        "lifecycleId": lifecycle_attr.get("id"),
        "lifecycleBytes": len(lifecycle.encode()),
        "lifecycleSha256": hashlib.sha256(lifecycle.encode()).hexdigest() if lifecycle else None,
        "file": launchable.get("file"),
        "rawURL": launchable.get("rawURL"),
    }


def validate_payload(payload: dict) -> None:
    summary = public_summary(payload)
    if summary["viewAccess"] != "public":
        raise SystemExit("Payload must create an anyone-with-link Launchable with viewAccess=public.")
    if summary["ports"]:
        raise SystemExit("Payload must not expose inbound ports for the Outposts worker.")
    if summary["parameterNames"] != EXPECTED_PARAMETERS:
        raise SystemExit("Payload parameter names or ordering changed unexpectedly.")
    if summary["parameterRequirements"] != EXPECTED_PARAMETER_REQUIREMENTS:
        raise SystemExit("Payload parameter required/optional settings changed unexpectedly.")
    if summary["lifecycleBytes"] <= 0 or summary["lifecycleBytes"] > SCRIPT_LIMIT:
        raise SystemExit("Payload lifecycle script is missing or over the API limit.")


def deployed_lifecycle_script(launchable_id: str, lifecycle_id: str) -> str:
    response = request(
        "GET",
        "/launchable/lifecycle-script?"
        + urllib.parse.urlencode({"envId": launchable_id, "scriptId": lifecycle_id}),
    )
    try:
        script = response["attrs"]["script"]
    except (KeyError, TypeError) as exc:
        raise SystemExit("The lifecycle-script endpoint returned no script.") from exc
    if not isinstance(script, str) or not script:
        raise SystemExit("The deployed lifecycle script is empty.")
    return script


def verify(launchable_id: str, expected_lifecycle: str | None = None) -> dict:
    launchable = request(
        "GET",
        f"/launchables/{urllib.parse.quote(launchable_id)}",
        not_found_retries=4,
    )
    lifecycle_id = (
        launchable.get("buildRequest", {})
        .get("vmBuild", {})
        .get("lifeCycleScriptAttr", {})
        .get("id")
    )
    if not lifecycle_id:
        raise SystemExit("The Launchable has no lifecycle script ID.")
    lifecycle = deployed_lifecycle_script(launchable_id, lifecycle_id)
    summary = public_summary(launchable, lifecycle)
    print(json.dumps(summary, indent=2))

    if summary["viewAccess"] != "public":
        raise SystemExit("Expected an anyone-with-link Launchable with viewAccess=public.")
    if summary["ports"]:
        raise SystemExit("Expected no inbound ports for the Outposts worker.")
    if summary["instanceType"] != "g2-standard-8:nvidia-l4:1":
        raise SystemExit("The created Launchable does not have the expected L4 default.")
    if summary["parameterNames"] != EXPECTED_PARAMETERS:
        raise SystemExit("The created Launchable does not expose the expected parameters.")
    if summary["parameterRequirements"] != EXPECTED_PARAMETER_REQUIREMENTS:
        raise SystemExit("The created Launchable has incorrect required/optional parameter settings.")
    if expected_lifecycle is not None and lifecycle.encode() != expected_lifecycle.encode():
        raise SystemExit("The deployed lifecycle script does not match the local rendered wrapper.")
    return summary


def build_edit_payload(current: dict, desired: dict, launchable_id: str) -> dict:
    current_build = copy.deepcopy(current.get("buildRequest", {}))
    current_workspace = copy.deepcopy(current.get("createWorkspaceRequest", {}))
    if current_workspace.get("cloudCredId") and current_workspace.get("workspaceGroupId"):
        current_workspace.pop("workspaceGroupId")
    lifecycle_attr = current_build.get("vmBuild", {}).get("lifeCycleScriptAttr", {})
    lifecycle_id = lifecycle_attr.get("id")
    if not lifecycle_id:
        raise SystemExit("The existing Launchable has no lifecycle script ID to preserve.")

    desired_build = desired["buildRequest"]
    current_build["parameters"] = copy.deepcopy(desired_build["parameters"])
    current_build.setdefault("vmBuild", {})["lifeCycleScriptAttr"] = {
        "id": lifecycle_id,
        "name": desired_build["vmBuild"]["lifeCycleScriptAttr"]["name"],
        "script": desired_build["vmBuild"]["lifeCycleScriptAttr"]["script"],
    }

    edit_payload = {
        "id": launchable_id,
        "name": current.get("name") or desired["name"],
        "description": desired["description"],
        "createWorkspaceRequest": current_workspace,
        "buildRequest": current_build,
    }
    if current.get("file"):
        edit_payload["file"] = copy.deepcopy(current["file"])
    if "viewAccess" in edit_payload:
        raise SystemExit("Edit payload must not include viewAccess.")
    return edit_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["render", "create", "edit", "verify"], required=True)
    parser.add_argument("--launchable-id", default=DEFAULT_LAUNCHABLE_ID)
    parser.add_argument("--org-id", default=DEFAULT_ORG_ID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.mode == "verify":
        verify(args.launchable_id, build_bootstrap())
        return 0

    payload = build_payload()
    validate_payload(payload)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_payload={args.output}")

    if args.mode == "render":
        print(
            json.dumps(
                {
                    "orgId": args.org_id,
                    "payload": public_summary(payload),
                    "viewAccess": payload["viewAccess"],
                },
                indent=2,
            )
        )
        return 0

    token = load_access_token()
    seconds_left = token_seconds_left(token)
    if seconds_left is not None:
        print(f"brev_token_seconds_left={seconds_left}")
        if seconds_left <= 0:
            raise SystemExit("Brev token is expired. Run `brev login` and retry.")

    if args.mode == "edit":
        current = request(
            "GET",
            f"/launchables/{urllib.parse.quote(args.launchable_id)}",
        )
        edit_payload = build_edit_payload(current, payload, args.launchable_id)
        response = request(
            "POST",
            f"/organizations/{urllib.parse.quote(args.org_id)}/v2/launchables/edit",
            token,
            edit_payload,
        )
        launchable_id = response.get("id", args.launchable_id)
    else:
        response = request(
            "POST",
            f"/organizations/{urllib.parse.quote(args.org_id)}/v2/launchables",
            token,
            payload,
        )
        launchable_id = response["id"]
    test_url = f"https://brev.nvidia.com/launchable/deploy?launchableID={urllib.parse.quote(launchable_id)}"
    result_key = "updatedId" if args.mode == "edit" else "createdId"
    print(json.dumps({result_key: launchable_id, "testUrl": test_url}, indent=2))
    summary = verify(
        launchable_id,
        payload["buildRequest"]["vmBuild"]["lifeCycleScriptAttr"]["script"],
    )
    print(
        json.dumps(
            {
                result_key: launchable_id,
                "testUrl": test_url,
                "verified": summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

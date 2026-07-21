# Devin Outposts on Brev Launchable

This Launchable provisions a Brev NVIDIA GPU VM and runs a small single-machine Outposts coordinator as a persistent `systemd` service. The coordinator polls one Outpost, verifies and caches each session's pinned official `devin-remote` binary, claims one session at a time, launches the runtime, and releases the claim afterward. It does not use the Devin CLI, expose an inbound port, or install Jupyter.

## Test inputs

- `DEVIN_WORKER_TOKEN` (required): the `cog_...` worker token copied when provisioning a Devin API service user whose role includes **Use outpost machine**. Enter the worker token only in the masked Launchable field.
- `DEVIN_OUTPOST_NAME` (required): the exact name shown under **Devin Settings > Environment > Outposts**. This is a name, not a token or env ID. The setup resolves and validates its internal ID before starting.
- `REPO_URL` (optional): a public GitHub HTTPS repository. If blank, Devin starts with an empty workspace. Private repositories and branch selection are intentionally not part of v1.

The Launchable form intentionally asks for the visible Outpost name. The setup script resolves that name to Devin's internal `outpost_env-...` identifier automatically.

The default is a stoppable GCP VM with 1 NVIDIA L4 (24 GB), 8 vCPU, 32 GB RAM, and a 256 GB disk. Region and zone are intentionally unpinned so Brev can place the VM based on capacity and the provider selected on the deploy screen. The deploy screen is the source of truth for current pricing.

## Create the anyone-with-link test Launchable

Authenticate once without putting credentials on the command line:

```bash
brev login
```

Then render and inspect the payload:

```bash
python3 .brev/devin-outposts/manage_launchable.py --mode render
```

Create the Launchable in the Brev Launchables organization:

```bash
python3 .brev/devin-outposts/manage_launchable.py --mode create
```

Update the existing link-accessible test Launchable while preserving its compute configuration and lifecycle ID:

```bash
python3 .brev/devin-outposts/manage_launchable.py --mode edit
```

The helper deliberately creates `viewAccess=public`, which means anyone with the link can open it. It does not make the Launchable discoverable in Explore. Publishing to Explore is a separate step after validation.

## Readable lifecycle bootstrap

The full installer is tracked at [`launchable-setup.sh`](launchable-setup.sh). Brev's lifecycle-script field is limited to 16 KiB, so the Launchable contains a short readable bootstrap instead of embedding a compressed base64 payload. The bootstrap downloads the installer from an immutable Git commit, verifies the exact file with a pinned SHA-256 digest, and only then runs it.

To publish an installer update:

1. Commit and push the updated `launchable-setup.sh`.
2. Verify the raw GitHub file matches the committed file byte-for-byte.
3. Update `SETUP_REVISION` and `SETUP_SHA256` in `manage_launchable.py`.
4. Render, inspect, and edit the Launchable. Never point the public Launchable at a mutable branch such as `main`.

## What the VM setup does

1. Fails clearly unless an NVIDIA GPU and driver are available.
2. Installs only base OS tools needed for the coordinator; it does not install the Devin CLI.
3. Optionally clones a public GitHub repository under `/home/ubuntu/devin-workspace/repos`, without initializing submodules or executing repository content.
4. Lists the account's Outposts through Devin's public API and requires an exact name match. An invalid worker token fails before the service starts.
5. Stores the machine-scoped token in `/etc/devin-outposts/worker.env` as root-only mode `0600` and keeps it out of the session runtime's environment.
6. Downloads the latest direct runtime as a preflight check, verifies its published SHA-256 checksum, and caches it under `/var/lib/devin-outposts-launchable/binaries`.
7. Starts `devin-outposts-worker.service`. The root coordinator polls for pending Linux sessions, downloads the session-pinned runtime before claiming, runs `devin-remote serve` as `ubuntu` with a clean environment, and releases the claim when the session ends or fails.
8. Writes `/var/lib/devin-outposts-launchable/ready` only after the service has verified a runtime and successfully parsed the selected Outpost's queue.

This direct path intentionally implements the minimal one-VM orchestration that `devin-remote` needs. One VM serves one session at a time and returns to the queue afterward. Launch more workers against the same Outpost for concurrency.

## VM checks

```bash
nvidia-smi
sudo systemctl status devin-outposts-worker.service
sudo journalctl -u devin-outposts-worker.service -f
sudo test -f /var/lib/devin-outposts-launchable/ready
```

After testing, stop or delete the Brev VM to stop charges. Revoke the test Outposts token in Devin when the test is finished.

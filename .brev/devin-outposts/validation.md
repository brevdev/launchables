# Devin Outposts Launchable validation

## Static checks

- [ ] `bash -n launchable-setup.sh` passes.
- [ ] The embedded `/usr/local/bin/devin-outposts-worker` heredoc also passes `bash -n` after extraction.
- [ ] `python3 -m py_compile manage_launchable.py` passes.
- [ ] The rendered readable lifecycle bootstrap is non-empty and below 16 KiB.
- [ ] The bootstrap pins a full Git commit and SHA-256 digest, and the raw GitHub installer matches both the committed file and digest.
- [ ] The payload has `viewAccess=public`, `ports=[]`, `firewallRules=[]`, VM mode, and Jupyter disabled.
- [ ] Only the clearly named worker token and Outpost name are required; API URL and public repository are optional.
- [ ] The setup and rendered lifecycle contain no Devin CLI download or `devin worker start` command.
- [ ] No token is present in tracked files, lifecycle logs, command arguments, or the readiness record.

## First deployment

- [ ] Use a newly generated machine-scoped worker token and exact Outpost name from the NVIDIA beta Devin account.
- [ ] Set `DEVIN_API_URL=https://nvidia.beta.devinenterprise.com/api`.
- [ ] Leave `REPO_URL` blank.
- [ ] Confirm the deploy page shows the L4 default and allows compute editing.
- [ ] Confirm the VM becomes ready without a Secure Link or Jupyter CTA.
- [ ] Run `nvidia-smi`; record GPU name, VRAM, and driver.
- [ ] Run `nvcc --version` if the selected image advertises a compiler toolkit. Record clearly if only the CUDA driver/runtime is present.
- [ ] Confirm `/etc/devin-outposts/worker.env` is owned by root and mode `0600`.
- [ ] Confirm `devin-outposts-worker.service` remains active for at least two minutes and the readiness record says `worker_mode=direct-devin-remote`.
- [ ] Confirm the service preflights a checksum-verified runtime and caches it under `/var/lib/devin-outposts-launchable/binaries` before reporting ready.
- [ ] In Devin, run a session on this Outpost and have it execute `nvidia-smi` on the worker.
- [ ] Confirm the session uses its `spec.remote_binary_sha`, reaches `running`, and releases its claim after sleep or termination.
- [ ] Run a second sequential Devin session to ensure the coordinator returns to the queue after session one.

## Repository deployment

- [ ] Deploy a fresh VM with a small public GitHub `REPO_URL` and confirm the default branch is cloned as a child of the worker's working directory.
- [ ] Confirm submodules were not initialized and repository content was not executed during setup.
- [ ] Confirm an invalid or private repository fails with a clear error and does not prompt for credentials.
- [ ] Confirm a branch cannot be supplied independently because v1 intentionally has no branch parameter.

## Negative and security checks

- [ ] Missing worker token fails before the worker is installed.
- [ ] Missing Outpost name fails before the worker is installed.
- [ ] A worker-token/API mismatch fails with only an HTTP status, not response contents or token text.
- [ ] An unknown Outpost name fails before service creation and lists only the available non-secret Outpost names.
- [ ] A `cog_...` value is rejected as an Outpost rather than being accepted by an empty queue response.
- [ ] A bad runtime checksum is never executed or claimed; an HTTP 409 claim race is treated as non-fatal.
- [ ] SIGTERM or a runtime failure performs one bounded best-effort release using the same persisted acceptor ID.
- [ ] The session runtime runs as the target VM user and does not inherit `DEVIN_OUTPOSTS_TOKEN`, `DEVIN_API_URL`, or unrelated service environment variables.
- [ ] An HTTP, credential-bearing, query-bearing, or fragment-bearing Devin API URL is rejected.
- [ ] A CPU VM or VM with a broken NVIDIA driver fails with an actionable GPU error.
- [ ] No inbound firewall rule, public port, Jupyter server, or passwordless-sudo change is added.
- [ ] The optional partner redirect flow is absent from this v1 Launchable.

## Cleanup and promotion

- [ ] Stop or delete every test VM and verify billing has stopped.
- [ ] Revoke the test token in Devin.
- [ ] Only after the checks above pass, change access from anyone-with-link to `published` for Explore/catalog discovery.

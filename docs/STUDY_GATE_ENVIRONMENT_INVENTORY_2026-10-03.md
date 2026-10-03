# Environment inventory annex — 2026-10-03

The existing gate amendment, boundaries and aggregate decision remain unchanged.
Before any real gate, generate `environment_identity.json` outside the checkout
with `scripts/environment_identity.py generate --settings <source-settings>
--output <external-directory>/environment_identity.json`. The resulting
`settings-effective.json` carries the generated inventory path and its seal.
Never overwrite either artifact. Generate after the final source commit and
definitive image build; this annex documents the procedure without a circular
commit or image hash entered manually as evidence.

The inventory contains full source commit and tree, study fingerprint, effective
recipes and distributions, lock and Dockerfile hashes, platform and GPU/driver,
Ollama version and full model digest, complete HF and index file manifests,
preregistration hash and cloud instance identity. For Docker, the host creates a
read-only receipt from image/container inspect with matching full image IDs and
injects that ID into the container launch environment. This receipt is a host
trust boundary; container code does not independently attest the Docker daemon.
Retain that inspect evidence outside the repository and report.

Container startup, live preflight and the runner verify the sealed inventory
against the live runtime. Missing inventory, changed seal, source, recipes,
packages, driver, image receipt, Ollama or artifacts reject admission. Reports
consume `environment_identity.py report`; they never transcribe identity hashes
as proof. Window identities include the generated inventory seal and must match
exactly. A platform or hardware difference from exp12 is recorded, not treated
as demonstrated output equivalence. Synthetic checks cannot grant GO.

No participant-visible controls or wording change. No participant recruitment is
authorized; ethical approval and the author's document updates remain pending.

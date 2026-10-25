# Study cloud deployment

The image installs the exact `requirements-lock.txt` pins on Python 3.14.3 Linux.
The Windows app environment has a different installed package inventory; equality
of generated outputs across platforms is **not established**. Record `pip freeze`,
driver version, image IDs, commit, model digests and artifact hashes before any gate.
The `multidimensional_scoring` and `terminology_normalization` flags remain present
and unused by the pipeline; this deployment does not implement them.

Build context contains `repository/` (clean checkout including `.git`) and
`vendor/thesis-paper-agents/` (tracked files of private commit
`5db68deb53e6e88a668670101d8372a5cff0dcf7`, no `.git` or credentials), plus
`vendor/manifest.json` (revision and SHA-256 for every tracked file). Copy the
repository `.dockerignore` to the context root, then run
`docker build -f repository/Dockerfile -t cloudrag-study:locked CONTEXT`.
The source snapshot is obtained with the operator's existing authorized Git
session, scanned for secrets, verified and privately transferred. No credential
is installed in the VM. The build changes only this dependency's transport from
the locked Git URL to a verified local editable source (version 3.2.0); every
other lock line is preserved. The original lock is retained unmodified. Its
installed editable source resides outside the app checkout so that the deployed
Git identity stays clean. Record this transport difference in the inventory.

The image retains a clean Git checkout at `/opt/cloudrag/repository`. Mount models
and indices read-only at its data directories. Mount private configuration and
session storage outside the checkout at `/srv/cloudrag`. The entrypoint verifies
the exact commit, clean checkout, reviewed seal, complete artifact manifest,
Ollama version and model digest before serving or measuring. No service-account
keys belong in the image, repository or deployment files. GCS uses VM metadata
credentials and immutable object generations, then reads the same generation to
verify SHA-256. Failed copies leave a pending state and prevent admission.

Use us-central1-a, g2-standard-4 with one L4, a retained pd-balanced boot disk,
VM deletion protection, boot auto-delete disabled and native maximum runtime
with termination action STOP. Compute disks do not expose independent deletion
protection: retention is enforced by attachment policy and the audited interlock.
Initial auxiliaries use CPU, Ollama uses L4. Remedies require the committed
amendment, new data and any applicable equivalence prerequisite.

Internet ingress is only HTTPS 443 with a valid certificate and app invitations.
Bind Streamlit and Ollama to loopback. Use a private backend behind the HTTPS load
balancer; restrict its firewall to Google's documented proxy/health ranges.
Administrative SSH uses IAP only, with a temporary rule for 35.235.240.0/20 and
an expiring key. Preserve its configuration before removing the rule at closure.
No public SSH, HTTP frontend, TLS bypass or public storage bucket is acceptable.

Operation order: start with a checked cost reserve and hard STOP deadline; verify
image/driver/assets; verify identities; prepare models; smoke or gate; verify and
download private session backups; stop the VM; confirm TERMINATED and protection.
No gate may run until both platform suites and the cloud smoke have passed.
Stopped VMs still incur disk/storage and load-balancer charges; account for these
in the cost reserve and handover. Resource IDs and commands belong in the private
audit package, never in this public runbook.

UX note: no question, instrument or generated-answer recipe changes here.
Finalization waits for a verified private backup before displaying the existing
completion message. If copying fails, the existing coordinator error is shown;
the error is preserved and new admission is blocked. Backup time is outside the
preregistered server query boundary.

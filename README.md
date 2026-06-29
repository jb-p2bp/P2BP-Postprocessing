# EC2 Queue Worker

## Overview

This repository contains the AWS EC2 instance responsible for consuming jobs from a Cloudflare Queue.

At the current stage of development, the instance can:

* Connect to a Cloudflare Queue
* Pull messages from the queue
* Validate `MESH_JOBS` messages
* Download zipped `.scanproject` archives from R2
* Merge zone scans into project LAZ outputs
* Acknowledge messages after successful processing

This repository serves as the foundation for point cloud postprocessing and meshing functionality.

---

# Repository Structure

```text
.
├── config.py
├── mesh_jobs.py               # pydantic models for MESH_JOBS queue messages
├── pull_queue.py
├── r2.py
├── scanproject_merger/        # point cloud registration & merging library
│   ├── __init__.py
│   ├── format.py
│   ├── registration.py
│   ├── visual.py
│   ├── export.py
│   └── merge.py
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

## File Descriptions

### pull_queue.py

Queue worker entry point.

Responsibilities:

* Authenticate with Cloudflare
* Pull messages from the configured queue
* Validate and dispatch mesh job messages
* Stage zipped scanproject archives from R2
* Upload merged project point-cloud outputs
* Acknowledge messages only after successful processing

### r2.py

Utilities for uploading objects to and downloading objects from Cloudflare R2 (S3-compatible, via boto3).

Responsibilities:

* Build an R2 client from the environment
* Upload local files to R2 (with overwrite protection via a HEAD check)
* Download objects to a given path, directory, or a unique temp directory
* Manage collision-free, private temp download directories

### config.py

Shared configuration helpers (`require_env`, `ConfigError`) used by the other
modules to read required environment variables.

### mesh_jobs.py

Pydantic models for the messages the worker consumes from the `MESH_JOBS`
queue. Each message carries a `type` for dispatch and a `version` pinned per
variant, so a consumer validates only the exact schema it was built for and
rejects any other version. The discriminated union (`MeshGenerateJob`,
`MeshRefineJob`) mirrors the `MeshJobQueueMessage` type defined in
`p2bp-cf-worker`; field names are camelCase to match that JSON wire contract
exactly (do not rename them to snake_case). Use `parse_mesh_job_message()` to
validate an arbitrary payload into a typed variant.

`mesh.generate` messages are processed by the queue worker. `mesh.refine`
messages are validated but intentionally rejected until refinement is implemented,
so they are retried instead of being acknowledged as completed.

### scanproject_merger/

Library that registers overlapping ScannerConsolidator `.scanproject` zones into a
single georeferenced LAS/LAZ point cloud. It is a port of the standalone
[`scanproject-merger`](https://github.com/ChrisMGeo/ScannerConsolidator) CLI as an
importable library (no command-line interface). See
[Point Cloud Registration Library](#point-cloud-registration-library) for usage.

The queue worker calls this library for `mesh.generate` jobs after extracting the
zipped `.scanproject` archives from R2.

### pyproject.toml

Project metadata and Python package dependencies, managed by [uv](https://docs.astral.sh/uv/).

### uv.lock

Pinned, reproducible dependency versions resolved by uv. Commit this file.

### README.md

Setup and development documentation.

---

# Prerequisites

Before running the worker, ensure the following are installed:

* [uv](https://docs.astral.sh/uv/) (manages Python and dependencies)
* Git
* Access to the Cloudflare account containing the queue

Verify uv installation:

```bash
uv --version
```

uv will automatically download a compatible Python (3.12+) if one is not present.

---

# Setup

## Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

## Install Dependencies

uv creates the virtual environment and installs all dependencies from `uv.lock`:

```bash
uv sync
```

---

# Cloudflare Configuration

Create a `.env` file in the repository root:

```env
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_QUEUE_ID=
CLOUDFLARE_API_TOKEN=

# R2 (object downloads)
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=
```

## Environment Variables

### CLOUDFLARE_ACCOUNT_ID

Cloudflare Account ID that owns the queue. Also used to build the R2 endpoint
URL, so it must be the 32-character hex account id.

### CLOUDFLARE_QUEUE_ID

Unique Cloudflare Queue ID.

Note:

This is **not** the queue name.

Example:

```text
Correct:
8f3d2d8a7aab4d8bbcd123456789abcd

Incorrect:
mesh-jobs
```

### CLOUDFLARE_API_TOKEN

Cloudflare API token with Queue permissions.

Required permissions:

* Queues Read
* Queues Write

### R2_ACCESS_KEY_ID

Access key id of an R2 API token (used to upload and download objects to/from R2).

### R2_SECRET_ACCESS_KEY

Secret for the R2 API token above.

### R2_BUCKET

Default R2 bucket, so upload and download calls can omit the bucket argument.
Required unless every call passes an explicit bucket argument — when omitted, the
transfer helpers fall back to this variable and raise `ConfigError` if it is unset.

### P2BP_TMP_DIR

Optional. Base directory for temporary downloads. Must be an **absolute** path,
and its parent directory must already exist. Defaults to `<system temp>/p2bp-tmp`
(i.e. `/tmp/p2bp-tmp` on the EC2 Linux host).

For security, every ancestor directory of this path must be trusted — owned by
the worker user, or under a sticky directory such as `/tmp`. Do **not** point it
under a world- or group-writable, non-sticky directory: the download tree is
created with strict ownership/permission checks, but those checks cannot protect
against a hostile ancestor that could redirect the path.

### MERGED_POINT_CLOUD_DEDUPLICATE_VOXEL

Optional. Output voxel size, in meters, for the canonical
`merged-point-cloud.laz`. Defaults to `0.02`.

### PREVIEW_POINT_CLOUD_DEDUPLICATE_VOXEL

Optional. Output voxel size, in meters, for the downsampled
`merged-point-cloud.preview.laz`. Defaults to `0.10`.

---

# Queue Requirements

The queue must have HTTP Pull enabled.

Example:

```bash
wrangler queues consumer http add mesh-jobs
```

Without HTTP Pull enabled, the worker will receive:

```text
messages cannot be pulled unless http_pull mode is enabled
```

---

# Running the instance

Run with uv (no manual activation needed):

```bash
uv run pull_queue.py
```

Or via the installed entry point:

```bash
uv run pull-queue
```

---

# Expected Behavior

When a `mesh.generate` message exists:

```text
Processing message with body type: dict
Downloading r2://<bucket>/<zone-scan-object-key> -> <workspace>/archives/000-zone.zip
Merging 1 scanproject archive(s) for organization=org_test project=proj_test
Uploading merged cloud (...) and preview (...)

Acknowledging message...

Message acknowledged successfully.
```

The worker uploads:

* `organizations/{orgId}/projects/{projectId}/merged-point-cloud.laz`
* `organizations/{orgId}/projects/{projectId}/merged-point-cloud.preview.laz`

When no message exists:

```text
Pulling one message from queue...

No messages found in queue.
```

---

# Troubleshooting

## Authentication Error (401)

Verify:

* API token is correct
* API token belongs to the correct Cloudflare account
* Account ID is correct
* Queue ID is correct

Validate token:

```bash
curl "https://api.cloudflare.com/client/v4/user/tokens/verify" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Expected:

```json
{
  "success": true
}
```

---

## HTTP Pull Not Enabled

Error:

```text
messages cannot be pulled unless http_pull mode is enabled
```

Fix:

```bash
wrangler queues consumer http add <queue-name>
```

---

# Point Cloud Registration Library

`scanproject_merger` registers overlapping `.scanproject` zones and writes one merged
LAS/LAZ point cloud plus a JSON registration report. Source packages are never
modified; the first input scan anchors the output frame.

```python
from scanproject_merger import merge_scan_projects

outputs = merge_scan_projects(
    ["path/to/scans"],                  # .scanproject dirs, or parent dirs containing them
    "path/to/results/merged.laz",       # merged LAS/LAZ output
    transformed_scans_dir="path/to/results/aligned",  # optional: one aligned LAZ per source
)

print(outputs.point_count, "points ->", outputs.output)
print("report:", outputs.report)
```

`merge_scan_projects` accepts keyword overrides for the registration and export
tuning (voxel sizes, confidence thresholds, ICP gates, loop tolerances, and
`use_visual_registration`); see its docstring for the full set. For finer control,
the lower-level `register_scans` and `export_*` functions are exported directly.

The queue worker invokes this library for `mesh.generate` messages after
downloading and extracting the R2 scan archives.

---

# Future Development

Planned additions include:

* Point cloud meshing (consuming the `scanproject_merger` registration output)
* Mesh refinement job processing

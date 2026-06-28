# EC2 Queue Worker

## Overview

This repository contains the AWS EC2 instance responsible for consuming jobs from a Cloudflare Queue.

At the current stage of development, the instance can:

* Connect to a Cloudflare Queue
* Pull messages from the queue
* Display message contents
* Acknowledge messages after processing

This repository serves as the foundation for future point cloud processing and meshing functionality.

---

# Repository Structure

```text
.
├── config.py
├── pull_queue.py
├── r2.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## File Descriptions

### pull_queue.py

Demonstration queue consumer.

Responsibilities:

* Authenticate with Cloudflare
* Pull messages from the configured queue
* Display message contents
* Acknowledge messages

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

Optional. Default R2 bucket, so upload and download calls can omit the bucket argument.

### P2BP_TMP_DIR

Optional. Base directory for temporary downloads. Must be an **absolute** path,
and its parent directory must already exist. Defaults to `<system temp>/p2bp-tmp`
(i.e. `/tmp/p2bp-tmp` on the EC2 Linux host).

For security, every ancestor directory of this path must be trusted — owned by
the worker user, or under a sticky directory such as `/tmp`. Do **not** point it
under a world- or group-writable, non-sticky directory: the download tree is
created with strict ownership/permission checks, but those checks cannot protect
against a hostile ancestor that could redirect the path.

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

When a message exists:

```text
Pulling one message from queue...

Captured message body:
{
  "organizationId": "org_test",
  "projectId": "proj_test"
}

Acknowledging message...

Message acknowledged successfully.
```

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

# Future Development

Planned additions include:

* Continuous queue polling
* Automatic EC2 shutdown after inactivity
* Point cloud meshing

```
```

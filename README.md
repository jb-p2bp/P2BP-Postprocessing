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
├── pull_queue.py
├── requirements.txt
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

### requirements.txt

Python package dependencies required to run the worker.

### README.md

Setup and development documentation.

---

# Prerequisites

Before running the worker, ensure the following are installed:

* Python 3.10+
* Git
* Access to the Cloudflare account containing the queue

Verify Python installation:

```bash
python3 --version
```

---

# Setup

## Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

## Create Virtual Environment

```bash
python3 -m venv venv
```

## Activate Virtual Environment

Linux / Ubuntu:

```bash
source venv/bin/activate
```

Deactivate:

```bash
deactivate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Cloudflare Configuration

Create a `.env` file in the repository root:

```env
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_QUEUE_ID=
CLOUDFLARE_API_TOKEN=
```

## Environment Variables

### CLOUDFLARE_ACCOUNT_ID

Cloudflare Account ID that owns the queue.

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

Activate the virtual environment:

```bash
source venv/bin/activate
```

Run:

```bash
python pull_queue.py
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
* Cloudflare R2 downloads
* Point cloud meshing
* Cloudflare R2 uploads

```
```

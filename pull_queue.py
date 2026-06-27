import json
import os
import sys
import time
import signal
import random
import logging
from typing import Any, Optional

import boto3
import requests
from cloudflare import Cloudflare
from dotenv import load_dotenv


# =========================
# CONFIG
# =========================

IDLE_LIMIT_SECONDS = int(os.getenv("IDLE_LIMIT_SECONDS", "60"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "10"))
MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "300"))
MAX_RUNTIME_SECONDS = int(os.getenv("MAX_RUNTIME_SECONDS", "43200"))  # 12h default

running = True


# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("queue-worker")


# =========================
# SIGNAL HANDLING
# =========================

def handle_signal(signum, frame):
    global running
    logger.warning(f"Received signal {signum}. Shutting down gracefully...")
    running = False


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# =========================
# HELPERS
# =========================

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logger.error(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def parse_body(body: Any) -> Any:
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body
    return body


def get_instance_id() -> Optional[str]:
    try:
        token_response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=2,
        )
        token_response.raise_for_status()
        token = token_response.text

        if not token:
            return None

        response = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        )
        response.raise_for_status()

        return response.text or None

    except requests.RequestException as e:
        logger.warning(f"IMDS unavailable: {e}")
        return None


def shutdown_self():
    instance_id = get_instance_id()

    if not instance_id:
        logger.warning("No instance ID found. Skipping shutdown.")
        sys.exit(0)

    try:
        logger.warning(f"Stopping EC2 instance {instance_id}...")

        region = require_env("AWS_REGION")

        ec2 = boto3.client(
            "ec2",
            region_name=region,
        )

        ec2.stop_instances(InstanceIds=[instance_id])

        logger.warning("Shutdown request sent successfully.")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"FAILED to stop EC2 instance: {e}")
        sys.exit(1)


# =========================
# CORE LOGIC
# =========================

def process_message(body: Any) -> None:
    """
    MUST raise Exception on failure.

    If this function does NOT raise, the message will be ACKed
    and considered permanently completed.
    """

    logger.info("Processing message:")

    # Replace this with real logic
    logger.info(json.dumps(body, indent=2))

    # Example contract enforcement (optional but recommended):
    # if something fails:
    #     raise Exception("processing failed")


def ack_message(client, queue_id, account_id, lease_id: str) -> None:
    client.queues.messages.ack(
        queue_id,
        account_id=account_id,
        acks=[{"lease_id": lease_id}],
        retries=[],
    )


# =========================
# MAIN LOOP
# =========================

def main():
    load_dotenv()

    account_id = require_env("CLOUDFLARE_ACCOUNT_ID")
    queue_id = require_env("CLOUDFLARE_QUEUE_ID")
    api_token = require_env("CLOUDFLARE_API_TOKEN")
    require_env("AWS_REGION")

    client = Cloudflare(api_token=api_token)

    consecutive_failures = 0
    start_time = time.time()

    last_successful_ack_time = time.time()

    logger.info("Queue worker started")
    logger.info(
        f"Configuration: idle_limit={IDLE_LIMIT_SECONDS}s, "
        f"max_runtime={MAX_RUNTIME_SECONDS}s, "
        f"poll_interval={POLL_INTERVAL_SECONDS}s"
    )

    while running:

        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            logger.warning("Max runtime reached. Shutting down.")
            shutdown_self()

        try:
            logger.debug("Polling queue...")

            pull_response = client.queues.messages.pull(
                queue_id,
                account_id=account_id,
                batch_size=1,
                visibility_timeout_ms=30_000,
            )

            messages = getattr(pull_response, "messages", [])
            consecutive_failures = 0

            if not messages:
                idle_for = time.time() - last_successful_ack_time

                logger.info(f"No messages. Idle since last work: {idle_for:.0f}s")

                if idle_for >= IDLE_LIMIT_SECONDS:
                    logger.warning("Idle limit reached. Shutting down EC2...")
                    shutdown_self()

                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            message = messages[0]
            body = parse_body(message.body)
            lease_id = getattr(message, "lease_id", None)

            if not lease_id:
                logger.warning("Message missing lease_id. Skipping.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # =========================
            # PROCESS (MUST RAISE ON FAILURE)
            # =========================
            process_message(body)

            # =========================
            # ACK ONLY ON SUCCESS
            # =========================
            logger.info("Acknowledging message...")
            ack_message(client, queue_id, account_id, lease_id)
            logger.info("Message acknowledged.")

            last_successful_ack_time = time.time()

        except Exception as e:
            consecutive_failures += 1

            backoff = min(
                POLL_INTERVAL_SECONDS * (2 ** (consecutive_failures - 1)),
                MAX_BACKOFF_SECONDS,
            )

            backoff *= random.uniform(0.8, 1.2)

            logger.exception(
                f"Error ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}"
            )

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error("Too many failures. Shutting down.")
                shutdown_self()

            time.sleep(backoff)

    logger.info("Worker exited cleanly.")


if __name__ == "__main__":
    main()
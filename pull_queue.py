import json
import os
import sys
import time
import signal
import random
import logging
import subprocess
from typing import Any, Optional

import boto3
import requests
from cloudflare import Cloudflare
from dotenv import load_dotenv


# =========================
# CONFIG
# =========================

load_dotenv()

IDLE_LIMIT_SECONDS = int(os.getenv("IDLE_LIMIT_SECONDS", "60"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "10"))
MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "300"))
MAX_RUNTIME_SECONDS = int(os.getenv("MAX_RUNTIME_SECONDS", "43200"))  # 12h default
SHUTDOWN_RETRY_SECONDS = int(os.getenv("SHUTDOWN_RETRY_SECONDS", "30"))

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


def get_instance_id(max_attempts: int = 3) -> Optional[str]:
    for attempt in range(1, max_attempts + 1):
        try:
            token_response = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=2,
            )
            token_response.raise_for_status()
            token = token_response.text.strip()

            if not token:
                raise requests.RequestException("IMDS returned an empty token")

            response = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
                timeout=2,
            )
            response.raise_for_status()

            instance_id = response.text.strip()
            if not instance_id:
                raise requests.RequestException("IMDS returned an empty instance ID")

            return instance_id

        except requests.RequestException as e:
            logger.warning(
                f"IMDS request failed ({attempt}/{max_attempts}): {e}"
            )

            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))

    return None


def power_off_os() -> None:
    subprocess.run(
        ["sudo", "-n", "systemctl", "poweroff"],
        check=True,
        timeout=10,
    )


def ensure_shutdown(reason: str) -> None:
    logger.critical(f"Entering shutdown mode: {reason}")

    while True:
        try:
            instance_id = get_instance_id()
            if not instance_id:
                raise RuntimeError("Unable to determine EC2 instance ID")

            region = os.getenv("AWS_REGION")
            if not region:
                raise RuntimeError("Missing required environment variable: AWS_REGION")

            logger.warning(f"Stopping EC2 instance {instance_id} via EC2 API...")

            ec2 = boto3.client("ec2", region_name=region)
            ec2.stop_instances(InstanceIds=[instance_id])

            logger.warning("EC2 stop request sent successfully.")
            sys.exit(0)

        except Exception:
            logger.exception("EC2 API shutdown failed")

        try:
            logger.critical("Attempting operating-system poweroff...")
            power_off_os()
            sys.exit(0)

        except Exception:
            logger.exception("Operating-system poweroff failed")

        logger.critical(
            f"All shutdown methods failed. Retrying in "
            f"{SHUTDOWN_RETRY_SECONDS}s."
        )
        time.sleep(SHUTDOWN_RETRY_SECONDS)


# =========================
# CORE LOGIC
# =========================

def process_message(body: Any) -> None:
    """
    MUST raise Exception on failure.

    If this function does NOT raise, the message will be ACKed
    and considered permanently completed.
    """

    # Avoid logging the full payload because it may contain sensitive data.
    logger.info("Processing message with body type: %s", type(body).__name__)

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
    account_id = require_env("CLOUDFLARE_ACCOUNT_ID")
    queue_id = require_env("CLOUDFLARE_QUEUE_ID")
    api_token = require_env("CLOUDFLARE_API_TOKEN")
    require_env("AWS_REGION")

    client = Cloudflare(api_token=api_token)

    consecutive_poll_failures = 0
    consecutive_processing_failures = 0
    start_time = time.monotonic()

    last_completed_work_time = time.monotonic()

    logger.info("Queue worker started")
    logger.info(
        f"Configuration: idle_limit={IDLE_LIMIT_SECONDS}s, "
        f"max_runtime={MAX_RUNTIME_SECONDS}s, "
        f"poll_interval={POLL_INTERVAL_SECONDS}s"
    )

    while running:

        if time.monotonic() - start_time > MAX_RUNTIME_SECONDS:
            logger.warning("Max runtime reached. Shutting down.")
            ensure_shutdown("maximum runtime reached")

        failure_stage = "poll"

        try:
            logger.debug("Polling queue...")

            pull_response = client.queues.messages.pull(
                queue_id,
                account_id=account_id,
                batch_size=1,
                visibility_timeout_ms=30_000,
            )

            messages = getattr(pull_response, "messages", [])
            consecutive_poll_failures = 0

            if not messages:
                idle_for = time.monotonic() - last_completed_work_time

                logger.info(f"No messages. Idle since last work: {idle_for:.0f}s")

                if idle_for >= IDLE_LIMIT_SECONDS:
                    logger.warning("Idle limit reached. Shutting down EC2...")
                    ensure_shutdown("idle limit reached")

                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            failure_stage = "processing"
            message = messages[0]
            body = parse_body(message.body)
            lease_id = getattr(message, "lease_id", None)

            if not lease_id:
                raise RuntimeError("Pulled message is missing lease_id")

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

            last_completed_work_time = time.monotonic()
            consecutive_processing_failures = 0

        except Exception as e:
            if failure_stage == "poll":
                consecutive_poll_failures += 1
                active_failures = consecutive_poll_failures
            else:
                consecutive_processing_failures += 1
                active_failures = consecutive_processing_failures

            backoff = min(
                POLL_INTERVAL_SECONDS
                * (2 ** (active_failures - 1))
                * random.uniform(0.8, 1.2),
                MAX_BACKOFF_SECONDS,
            )

            logger.exception(
                f"{failure_stage.capitalize()} error "
                f"({active_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}"
            )

            if active_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"Too many consecutive {failure_stage} failures.")
                ensure_shutdown(
                    f"too many consecutive {failure_stage} failures"
                )

            time.sleep(backoff)

    logger.info("Worker exited cleanly.")


if __name__ == "__main__":
    main()

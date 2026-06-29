"""EC2-hosted worker that drains a Cloudflare queue and self-stops when idle.

The instance is powered on by a Cloudflare worker that first enqueues a message,
so the worker assumes that being up means there should be work. It polls the
queue, processes one message at a time, and stops the EC2 instance when the
queue stays empty past the idle limit, the max runtime is hit, or failures pile
up. See ensure_shutdown for the EC2-API-then-OS-poweroff shutdown strategy.

Runtime prerequisites:
  * Environment variables: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_QUEUE_ID,
    CLOUDFLARE_API_TOKEN, AWS_REGION (loaded from the environment / a .env file).
  * Process supervisor: run under a supervisor (e.g. systemd). The worker exits
    on graceful signals and expects the supervisor NOT to relaunch it once the
    instance is stopping (it absorbs stop signals during the EC2 stop window).
  * IAM role permissions: ec2:StopInstances and ec2:DescribeInstanceAttribute.
  * Instance metadata: IMDSv2 must be reachable at IMDS_BASE_URL.
  * Shutdown behavior: the instance's InstanceInitiatedShutdownBehavior must be
    "stop" (verified at startup) so the OS-poweroff fallback cannot terminate it.
  * sudo: a passwordless rule allowing `sudo -n systemctl poweroff`, scoped to
    exactly that command.
"""

import json
import os
import sys
import time
import signal
import random
import logging
import subprocess
from typing import Any, NoReturn, Optional

import boto3
import requests
from cloudflare import Cloudflare
from dotenv import load_dotenv


# =========================
# CONFIG
# =========================

load_dotenv()

# Seconds with no completed work before the instance shuts itself down. The
# idle clock starts at boot (see last_completed_work_time in main), so an
# instance that boots to an empty queue stops itself after this window. This is
# intentional: a Cloudflare worker enqueues a message *before* powering this
# instance on, so an instance being up implies work should be waiting. If none
# is found, the instance is unexpected and should shut down promptly.
IDLE_LIMIT_SECONDS = int(os.getenv("IDLE_LIMIT_SECONDS", "60"))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "10"))
MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "300"))
MAX_RUNTIME_SECONDS = int(os.getenv("MAX_RUNTIME_SECONDS", "43200"))  # 12h default
SHUTDOWN_RETRY_SECONDS = int(os.getenv("SHUTDOWN_RETRY_SECONDS", "30"))

# How long a pulled message stays invisible to other pulls before redelivery.
VISIBILITY_TIMEOUT_MS = int(os.getenv("VISIBILITY_TIMEOUT_MS", "30000"))

# EC2 Instance Metadata Service (IMDSv2). These are fixed infrastructure facts,
# not per-deploy tunables.
IMDS_BASE_URL = "http://169.254.169.254"
IMDS_TOKEN_TTL_SECONDS = "21600"  # 6h; only bounds a single metadata fetch
IMDS_REQUEST_TIMEOUT_SECONDS = 2

running = True


# =========================
# LOGGING
# =========================

logger = logging.getLogger("queue-worker")


# =========================
# SIGNAL HANDLING
# =========================

def handle_signal(signum: int, frame: Any) -> None:
    global running
    logger.warning(f"Received signal {signum}. Shutting down gracefully...")
    running = False


def configure_runtime() -> None:
    """Apply process-global side effects.

    Kept out of module import so the module can be imported (e.g. by tests)
    without reconfiguring logging or replacing the process signal handlers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
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
                f"{IMDS_BASE_URL}/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": IMDS_TOKEN_TTL_SECONDS},
                timeout=IMDS_REQUEST_TIMEOUT_SECONDS,
            )
            token_response.raise_for_status()
            token = token_response.text.strip()

            if not token:
                raise requests.RequestException("IMDS returned an empty token")

            response = requests.get(
                f"{IMDS_BASE_URL}/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
                timeout=IMDS_REQUEST_TIMEOUT_SECONDS,
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
    # IMPORTANT: This is an OS-initiated shutdown. AWS applies the instance's
    # InstanceInitiatedShutdownBehavior attribute here, so this only *stops* the
    # instance when that attribute is "stop". If it were "terminate", this would
    # DESTROY the instance instead of stopping it. verify_shutdown_behavior()
    # is called at startup to guarantee the attribute is "stop" before the
    # worker ever reaches this fallback.
    subprocess.run(
        ["sudo", "-n", "systemctl", "poweroff"],
        check=True,
        timeout=10,
    )


def verify_shutdown_behavior(instance_id: Optional[str], region: str) -> None:
    """Refuse to run unless OS-initiated shutdown will *stop* the instance.

    The power_off_os() fallback triggers an OS-initiated shutdown, which AWS
    resolves using the instance's InstanceInitiatedShutdownBehavior attribute.
    If that attribute is "terminate", the fallback would destroy the instance
    instead of stopping it, leaving nothing for the launcher to start again.
    Fail fast at startup rather than risk that during a later shutdown.

    Requires the ec2:DescribeInstanceAttribute IAM permission. This is
    fail-closed: if the attribute cannot be positively confirmed to be "stop"
    (unknown instance ID, missing permission, transient API error, or any other
    value), the worker exits rather than risk destroying the instance.
    """
    if not instance_id:
        logger.critical(
            "Cannot verify InstanceInitiatedShutdownBehavior without an "
            "instance ID. Refusing to run."
        )
        sys.exit(1)

    try:
        ec2 = boto3.client("ec2", region_name=region)
        attribute = ec2.describe_instance_attribute(
            InstanceId=instance_id,
            Attribute="instanceInitiatedShutdownBehavior",
        )
        behavior = attribute["InstanceInitiatedShutdownBehavior"]["Value"]
    except Exception:
        logger.critical(
            "Could not verify InstanceInitiatedShutdownBehavior (check the "
            "ec2:DescribeInstanceAttribute permission). Refusing to run.",
            exc_info=True,
        )
        sys.exit(1)

    if behavior != "stop":
        logger.critical(
            "InstanceInitiatedShutdownBehavior is %r, not 'stop'. The OS "
            "poweroff fallback would destroy this instance. Refusing to run; "
            "set the shutdown behavior to 'stop' on the launch template.",
            behavior,
        )
        sys.exit(1)

    logger.info("Verified InstanceInitiatedShutdownBehavior=stop.")


def ensure_shutdown(
    reason: str,
    instance_id: Optional[str],
    region: str,
) -> NoReturn:
    logger.critical(f"Entering shutdown mode: {reason}")

    # Once we have committed to shutting down, ignore SIGTERM/SIGINT for the
    # entire shutdown window -- the EC2 API attempt(s), the OS-poweroff
    # fallback, and the retry sleeps. Absorbing a supervisor's stop signal
    # prevents it from observing an exit and launching a replacement worker on
    # an instance that is already going down. The OS still terminates us at the
    # end of its own shutdown sequence via SIGKILL, which cannot be ignored.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shutdown_instance_id = instance_id

    while True:
        try:
            if not shutdown_instance_id:
                shutdown_instance_id = get_instance_id()

            if not shutdown_instance_id:
                raise RuntimeError("Unable to determine EC2 instance ID")

            logger.warning(
                f"Stopping EC2 instance {shutdown_instance_id} via EC2 API..."
            )

            ec2 = boto3.client("ec2", region_name=region)
            ec2.stop_instances(InstanceIds=[shutdown_instance_id])

            logger.warning(
                "EC2 stop request accepted. Waiting for the instance to stop."
            )

            # Stay alive (signals already ignored above) until the OS shutdown
            # sequence terminates us.
            while True:
                time.sleep(60)

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

    # ----------------------------------------------------------------------
    # TODO: IMPLEMENT REAL MESSAGE PROCESSING HERE.
    #
    # WARNING: This function is currently a no-op. Because it returns without
    # raising, the caller will ACK every message, which PERMANENTLY DELETES it
    # from the queue without doing any work. Do not run this against a queue
    # carrying real messages until processing is implemented.
    #
    # Contract reminder when implementing: raise on failure (so the message is
    # retried), return normally only on success (so the message is acked).
    # ----------------------------------------------------------------------


def ack_message(
    client: Cloudflare, queue_id: str, account_id: str, lease_id: str
) -> None:
    client.queues.messages.ack(
        queue_id,
        account_id=account_id,
        acks=[{"lease_id": lease_id}],
        retries=[],
    )


def pull_one(client: Cloudflare, queue_id: str, account_id: str) -> list:
    """Pull a single message from the queue, returning a (possibly empty) list."""
    pull_response = client.queues.messages.pull(
        queue_id,
        account_id=account_id,
        batch_size=1,
        visibility_timeout_ms=VISIBILITY_TIMEOUT_MS,
    )
    return getattr(pull_response, "messages", [])


def handle_message(
    client: Cloudflare, queue_id: str, account_id: str, message: Any
) -> None:
    """Process a single message and ack it only on success.

    Raises on any failure so the caller can record a processing failure and
    leave the message un-acked for redelivery.
    """
    body = parse_body(message.body)
    lease_id = getattr(message, "lease_id", None)

    if not lease_id:
        raise RuntimeError("Pulled message is missing lease_id")

    # PROCESS (MUST RAISE ON FAILURE)
    process_message(body)

    # ACK ONLY ON SUCCESS
    logger.info("Acknowledging message...")
    ack_message(client, queue_id, account_id, lease_id)
    logger.info("Message acknowledged.")


# =========================
# MAIN LOOP
# =========================

def main() -> None:
    configure_runtime()

    account_id = require_env("CLOUDFLARE_ACCOUNT_ID")
    queue_id = require_env("CLOUDFLARE_QUEUE_ID")
    api_token = require_env("CLOUDFLARE_API_TOKEN")
    region = require_env("AWS_REGION")

    instance_id = get_instance_id()
    if instance_id:
        logger.info(f"Cached EC2 instance ID: {instance_id}")
    else:
        logger.warning(
            "Could not cache the EC2 instance ID during startup. "
            "Shutdown mode will retry metadata lookup."
        )

    verify_shutdown_behavior(instance_id, region)

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
            ensure_shutdown("maximum runtime reached", instance_id, region)

        failure_stage = "poll"

        try:
            logger.debug("Polling queue...")

            messages = pull_one(client, queue_id, account_id)
            consecutive_poll_failures = 0

            if not messages:
                idle_for = time.monotonic() - last_completed_work_time

                logger.info(f"No messages. Idle since last work: {idle_for:.0f}s")

                if idle_for < IDLE_LIMIT_SECONDS:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Final confirming poll before stopping. This closes the window
                # between observing an empty queue and the asynchronous
                # stop_instances call: a producer can enqueue in that gap and,
                # because the instance still appears "on", nothing else would
                # start a worker to pick the message up. If the confirming poll
                # is also empty we stop; otherwise we process the new message.
                logger.warning(
                    "Idle limit reached. Performing a final confirming poll..."
                )
                messages = pull_one(client, queue_id, account_id)

                if not messages:
                    logger.warning("Queue confirmed empty. Shutting down EC2...")
                    ensure_shutdown("idle limit reached", instance_id, region)

                logger.info(
                    "Message arrived during idle confirmation; "
                    "processing instead of shutting down."
                )

            failure_stage = "processing"
            handle_message(client, queue_id, account_id, messages[0])

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
                    f"too many consecutive {failure_stage} failures",
                    instance_id,
                    region,
                )

            time.sleep(backoff)

    logger.info("Worker exited cleanly.")


if __name__ == "__main__":
    main()

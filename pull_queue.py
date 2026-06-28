import json
import os
import sys
from typing import Any

from cloudflare import Cloudflare
from dotenv import load_dotenv


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        print(f"Missing required environment variable: {name}")
        sys.exit(1)

    return value


def parse_body(body: Any) -> Any:
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    return body


def main() -> None:
    load_dotenv()

    account_id = require_env("CLOUDFLARE_ACCOUNT_ID")
    queue_id = require_env("CLOUDFLARE_QUEUE_ID")
    api_token = require_env("CLOUDFLARE_API_TOKEN")

    client = Cloudflare(api_token=api_token)

    print("Pulling one message from queue...")

    pull_response = client.queues.messages.pull(
        queue_id,
        account_id=account_id,
        batch_size=1,
        visibility_timeout_ms=30_000,
    )

    messages = pull_response.messages or []

    if not messages:
        print("No messages found in queue.")
        return

    message = messages[0]
    body = parse_body(message.body)
    lease_id = message.lease_id

    print("\nCaptured message body:")
    print(json.dumps(body, indent=2))

    if not lease_id:
        print("No lease_id found. Cannot acknowledge message.")
        sys.exit(1)

    print("\nAcknowledging message...")

    client.queues.messages.ack(
        queue_id,
        account_id=account_id,
        acks=[
            {
                "lease_id": lease_id,
            }
        ],
        retries=[],
    )

    print("Message acknowledged successfully.")


if __name__ == "__main__":
    main()
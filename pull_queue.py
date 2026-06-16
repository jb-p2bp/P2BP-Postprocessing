import os
import sys
import json
import requests
from dotenv import load_dotenv


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def main() -> None:
    load_dotenv()

    account_id = require_env("CLOUDFLARE_ACCOUNT_ID")
    queue_id = require_env("CLOUDFLARE_QUEUE_ID")
    api_token = require_env("CLOUDFLARE_API_TOKEN")

    base_url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/queues/{queue_id}/messages"
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    print("Pulling one message from queue...")

    pull_response = requests.post(
        f"{base_url}/pull",
        headers=headers,
        json={
            "batch_size": 1,
            "visibility_timeout_ms": 30000,
        },
        timeout=30,
    )

    print(f"Pull status: {pull_response.status_code}")

    if not pull_response.ok:
        print("Pull failed:")
        print(pull_response.text)
        sys.exit(1)

    data = pull_response.json()
    print("Raw pull response:")
    print(json.dumps(data, indent=2))

    messages = data.get("result", {}).get("messages", [])

    if not messages:
        print("No messages found in queue.")
        return

    message = messages[0]
    body = message.get("body")
    lease_id = message.get("lease_id")

    print("\nCaptured message body:")
    print(json.dumps(body, indent=2))

    if not lease_id:
        print("No lease_id found. Cannot acknowledge message.")
        sys.exit(1)

    print("\nAcknowledging message...")

    ack_response = requests.post(
        f"{base_url}/ack",
        headers=headers,
        json={
            "acks": [
                {
                    "lease_id": lease_id,
                }
            ]
        },
        timeout=30,
    )

    print(f"Ack status: {ack_response.status_code}")

    if not ack_response.ok:
        print("Ack failed:")
        print(ack_response.text)
        sys.exit(1)

    print("Message acknowledged successfully.")


if __name__ == "__main__":
    main()

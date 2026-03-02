#!/usr/bin/env python3
"""
Originate an outbound call via Asterisk ARI.
When the called person picks up, they are connected to the AI voice agent
which talks using the cloned voice.

Usage:
  python3 make_call.py 18573948674
  python3 make_call.py +1-857-394-8674
  python3 make_call.py <any_phone_number>

The number is dialed out through the SIP trunk.
"""

import sys
import re
import requests

ARI_URL  = "http://127.0.0.1:8088/ari"
ARI_USER = "voiceagent"
ARI_PASS = "voiceagent123"
ARI_APP  = "voiceagent"


def clean_number(raw: str) -> str:
    """Strip everything except digits from the phone number."""
    return re.sub(r"[^\d]", "", raw)


def make_call(number: str):
    """Originate an outbound call to `number` via the SIP trunk.
    When the remote party answers, Asterisk puts the channel into
    the Stasis(voiceagent) app — the ARI agent takes over from there."""

    number = clean_number(number)
    if not number:
        print("Error: no valid phone number provided")
        sys.exit(1)

    # Ensure US number has leading 1 (e.g. 4099953437 → 14099953437)
    if len(number) == 10:
        number = "1" + number

    # Provider requires 9871 prefix for US outbound calls
    dial_str = f"9871{number}"
    print(f"\U0001f4de  Dialing {number} (→ {dial_str}) via SIP trunk ...")

    # ARI originate: create a channel that dials out through the trunk,
    # and when answered, drop it into our Stasis app.
    resp = requests.post(
        f"{ARI_URL}/channels",
        auth=(ARI_USER, ARI_PASS),
        params={
            "endpoint":    f"PJSIP/{dial_str}@trunk",
            "app":         ARI_APP,
            "callerId":    "Voice Agent <1760990923>",
            "timeout":     60,
        },
    )

    if resp.status_code in (200, 201):
        data = resp.json()
        channel_id = data.get("id", "unknown")
        print(f"✅  Call originated! Channel: {channel_id}")
        print(f"    Waiting for {number} to pick up ...")
        print(f"    When they answer, the AI voice agent will start talking.")
        print(f"    The agent uses your cloned voice.")
    else:
        print(f"❌  Failed to originate call: HTTP {resp.status_code}")
        print(f"    {resp.text[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 make_call.py <phone_number>")
        print("Example: python3 make_call.py 18573948674")
        sys.exit(1)

    make_call(sys.argv[1])

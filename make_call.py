#!/usr/bin/env python3
"""
Originate an outbound call via Asterisk ARI.
When the called person picks up, the AI voice clone agent takes over.

Usage:
  python3 make_call.py +447449868303          # UK via Naira (working route)
  python3 make_call.py 447449868303           # same, auto-normalised
  python3 make_call.py +18573948674 --route astpp   # US via ASTPP trunk

Routes:
  naira  (default) — NairaCalls trunk; UK calls confirmed working
  astpp            — legacy ASTPP trunk (US, may need provider enablement)
"""

import sys
import re
import argparse
import requests

ARI_URL  = "http://127.0.0.1:8088/ari"
ARI_USER = "voiceagent"
ARI_PASS = "voiceagent123"
ARI_APP  = "voiceagent"


def normalize_to_e164(raw: str) -> str:
    """Normalize messy user input to best-effort E.164 (e.g. +447449868303)."""
    raw = (raw or "").strip()
    digits = re.sub(r"[^\d]", "", raw)
    if not digits:
        return ""
    # 00-prefixed international (e.g. 00447449868303)
    if raw.startswith("00") and len(digits) > 2:
        return "+" + digits[2:]
    # Already international with +
    if raw.startswith("+"):
        return "+" + digits
    # Looks like UK (44...) with enough digits
    if digits.startswith("44") and len(digits) >= 11:
        return "+" + digits
    # 11-digit US
    if digits.startswith("1") and len(digits) == 11:
        return "+" + digits
    # 10-digit US local → assume +1
    if len(digits) == 10:
        return "+1" + digits
    # Generic fallback
    return "+" + digits


def build_endpoint(e164: str, route: str) -> tuple[str, str]:
    """
    Return (ARI endpoint string, dial_string) for the chosen route.
    e164   — normalised number like +447449868303
    route  — 'naira' or 'astpp'
    """
    route = (route or "naira").lower().strip()
    digits = e164.lstrip("+")

    if route in ("naira", "uk"):
        # Dial the E.164 number directly through the naira trunk
        endpoint = f"PJSIP/{e164}@naira"
        return endpoint, e164

    if route in ("astpp", "us"):
        # ASTPP needs 9871 prefix and digits without +
        if len(digits) == 10:
            digits = "1" + digits
        dial = f"9871{digits}"
        endpoint = f"PJSIP/{dial}@trunk"
        return endpoint, dial

    raise ValueError(f"Unknown route '{route}'. Use 'naira' or 'astpp'.")


def make_call(number: str, route: str = "naira", caller_id: str = "1001"):
    """Originate a call; when remote answers, ARI voice-clone agent takes over."""
    e164 = normalize_to_e164(number)
    if not e164:
        print("Error: no valid phone number provided")
        sys.exit(1)

    try:
        endpoint, dial_str = build_endpoint(e164, route)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"📞  Dialing {e164} via route={route} (endpoint={endpoint}) ...")

    resp = requests.post(
        f"{ARI_URL}/channels",
        auth=(ARI_USER, ARI_PASS),
        params={
            "endpoint":  endpoint,
            "app":       ARI_APP,
            "appArgs":   "outbound",   # tells voice agent this is outbound
            "callerId":  caller_id,
            "timeout":   60,
        },
    )

    if resp.status_code in (200, 201):
        data = resp.json()
        print(f"✅  Call originated! Channel: {data.get('id', 'unknown')}")
        print(f"    Waiting for {e164} to pick up ...")
        print(f"    When answered, the AI voice agent talks in the cloned voice.")
    else:
        print(f"❌  Failed: HTTP {resp.status_code}")
        print(f"    {resp.text[:500]}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Originate outbound call into ARI cloned-voice agent")
    parser.add_argument("number",
        help="Target phone number (e.g. +447449868303 or 447449868303)")
    parser.add_argument("--route", default="naira",
        choices=["naira", "uk", "astpp", "us"],
        help="Outbound trunk: naira=UK working, astpp=legacy US (default: naira)")
    parser.add_argument("--caller-id", default="1001",
        help="Caller ID presented to the receiver")
    args = parser.parse_args()
    make_call(args.number, args.route, args.caller_id)

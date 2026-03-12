# Asterisk/SIP Incident Log (March 2026)

This file records what worked, what failed, and how each issue was resolved during the deployment.

## Successes

- MicroSIP extension registration to Asterisk was restored.
- SIP trunk to `sip.siptraffic.com` reached `Registered/Avail` state.
- SIP trunk to `sip.nairacalls.com` reached `Registered/Avail` state.
- UK outbound test calling succeeded through both trunks.
- Dialplan cleanup removed mixed-provider confusion and restored predictable routing.

## Failures and Root Causes

1. Failure: MicroSIP timeout when pointed to provider.
- Root cause: Softphone was configured to register directly to a trunk provider.
- Fix: Register softphone to Asterisk extension endpoint instead.

2. Failure: Dialed number not found in dialplan.
- Root cause: New outbound routes were appended outside `[from-internal]`.
- Fix: Move all user-originated outbound routes into `[from-internal]`.

3. Failure: US calls rejected, UK calls succeeded.
- Root cause: Provider account routing/caller-ID policy restrictions for US destinations.
- Fix attempted: caller ID changes, dialing format retries (`+1`, `1`, `001`).
- Final status: still provider-side restricted.

4. Failure: Toll-free US routes returned temporary unavailable.
- Root cause: Destination class not enabled or blocked by provider policy.
- Final status: unresolved at account level.

5. Failure: Exact override route did not trigger.
- Root cause: Override landed in wrong context (`from-trunk` instead of `from-internal`).
- Fix: relocate override into `[from-internal]`.

## High-Value Lessons

- Trunk registration success is not enough; destination entitlements must be confirmed.
- Keep routing logic simple while debugging. Avoid multi-provider failovers until one provider is proven.
- Always run `dialplan show from-internal` after edits to confirm pattern placement.
- For policy rejections, SIP traces (`From`, `P-Asserted-Identity`) are required evidence for support teams.

## What To Send Provider Support

Use this template:

```text
Please enable outbound US (+1) routing on our SIP account.
Tested destination: +18573948674
Observed result: rejected due to caller ID policy / temporarily unavailable.
Please confirm approved caller ID formats and required verified ANI.
```

## Known Good Call Paths

- Internal: `1001 -> 1002`
- Outbound UK via test prefix: working

## Known Blocked Paths (at time of writing)

- Outbound US mobile: blocked by provider policy
- Outbound US toll-free: blocked by provider policy

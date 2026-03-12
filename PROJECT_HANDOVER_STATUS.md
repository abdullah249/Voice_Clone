# Project Handover Status

Last updated: 2026-03-12

## Scope Completed

- Voice cloning app is in repository and deployable.
- Server deployment guide exists (`DEPLOYMENT_GUIDE.md`).
- Asterisk + SIP operational documentation added:
  - `ASTERISK_SIP_PLAYBOOK.md`
  - `ASTERISK_SIP_INCIDENTS_MAR2026.md`

## Operational Outcome

- UK outbound calling path validated.
- US outbound path remains provider-restricted and requires provider-side enablement.

## Immediate Next Actions For New Engineer

1. Reuse `ASTERISK_SIP_PLAYBOOK.md` to deploy PBX on a new server.
2. Confirm trunk destination entitlements before production cutover.
3. Validate caller ID policy with provider using one US mobile and one US toll-free test number.
4. Keep SIP logger evidence for every failed test attempt.

## Risks

- Current provider account may not support required US destinations without manual enablement.
- Business continuity depends on provider escalation response time.

## Suggested Provider Evaluation Criteria

- Explicit US mobile + toll-free outbound support.
- Clear caller-ID and CLI verification policy.
- Stable PJSIP interoperability with Asterisk.
- Fast support SLA for route policy changes.

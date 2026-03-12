# Asterisk + SIP Playbook

This document is the repeatable setup guide for deploying the calling stack on a new server.

## 1. Target Architecture

Call flow:

`MicroSIP client -> Asterisk (PJSIP) -> SIP trunk provider (NairaCalls/Telnyx/etc.)`

Important rule:

- Do not register MicroSIP directly to trunk providers.
- Register MicroSIP to Asterisk extension (for example `1001`).

## 2. Server Prerequisites

```bash
sudo apt update && sudo apt install -y asterisk ufw fail2ban
sudo systemctl enable asterisk
sudo systemctl start asterisk
```

Recommended open ports:

- `5060/udp` (SIP)
- `5080/udp` (if used for client-facing SIP transport)
- `10000:20000/udp` (RTP)

## 3. Core Asterisk Config Files

Paths:

- `/etc/asterisk/pjsip.conf`
- `/etc/asterisk/extensions.conf`
- `/etc/asterisk/modules.conf`
- `/etc/asterisk/rtp.conf`
- `/etc/asterisk/http.conf`
- `/etc/asterisk/ari.conf`

### 3.1 `pjsip.conf` baseline

Use your real values for usernames/passwords/IPs.

```ini
[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0:5060

[1001]
type=endpoint
transport=transport-udp
context=from-internal
disallow=all
allow=ulaw,alaw
aors=1001
auth=1001-auth
direct_media=no
rewrite_contact=yes
rtp_symmetric=yes
force_rport=yes

[1001-auth]
type=auth
auth_type=userpass
username=1001
password=CHANGE_ME

[1001]
type=aor
max_contacts=5
remove_existing=yes
qualify_frequency=30

; Example trunk
[naira]
type=endpoint
transport=transport-udp
context=from-trunk
disallow=all
allow=ulaw,alaw
aors=naira
outbound_auth=naira-auth
from_domain=sip.nairacalls.com
from_user=YOUR_TRUNK_USERNAME
rewrite_contact=yes
rtp_symmetric=yes
force_rport=yes

[naira-auth]
type=auth
auth_type=userpass
username=YOUR_TRUNK_USERNAME
password=YOUR_TRUNK_PASSWORD

[naira]
type=aor
contact=sip:sip.nairacalls.com:5060
qualify_frequency=30

[naira-reg]
type=registration
transport=transport-udp
outbound_auth=naira-auth
server_uri=sip:sip.nairacalls.com:5060
client_uri=sip:YOUR_TRUNK_USERNAME@sip.nairacalls.com
retry_interval=60
forbidden_retry_interval=300
expiration=300
line=yes
endpoint=naira
```

### 3.2 `extensions.conf` baseline

Keep outbound routes inside `[from-internal]`. This prevented multiple failures during this project.

```ini
[from-internal]
exten => 1001,1,Dial(PJSIP/1001,20)
exten => 1002,1,Dial(PJSIP/1002,20)

; Prefix 6 -> trunk test
exten => _6X.,1,NoOp(Outbound via naira: ${EXTEN})
 same => n,Set(NUM=${EXTEN:1})
 same => n,Dial(PJSIP/${NUM}@naira,45)
 same => n,Hangup()

[from-trunk]
exten => _X.,1,NoOp(Inbound from trunk)
 same => n,Stasis(voiceagent)
 same => n,Hangup()
```

### 3.3 `modules.conf` minimum

```ini
[modules]
autoload=yes
noload => chan_sip.so
```

### 3.4 `rtp.conf` minimum

```ini
[general]
rtpstart=10000
rtpend=20000
```

### 3.5 `http.conf` for ARI

```ini
[general]
enabled=yes
bindaddr=0.0.0.0
bindport=8088
```

### 3.6 `ari.conf` minimum

```ini
[general]
enabled = yes
pretty = yes

[voicebot]
type = user
read_only = no
password = CHANGE_ME
```

## 4. Firewall Rules

```bash
sudo ufw allow 22/tcp
sudo ufw allow 5060/udp
sudo ufw allow 5080/udp
sudo ufw allow 10000:20000/udp
sudo ufw allow 8088/tcp
sudo ufw status
```

Also allowlist provider signaling IPs if your policy requires strict source filtering.

## 5. Reload + Verify Commands

```bash
sudo asterisk -rx "pjsip reload"
sudo asterisk -rx "dialplan reload"
sudo asterisk -rx "module reload"

sudo asterisk -rx "pjsip show endpoints"
sudo asterisk -rx "pjsip show registrations"
sudo asterisk -rx "dialplan show from-internal"
```

Expected checks:

- Extension `1001` appears and can become `Avail`.
- Trunk registration shows `Registered`.
- Prefix route (for example `_6X.`) is present in `from-internal`.

## 6. MicroSIP Client Profile

Use these fields for the client softphone:

- SIP Server: `ASTERISK_PUBLIC_IP:5080` (or `5060` if that is your client SIP port)
- SIP Proxy: `ASTERISK_PUBLIC_IP:5080`
- Username/Login: `1001`
- Domain: `ASTERISK_PUBLIC_IP`
- Password: extension password from `pjsip.conf`
- Transport: UDP (or TCP if your network requires it)

If this is pointed to the trunk provider instead of Asterisk, registration and calls will fail.

## 7. Standard Test Plan

1. Register MicroSIP extension `1001` to Asterisk.
2. Place internal call `1001 -> 1002`.
3. Place outbound UK test call via prefix route.
4. Place outbound US mobile test call.
5. Place outbound US toll-free test call.
6. Capture SIP trace if any failure appears.

Trace commands:

```bash
sudo asterisk -rvvvvv
pjsip set logger on
```

## 8. Provider Caveats

- Successful trunk registration does not guarantee all destination countries are enabled.
- Some carriers block US dialing until caller ID policy is satisfied.
- Verified portal number is not always equal to full outbound route entitlement.

## 9. Quick Clone To New Server

1. Install Asterisk + firewall tools.
2. Copy baseline configs from this playbook.
3. Replace placeholders with new credentials.
4. Open SIP/RTP/ARI ports.
5. Reload and run verification commands.
6. Test internal call, then outbound UK and US.
7. If US fails with caller ID policy, contact provider support with exact sample number and timestamp.

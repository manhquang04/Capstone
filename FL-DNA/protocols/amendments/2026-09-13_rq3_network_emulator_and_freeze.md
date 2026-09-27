# Amendment — RQ3 network emulator and official benchmark freeze

**Amendment ID:** `2026-09-13-RQ3-NETWORK-EMULATOR-FREEZE`  
**Timestamp:** 2026-09-13 (before official RQ3 measurements)  
**Approved scope:** implementation choice required by the already approved RQ3
protocol and the supervisor's instruction to execute Group 1  

## Tool decision

RQ3 uses the versioned user-space emulator in
`experiments/rq3_user_space_network.py`. It transfers the complete application
payload through a local socket pair, verifies the received bytes, and applies
the frozen uplink, RTT and packet-loss profile with a monotonic-clock delay.
Loss is represented as deterministic seeded packet loss followed by reliable
retransmission; the raw loss/retransmission counts are retained.

The host provides `/usr/sbin/dnctl` and `/sbin/pfctl`, but both returned an
operation/permission error during the preflight. They require privileged,
machine-global network state, which is unavailable in this execution context
and would weaken per-cell isolation. The user-space choice is cross-platform,
requires no global state, is hashable, and can be replayed exactly.

This benchmark therefore reports `APPLICATION_PAYLOAD_BYTES`; it does not call
the value total wire traffic and does not include TCP/IP, TLS or link-layer
headers.

## Frozen transport semantics

- One cell runs at a time and no RQ1/RQ2/RQ3 job runs concurrently.
- Each client upload is executed sequentially, matching the current simulated
  FL implementation's client loop. Round transfer time is the sum of client
  uploads.
- Payloads are segmented into 1200-byte emulation packets. Each packet attempt
  is independently lost with the profile probability and reliable transport
  retransmits until success. Modeled retransmission bytes and the maximum
  retransmission round are retained. A transfer is a timeout only when its
  complete modeled duration exceeds the already frozen 30-second limit.
- `RAW_FLOAT32` and `DNA_TRANSFORM_TRANSPORT` use the same compact binary tensor
  framing. `LOSSLESS_DNA_AES_GCM` uses the implemented AES-256-GCM Base64/JSON
  payload including nonce, authentication tag, shape, dtype and field names.
- DNA Transform uses the approved conservative configuration
  `(block_size=256, mix_ratio=0.08, keep_ratio=0.88,
  shrink_factor=0.45)` and then the compact binary transport.
- All cells use identical deterministic FraudMLP update tensors. Client count
  changes the number of equal-sized model updates, so the official 27-cell
  matrix is the supervisor-specified 3 profiles × 3 methods × 3 client counts.
  The earlier optional `fixed_data_per_client_if_compute_permits` wording does
  not create a second 27-cell matrix.

## Frozen uncertainty method

Paired uncertainty uses a percentile bootstrap over repetition indices with
10,000 resamples and seed `271828`. The entire 95% CI must be within a paired
threshold. Exact payload ratios and failure counts are not bootstrapped.

## Smoke-test boundary

Before freezing the official config, an implementation-only smoke test may run
one warm-up and one measured repetition on a synthetic fixture in a separately
labeled directory. It may test byte equality, AES authentication, transform
framing, timeout handling and output schema only. Its timing values may not be
used to change a threshold, profile, repetition count, input size or method.

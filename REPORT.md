# glm-4.5-air-106b — GLM-4.5-Air (MoE, no spec-decode)

Zhipu `GLM-4.5-Air` MoE (106B total / ~12B active) at **Q4_K_M (73 GB, 2 shards)**. Doesn't fit 48 GB → offload-bound.
**Decode ~12 tok/s, 7/7 clean** (no crashes — no MTP draft buffer competing for the 5080's headroom, unlike 122B).

## Decode throughput (7 workloads, `-fit on`)
| Workload | decode tok/s |
|---|---|
| Translation (multilingual) | **13.1** |
| Math / reasoning | 12.9 |
| Summarization | 12.9 |
| Chat / dialogue | 12.6 |
| JSON / structured | 12.5 |
| Code generation | 11.9 |
| Free-form prose | 9.0 |
| **Average** | **~12.1** (prefill ~26.9) |

## Serving configuration
| Param | Value |
|---|---|
| Backend | llama.cpp build-master, `mctl switch glm-air` :8017 |
| Quant | Q4_K_M, 73 GB, 2 shards (unsloth) |
| Placement | **`-fit on`** — active path on GPU, routed experts → CPU RAM |
| KV / ctx / fa | `--cache-type-k/v q8_0` · 4096 · `-fa on` |
| VRAM | 5090 ~31 GB + 5080 ~14 GB on GPU; rest of 73 GB in CPU RAM |

## Tuning-research + config rationale
- **`-fit on`** is the right default for a 73 GB model on 48 GB VRAM — keeps attention/shared/dense on GPU, spills
  only routed experts. No hand-tuned split needed; loads clean.
- **No native speculative decoding** (unlike Qwen3.5-122B's MTP) → no draft-accept speedup, so decode is the raw
  offload-bound rate. This is why GLM-Air (~12) is slower than the larger 122B-MTP (~30): MTP hides the offload latency,
  GLM-Air pays it in full.
- **Filename gotcha (fixed):** multi-shard GGUFs MUST keep the `-00001-of-00002` suffix or llama.cpp aborts with
  `invalid split file name`. Don't rename shards on download.

## Analysis — limiter
73 GB ≫ 48 GB → deep on the wrong side of the VRAM-fit cliff. ~25 GB of routed experts in CPU RAM, read per token.
**Limiter = CPU-RAM bandwidth**, with no MTP/spec assist to amortize it → ~12 tok/s, the floor for a no-spec 100B-class
MoE on this 48 GB + 90 GB rig. The flat per-workload curve (9–13, except prose) confirms it's bandwidth-bound, not
compute-bound (compute-bound models show a wider spread).

## Failures → fixes
- **First load: `invalid split file name`** — shards had been renamed without the `-of-` suffix. Fixed by restoring the
  canonical `GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf` naming.
- **No stability crashes** — unlike 122B, GLM-Air ran all 7 workloads continuously without OOM. No MTP draft buffer
  means the 5080 keeps more headroom for KV spikes.

## Verdict
✅ **Works, 7/7 clean, ~12 tok/s.** A capable 106B reasoning model but **offload-bound and spec-decode-less → ~2.5×
slower than 122B-MTP** on the same rig. Use when you want GLM's specific capabilities; for raw speed at this size the
MTP-equipped 122B wins. **Disk: candidate for cleanup** (not a designated keeper — only Qwen3.5-122B + Leanstral stay).

_2026-06-27 · `-fit on`, 7-workload clean run · keyingd · part of the >120B offload-class survey._

# serve-glm-4.5-air — how to serve **GLM-4.5-Air-106B** efficiently (any GPU)

*A GPU-agnostic operations manual for one model. The reference tok/s is from one rig; the **recipe applies to any GPU**
— figure out your VRAM, then follow it.*

- **Architecture:** MoE (106B total / ~12B active), no spec-decode
- **Fits in VRAM?** NO — Q4_K_M ≈73 GB; offload-bound

## Recipe
**Plain offload:** **`-fit on`** + `-fa on` + q8 KV. No native draft head, so no spec-decode lever — decode is the raw offload-bound rate. `-fit on` keeps the active path on GPU and spills routed experts to RAM; don't hand-tune `--n-cpu-moe` (fragile on a 100B). Multi-shard GGUF: keep the `-00001-of-00002` suffix.

**Serving flags (llama.cpp):**
```
-fit on -fa on --cache-type-k q8_0 --cache-type-v q8_0 -c 4096
```

## Reference throughput
~12 tok/s decode, 7/7 workloads clean (no crashes — no MTP draft buffer competing for the small card's headroom). ~2.5× slower than the MTP-equipped 122B because it pays the offload latency in full.

## Failures → fixes
- `invalid split file name` → shards were renamed without the `-of-` suffix; restore canonical naming.
- Slow vs a 122B? Expected — no spec-decode. The size, not a misconfig, is the limiter.

## Verdict
Capable 106B reasoning model; offload-bound + no spec-decode → the floor for a no-spec 100B MoE. Use for GLM's specific strengths, not raw speed.

---
## The one decision: does it FIT in your VRAM?
Estimate size ≈ params × bytes/weight (Q4≈0.5, Q8≈1, FP16≈2 B/param) + KV + ~2–3 GB overhead.
- **Fits** → full GPU residency, **no offload**, single card if it fits on one → *bandwidth-bound, fast.*
- **Doesn't fit** → offload experts to RAM (use `-fit on`), keep the active path on GPU → *RAM-bandwidth-bound, slower.*

## Measure honestly
Use the server's **`/completion` decode timings** (`predicted_per_second`), greedy, cache off, multiple workloads —
NOT short OpenAI wall-time (it understates decode). See `bench_decode.py`.

## Files
- `REPORT.md` — the detailed benchmark (throughput · config · tuning-research+sources · analysis · failures), if present.
- `bench_decode.py` — honest decode-tok/s measurement (`/completion` timings).
- `launcher-entry.json` — a ready-to-paste config (name + serving flags) for whatever model server you run.

*Part of a per-model serving-playbook set.*

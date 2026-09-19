# Experiments and negative results

All numbers below are from the reference Windows machine in the README. They
should be treated as engineering evidence, not universal performance claims.

## Selected: workload-aware placement

On controlled no-MTP tasks, identity placement measured 23.2–24.3 tok/s while
the coding profile measured 30.8–31.4 tok/s. With MTP, held-out mixed tasks
measured 33.6–51.1 tok/s depending mainly on draft acceptance.

## Selected: 64K Q4 cache

The 48K profile kept 160 experts per layer on GPU. The 64K cache required four
more CPU experts per layer, leaving 156 GPU experts. Four warm-task means were
47.18 tok/s at 48K and 47.25 tok/s at 64K, which is indistinguishable within
run-to-run noise. The 64K profile became the default.

## Rejected: separate MTP placement

A separately profiled MTP permutation required reducing main-model GPU expert
capacity. Held-out results fell to 44.5/34.6/43.5/44.2 tok/s, so preserving
main-model slots was more valuable.

## Rejected: variable per-layer capacity

A prefix-safe allocator redistributed the same total expert slots according to
marginal router-hit gain. It loaded successfully but fell to
44.4/31.1/36.9/38.4 tok/s. Aggregate hit maximization ignored the sequential
per-layer critical path.

## Rejected: fine-grained peer-GPU expert execution

Direct cross-device activation round trips measured about 0.097 ms median on
Windows/WDDM. Adding peer experts to eight layers covered too few routed
assignments and reduced the controlled mean by 6.9–7.7%. Kernel/event/transfer
overhead exceeded the CPU work removed.

## Rejected: adjacent-layer top-1 prefetch

Offline transition recall for CPU-tail assignments was only 9.1%. The canary
also introduced a scalar GPU-to-host synchronization and sometimes computed an
expert that was not selected. Mean decode regressed from 26.85 to 25.28 tok/s.

Machine-readable values are in `benchmarks/benchmark-summary.json`.

## Selected: 96K Q4 cache

The latest profile moved another eight routed experts per layer to CPU, leaving
148 GPU-resident and 364 CPU-resident routed experts. On the same boot, four
matched short coding requests averaged 46.1 tok/s at 96K versus 46.6 tok/s at
64K, a 1.1% difference. A 70,019-token cold prompt completed at 1,154 prompt
tok/s with 60.7 seconds to first token and returned the expected result.

After that long prompt, reported free VRAM was approximately 322 MiB on GPU0
and 1,367 MiB on GPU1. The profile is therefore validated on the reference
machine but has little margin for unrelated GPU users. See
`benchmarks/context-96k-trial.json`.

## Kernel, MTP and CPU-worker matrix

The 96K profile was exercised across activation GEMV modes 0/1/2, automatic
and forced MoE tile selection, MTP draft ceilings 3/4/5, and 8/12/16/20 CPU
workers. Important results:

- Forced wide or narrow MoE tiles did not beat automatic selection in a
  fixed-seed confirmation.
- Draft 4 averaged 42.55 tok/s and regressed clearly. Draft 5 tied draft 3 in
  one short matrix but had lower acceptance (72.19% versus 84.37%) and no
  reliable advantage.
- Eight and twenty CPU workers were slower. A final 512-token confirmation
  measured 47.83 tok/s with 16 workers versus 46.08 with 12.
- INT8 GEMV mode 2 measured 47.60 tok/s in the final 512-token comparison,
  versus 47.83 with it disabled. The default therefore uses
  `EXL3_INT8_GEMV=0` and avoids a small activation approximation without
  giving up measured speed.

The selected default remains dynamic MTP with a ceiling of three draft tokens,
16 CPU workers and automatic MoE tile selection. Aggregate values are in
`benchmarks/tuning-matrix-summary.json`; per-case values are in
`benchmarks/tuning-matrix-raw.jsonl`.

## Claude Code: backend speed versus agent speed

Warm long Claude Code turns measured 33.5-41.8 output tok/s and 38.8 tok/s as
a weighted wall-rate mean. A large warm turn with 68,635 input tokens reused
97.7% of its prefix and emitted 5,745 tokens in 145 seconds (39.6 output
tok/s). A cold post-compaction turn looked much slower because it had to
prefill 48,730 uncached input tokens.

This distinction matters: the fixed backend benchmark can approach 48 tok/s,
but a real agent loop includes prefill, gateway work, tool execution and short
generations. The aggregate measurements and compaction policy are documented
in `benchmarks/claude-code-96k-summary.json` and `docs/CLAUDE_CODE.md`.

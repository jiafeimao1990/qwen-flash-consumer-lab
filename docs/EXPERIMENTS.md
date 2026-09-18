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

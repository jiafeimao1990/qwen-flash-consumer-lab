# Architecture

Qwen3.8-Flash-Next has 512 routed experts in each of 48 MoE layers. The current
96K profile keeps attention, recurrent components and 148 routed experts per
layer on the two GPUs. The other 364 routed experts remain in system memory
and use ExLlamaV3's CPU MoE path.

The patch adds four layers around that upstream path:

1. **Router tracing** records selected expert IDs and placement without storing
   prompt text.
2. **Static workload placement** permutes each layer so the most frequently
   selected experts occupy the available GPU slots.
3. **Bounded adaptation** accumulates routing counts during a request and can
   exchange a limited number of hot/cold experts only between requests. It
   never changes placement mid-generation.
4. **Profiling and research hooks** expose split timing, variable per-layer
   capacity, peer-GPU dispatch and transition-driven prefetch experiments.

The stable launcher begins with a mixed workload profile and permits at most
two promotions per layer in a sweep. This prevents early layers from consuming
the global swap budget.

The implementation deliberately avoids traditional tensor parallelism for the
CPU-expert split. GPU-resident experts use the existing ExLlamaV3 kernels, CPU
experts use the existing fused worker, and both portions overlap within a
layer. MTP uses the model's built-in component with dynamic draft length.

# Qwen Flash Consumer Lab

Reproducible Windows experiments for running **Qwen3.8-Flash-Next EXL3** on
two 16 GB consumer NVIDIA GPUs with CPU/GPU expert splitting, workload-aware
expert placement, MTP, 64K context and real agent-tool validation.

> This is not a new inference engine. It is an independently developed patch,
> profiler, launch profile and benchmark suite built on ExLlamaV3 v1.5.0.

## What was validated

Reference machine:

- Windows 11
- Ryzen 9 9950X
- 128 GB DDR5-5200
- 2 × RTX 5060 Ti 16 GB, no NVLink
- Qwen3.8-Flash-Next EXL3 4.05 bpw

Stable profile:

| Setting | Value |
|---|---:|
| Context | 65,536 tokens |
| KV cache | Q4 |
| GPU budget | 14.5 + 14.5 GiB |
| GPU-resident routed experts | 156 per layer |
| CPU routed experts | 356 per layer |
| CPU workers | 16 |
| MTP | dynamic, draft ceiling 3 |
| Reasoning budget | 1,024 tokens |

Measured results:

| Workload | Result |
|---|---:|
| Four warm coding prompts | 47.25 tok/s mean |
| DeepSeek Harness tool task | 38.3–40.4 tok/s decode |
| Harness initial 8,037-token prefill | 721 tok/s |
| Harness prefix reuse | 71–97% |
| Harness quality check | 10/10 independent tests passed |

The Harness task created a dependency-free Python CLI, wrote its tests, ran
them and completed a manual smoke test without intervention. Raw structured
results are in [`benchmarks/`](benchmarks/).

## What is original here

- Router tracing and workload profile generation.
- Frequency-guided static expert placement.
- Bounded between-request hot/cold placement adaptation.
- Per-layer capacity allocation and transition-profile analysis tools.
- Experimental peer-GPU and adjacent-layer prefetch implementations, retained
  behind opt-in environment variables.
- Parameterized TabbyAPI, llama-swap and DeepSeek Harness integration.
- Controlled positive and negative benchmark results.

The stable speedup came primarily from profiling and placement. Fine-grained
peer-GPU expert dispatch and adjacent-layer single-expert prefetch were slower
on this Windows/WDDM PCIe machine; those failures are documented rather than
hidden.

## Repository layout

```text
patches/       Patch against ExLlamaV3 v1.5.0
scripts/       Launch, patch, trace-analysis and allocation tools
configs/       Parameterized TabbyAPI template
profiles/      Example workload-derived expert profiles
integrations/  llama-swap and DeepSeek Harness examples
benchmarks/    Machine-readable measurements
docs/          Architecture and rejected experiments
tests/         Analysis-tool tests
```

## Manual setup

This first release intentionally does not provide a one-click installer.

1. Clone the exact upstream base and apply the patch:

```powershell
git clone --branch v1.5.0 https://github.com/turboderp-org/exllamav3.git
git clone https://github.com/jiafeimao1990/qwen-flash-consumer-lab.git
cd qwen-flash-consumer-lab
.\scripts\apply-patch.ps1 -ExllamaRoot ..\exllamav3
```

2. Install ExLlamaV3 and TabbyAPI according to their upstream documentation.
   Obtain a compatible Qwen3.8-Flash-Next EXL3 pack separately; weights are not
   included here.

3. Start the tested profile with your own paths:

```powershell
.\scripts\start-server.ps1 `
  -TabbyApiRoot C:\path\to\TabbyAPI `
  -ExllamaRoot C:\path\to\patched\exllamav3 `
  -ModelRoot E:\models `
  -ModelName Qwen3.8-Flash-Next-EXL3-4.05bpw
```

Use `-Port`, `-ContextTokens`, `-CpuExperts`, `-CpuThreads` and `-GpuSplit`
to adapt the profile. The 64K defaults are specific to two 16 GB cards; start
more conservatively on different hardware.

Add `-DryRun` to render and inspect the generated TabbyAPI configuration
without starting a server.

## llama-swap and Harness

The examples in [`integrations/`](integrations/) contain placeholders rather
than machine-specific paths. Add the llama-swap model block to your own config,
then add the Harness provider block to your settings. Both route through the
model ID `qwen3.8-flash-next-exl3`.

## Rebuilding a workload profile

Enable tracing, run representative prompts, then analyze the JSONL output:

```powershell
python .\scripts\analyze_moe_trace.py .\results-private\routing.jsonl `
  --placement-stats .\profiles\my-workload.json
```

Do not assume the bundled profile is universally optimal. Expert popularity
depends on prompt language and workload.

## Limits

- Tested on one Windows dual-5060-Ti system; outside results are welcome.
- Text and tool use were tested. Vision is disabled and not claimed.
- The 4.05-bpw model and 128 GB RAM were used for the published result.
- The patch is pinned to ExLlamaV3 v1.5.0 and may need rebasing for newer tags.
- Experimental peer/prefetch paths are opt-in and are not recommended defaults.

## Safety and privacy

The server defaults to `127.0.0.1`. Do not expose an unauthenticated endpoint
to a network. The repository contains no model weights, API keys, user prompts,
home-directory paths or raw private logs.

## License

Project-authored code and documentation are MIT licensed. See
[`NOTICE.md`](NOTICE.md) for ExLlamaV3 attribution.

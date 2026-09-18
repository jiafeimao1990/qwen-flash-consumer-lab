# Contributing

Hardware reports and narrowly scoped patches are welcome. Please include:

- OS, CPU, RAM capacity/speed, GPU model/count and PCIe topology.
- Exact ExLlamaV3 commit, quantization, context and KV mode.
- Cold versus warm runs, prompt tokens, generated tokens, prefill, decode,
  first-token latency and MTP acceptance.
- Whether results are single-stream or aggregate throughput.

Do not commit model weights, API keys, personal prompts or absolute paths from
your machine. Run `scripts/check-public-tree.ps1` before opening a PR.

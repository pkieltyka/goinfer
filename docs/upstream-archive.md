# Upstream archive

This fork keeps the inference library, chat CLI, server, and the CUDA, Metal and
Vulkan backends. For the removed components, use the copies at the last merged
upstream revision:

- [WebGPU backend](https://github.com/townsendmerino/goinfer/tree/36dbfbaf149aabdff43dce5766b49a65183113a7/backend/webgpu)
- [Website generator](https://github.com/townsendmerino/goinfer/tree/36dbfbaf149aabdff43dce5766b49a65183113a7/apps/site)
- [Agent demo](https://github.com/townsendmerino/goinfer/tree/36dbfbaf149aabdff43dce5766b49a65183113a7/demo/agent)
- [SDK integration harness](https://github.com/townsendmerino/goinfer/tree/36dbfbaf149aabdff43dce5766b49a65183113a7/apps/sdktest)

## Measurements

The [upstream measurement tree](https://github.com/townsendmerino/goinfer/tree/36dbfbaf149aabdff43dce5766b49a65183113a7/docs/measurements)
holds the archived raw logs, traces and profiles at their original relative paths.
Every removed raw artifact was checked against the Git blob at that upstream
revision. Measurement summaries, fork-specific records, release inputs and test
fixtures are kept in the fork.

For a historical replay that reads one of the archived files, retrieve it by path
from the pinned upstream tree. For example, with a local upstream clone, use
`git -C <upstream-clone> show 36dbfbaf149aabdff43dce5766b49a65183113a7:<path>`.

# CUDA loader path on NixOS

Scope: a Linux host running a CUDA-linked goinfer binary (`goinfer-serve`, `goinfer-chat`, or a CUDA
submodule entrypoint) where `check --hardware` reports a CUDA line such as:

```
cuda: no usable device: ... libcuda.so.1: cannot open shared object file: No such file or directory
```

This is a loader-path problem, not a kernel or parity result. NixOS keeps the NVIDIA driver library in
`/run/opengl-driver/lib`, which a plain binary's loader does not search by default. The probe therefore
cannot open `libcuda.so.1`, and the GPU is unused.

## Confirm the cause

`check --hardware` prints the registered hardware facts and the startup self-test results. For a loader
failure the CUDA line names `libcuda.so.1` (or another shared object) as missing. If the line says only
that no device exists, first check whether the environment hides it:

```sh
printf '%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}"
```

An empty `CUDA_VISIBLE_DEVICES` intentionally hides every device; unset it and re-run before treating the
result as a loader problem.

## Fix the loader path

Put the driver directory on the loader path for the command, then re-run the report:

```sh
LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
    ./goinfer-serve check --hardware
```

Equivalently, run inside a Nix shell or wrapper that provides the NVIDIA driver (for example one built
from `nvidia_x11`/the matching `cudaPackages` driver), so the library is on the path the binary inherits.
Do not copy or pin a driver library into the repository.

A successful re-run prints a `cuda: device: ...` line and a CUDA self-test result. The report is evidence
of what this process could open, not a claim that a previous run discovered or used the GPU; a run with
`CUDA_VISIBLE_DEVICES` empty explicitly did neither.

## What the self-test does with an unavailable device

`check --hardware` asks the backend's own hardware probe before it loads the four GPU self-test fixtures.
When that probe returns a definite `no usable device`, the report records one `skipped` CUDA self-test
with the probe's reason and goes on to run the CPU self-test; it does not repeat the same model-load
decline once per fixture. A GPU that opens still runs its fixtures, and a genuine kernel disagreement is
reported as `declined`, not hidden by the availability check. A zero or unknown free-memory figure from a
usable device is not treated as a missing device.

## Verified on NixOS

On 2026-11-26, a separate NixOS run on an RTX 5060 used the **published v0.6.9** binary and established only
the lookup/device/hidden split and the original noise: the bare binary could not load `libcuda.so.1`, and
its bare and hidden arms each printed **four `[resident] BuildResident declined` lines**; a command-local
`/run/opengl-driver/lib` saw the GPU and passed four fixtures (cosine 0.99939, relative L2 0.0350); the same
path with `CUDA_VISIBLE_DEVICES=` empty reported no device. That run is not evidence about the candidate
self-test change below.

The candidate change (source commit `3fb1c031`; tests
`TestGPUProbe_standaloneSkipsFixturesWhenTheDeviceIsUnavailable`,
`TestGPUProbe_standaloneRunsEveryFixtureAndStillDeclinesAMismatch`,
`TestGPUProbe_modelLoadDoesNotUseTheStandaloneDeviceCheck`, and
`TestRunSelfTests_keepsCPUTestedWhenAGPUIsUnavailable`) was exercised on NixOS with an **RTX PRO 6000** on
2026-11-26:

- the bare command records one `cuda skipped` self-test with the driver's `libcuda.so.1` error, builds
  **zero GPU fixtures**, prints **zero `[resident]` lines**, and runs the CPU self-test;
- `LD_LIBRARY_PATH=/run/opengl-driver/lib ... check --hardware` prints the device and
  `cuda pass ... over 4 fixtures` beside the CPU pass;
- the same loader path with `CUDA_VISIBLE_DEVICES=` empty records the typed
  `cuda: driver init: cuInit: CUDA_ERROR_NO_DEVICE`, one `skipped` result, and builds no GPU fixture.

The zero-fixture and no-repeated-warning claims are this candidate RTX PRO 6000 evidence, not the v0.6.9
RTX 5060 run. The raw outputs are retained with the release evidence.

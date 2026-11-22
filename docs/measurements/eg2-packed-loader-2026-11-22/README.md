# EmbeddingGemma 2 GGUF startup on a laptop CPU

On 2026-11-22, an Intel Core Ultra 7 255H laptop ran the resident-byte startup
benchmark with Linux/amd64, Go 1.27.3, logical CPUs 2–5 and `GOMAXPROCS=4`.
Each of the 18 cells ran in a fresh process with one timed iteration. See the
[raw log](laptop.log) for per-cell load and available memory.

| GGUF | Loader | Weights, ms | Verify + load + query, ms | Retained heap, MiB | Settled RSS, MiB | Peak RSS, MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | Dense serial | 485 | 811 | 1072 | 1395 | 1395 |
| Q8_0 | Packed serial | 320 | 660 | 696 | 1026 | 1026 |
| Q8_0 | Packed parallel | 134 | 469 | 696 | 1030 | 1030 |
| UD-Q6_K_XL | Dense serial | 617 | 946 | 1072 | 1339 | 1339 |
| UD-Q6_K_XL | Packed serial | 350 | 660 | 665 | 932 | 932 |
| UD-Q6_K_XL | Packed parallel | 136 | 448 | 665 | 933 | 933 |

Each value is the median of three cells. The total uses the per-process
benchmark's elapsed time. The weights interval is wall time for all tensor
jobs. In the log, token-table work and `ms/other-wall` can overlap on the
parallel arm. Settled RSS is Linux `VmRSS` after GC with the model alive,
and peak RSS is the process's `VmHWM`. Retained heap is the Go heap increase
over the resident input bytes after GC.

## Reproduce

Use the Unsloth EmbeddingGemma 2 GGUFs from revision
`031f0d4b35536f69ab3509d4893c923264fcf253`. The benchmark verifies these
files by SHA-256 before every timed load.

| File | SHA-256 |
| --- | --- |
| `embeddinggemma-2-Q8_0.gguf` | `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc` |
| `embeddinggemma-2-UD-Q6_K_XL.gguf` | `dc84f042bc3ffbe6122b3b38b2fb299d66f5f37b276c9d458db85d4a4f1e6ba3` |

Set `SNAPSHOT` to the local model directory. From the checkout, build the
test binary under the shared timing lock. Run the cells under the exclusive
lock after the normal fit and quiet-host checks.

```sh
python3 scripts/timing_lock.py run --shared --label eg2-startup-build -- \
  go test -c -o ./reader.test ./embeddinggemma2
python3 scripts/timing_lock.py run --label eg2-startup -- sh -c '
  for quant in Q8_0 UD-Q6_K_XL; do
    for arm in dense-serial packed-serial packed-parallel \
      packed-parallel packed-serial dense-serial \
      dense-serial packed-serial packed-parallel; do
      date -Iseconds
      cat /proc/loadavg
      awk "/^MemAvailable:/" /proc/meminfo
      GOMAXPROCS=4 taskset -c 2-5 ./reader.test -test.run "^\$" \
        -test.bench "^BenchmarkGGUFVerifiedStartup\$/$quant/$arm\$" \
        -test.benchtime=1x -eg2-reader-gguf-dir "$1" || exit
    done
  done
' sh "$SNAPSHOT"
```

The fixed CPU query is “How long can results be before they get cut off?”
Each cell preloads its GGUF bytes before the timer. The measured stages are
digest verification, construction and one query. The source I/O and pbrain
application startup are outside that interval.

This is a diagnostic result. The recorded one-minute load was in the 4s,
above the normal 1.0 quiet-host gate. AC and temperature were not sampled
during the run. The desktop pbrain startup target remains unqualified.

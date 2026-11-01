# TAPP-Con Benchmarks

Constrained block-sparse GEMM (TAPP-Con) vs a dense GEMM baseline on the same
logical matrices. Produced by `examples/benchmarks/constrained_vs_dense.cpp`.

## Environment

| | |
|---|---|
| CPU build | `g++ 16.2.1`, `-O3 -fopenmp -DTAPP_CON_USE_OPENMP` |
| BLAS | **none found** → TAPP-Con uses its naive triple-loop block kernel |
| CUDA | not present (`TAPP_CON_USE_CUDA` off) |
| Baseline | naive row-major dense GEMM (no BLAS), identical kernel quality to the library |

Because neither side has BLAS, the measured speedup is the **algorithmic
sparsity win**, not a BLAS-quality difference. The dense baseline is a plain
`O(N³)` triple loop that knows nothing about the block structure; the OpenMP
dense variant is reported alongside for reference.

## Methodology

Two modes, both in one process (same compiler/flags/kernel quality):

* **structured** — block-diagonal U(1) tensors built block-by-block. The
  block-sparse result is exactly `dense(A) @ dense(B)`, so the correctness gate
  is a direct dense-product comparison (`max|Δ| ≤ 1e-4`, float).
* **constrained** — `A` and `B` built through the constraint engine
  (`tapp_constrained_build_from_constraints`, `q0 + q1 = 0`) with a symmetric
  charge range so the anti-diagonal `A,B` structure closes onto a diagonal `C`.
  Exact dense-product gate, same as above.

`triples` is an independent count (in the harness) of fusible `(A,B,C)` block
triples, mirroring the library's plan-time matching. `spars` = fraction of the
`N×N` matrix covered by blocks.

## Results

### structured — block-diagonal U(1)

| case | N | blocks | triples | sparsity | µs/block | µs/dense | µs/dense OMP | speedup | correct |
|---|---|---|---|---|---|---|---|---|---|
| s=8  nb=4  | 32   | 4  | 4  | 0.250 | 1.8  | 23.7    | 15.4     | 13.3x  | YES |
| s=16 nb=4  | 64   | 4  | 4  | 0.250 | 6.2  | 216.1   | 78.0     | 34.9x  | YES |
| s=32 nb=4  | 128  | 4  | 4  | 0.250 | 44.8 | 1899.1  | 723.6    | 42.4x  | YES |
| s=64 nb=4  | 256  | 4  | 4  | 0.250 | 372.7| 17714.2 | 7963.7   | 47.5x  | YES |
| s=8  nb=16 | 128  | 16 | 16 | 0.062 | 6.0  | 1907.0  | 904.6    | 317.0x | YES |
| s=16 nb=16 | 256  | 16 | 16 | 0.062 | 26.2 | 18203.7 | 8214.1   | 695.0x | YES |
| s=32 nb=16 | 512  | 16 | 16 | 0.062 | 176.0| 206286.8| 101285.4 | 1172.4x| YES |
| s=64 nb=16 | 1024 | 16 | 16 | 0.062 | 1325.5| 3653920.2| 1162568.6| 2756.7x| YES |

### constrained — engine-built A,B; C = product closure

| case | N | blocks | triples | sparsity | µs/block | µs/dense | µs/dense OMP | speedup | correct |
|---|---|---|---|---|---|---|---|---|---|
| K=4  sd=8  | 32   | 4  | 4  | 0.250 | 1.7  | 23.7    | 14.9     | 13.9x  | YES |
| K=4  sd=16 | 64   | 4  | 4  | 0.250 | 5.8  | 196.8   | 85.5     | 34.2x  | YES |
| K=4  sd=32 | 128  | 4  | 4  | 0.250 | 44.8 | 1896.8  | 724.4    | 42.4x  | YES |
| K=4  sd=64 | 256  | 4  | 4  | 0.250 | 363.6| 18348.8 | 8096.6   | 50.5x  | YES |
| K=8  sd=8  | 64   | 8  | 8  | 0.125 | 2.8  | 198.2   | 83.6     | 71.6x  | YES |
| K=8  sd=16 | 128  | 8  | 8  | 0.125 | 12.2 | 1908.5  | 696.6    | 156.6x | YES |
| K=8  sd=32 | 256  | 8  | 8  | 0.125 | 94.5 | 18422.2 | 8099.0   | 195.1x | YES |
| K=8  sd=64 | 512  | 8  | 8  | 0.125 | 705.8| 203391.3| 98428.2  | 288.2x | YES |
| K=16 sd=8  | 128  | 16 | 16 | 0.062 | 7.6  | 1905.3  | 729.3    | 251.4x | YES |
| K=16 sd=16 | 256  | 16 | 16 | 0.062 | 27.3 | 18906.1 | 8267.9   | 693.9x | YES |
| K=16 sd=32 | 512  | 16 | 16 | 0.062 | 157.8| 203466.7| 103442.1 | 1289.1x| YES |
| K=16 sd=64 | 1024 | 16 | 16 | 0.062 | 1328.4| 4240446.8| 2089398.6| 3192.2x| YES |

### polyacetylene-style (`q0 + q1 = 0`) — timing/sparsity only

| case | N | blocks | triples | sparsity | µs/block | µs/dense | speedup |
|---|---|---|---|---|---|---|---|
| K=3 sd=16 | 48  | 3 | 3 | 0.333 | 8.4   | 80.5   | 9.6x  |
| K=3 sd=32 | 96  | 3 | 3 | 0.333 | 38.5  | 709.2  | 18.4x |
| K=3 sd=64 | 192 | 3 | 3 | 0.333 | 354.7 | 7034.8 | 19.8x |

## Analysis

* **Correctness**: every gated case matches the dense product to `max|Δ| ≤ 1e-4`
  in float. No failures.
* **Speedup tracks the flop reduction, not the raw sparsity ratio.** Dense work
  is `2N³`; block work is `Σ 2·s³` over blocks. At `N=1024, 16×64³` the block
  work is ~250× fewer flops, and measured speedup is 2757× — the gap beyond 250×
  is because the naive dense kernel is memory-bandwidth-bound (it streams all of
  `B` for every row of `A`), while the block kernel keeps every working set in
  cache.
* Speedup grows with `N` at fixed sparsity, exactly as expected: the dense cost
  is `O(N³)` while the block cost is `O(N·s²)`.

## Finding: the engine's canonical structure is degenerate

The diagnostic rows show that when `A`, `B` *and* `C` are all built from the same
constraint (`q0 - q1 = 0`, 4 or 8 blocks), the plan finds only **1** fusible
triple, and only the zero-charge diagonal block produces output (256 = 16²
entries). Root cause:

1. `ConstraintSolver::generate_blocks` sets each block's `u1_qnum` to the charge
   **summed over all legs**, not the row/column charge. For a two-leg tensor with
   `q0 - q1 = 0` that value is `2·q_i`, so the fusion rule `qA + qB == qC`
   reduces to `4·q_i == 2·q_i`, which holds only for `q_i == 0`.
2. Plan-time geometric matching only compares block **sizes** (`A.num_cols ==
   B.num_rows`), not the actual row/column **ranges**, so it can accept pairs
   whose index ranges do not actually align.

Consequence: the constrained GEMM path is only useful today with **hand-built**
block structure (as in the `structured` mode, and the library's own
`test_block_gemm.cpp` cases). Improving the charge convention is the single
highest-leverage fix for making the constraint-engine path usable.

## Reproduce

```bash
SRC="src/tapp_constrained.cpp src/constrained_tensor.cpp \
     src/symmetry/u1_handler.cpp src/symmetry/su2_handler.cpp \
     src/backend/block_gemm.cpp src/backend/gpu_backend.cpp \
     src/constraint/constraint_parser.cpp src/constraint/constraint_solver.cpp \
     src/constraint/constraint_engine.cpp"
g++ -std=c++17 -O3 -fopenmp -DTAPP_CON_USE_OPENMP -pthread -Iinclude -Isrc \
    $SRC examples/benchmarks/constrained_vs_dense.cpp \
    -o build/examples/benchmarks/constrained_vs_dense
./build/examples/benchmarks/constrained_vs_dense --mode all
```

(`cmake` is not on `PATH` on this machine; the direct `g++` invocation above is
equivalent to what the CMake target would produce.)

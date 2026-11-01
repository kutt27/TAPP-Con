# tapp_con — NumPy port

A faithful NumPy reimplementation of the TAPP-Con constrained block-sparse
tensor library (the C++ code in this repository). It mirrors both the Pythonic
`ConstrainedTensor` class and the C API in `include/tapp_constrained.h`.

## Install

```bash
cd python
pip install -e .          # runtime dependency: numpy
```

## Usage

```python
import numpy as np
import tapp_con as tc

# --- class API ---
a = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
a.add_block(0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
a.add_block(2, 2, 2, 2, 1, 0, [[5, 6], [7, 8]])

b = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
b.add_block(0, 0, 2, 2, 1, 0, [[9, 10], [11, 12]])
b.add_block(2, 2, 2, 2, 0, 0, [[13, 14], [15, 16]])

c = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
c.add_block(0, 0, 2, 2, 1, 0, None)
c.add_block(2, 2, 2, 2, 1, 0, None)

plan = tc.PlanData(a, b, c, tc.DataType.FLOAT)
plan.compute_block_map()
tc.BlockGEMM.multiply_parallel(c, a, b, plan.compatible_triples, plan.c_block_work, 1.0, 0.0)
print(c.to_dense())

# --- C-API mirror ---
status, t = tc.build_from_constraints(
    2, ["q0 + q1 = 0"], [3, 3], [0], [1], [6, 6],
    tc.TAPP_CON_FLOAT, tc.TAPP_CON_SYMMETRY_U1, -1, 1,
)
assert status == tc.Status.SUCCESS and t.num_blocks == 3
```

## Tests

```bash
cd python
python -m unittest discover -s tests -t . -v
```

The suite includes `test_parity.py`, which runs the C++ reference dumper
(`build/tools/dump_reference`, built from `tools/dump_reference.cpp`) and asserts
the port reproduces parser output, block enumeration, tensor construction and
GEMM results value-for-value. If that binary is not built the parity test is
skipped and the rest of the suite still runs.

## Faithfulness / known divergences

The port reproduces the C++ behaviour, including its quirks. Sites that
deliberately mirror surprising C++ behaviour are marked `FAITHFUL-QUIRK`.
Preserved quirks:

* **`beta` is applied per contributing triple** (`multiply_parallel`), not once
  per C block. With `beta == 0` the whole C buffer is zeroed first. For
  multi-contributor blocks `beta != 0` is therefore not plain BLAS semantics.
* **Transpose is ignored on the plan path.** `multiply_parallel` hardcodes
  `false, false`; only the serial `multiply` honours `trans_a`/`trans_b`.
* **`generate_blocks` stores `u1_vals[0]` in the SU(2) spin field**, not a spin.
* **Multi-row/col-leg offset stride** uses the current leg's `sector_dim` rather
  than the product of the already-visited legs' sizes.
* The constraint engine sets each block's `u1_qnum` to the charge **summed over
  all legs**, which makes `qA + qB == qC` fail for most engine-built tensors.

Intentional divergence:

* **Complex dtypes are supported.** The C++ CPU kernel falls through to a
  `double` reinterpretation for `COMPLEX_FLOAT`/`COMPLEX_DOUBLE`; here they map
  to `np.complex64`/`np.complex128` and compute correctly.

## Layout

```
tapp_con/
  _types.py        enums, element sizes, numpy dtype mapping
  errors.py        exceptions + status mapping
  tensor.py        Block, ConstrainedTensor
  symmetry.py      U(1) / SU(2) selection rules
  backend.py       BlockTriple, PlanData, BlockGEMM
  api.py           C-API mirrors
  constraint/
    parser.py      tokenizer + parse_u1 + parse_system
    solver.py      generate_blocks (Cartesian enumeration)
    engine.py      build_tensor / build_tensor_from_strings
```

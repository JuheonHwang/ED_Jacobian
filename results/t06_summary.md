### (a) Cost of one Jacobian, f10k (5027 vertices, 50 nodes, 606 parameters)

| method | time [s] | x analytic F+J |
|---|---|---|
| residual only | 0.0033 | 0.08 |
| analytic F + J | 0.0423 | 1 |
| finite differences, serial (606 residual evaluations, measured) | 0.713 | 16.9 |
| finite differences, parfor (10 workers, measured, pool already started) | 0.350 | 8.3 |

Forward differences, step 1e-5, fixed correspondences; max abs(J - J_fd) / max abs(J) = 4.75e-06.

### (b) Mesh size (SampleRate 0.01)

| mesh | vertices | nodes | parameters | rows | nnz(J) | fill [%] | residual [s] | F+J [s] | FD serial [s] (estimated) | FD / (F+J) |
|---|---|---|---|---|---|---|---|---|---|---|
| f10k | 5027 | 50 | 606 | 30830 | 1681702 | 9.001 | 0.0043 | 0.0368 | 0.86 | 23 |
| f20k | 10038 | 100 | 1206 | 61540 | 3387987 | 4.565 | 0.0060 | 0.0658 | 3.01 | 46 |

f80k not measured: data not present in data/.

### (b, c) Number of ED nodes (f10k)

| SampleRate | vertices | nodes | parameters | rows | nnz(J) | fill [%] | residual [s] | F+J [s] | FD serial [s] (estimated) | FD / (F+J) |
|---|---|---|---|---|---|---|---|---|---|---|
| rate 0.0025 | 5027 | 13 | 162 | 30349 | 1592245 | 32.385 | 0.0040 | 0.0482 | 0.28 | 6 |
| rate 0.005 | 5027 | 25 | 306 | 30505 | 1636877 | 17.536 | 0.0042 | 0.0337 | 0.42 | 12 |
| rate 0.01 | 5027 | 50 | 606 | 30830 | 1681702 | 9.001 | 0.0034 | 0.0342 | 0.75 | 22 |
| rate 0.02 | 5027 | 101 | 1218 | 31493 | 1762237 | 4.594 | 0.0038 | 0.0320 | 1.32 | 41 |
| rate 0.04 | 5027 | 201 | 2418 | 32793 | 1869858 | 2.358 | 0.0031 | 0.0296 | 2.62 | 89 |

FD serial values in these tables are ESTIMATED: the time of 20 columns scaled to all columns.

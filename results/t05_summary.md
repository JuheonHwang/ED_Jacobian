| solver | iters | final | best | time [s] | #fun | dist mean | dist median | dist p95 | exit |
|---|---|---|---|---|---|---|---|---|---|
| lsqnonlin (reference) | 31 | 151.3867 | 151.3867 | 15.14 | 32 | 9.852 | 9.279 | 18.292 | lsqnonlin flag 0 |
| gradientDescent (1e-3, legacy) | 100 | 306.8758 | 243.5140 | 3.09 | 101 | 15.459 | 14.193 | 29.601 | MaxIterations |
| gradientDescent (2e-4) | 100 | 187.0997 | 187.0997 | 2.93 | 101 | 11.186 | 10.558 | 20.137 | MaxIterations |
| gaussNewton (normal eq.) | 30 | 150.3727 | 150.3727 | 2.38 | 31 | 9.778 | 9.162 | 18.356 | GradientTolerance |
| gaussNewton (lsq, legacy _re) | 19 | 16720981647335426.0000 | 197.5322 | 2.36 | 20 | 163929.142 | 158667.313 | 309829.699 | GradientTolerance |
| LM hjh2 (direct) | 30 | 150.7190 | 150.7190 | 2.29 | 75 | 9.794 | 9.203 | 18.343 | MaxIterations |
| LM hjh1 (legacy 3 trials) | 30 | 150.3316 | 150.3316 | 2.38 | 73 | 9.795 | 9.202 | 18.379 | MaxIterations |
| LM-CG | 30 | 151.4166 | 151.4166 | 2.78 | 61 | 9.825 | 9.286 | 18.390 | MaxIterations |
| LM-PCG (Jacobi) | 30 | 150.2511 | 150.2511 | 3.18 | 76 | 9.775 | 9.182 | 18.282 | MaxIterations |

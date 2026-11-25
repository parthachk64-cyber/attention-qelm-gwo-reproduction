# GWO search contract (generated from `optim/gwo.py`)

| dim | name | bounds | decoding |
|---|---|---|---|
| 0 | n_layers | [1, 5] | round half up, clip 1..5 |
| 1 | width | [64, 1024] | round half up, clip 64..1024 |
| 2 | activation | [0, 4] | floor; x = 4 -> index 3 of (sigmoid, tanh, relu, gelu) |
| 3 | log10_lambda1 | [-6, -1] | 10 ** x |
| 4 | log10_lambda2 | [-6, -1] | 10 ** x |
| 5 | theta | [0, 6.28319] | as is |
| 6 | rho | [0, 1] | as is |

```
initialise N wolves uniformly in the bounds; evaluate (iteration 0)
leaders = three best DISTINCT positions seen so far (stable ties)
for t = 1..T:  a = 2 (1 - t/T)
    for each wolf X: for each leader L in (alpha, beta, delta):
        A = 2 a r1 - a;  C = 2 r2;  X_L = L - A |C L - X|
        X <- clip(mean(X_alpha, X_beta, X_delta), bounds)
    evaluate decoded candidates (memoised by decoded config); update leaders
    stop when `patience` consecutive iterations improve the best by <= tol
return decoded alpha position
```

Candidate evaluations: N for the initial population + N per iteration (<= N (T+1) = 1,020 for N=20, T=50); each costs `n_folds` fold fits (the article's nominal 1,000 = N T does not distinguish the two).
Fitness is five-fold macro-F1 with the encoder fitted per fold on that fold's fit partition only. GWO never runs at inference.

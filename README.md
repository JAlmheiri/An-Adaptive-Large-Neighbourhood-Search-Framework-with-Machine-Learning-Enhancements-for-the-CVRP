# An Adaptive Large Neighbourhood Search Framework with Machine-Learning Enhancements for the CVRP

## Abstract
We present a systematic, iterative study of learning-augmented Adaptive Large Neighbourhood Search (ALNS) for the Capacitated Vehicle Routing Problem (CVRP). Starting from classical Clarke-Wright savings, we incrementally introduce increasingly sophisticated components including simulated annealing, enriched operator portfolios, neural operator selectors, novel hybrid selection mechanisms, DR-ALNS-inspired state representations, and a novel learned acceptance criterion based on dynamically weighted NV-TD objectives. We benchmark each variant against 59 instances (X-n101 to X-n401) under the ML4VRP competition objective 1000·NV + TD. Our best result, a DR-ALNS hybrid selector with two-operator ALNS (avg. objective 71,047.47), achieves a 1.80% gap to the Best Known Solution (BKS) and trails PyVRP (70,505.15, 0.82% gap) by 542 units on average: a meaningful gap that reflects the sophistication of PyVRP's hybrid genetic algorithm versus our online-learned metaheuristic. Our analysis reveals several non-obvious findings: SA consistently hurts performance due to the near-full capacity structure of these instances; the most impactful single contribution is operator diversification (shaw removal + regret insert); neural operator selection requires richer state representations to compete with roulette; and the learned acceptance criterion, despite not being the top-performing standalone variant, provides consistent improvement when applied to any base solver, suggesting it would benefit from combination with the best hybrid selector, a direction we identify for future work.

## Directory structure
```
/src
    classical_baseline.py
    /alns
        alns_base_simple_descent.py
        alns_sa.py   # ALNS with simulated annealing acceptance, roulette wheel
        alns_plus.py
        alns_plus_sa.py
    /ml
        alns_plus_neural.py
        alns_plus_hybrid.py
        alns_plus_adaptive_q.py
        alns_plus_dr_alns.py
        alns_plus_dr_alns_sa.py
        alns_plus_dr_alns_hybrid.py
        alns_plus_learned_accept.py
/results
    benchmarking_spreadsheet.xlsx
    results_table.xlsx
```
## The Team
- Aysha Aldarmaki					U22104299
- Jawaher Almeheiri					U22103653
- Mariam Alhosani					U22101372
- Sara Alrumaitha					U21200190

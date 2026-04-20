'''
ALNS_plus_SA Adaptive Large Neighborhood Search with Simulated Annealing
based on: S. Ropke and D. Pisinger, "An Adaptive Large Neighborhood Search Heuristic
for the Pickup and Delivery Problem with Time Windows," Transportation Science,
vol. 40, no. 4, pp. 455–472, Nov. 2006, doi: 10.1287/trsc.1050.0135.

1 additional destroy -> shaw removal
1 additional repair  -> regret insert
+ Simulated Annealing acceptance criterion
'''

class ALNS_plus_SA:
    def __init__(self, inst: CVRPInstance, selector='roulette'):
        self.inst = inst
        self.selector = selector

        # destroy and repair operation lists
        self.D = [self.random_removal, self.worst_removal, self.shaw_removal] # random 0, worst 1, shaw 2
        self.R = [self.greedy_insert, self.regret_insert]                     # greedy 0, regret 1

        # weights for each heuristic
        # 1 because at the start they're all equally likely to be selected
        self.dest_weights = [1.0] * len(self.D)
        self.rep_weights  = [1.0] * len(self.R)

        # initializing scores
        self.dest_scores = [0.0] * len(self.D)
        self.rep_scores  = [0.0] * len(self.R)
        self.dest_usage  = [0]   * len(self.D)
        self.rep_usage   = [0]   * len(self.R)

        # SA parameters — w tuned via grid search, while T_start auto-calibrated per instance
        self.T       = None
        self.w       = 0.01
        self.T_start = None
        self.c       = 0.9997

    def calibrate_T_start(self, initial_sol, n_samples=200, target_accept=0.5):
        """
        sample n_samples random destroy/repair pairs on the initial solution.
        set T_start so that a move of average worsening delta is accepted
        with probability target_accept (default 0.5).
        
        T_start = -delta_avg / ln(target_accept)
        """
        deltas   = []
        init_obj = objective(initial_sol, self.inst.dist)
        for _ in range(n_samples):
            q             = random.randint(5, 15)
            temp, removed = self.random_removal(initial_sol, q)
            candidate     = self.greedy_insert(temp, removed)
            delta         = objective(candidate, self.inst.dist) - init_obj
            if delta > 0:
                deltas.append(delta)
        if not deltas:
            return -(self.w * init_obj) / math.log(target_accept)
        return -sum(deltas) / len(deltas) / math.log(target_accept)

    def solve(self, initial_solution, n_iter=1000, n_seg=100):
        current_sol = copy_solution(initial_solution)
        best_sol    = copy_solution(initial_solution)
        best_obj    = objective(best_sol, self.inst.dist)
        current_obj = best_obj

        # auto-calibrate T_start from actual cost deltas
        self.T_start = self.calibrate_T_start(initial_solution)
        self.T       = self.T_start

        stagnation = 0 # iterations since last best improvement

        for i in range(n_iter):
            self.reset_scoresNusage()
            q = random.randint(5, 15)

            for j in range(n_seg):
                dest_index = self.select_operator(self.dest_weights, op_type='destroy')
                rep_index  = self.select_operator(self.rep_weights,  op_type='repair')

                self.dest_usage[dest_index] += 1
                self.rep_usage[rep_index]   += 1

                temp_dest, removed = self.destroy(current_sol, dest_index, q)
                candidate_sol      = self.repair(temp_dest, rep_index, removed)
                candidate_obj      = objective(candidate_sol, self.inst.dist)

                accepted = self.accept(current_obj, candidate_obj)
                reward   = self.get_reward(candidate_obj, current_obj, best_obj, accepted)

                if accepted in ('improved', 'sa_accepted'):
                    current_sol = candidate_sol
                    current_obj = candidate_obj

                    if current_obj < best_obj:
                        best_sol = copy_solution(current_sol)
                        best_obj = current_obj
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1

                # update scores & weights
                self.dest_scores[dest_index] += reward
                self.rep_scores[rep_index]   += reward

                # cool temperature after each destroy & repair cycle
                self.T *= self.c

            self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)
            self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)

        return best_sol

    def select_operator(self, op_weights, op_type):
        if self.selector == 'roulette':
            return self.roulette_wheel(op_weights)
        else:
            raise ValueError(f"invalid selector type: {self.selector}")

    def roulette_wheel(self, op_weights):
        probs = [w / sum(op_weights) for w in op_weights]
        return np.random.choice(range(len(op_weights)), p=probs)

    def update_weights(self, weights, scores, usage, r=0.1):
        for w in range(len(weights)):
            if usage[w] > 0:
                weights[w] = weights[w] * (1 - r) + r * (scores[w] / usage[w])

    def random_removal(self, solution, q):
        sol      = copy_solution(solution)
        flat_sol = [cust for route in sol for cust in route]
        removed  = random.sample(flat_sol, q)

        for i in range(len(sol)):
            sol[i] = [cust for cust in sol[i] if cust not in removed]
        sol = [route for route in sol if route]

        return sol, removed

    def worst_removal(self, solution, q):
        p   = 6
        sol = copy_solution(solution)
        removed = []
        costs   = self.compute_costs4worst(sol)

        while q > 0:
            y     = random.random()
            index = min(int(y**p * len(costs)), len(costs) - 1)

            selected_cust = costs[index][1]
            removed.append(selected_cust)

            for j in range(len(sol)):
                sol[j] = [cust for cust in sol[j] if cust != selected_cust]
            sol   = [route for route in sol if route]
            costs = self.compute_costs4worst(sol)
            q    -= 1

        return sol, removed

    def shaw_removal(self, solution, q):
        p        = 6  # same p as worst removal, controls how biased we are toward similar customers
        sol      = copy_solution(solution)
        removed  = []
        flat_sol = [cust for route in sol for cust in route]

        # pick a random starting customer as our anchor
        # everyone we remove after this will be "similar" to someone already removed
        r = random.choice(flat_sol)
        removed.append(r)

        while len(removed) < q:
            # pick someone already marked as our reference point this round
            r = random.choice(removed)

            # everyone not yet marked is a candidate
            L = [cust for cust in flat_sol if cust not in removed]

            # sort by distance to r — closest customers come first
            # for plain CVRP similarity is just distance, no time windows or capacity stuff
            L.sort(key=lambda cust: self.inst.dist[r][cust])

            # same biased pick as worst removal — usually grabs from the front
            # (most similar) but occasionally picks further down
            y     = random.random()
            index = min(int(y**p * len(L)), len(L) - 1)
            removed.append(L[index])

        # remove everyone at once after the loop
        # different from worst removal which removes one by one and recomputes each time
        for i in range(len(sol)):
            sol[i] = [cust for cust in sol[i] if cust not in removed]
        sol = [route for route in sol if route]

        return sol, removed

    def greedy_insert(self, temp_dest, removed):
        sol            = copy_solution(temp_dest)
        sorted_removed = sorted(removed, key=lambda c: self.inst.demands[c], reverse=True)

        # inserting the hardest customers first
        for cust in sorted_removed:
            best_cost  = float('inf')
            best_route = None
            best_pos   = None

            for r, route in enumerate(sol):
                if route_demand(route, self.inst.demands) + self.inst.demands[cust] > self.inst.capacity: # capacity pre-check
                    continue  # skip infeasible routes early
                for pos in range(len(route) + 1):
                    cost = self.insert_cost(cust, route, pos)
                    if cost < best_cost:
                        best_cost  = cost
                        best_route = r
                        best_pos   = pos

            if best_route is not None:
                sol[best_route].insert(best_pos, cust)
            else:
                sol.append([cust])  # no feasible position -> solo route

        return sol

    def regret_insert(self, temp_dest, removed_customers):
        sol     = copy_solution(temp_dest)
        removed = list(removed_customers)  # copy to avoid modifying original

        while removed:
            regret = []
            for cust in removed:
                best_cost        = float('inf')
                best_route       = None
                best_pos         = None
                second_best_cost = float('inf')

                for r, route in enumerate(sol):
                    if route_demand(route, self.inst.demands) + self.inst.demands[cust] > self.inst.capacity:
                        continue  # skip infeasible routes entirely
                    for pos in range(len(route) + 1):
                        cost = self.insert_cost(cust, route, pos)
                        if cost < best_cost:
                            second_best_cost = best_cost  # old best becomes second best
                            best_cost  = cost
                            best_route = r
                            best_pos   = pos
                        elif cost < second_best_cost:
                            second_best_cost = cost  # no route restriction now

                if best_cost == float('inf'):          # no feasible insertion found
                    regret.append((0, cust, None, None))
                elif second_best_cost == float('inf'): # only one feasible route exists
                    regret.append((0, cust, best_route, best_pos))
                else:
                    regret.append((second_best_cost - best_cost, cust, best_route, best_pos))

            sorted_regret = sorted(regret, key=lambda x: x[0], reverse=True)
            _, most_regret_cust, best_route, best_pos = sorted_regret[0]

            if best_route is not None:
                sol[best_route].insert(best_pos, most_regret_cust)
            else:
                sol.append([most_regret_cust])  # no feasible position -> solo route
            removed.remove(most_regret_cust)

        return sol

    def insert_cost(self, cust, route, pos):
        prev  = route[pos - 1] if pos > 0         else 0
        after = route[pos]     if pos < len(route) else 0

        demand = route_demand(route, self.inst.demands)
        if demand + self.inst.demands[cust] > self.inst.capacity:
            return float('inf')

        return self.inst.dist[prev][cust] + self.inst.dist[cust][after] - self.inst.dist[prev][after]

    def destroy(self, sol, dest_index, q):
        return self.D[dest_index](sol, q)

    def repair(self, sol, rep_index, removed):
        return self.R[rep_index](sol, removed)

    def accept(self, current_obj, candidate_obj):
        if candidate_obj < current_obj:
            return 'improved'
        r = random.random()
        P = math.exp(-(candidate_obj - current_obj) / self.T)
        if r < P:
            return 'sa_accepted'
        return 'rejected'

    def get_reward(self, candidate_obj, current_obj, best_obj, accepted):
        sigma1, sigma2, sigma3 = 33, 9, 13
        if accepted == 'improved':
            return sigma1 if candidate_obj < best_obj else sigma2
        elif accepted == 'sa_accepted':
            return sigma3
        return 0

    def reset_scoresNusage(self):
        self.dest_scores = [0.0] * len(self.D)
        self.rep_scores  = [0.0] * len(self.R)
        self.dest_usage  = [0]   * len(self.D)
        self.rep_usage   = [0]   * len(self.R)

    def compute_costs4worst(self, sol):
        # cost(i) = how much does the route shorten if we remove customer i
        # formula: dist(prev, i) + dist(i, after) - dist(prev, after)
        costs = []
        for route in sol:
            for pos in range(len(route)):
                cust_id = route[pos]
                prev    = route[pos - 1] if pos > 0              else 0
                after   = route[pos + 1] if pos < len(route) - 1 else 0
                cost_value = self.inst.dist[prev][cust_id] + self.inst.dist[cust_id][after] - self.inst.dist[prev][after]
                costs.append((cost_value, cust_id))

        costs.sort(reverse=True)
        return costs
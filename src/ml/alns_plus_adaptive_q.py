'''
ALNS_plus_Q : ALNS with additional operators + learned destruction size (q)

the key idea: instead of picking q randomly from [5,15] every segment,
a small neural network reads the current solution state and predicts
how aggressively we should destroy, more disruption when stuck,
less when we're improving nicely.

The network is trained via REINFORCE, same reward signal
as the roulette wheel (σ1=33 new best, σ2=9 improvement, 0=rejected). 

based on:
S. Ropke and D. Pisinger, "An Adaptive Large Neighborhood Search Heuristic
for the Pickup and Delivery Problem with Time Windows," Transportation Science,
vol. 40, no. 4, pp. 455–472, Nov. 2006, doi: 10.1287/trsc.1050.0135.
'''

# QPredictor network 
# same shallow arch as OperatorSelector we did earlier
# input (4 features) -> hidden (16, ReLU) -> output (1, Sigmoid)
# sigmoid squashes output to (0,1) which we then scale to [q_min, q_max]

class QPredictor(nn.Module):
    def __init__(self, n_features=4, n_hidden=16, q_min=5, q_max=15):
        super().__init__()
        self.fc1   = nn.Linear(n_features, n_hidden)  # input -> hidden (theta weights in textbook)
        self.fc2   = nn.Linear(n_hidden, 1)            # hidden -> output (phi weights in textbook)
        self.q_min = q_min
        self.q_max = q_max

    def forward(self, x):
        x = F.relu(self.fc1(x))           # hidden layer + ReLU
        x = torch.sigmoid(self.fc2(x))    # output: single value in (0,1)
        return x

    def predict(self, features):
        '''
        takes 4 features, outputs a q value in [q_min, q_max]
        uses normal distribution around the predicted mean so we can get log_prob for REINFORCE
        std=1.5 gives some spread around the predicted q value
        '''
        mean     = self.q_min + self.forward(features) * (self.q_max - self.q_min)
        std      = torch.tensor(1.5)  # fixed spread — could tune this
        dist     = torch.distributions.Normal(mean, std)
        sample   = dist.sample()
        log_prob = dist.log_prob(sample)

        # clamp to valid range and round to integer
        q = int(round(sample.item()))
        q = max(self.q_min, min(self.q_max, q))

        return q, log_prob


# ALNS_plus_Q 

class ALNS_plus_Q:
    def __init__(self, inst: CVRPInstance):
        self.inst = inst

        # destroy operators 
        self.D = [self.random_removal, self.worst_removal, self.shaw_removal]  # random=0, worst=1, shaw=2

        # repair operators 
        self.R = [self.greedy_insert, self.regret_insert]  # greedy=0, regret=1

        # roulette wheel weights we start equal, adapt via scores
        self.dest_weights = [1.0] * len(self.D)
        self.rep_weights  = [1.0] * len(self.R)

        # scores + usage which reset every segment for weight update
        self.dest_scores = [0.0] * len(self.D)
        self.rep_scores  = [0.0] * len(self.R)
        self.dest_usage  = [0]   * len(self.D)
        self.rep_usage   = [0]   * len(self.R)

        # ML component: QPredictor learns how much to destroy each segment
        # instead of random q ~ [5,15], network predicts q based on solution state
        self.q_net       = QPredictor(q_min=5, q_max=15)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.001) # adam optimizer with lr=0.001


    def solve(self, initial_solution, n_iter=1000, n_seg=100):
        current_sol = copy_solution(initial_solution)
        best_sol    = copy_solution(initial_solution)
        best_obj    = objective(best_sol, self.inst.dist)
        current_obj = best_obj

        for i in range(n_iter):
            self.reset_scoresNusage()  # fresh scores at start of every segment

            # compute features once per segment (progress is at segment level)
            # note: we use i as iteration count — progress goes from 0 to 1 over segments
            features = extract_features(current_sol, current_obj, i, n_iter, self.inst)

            # predict q from network instead of random.randint(5, 15)
            # network reads solution state and decides how aggressively to destroy
            q, log_prob_q = self.q_net.predict(features)

            for j in range(n_seg):

                # roulette wheel picks destroy + repair operators as before
                dest_index = self.roulette_wheel(self.dest_weights)
                rep_index  = self.roulette_wheel(self.rep_weights)

                self.dest_usage[dest_index] += 1
                self.rep_usage[rep_index]   += 1

                # destroy q customers, repair them back in
                temp_dest, removed = self.destroy(current_sol, dest_index, q)
                candidate_sol      = self.repair(temp_dest, rep_index, removed)
                candidate_obj      = objective(candidate_sol, self.inst.dist)

                accepted = self.accept(current_obj, candidate_obj)
                reward   = self.get_reward(candidate_obj, current_obj, best_obj, accepted)

                if accepted:
                    current_sol = candidate_sol
                    current_obj = candidate_obj

                    if current_obj < best_obj:
                        best_sol = copy_solution(current_sol)
                        best_obj = current_obj

                # track scores for roulette weight update
                self.dest_scores[dest_index] += reward
                self.rep_scores[rep_index]   += reward

            # after each segment:
            # 1) update roulette weights based on accumulated scores
            self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)
            self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)

            # 2) REINFORCE update for q network
            # we use the total reward accumulated this segment as the signal
            # if the segment went well (high reward) -> reinforce the q choice
            # if the segment went badly (reward=0) -> no update needed
            segment_reward = sum(self.dest_scores)  # proxy for how well this segment went
            if segment_reward > 0:
                reinforce_update(log_prob_q, segment_reward, self.q_optimizer)

        return best_sol


    def roulette_wheel(self, op_weights):
        # prob of each op = its weight / total weight
        # higher weight -> bigger slice -> more likely to be picked
        probs = [w / sum(op_weights) for w in op_weights]
        return np.random.choice(range(len(op_weights)), p=probs)

    def update_weights(self, weights, scores, usage, r=0.1):
        for w in range(len(weights)):
            if usage[w] > 0:  # avoid div by 0
                weights[w] = weights[w] * (1 - r) + r * (scores[w] / usage[w])

    def random_removal(self, solution, q):
        sol      = copy_solution(solution)
        flat_sol = [cust for route in sol for cust in route]
        removed  = random.sample(flat_sol, q)  # pick q customers at random

        for i in range(len(sol)):
            sol[i] = [cust for cust in sol[i] if cust not in removed]
        sol = [route for route in sol if route]

        return sol, removed

    def worst_removal(self, solution, q):
        # removes the q most "expensive" customers to keep
        # biased random selection using p=6 (usually picks from front of sorted list)
        p       = 6
        sol     = copy_solution(solution)
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
        # removes geographically similar customers — cluster-based destruction
        # starts from a random anchor then grabs nearby customers
        # the idea: if we remove a cluster we might be able to reorganize routes better
        p        = 6
        sol      = copy_solution(solution)
        removed  = []
        flat_sol = [cust for route in sol for cust in route]

        r = random.choice(flat_sol)  # random anchor customer
        removed.append(r)

        while len(removed) < q:
            r = random.choice(removed)  # pick reference from already removed
            L = [cust for cust in flat_sol if cust not in removed]
            L.sort(key=lambda cust: self.inst.dist[r][cust])  # sort by distance to r

            y     = random.random()
            index = min(int(y**p * len(L)), len(L) - 1)
            removed.append(L[index])

        for i in range(len(sol)):
            sol[i] = [cust for cust in sol[i] if cust not in removed]
        sol = [route for route in sol if route]

        return sol, removed

    def greedy_insert(self, temp_dest, removed):
        sol            = copy_solution(temp_dest)
        # hardest customers first (high demand) — less likely to be squeezed out
        sorted_removed = sorted(removed, key=lambda c: self.inst.demands[c], reverse=True)

        for cust in sorted_removed:
            best_cost  = float('inf')
            best_route = None
            best_pos   = None

            for r, route in enumerate(sol):
                if route_demand(route, self.inst.demands) + self.inst.demands[cust] > self.inst.capacity:
                    continue  # skip full routes early
                for pos in range(len(route) + 1):
                    cost = self.insert_cost(cust, route, pos)
                    if cost < best_cost:
                        best_cost  = cost
                        best_route = r
                        best_pos   = pos

            if best_route is not None:
                sol[best_route].insert(best_pos, cust)
            else:
                sol.append([cust])  # no room anywhere -> solo route

        return sol

    def regret_insert(self, temp_dest, removed_customers):
        # inserts customer with highest regret first
        # regret = difference between best and second best insertion cost
        # high regret -> inserting elsewhere would be much worse -> do it now
        sol     = copy_solution(temp_dest)
        removed = list(removed_customers)

        while removed:
            regret = []
            for cust in removed:
                best_cost        = float('inf')
                best_route       = None
                best_pos         = None
                second_best_cost = float('inf')

                for r, route in enumerate(sol):
                    if route_demand(route, self.inst.demands) + self.inst.demands[cust] > self.inst.capacity:
                        continue  # skip full routes
                    for pos in range(len(route) + 1):
                        cost = self.insert_cost(cust, route, pos)
                        if cost < best_cost:
                            second_best_cost = best_cost  # old best -> second best
                            best_cost  = cost
                            best_route = r
                            best_pos   = pos
                        elif cost < second_best_cost:
                            second_best_cost = cost

                if best_cost == float('inf'):           # no feasible position at all
                    regret.append((0, cust, None, None))
                elif second_best_cost == float('inf'):  # only one option, no regret
                    regret.append((0, cust, best_route, best_pos))
                else:
                    regret.append((second_best_cost - best_cost, cust, best_route, best_pos))

            sorted_regret = sorted(regret, key=lambda x: x[0], reverse=True)
            _, most_regret_cust, best_route, best_pos = sorted_regret[0]

            if best_route is not None:
                sol[best_route].insert(best_pos, most_regret_cust)
            else:
                sol.append([most_regret_cust])
            removed.remove(most_regret_cust)

        return sol

    def insert_cost(self, cust, route, pos):
        prev   = route[pos - 1] if pos > 0         else 0
        after  = route[pos]     if pos < len(route) else 0
        demand = route_demand(route, self.inst.demands)

        if demand + self.inst.demands[cust] > self.inst.capacity:
            return float('inf')

        return self.inst.dist[prev][cust] + self.inst.dist[cust][after] - self.inst.dist[prev][after]

    def destroy(self, sol, dest_index, q):
        return self.D[dest_index](sol, q)

    def repair(self, sol, rep_index, removed):
        return self.R[rep_index](sol, removed)

    def accept(self, current_obj, candidate_obj):
        return candidate_obj < current_obj  # simple descent — only accept improvements

    def get_reward(self, candidate_obj, current_obj, best_obj, accepted):
        sigma1 = 33  # new global best — big reward
        sigma2 = 9   # better than current — small reward
        if accepted:
            return sigma1 if candidate_obj < best_obj else sigma2
        return 0  # rejected — no reward

    def reset_scoresNusage(self):
        self.dest_scores = [0.0] * len(self.D)
        self.rep_scores  = [0.0] * len(self.R)
        self.dest_usage  = [0]   * len(self.D)
        self.rep_usage   = [0]   * len(self.R)

    def compute_costs4worst(self, sol):
        # cost(i) = how much does the route shorten if we remove customer i
        # dist(prev, i) + dist(i, next) - dist(prev, next)
        # high cost -> expensive to keep -> good candidate for removal
        costs = []
        for route in sol:
            for pos in range(len(route)):
                cust_id = route[pos]
                prev    = route[pos - 1] if pos > 0              else 0
                after   = route[pos + 1] if pos < len(route) - 1 else 0
                cost_value = (self.inst.dist[prev][cust_id]
                              + self.inst.dist[cust_id][after]
                              - self.inst.dist[prev][after])
                costs.append((cost_value, cust_id))
        costs.sort(reverse=True)
        return costs
'''
ALNS+ with neural operator selector (REINFORCE).
 
Destroy operators: random removal, worst removal, shaw removal.
Repair operators:  greedy insertion, regret insertion.
 
Selector options:
  - 'roulette' : roulette wheel over accumulated segment scores
  - 'neural'   : two-network MLP selector trained online via REINFORCE
                 destroy net: 4 features -> 16 ReLU -> 3 outputs (Softmax)
                 repair  net: 4 features -> 16 ReLU -> 2 outputs (Softmax)
                 features: normalized cost, route load variance,
                           avg customers per route, search progress
                 optimizer: Adam, lr=0.001, updated every inner iteration when reward > 0
 
Based on:
  Ropke & Pisinger (2006), doi: 10.1287/trsc.1050.0135
  Williams (1992), REINFORCE
  Prince, Understanding Deep Learning (2023), Ch. 3, 19
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class OperatorSelector(nn.Module):
    def __init__(self, n_operators, n_features=4, n_hidden=16):
        super().__init__()
        # layer 1: input → hidden (equivalent to theta weights in textbook UDL)
        self.fc1 = nn.Linear(n_features, n_hidden)
        # layer 2: hidden → output (equivalent to phi weights in textbook)
        self.fc2 = nn.Linear(n_hidden, n_operators)

    def forward(self, x):
        # preactivation + ReLU (hidden layer)
        x = F.relu(self.fc1(x))
        # output layer (raw logits, no softmax yet)
        return self.fc2(x)

    def select(self, features):
        logits   = self.forward(features)
        # Categorical handles softmax internally + gives us log_prob
        dist     = Categorical(logits=logits)
        action   = dist.sample()           # sample operator index
        log_prob = dist.log_prob(action)   # log pi(a|s) needed for REINFORCE
        return action.item(), log_prob

# feature extractor

def extract_features(current_sol, current_obj, iteration, total_iterations, inst):
    # feature 1: normalized solution cost
    cost = current_obj / 100000.0

    # feature 2: route load variance (normalized by capacity^2)
    loads     = [route_demand(r, inst.demands) for r in current_sol]
    mean_load = sum(loads) / len(loads)
    variance  = sum((l - mean_load)**2 for l in loads) / len(loads)
    load_var  = variance / (inst.capacity ** 2)

    # feature 3: avg customers per route (normalized by total customers)
    avg_len = sum(len(r) for r in current_sol) / len(current_sol) / inst.n

    # feature 4: progress through solve (0 = start, 1 = end)
    progress = iteration / total_iterations

    return torch.tensor([cost, load_var, avg_len, progress], dtype=torch.float32)

# REINFORCE update
def reinforce_update(log_prob, reward, optimizer):
    # REINFORCE loss: -log_prob × reward
    # negative because PyTorch minimizes, but we want to maximize reward
    loss = -log_prob * reward
    optimizer.zero_grad()   # clear previous gradients
    loss.backward()         # compute gradients
    optimizer.step()        # update weights


# ALNS+ with configurable operator selector.
# usage: ALNS_plus(inst, selector='roulette') or ALNS_plus(inst, selector='neural')

class ALNS_plus:
    def __init__(self, inst: CVRPInstance, selector='roulette'):
        self.inst = inst
        self.selector = selector # either 'roulette' or 'neural'

        # destory and repair operation lists
        self.D = [self.random_removal, self.worst_removal, self.shaw_removal] # ya3ni random index 0, worst 1, shaw 2
        self.R = [self.greedy_insert, self.regret_insert] # greedy index 0, regret? index 1

        # weights for each heuristic
        # 1 because at the start they're all equally likely to be selected
        self.dest_weights = [1.0] * len(self.D)
        self.rep_weights = [1.0] * len(self.R)

        # initializing scores
        self.dest_scores = [0.0] * len(self.D)
        self.rep_scores = [0.0] * len(self.R)
        self.dest_usage = [0] * len(self.D)
        self.rep_usage = [0] * len(self.R)

        # neural selector networks + optimizers
        if selector == 'neural':
            self.destroy_net       = OperatorSelector(n_operators=len(self.D))
            self.repair_net        = OperatorSelector(n_operators=len(self.R))
            self.destroy_optimizer = torch.optim.Adam(self.destroy_net.parameters(), lr=0.001)
            self.repair_optimizer  = torch.optim.Adam(self.repair_net.parameters(),  lr=0.001)


    def solve(self, initial_solution, n_iter=1000, n_seg=100):
        # before loop initialize solution first
        current_sol = copy_solution(initial_solution)
        best_sol = copy_solution(initial_solution)
        best_obj = objective(best_sol, self.inst.dist)
        current_obj = best_obj

        for i in range(n_iter):
            self.reset_scoresNusage() # reset current score (to 0 as the start of every segment)
            # picking new q for each segment
            q = random.randint(5,15)

            for j in range(n_seg):

                # compute features for neural selector (used only if selector='neural')
                features = extract_features(current_sol, current_obj,i * n_seg + j, n_iter * n_seg, self.inst) if self.selector == 'neural' else None # so it doesnt compute for roulette

                dest_index, log_prob_d = self.select_operator(self.dest_weights, op_type='destroy', features=features)
                rep_index,  log_prob_r = self.select_operator(self.rep_weights,  op_type='repair',  features=features)

                # increment usage after selection
                self.dest_usage[dest_index] += 1
                self.rep_usage[rep_index] += 1

                # apply destroy + repair
                temp_dest, removed = self.destroy(current_sol, dest_index, q)
                candidate_sol = self.repair(temp_dest, rep_index, removed) # we gotta send in the list of removed customers
                candidate_obj = objective(candidate_sol, self.inst.dist)

                accepted = self.accept(current_obj, candidate_obj)
                reward = self.get_reward(candidate_obj, current_obj, best_obj, accepted)

                if accepted:
                    current_sol = candidate_sol # updating current to candidate
                    current_obj = candidate_obj

                    if current_obj < best_obj: # compare to global best
                        best_sol = copy_solution(current_sol)
                        best_obj = current_obj


                # REINFORCE update — only if neural selector and reward > 0
                if self.selector == 'neural' and reward > 0:
                    reinforce_update(log_prob_d, reward, self.destroy_optimizer)
                    reinforce_update(log_prob_r, reward, self.repair_optimizer)


                # update regardless of acceptance of rejection
                self.dest_scores[dest_index] += reward
                self.rep_scores[rep_index] += reward

            # after each segment
            # weight update — roulette only (neural uses REINFORCE instead)
            if self.selector == 'roulette':
                self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)
                self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)

        return best_sol

    def select_operator(self, op_weights, op_type, features=None):
        if self.selector == 'roulette':
            return self.roulette_wheel(op_weights), None  # None = no log_prob needed
        elif self.selector == 'neural':
            return self.neural_network(op_weights, op_type, features)
        else:
            raise ValueError(f"invalid selector type: {self.selector}")

    def neural_network(self, op_weights, op_type, features):
        if op_type == 'destroy':
            action, log_prob = self.destroy_net.select(features)
        elif op_type == 'repair':
            action, log_prob = self.repair_net.select(features)
        return action, log_prob

    def roulette_wheel(self, op_weights):
        # getting probabilities of each op based on weight formula
        # weight of op / sum of all weights of all ops
        probs = [0.0] * len(op_weights) # same length

        for i in range(len(op_weights)):
            probs[i] = op_weights[i] / sum(op_weights)

        # using random.choice p tells numpy the prob of picking that index
        # random value is generated by numpy b/w 0 - 1 and maps it to the index
        # based on those probs (higher prob -> bigger slice in wheel -> high likelyhood of being landed on)
        index = np.random.choice(range(len(op_weights)), p=probs)

        return index # will be used to pick the op from self.D or self.R

    def update_weights(self, weights, scores, usage,r=0.1):
        for w in range(len(weights)):
            if usage[w] > 0: # to avoid division by 0
                weights[w] = weights[w]*(1-r) + r*(scores[w]/usage[w])

    def random_removal(self, solution, q):
        sol = copy_solution(solution)
        removed = []
        # simple to flatten all customers from sol into one list
        flat_sol = [cust for route in sol for cust in route]
        # picking q customers randomly
        removed = random.sample(flat_sol,q)

        # iterating over to remove them from their routes
        for i in range(len(sol)):
            # for each route only keeping cust that aren't in removed
            sol[i] = [cust for cust in sol[i] if cust not in removed]

        sol = [route for route in sol if route] # filtering out the empty routes []
        # recall: route if nonempty evaluates to true

        return sol, removed

    def worst_removal(self, solution, q):
        p = 6 # randomization parameter
        sol = copy_solution(solution)
        removed = []

        costs = self.compute_costs4worst(sol)

        while q > 0:
            # getting selected customer from the costs list using index
            y = random.random() # random float value b/w 0.0 & 1.0
            index = min(int(y**p * len(costs)), len(costs) - 1) # note this clamps any issues when y gets close to 1.0 which could = len(costs)

            # getting selected customer from the costs list
            selected_cust = costs[index][1]
            removed.append(selected_cust)

            for j in range(len(sol)):
                sol[j] = [cust for cust in sol[j] if cust != selected_cust]

            # empty route removal
            sol = [route for route in sol if route]

            # recomputing costs -> rebuilding
            costs = self.compute_costs4worst(sol)

            q -= 1

        return sol, removed

    def shaw_removal(self, solution, q):
        p = 6  # same p as worst removal, controls how biased we are toward similar customers
        sol = copy_solution(solution)
        removed = []
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
            y = random.random()
            index = min(int(y**p * len(L)), len(L) - 1)
            removed.append(L[index])

        # remove everyone at once after the loop
        # different from worst removal which removes one by one and recomputes each time
        for i in range(len(sol)):
            sol[i] = [cust for cust in sol[i] if cust not in removed]

        sol = [route for route in sol if route]  # clear empty routes

        return sol, removed


    def greedy_insert(self, temp_dest, removed):
        sol = copy_solution(temp_dest)

        # inserting the hardest customers first
        sorted_removed = sorted(removed, key=lambda c: self.inst.demands[c], reverse=True)

        for cust in sorted_removed:
            best_cost = float('inf')
            best_route = None
            best_pos = None

            for r, route in enumerate(sol):
                for pos in range(len(route) + 1):
                    cost = self.insert_cost(cust, route, pos)
                    if cost < best_cost: # is cheap to insert?
                        best_cost = cost
                        best_route = r
                        best_pos = pos

            # inserting at best_pos found
            if best_route is not None:
              sol[best_route].insert(best_pos, cust)
            else:
              # there is no feasible position in the existing routes -> creating a solo route
              sol.append([cust])
              #print(f'Warning! no feasible position for customer {cust} so creating solo route')

        return sol

    def regret_insert(self, temp_dest, removed_customers):
        sol = copy_solution(temp_dest)
        removed = list(removed_customers) # Make a copy to avoid modifying the original list

        while removed:
            regret = []
            for cust in removed:
                best_cost = float('inf')
                best_route = None
                best_pos = None
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

                # If only one feasible position found, second_best_cost might still be inf. Handle this.
                if best_cost == float('inf'): # No feasible insertion found
                    regret.append((0, cust, None, None)) # Regret is 0, will create new route
                elif second_best_cost == float('inf'): # Only one feasible insertion point
                    regret.append((0, cust, best_route, best_pos)) # No regret (cost difference is infinite)
                else:
                    regret.append((second_best_cost - best_cost, cust, best_route, best_pos))

            # sorting regret for each customer (descending)
            sorted_regret = sorted(regret, key=lambda x: x[0], reverse=True)

            # picking customer with most regret and inserting at best position
            _, most_regret_cust, best_route, best_pos = sorted_regret[0]

            # inserting at best_pos found
            if best_route is not None:
                sol[best_route].insert(best_pos, most_regret_cust)
            else:
                # If best_route is None, it means no feasible position in existing routes
                # (this can happen if regret was 0 due to no feasible insertions)
                sol.append([most_regret_cust])
            removed.remove(most_regret_cust) # remove from removed list in both cases

        return sol

    def insert_cost(self, cust, route, pos): # similar to cost4worst only capacity consideration
      prev  = route[pos-1] if pos > 0 else 0
      after = route[pos] if pos < len(route) else 0

      demand = route_demand(route, self.inst.demands)
      # debugging print(f"  cust={cust} demand={self.inst.demands[cust]} route_demand={demand} capacity={self.inst.capacity}")

      if demand + self.inst.demands[cust] > self.inst.capacity:
          return float('inf')

      return self.inst.dist[prev][cust] + self.inst.dist[cust][after] - self.inst.dist[prev][after]

    def destroy(self, sol, dest_index, q):
        return self.D[dest_index](sol, q)

    def repair(self, sol, rep_index, removed):
        return self.R[rep_index](sol, removed)

    def accept(self, current_obj, candidate_obj):
        return candidate_obj < current_obj

    def get_reward(self, candidate_obj, current_obj, best_obj, accepted):
        # initializing the reward values
        sigma1 = 33
        sigma2 = 9
        # sigma3 = 13 not really used - because it requires simulated annealing acceptance

        if accepted:
            if candidate_obj < best_obj: # then new global best
                return sigma1
            else:
                return sigma2 # better than current
        else:
            return 0 # rejected

    def reset_scoresNusage(self):
        self.dest_scores = [0.0] * len(self.D)
        self.rep_scores = [0.0] * len(self.R)
        self.dest_usage = [0] * len(self.D)
        self.rep_usage = [0] * len(self.R)

    def compute_costs4worst(self, sol):
        # cost(i) = how much does the route shorten if we remove the customer i
        # formula from paper: dist(prev, i) + dist(i, after) - dist(prev, after)
        # basically the idea is what we pay to go through this cust i, minus what we'd pay skipping it

        costs = []
        # getting neighbors of customer on route + calcuating costs based on formula in paper
        for route in sol:
            for pos in range(len(route)):
                cust_id = route[pos] # customer id (not position)

                # depot is 0 so boundary customers connect to depot
                if pos == 0 : # if current cust is the first in the route
                    prev = 0
                else:
                    prev = route[pos-1]

                if pos == len(route) - 1: # if cust is the last in route
                    after = 0
                else:
                    after = route[pos+1]

                # high cost meaning its expensive to keep cust where they are -> good to remove

                cost_value = self.inst.dist[prev][cust_id] + self.inst.dist[cust_id][after] - self.inst.dist[prev][after]
                costs.append((cost_value, cust_id))


        costs.sort(reverse=True) # storing with most expensive at cust at the start

        return costs

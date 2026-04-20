import math  # math functions used for exponential and logarithmic calculations (mainly for simulated annealing)
import torch  # PyTorch core library for tensors and automatic differentiation
import torch.nn as nn  # neural network module (layers, models, etc.)
import torch.nn.functional as F  # functional utilities like activations (ReLU, Softplus)


class WeightNetwork(nn.Module):
    # this network learns how to produce alpha and beta parameters of a Beta distribution
    # from these, we sample a weight w1 between 0 and 1, and define w2 = 1 - w1
    # these weights determine how much importance we give to NV vs TD dynamically during search

    def __init__(self, input_dim=7, hidden_dim=16):
        super().__init__()  # initialize base PyTorch module

        # defining a small feedforward network
        # input is 7-dimensional feature vector describing search state
        # output is 2 values (alpha and beta)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # maps input features into hidden representation
            nn.ReLU(),  # adds non-linearity so the network can learn complex patterns
            nn.Linear(hidden_dim, 2),  # outputs two values corresponding to alpha and beta
            nn.Softplus()  # ensures outputs are positive (required for Beta distribution)
        )

        # initialize bias of final linear layer to 2.0
        # this biases alpha higher initially, meaning w1 tends toward 1
        # so early search prioritizes reducing NV strongly
        nn.init.constant_(self.net[2].bias, 2.0)

    def forward(self, x):
        # pass input through network to get raw alpha and beta values
        params = self.net(x) + 1.0  # adding 1 ensures alpha and beta are > 1 for stable unimodal distribution

        # split into alpha and beta components
        alpha, beta = params[:, 0], params[:, 1]

        # return both parameters for Beta distribution
        return alpha, beta


class ALNS_plus:
    # this is the main ALNS solver with multiple destroy and repair operators
    # it focuses heavily on reducing number of vehicles (NV) while maintaining good distance (TD)

    def __init__(self, inst: CVRPInstance, selector='roulette'):
        self.inst = inst  # instance containing problem data (distances, demands, capacity)
        self.selector = selector  # determines how operators are selected (roulette or neural)

        # list of destroy operators (ways to break the solution)
        self.D = [
            self.random_removal,        # random removal for exploration
            self.worst_removal,         # removes customers causing highest cost
            self.shaw_removal,          # removes geographically related customers
            self.small_route_removal,   # removes smallest route entirely (strong NV reduction)
            self.merge_routes_removal,  # removes two routes to force merging
            self.small_demand_removal,  # removes low-demand customers to free capacity
        ]

        # list of repair operators (ways to rebuild solution)
        self.R = [
            self.greedy_insert,  # cheapest insertion first
            self.regret_insert,  # insertion based on regret heuristic
        ]

        # weights, scores, and usage tracking for adaptive operator selection
        self.dest_weights = [1.0] * len(self.D)
        self.rep_weights  = [1.0] * len(self.R)
        self.dest_scores  = [0.0] * len(self.D)
        self.rep_scores   = [0.0] * len(self.R)
        self.dest_usage   = [0]   * len(self.D)
        self.rep_usage    = [0]   * len(self.R)

    def solve(self, initial_solution, n_iter=1000, n_seg=100):
        # copy initial solution to avoid modifying original
        current_sol = copy_solution(initial_solution)
        best_sol    = copy_solution(initial_solution)

        # compute initial objective
        best_obj = objective(best_sol, self.inst.dist)
        current_obj = best_obj

        # compute NV and TD separately for tracking
        current_nv, current_td = solution_cost(current_sol, self.inst.dist)
        best_nv, best_td = current_nv, current_td

        stagnation = 0  # counts how long we go without improvement

        for i in range(n_iter):  # outer loop
            self.reset_scoresNusage()  # reset operator stats

            # adaptive destruction size q
            # if current solution uses more vehicles than best, increase destruction
            if current_nv > best_nv:
                q = random.randint(10, min(25, self.inst.n // 4))
            else:
                q = random.randint(5, 15)

            for j in range(n_seg):  # inner loop
                dest_index = self.select_operator(self.dest_weights, op_type='destroy')  # choose destroy operator
                rep_index  = self.select_operator(self.rep_weights,  op_type='repair')  # choose repair operator

                self.dest_usage[dest_index] += 1  # track usage
                self.rep_usage[rep_index]   += 1

                temp_dest, removed = self.destroy(current_sol, dest_index, q)  # destroy part of solution
                candidate_sol = self.repair(temp_dest, rep_index, removed)  # rebuild solution

                candidate_obj = objective(candidate_sol, self.inst.dist)  # compute objective
                cand_nv, cand_td = solution_cost(candidate_sol, self.inst.dist)  # compute NV and TD

                accepted = self.accept(current_obj, candidate_obj)  # simple descent acceptance
                reward = self.get_reward(candidate_obj, current_obj, best_obj, accepted)  # compute reward

                if accepted:
                    current_sol = candidate_sol
                    current_obj = candidate_obj
                    current_nv  = cand_nv
                    current_td  = cand_td

                    if current_obj < best_obj:
                        best_sol = copy_solution(current_sol)
                        best_obj = current_obj
                        best_nv  = current_nv
                        best_td  = current_td
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1

                self.dest_scores[dest_index] += reward  # update destroy operator score
                self.rep_scores[rep_index]   += reward  # update repair operator score

            self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)  # update destroy weights
            self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)  # update repair weights

        return best_sol  # return best solution found


class ALNS_plus_learned_accept(ALNS_plus):
    # this class extends ALNS_plus by adding:
    # 1. learned acceptance using neural network
    # 2. simulated annealing fallback
    # 3. entropy-regularized policy gradient learning

    def __init__(self, inst: CVRPInstance, lr=0.01,
                 T_end=0.01, target_accept=0.1):
        super().__init__(inst, selector='roulette')

        self.weight_net = WeightNetwork(input_dim=7, hidden_dim=16)  # neural network for weights
        self.weight_opt = torch.optim.Adam(self.weight_net.parameters(), lr=lr)  # optimizer

        # simulated annealing parameters
        self.T_end = T_end  # final temperature
        self.target_accept = target_accept  # desired acceptance rate
        self.T = None  # current temperature
        self.T_start = None  # initial temperature
        self.c = None  # cooling factor

        # normalization references
        self._ref_nv = None
        self._ref_td = None
        self._best_nv = None

        # policy gradient tracking
        self._last_log_prob = None
        self._last_features = None
        self._stagnation = 0

    def calibrate_T_start(self, initial_sol, n_samples=200):
        # estimates a good starting temperature based on average worsening moves

        deltas = []
        init_nv, init_td = solution_cost(initial_sol, self.inst.dist)

        for _ in range(n_samples):
            q = random.randint(5, 15)
            temp, rem = self.random_removal(initial_sol, q)
            cand = self.greedy_insert(temp, rem)
            cand_nv, cand_td = solution_cost(cand, self.inst.dist)

            current_score = 0.5 * (init_nv / max(init_nv, 1)) + 0.5 * (init_td / max(init_td, 1))
            candidate_score = 0.5 * (cand_nv / max(init_nv, 1)) + 0.5 * (cand_td / max(init_td, 1))

            delta = candidate_score - current_score

            if delta > 0:
                deltas.append(delta)

        if not deltas:
            return 0.1  # fallback if no worsening moves found

        delta_avg = sum(deltas) / len(deltas)

        # compute temperature such that average worsening move is accepted with target probability
        return -delta_avg / math.log(self.target_accept)

    def get_weight_features(self, current_nv, current_td,
                             best_nv, best_td,
                             stagnation, iteration, total_iter):
        # builds feature vector representing current search state

        progress = iteration / max(total_iter, 1)
        nv_ratio = current_nv / max(self._ref_nv, 1)
        td_ratio = current_td / max(self._ref_td, 1)

        nv_from_best = max(0.0, (current_nv - best_nv) / max(self._ref_nv, 1))
        td_from_best = max(0.0, (current_td - best_td) / max(self._ref_td, 1))

        stagnation_norm = min(stagnation / max(total_iter, 1), 1.0)

        nv_above_optimal = max(0.0, (current_nv - self._best_nv) / max(self._ref_nv, 1))

        return [
            progress,
            nv_ratio,
            td_ratio,
            nv_from_best,
            td_from_best,
            stagnation_norm,
            nv_above_optimal,
        ]

    def learned_accept(self, current_nv, current_td,
                        candidate_nv, candidate_td,
                        best_nv, best_td,
                        iteration, total_iter):
        # performs two-stage acceptance: learned + simulated annealing fallback

        features = self.get_weight_features(
            current_nv, current_td,
            best_nv, best_td,
            self._stagnation, iteration, total_iter
        )

        self._last_features = features  # store features for training

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        alpha, beta = self.weight_net(x)
        dist = torch.distributions.Beta(alpha, beta)

        w1 = dist.rsample()  # differentiable sample
        w2 = 1.0 - w1

        self._last_log_prob = dist.log_prob(w1)

        w1_val = w1.item()
        w2_val = w2.item()

        current_score = w1_val * (current_nv / max(self._ref_nv, 1)) \
                      + w2_val * (current_td / max(self._ref_td, 1))

        candidate_score = w1_val * (candidate_nv / max(self._ref_nv, 1)) \
                        + w2_val * (candidate_td / max(self._ref_td, 1))

        if candidate_score < current_score:
            return True, 'learned'

        delta = candidate_score - current_score

        if self.T is not None and self.T > 1e-10:
            P = math.exp(-delta / self.T)
            if random.random() < P:
                return True, 'sa'

        return False, 'rejected'

    def update_weight_net(self, reward):
        # updates neural network using REINFORCE

        if reward == 0 or self._last_log_prob is None or self._last_features is None:
            return

        x = torch.tensor(self._last_features, dtype=torch.float32).unsqueeze(0)

        alpha, beta = self.weight_net(x)
        dist = torch.distributions.Beta(alpha, beta)

        entropy = dist.entropy().mean()  # encourages exploration

        loss = -self._last_log_prob * reward - 0.01 * entropy  # policy gradient + entropy regularization

        self.weight_opt.zero_grad()
        loss.backward()
        self.weight_opt.step()

    def solve(self, initial_solution, n_iter=1000, n_seg=100):
        current_sol = copy_solution(initial_solution)
        best_sol = copy_solution(initial_solution)

        current_nv, current_td = solution_cost(current_sol, self.inst.dist)
        best_nv, best_td = current_nv, current_td
        best_obj = 1000 * best_nv + best_td

        self._ref_nv = current_nv
        self._ref_td = current_td
        self._best_nv = current_nv

        self.T_start = self.calibrate_T_start(initial_solution)
        self.T = self.T_start

        total_iter = n_iter * n_seg

        self.c = (self.T_end / max(self.T_start, 1e-10)) ** (1.0 / total_iter)

        self._stagnation = 0
        self._last_features = None
        self._last_log_prob = None

        for i in range(n_iter):
            self.reset_scoresNusage()

            if current_nv > best_nv:
                q = random.randint(10, min(25, self.inst.n // 4))
            else:
                q = random.randint(5, 15)

            for j in range(n_seg):
                iteration = i * n_seg + j

                dest_index = self.select_operator(self.dest_weights, op_type='destroy')
                rep_index  = self.select_operator(self.rep_weights,  op_type='repair')

                self.dest_usage[dest_index] += 1
                self.rep_usage[rep_index]   += 1

                temp_dest, removed = self.destroy(current_sol, dest_index, q)
                candidate_sol = self.repair(temp_dest, rep_index, removed)

                cand_nv, cand_td = solution_cost(candidate_sol, self.inst.dist)
                cand_obj = 1000 * cand_nv + cand_td

                current_obj_comp = 1000 * current_nv + current_td

                accepted, accept_type = self.learned_accept(
                    current_nv, current_td,
                    cand_nv, cand_td,
                    best_nv, best_td,
                    iteration, total_iter
                )

                if accepted:
                    if cand_obj < best_obj:
                        reward = 33
                    elif cand_obj < current_obj_comp:
                        reward = 9
                    elif accept_type == 'sa':
                        reward = 3
                    else:
                        reward = 1
                else:
                    reward = 0

                if accepted:
                    current_sol = candidate_sol
                    current_nv  = cand_nv
                    current_td  = cand_td

                    if cand_obj < best_obj:
                        best_sol = copy_solution(current_sol)
                        best_nv = cand_nv
                        best_td = cand_td
                        best_obj = cand_obj
                        self._best_nv = cand_nv
                        self._stagnation = 0
                    else:
                        self._stagnation += 1
                else:
                    self._stagnation += 1

                self.dest_scores[dest_index] += reward
                self.rep_scores[rep_index]   += reward

                self.T *= self.c  # cool temperature

                self.update_weight_net(reward)

            self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)
            self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)

        return best_sol

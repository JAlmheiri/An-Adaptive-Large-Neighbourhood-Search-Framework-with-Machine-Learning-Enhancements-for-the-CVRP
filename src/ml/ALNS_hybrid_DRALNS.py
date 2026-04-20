import torch                         # core pytorch library (tensors, gradients, etc.)
import torch.nn as nn               # neural network module (layers, models)
import torch.nn.functional as F     # activation functions like relu, softmax, etc.


class AdjustmentNet(nn.Module):
    #this network does not directly choose an operator.

    #instead, it outputs a multiplier (adjustment factor) for each operator.
    #these multipliers are applied on top of the existing roulette weights.

    #so the idea is:
        #roulette already has weights
        # this network nudges them up or down depending on the current state

    #values:
    #- around 1.0 → neutral (no change)
    #- >1.0 → boost that operator
    #- <1.0 → suppress that operator

    def __init__(self, n_features=12, hidden=32, n_operators=3):
        super().__init__()  # initialize base nn.Module

        # define the neural network as a simple feedforward stack
        self.net = nn.Sequential(

            # first layer: map input features → hidden representation
            nn.Linear(n_features, hidden),

            # relu introduces non-linearity (otherwise network is just linear)
            nn.ReLU(),

            # second layer: hidden → one value per operator
            nn.Linear(hidden, n_operators),

            # softplus ensures outputs are always positive
            # this is important because these are multiplicative weights
            nn.Softplus()
        )

    def forward(self, x):
        # pass input through the network
        raw = self.net(x)

        # shift outputs so they are centered roughly around 1.0 instead of near 0
        # without this, most values would be too small and shrink roulette weights too much
        return 0.5 + raw  # center around 1.0




class ALNS_plus_hybrid2(ALNS_plus):
    #this is a hybrid version of alns:

    #instead of choosing operators purely via: roulette (classical alns), or neural policy (like reinforce),

    #we combine both: final probability = roulette_weight × neural_adjustment


    def __init__(self, inst: CVRPInstance, selector='hybrid', lr=0.001):
        # initialize base alns class (operators, weights, etc.)
        super().__init__(inst, selector=selector)

        # adjustment network for destroy operators
        # outputs one multiplier per destroy operator
        self.destroy_adj = AdjustmentNet(
            n_features=12, hidden=32, n_operators=len(self.D))

        # adjustment network for repair operators
        self.repair_adj  = AdjustmentNet(
            n_features=12, hidden=32, n_operators=len(self.R))

        # optimizer for destroy adjustment network
        self.destroy_adj_opt = torch.optim.Adam(
            self.destroy_adj.parameters(), lr=lr)

        # optimizer for repair adjustment network
        self.repair_adj_opt  = torch.optim.Adam(
            self.repair_adj.parameters(),  lr=lr)

        # store current state features
        self._features         = None

        # store last adjustment outputs (needed for learning update)
        self._last_destroy_adj = None
        self._last_repair_adj  = None

        # store which operator index was chosen last
        self._last_dest_index  = None
        self._last_rep_index   = None

        # same state flags as previous models (used as features)
        self._best_improved    = 0.0
        self._current_accepted = 0.0
        self._current_improved = 0.0

#features 
    def get_features(self, current_sol, current_obj, best_obj,
                     stagnation, iteration, total_iter):

        # whether current solution matches or beats best
        is_current_best = 1.0 if current_obj <= best_obj else 0.0

        # normalized gap between current and best
        cost_diff_best  = (current_obj - best_obj) / max(best_obj, 1.0)

        # clamp to keep stable
        cost_diff_best  = max(-1.0, min(1.0, cost_diff_best))

        # normalized stagnation (how long we've been stuck)
        stagnation_norm = min(stagnation / max(total_iter, 1), 1.0)

        # progress through search (0 → 1)
        search_budget   = iteration   / max(total_iter, 1)

        # compute normalized route loads
        loads    = [route_demand(r, self.inst.demands) / self.inst.capacity
                    for r in current_sol]

        # average load across routes
        avg_load = sum(loads) / len(loads)

        # variance of route loads (captures imbalance)
        variance = sum((l - avg_load)**2 for l in loads) / len(loads)

        # fraction of routes that are almost full (>90%)
        pct_full = sum(1 for l in loads if l > 0.9) / len(loads)

        # average route length (number of customers)
        avg_rl   = sum(len(r) for r in current_sol) / len(current_sol)

        # normalize route length by instance size
        avg_rl_norm = avg_rl / max(self.inst.n, 1)

        # normalize problem size
        size_norm   = self.inst.n / 400.0

        # return full 12-feature vector
        return [
            self._best_improved, self._current_accepted,
            self._current_improved, is_current_best,
            cost_diff_best, stagnation_norm, search_budget,
            avg_load, variance, pct_full, avg_rl_norm, size_norm,
        ]

    # hybrid selection
    def select_operator(self, op_weights, op_type):

        # decide selection method based on mode
        if self.selector == 'hybrid':
            return self.hybrid_select(op_weights, op_type)

        # fallback: pure roulette
        elif self.selector == 'roulette':
            return self.roulette_wheel(op_weights)

        # anything else is invalid
        else:
            raise ValueError(f"invalid selector: {self.selector}")

    def hybrid_select(self, op_weights, op_type):

      #pick correct adjustment network
      net = self.destroy_adj if op_type == 'destroy' else self.repair_adj

      # convert features into tensor
      x           = torch.tensor(self._features, dtype=torch.float32).unsqueeze(0)

      # get adjustment multipliers from network
      adjustments = net(x).squeeze(0)

      # store adjustments and reset last selected index
      if op_type == 'destroy':
          self._last_destroy_adj = adjustments
          self._last_dest_index  = None
      else:
          self._last_repair_adj  = adjustments
          self._last_rep_index   = None

      # convert tensor to numpy for easier combination with weights
      adj_np   = adjustments.detach().numpy()

      # multiply roulette weights by neural adjustments
      combined = [w * a for w, a in zip(op_weights, adj_np)]

      # safety fix:
      # ensure no value is zero or negative (would break probability distribution)
      combined = np.array([max(w, 1e-8) for w in combined], dtype=np.float64)

      # normalize so probabilities sum to exactly 1
      combined = combined / combined.sum()

      # sample operator index from resulting distribution
      index = np.random.choice(range(len(combined)), p=combined)

      # store chosen index for training later
      if op_type == 'destroy':
          self._last_dest_index = index
      else:
          self._last_rep_index  = index

      return index

    #policy gradient update
    def update_hybrid(self, reward):

        # skip if no useful signal
        if reward == 0:
            return

        # update both destroy and repair networks
        for adj, index, optimizer in [
            (self._last_destroy_adj, self._last_dest_index, self.destroy_adj_opt),
            (self._last_repair_adj,  self._last_rep_index,  self.repair_adj_opt),
        ]:

            # skip if no valid data
            if adj is None or index is None:
                continue

            # log of chosen adjustment (like log_prob in reinforce)
            log_adj  = torch.log(adj[index] + 1e-8)

            # normalize adjustments into distribution
            norm_adj = adj / (adj.sum() + 1e-8)

            # entropy term to encourage exploration
            entropy  = -(norm_adj * torch.log(norm_adj + 1e-8)).sum()

            # loss:
            # encourage high adjustment if reward is good
            # discourage if reward is bad
            loss     = -log_adj * reward - 0.01 * entropy

            # standard training steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # solve(): simple descent + hybrid updates 
    def solve(self, initial_solution, n_iter=1000, n_seg=100):

        # initialize current and best solutions
        current_sol = copy_solution(initial_solution)
        best_sol    = copy_solution(initial_solution)

        # compute initial objective
        best_obj    = objective(best_sol, self.inst.dist)
        current_obj = best_obj

        # total iterations
        total_iter = n_iter * n_seg

        # stagnation counter
        stagnation = 0

        # previous objective
        prev_obj   = current_obj

        # initialize flags
        self._best_improved    = 0.0
        self._current_accepted = 0.0
        self._current_improved = 0.0

        # outer loop
        for i in range(n_iter):

            # reset stats for segment
            self.reset_scoresNusage()

            # random destruction size
            q = random.randint(5, 15)

            # inner loop
            for j in range(n_seg):
                iteration = i * n_seg + j

                # compute features
                self._features = self.get_features(
                    current_sol, current_obj, best_obj,
                    stagnation, iteration, total_iter
                )

                # select operators
                dest_index = self.select_operator(self.dest_weights, op_type='destroy')
                rep_index  = self.select_operator(self.rep_weights,  op_type='repair')

                # update usage counts
                self.dest_usage[dest_index] += 1
                self.rep_usage[rep_index]   += 1

                # apply destroy and repair
                temp_dest, removed = self.destroy(current_sol, dest_index, q)
                candidate_sol      = self.repair(temp_dest, rep_index, removed)

                # evaluate candidate
                candidate_obj      = objective(candidate_sol, self.inst.dist)

                # accept or reject
                accepted = self.accept(current_obj, candidate_obj)

                # compute reward
                reward   = self.get_reward(candidate_obj, current_obj, best_obj, accepted)

                # update flags
                self._current_accepted = 1.0 if accepted else 0.0
                self._current_improved = 1.0 if (accepted and candidate_obj < prev_obj) else 0.0
                self._best_improved    = 0.0

                if accepted:
                    prev_obj    = current_obj
                    current_sol = candidate_sol
                    current_obj = candidate_obj

                    if current_obj < best_obj:
                        best_sol            = copy_solution(current_sol)
                        best_obj            = current_obj
                        stagnation          = 0
                        self._best_improved = 1.0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1

                # update classical scores
                self.dest_scores[dest_index] += reward
                self.rep_scores[rep_index]   += reward

                # neural update
                if self.selector == 'hybrid':
                    self.update_hybrid(reward)

            # update roulette weights
            self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)
            self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)

        # return best found solution
        return best_sol

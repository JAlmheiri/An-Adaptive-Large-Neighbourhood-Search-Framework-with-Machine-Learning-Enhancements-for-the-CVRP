import torch  # importing PyTorch main library which handles tensors, gradients, and all deep learning operations
import torch.nn as nn  # importing neural network module which gives us layers like Linear and model structure tools
import torch.nn.functional as F  # importing functional API for activations like ReLU and Softplus

class WeightNetwork(nn.Module):
    # this network learns how to dynamically balance NV vs TD instead of using fixed 1000*NV + TD
    # it outputs parameters of a Beta distribution (alpha, beta) which we use to sample a weight w1
    # then w2 is simply 1 - w1, so together they form a convex combination of NV and TD

    def __init__(self, input_dim=7, hidden_dim=16):
      super().__init__()  # initialize the base nn.Module so PyTorch can track parameters properly

      # defining a small feedforward neural network using Sequential
      # input: 7 features describing current search state
      # output: 2 values which will become alpha and beta for Beta distribution
      self.net = nn.Sequential(
          nn.Linear(input_dim, hidden_dim),  # first layer maps 7 input features into a 16-dimensional hidden space
          nn.ReLU(),  # non-linearity so the network can learn complex relationships instead of just linear ones
          nn.Linear(hidden_dim, 2),  # final layer outputs 2 numbers which will correspond to alpha and beta
          nn.Softplus()  # ensures outputs are strictly positive (important because Beta params must be > 0)
    )

      # manually initializing bias of the last Linear layer (index 2 in Sequential) to 2.0
      # this makes alpha and beta start > 1 so the Beta distribution is unimodal (not extreme or unstable)
      nn.init.constant_(self.net[2].bias, 2.0)


    def forward(self, x):
        # passing input features through the network to get raw alpha and beta parameters
        params = self.net(x) + 1.0  # adding 1 ensures alpha and beta are > 1 (stability, avoids weird distributions)

        # splitting the two outputs into alpha and beta
        alpha, beta = params[:, 0], params[:, 1]

        # returning both parameters which define the Beta distribution
        return alpha, beta


class ALNS_plus_learned_accept(ALNS_plus):
    # this class modifies the acceptance rule of ALNS
    # instead of fixed 1000*NV + TD, it learns weights (w1, w2) dynamically using a neural network
    # so the algorithm can adapt during search (e.g., focus on vehicles early, distance later)

    def __init__(self, inst: CVRPInstance, lr=0.01):
        super().__init__(inst, selector='roulette')  # initialize base ALNS with roulette operator selection

        # creating the neural network that will learn weights
        self.weight_net = WeightNetwork(input_dim=7, hidden_dim=16)

        # optimizer to update the network parameters using gradients
        self.weight_opt = torch.optim.Adam(self.weight_net.parameters(), lr=lr)

        # reference values for normalization (these stay fixed from start of solve)
        self._ref_nv  = None  # initial number of vehicles (used to normalize future values)
        self._ref_td  = None  # initial total distance (used for normalization)
        self._best_nv = None  # best NV seen so far (used for special feature)

        # variables to store last sampled action for REINFORCE update
        self._last_log_prob = None  # log probability of sampled weight
        self._last_w1       = None  # sampled weight for NV
        self._last_w2       = None  # complementary weight for TD

        # track how long we've gone without improvement
        self._stagnation    = 0

    def get_weight_features(self, current_nv, current_td,
                             best_nv, best_td,
                             stagnation, iteration, total_iter):
        # this function builds the 7 input features for the neural network
        # everything is normalized so the network doesn't depend on scale

        progress = iteration / max(total_iter, 1)  # how far along the search we are (0 to 1)

        nv_ratio = current_nv / max(self._ref_nv, 1)  # current vehicles relative to initial vehicles
        td_ratio = current_td / max(self._ref_td, 1)  # current distance relative to initial distance

        # how much worse current is compared to best (if negative, we clip later)
        nv_from_best = (current_nv - best_nv) / max(self._ref_nv, 1)
        td_from_best = (current_td - best_td) / max(self._ref_td, 1)

        # special feature: how far NV is above best NV (only positive part matters)
        nv_above_optimal = max(0, current_nv - self._best_nv) / max(self._ref_nv, 1)

        # normalized stagnation (how stuck we are)
        stagnation_norm = min(stagnation / max(total_iter, 1), 1.0)

        # returning final feature vector
        return [
            progress,
            nv_ratio,
            td_ratio,
            max(0.0, nv_from_best),  # ensure we don't give negative values (network focuses on regression only)
            max(0.0, td_from_best),
            stagnation_norm,
            nv_above_optimal
        ]

    def learned_accept(self, current_nv, current_td,
                        candidate_nv, candidate_td,
                        best_nv, best_td,
                        iteration, total_iter):
        # this function decides whether to accept a candidate solution
        # instead of fixed weights, we sample (w1, w2) from a Beta distribution

        features = self.get_weight_features(
            current_nv, current_td,
            best_nv, best_td,
            self._stagnation, iteration, total_iter
        )  # build state representation

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # convert features into tensor with batch dimension

        alpha, beta = self.weight_net(x)  # get parameters of Beta distribution

        dist = torch.distributions.Beta(alpha, beta)  # define Beta distribution

        w1 = dist.rsample()  # sample weight using reparameterization trick so gradients can flow
        w2 = 1.0 - w1  # second weight automatically complements the first

        self._last_log_prob = dist.log_prob(w1)  # store log probability for policy gradient update
        self._last_w1       = w1.item()  # store actual scalar value of w1
        self._last_w2       = w2.item()  # store actual scalar value of w2

        # compute score for current solution using learned weights
        current_score = self._last_w1 * (current_nv / max(self._ref_nv, 1)) \
                      + self._last_w2 * (current_td / max(self._ref_td, 1))

        # compute score for candidate solution
        candidate_score = self._last_w1 * (candidate_nv / max(self._ref_nv, 1)) \
                        + self._last_w2 * (candidate_td / max(self._ref_td, 1))

        # accept if candidate has lower weighted score (i.e., better)
        return candidate_score < current_score

    def update_weight_net(self, reward):
        # this function updates the neural network using REINFORCE

        if reward == 0 or self._last_log_prob is None:  # if no reward or no action stored, skip update
            return

        # recompute distribution just to calculate entropy (encourages exploration)
        x = torch.tensor(
            self.get_weight_features(0, 0, 0, 0, self._stagnation, 0, 1),
            dtype=torch.float32
        ).unsqueeze(0)

        alpha, beta = self.weight_net(x)  # get distribution parameters again
        dist = torch.distributions.Beta(alpha, beta)  # define distribution
        entropy = dist.entropy().mean()  # measure uncertainty of distribution

        # REINFORCE loss: maximize reward-weighted log probability
        # minus entropy bonus so distribution doesn't collapse to extremes
        loss = -self._last_log_prob * reward - 0.01 * entropy

        self.weight_opt.zero_grad()  # clear previous gradients
        loss.backward()  # compute gradients via backpropagation
        self.weight_opt.step()  # update network parameters

    def solve(self, initial_solution, n_iter=1000, n_seg=100):
        # main ALNS loop but with learned acceptance instead of fixed rule

        current_sol = copy_solution(initial_solution)  # copy initial solution so we don't modify original
        best_sol    = copy_solution(initial_solution)  # initialize best solution as starting solution

        current_nv, current_td = solution_cost(current_sol, self.inst.dist)  # compute NV and TD for current
        best_nv, best_td = current_nv, current_td  # initialize best as current
        best_obj = 1000 * best_nv + best_td  # compute competition metric

        self._ref_nv = current_nv  # store initial NV as normalization reference
        self._ref_td = current_td  # store initial TD as normalization reference
        self._best_nv = current_nv  # initialize best NV

        total_iter = n_iter * n_seg  # total number of iterations
        self._stagnation = 0  # reset stagnation counter

        for i in range(n_iter):  # outer ALNS loop
            self.reset_scoresNusage()  # reset operator statistics
            q = random.randint(5, 15)  # randomly choose number of nodes to remove

            for j in range(n_seg):  # inner loop
                iteration = i * n_seg + j  # compute global iteration index

                dest_index = self.select_operator(self.dest_weights, op_type='destroy')  # choose destroy operator
                rep_index  = self.select_operator(self.rep_weights,  op_type='repair')  # choose repair operator

                self.dest_usage[dest_index] += 1  # track usage
                self.rep_usage[rep_index]   += 1

                temp_dest, removed = self.destroy(current_sol, dest_index, q)  # apply destroy operator
                candidate_sol = self.repair(temp_dest, rep_index, removed)  # repair solution

                cand_nv, cand_td = solution_cost(candidate_sol, self.inst.dist)  # compute NV and TD
                cand_obj = 1000 * cand_nv + cand_td  # compute competition metric

                accepted = self.learned_accept(
                    current_nv, current_td,
                    cand_nv, cand_td,
                    best_nv, best_td,
                    iteration, total_iter
                )  # decide acceptance using learned weights

                # reward logic based on improvement
                if accepted:
                    if cand_obj < best_obj:
                        reward = 33  # strong reward for new global best
                    elif cand_obj < (1000 * current_nv + current_td):
                        reward = 9  # medium reward for improving current
                    else:
                        reward = 3  # small reward for accepted but not improving metric
                else:
                    reward = 0  # no reward if rejected

                if accepted:
                    current_sol = candidate_sol  # update current solution
                    current_nv  = cand_nv
                    current_td  = cand_td

                    if cand_obj < best_obj:
                        best_sol = copy_solution(current_sol)  # update best solution
                        best_nv  = cand_nv
                        best_td  = cand_td
                        best_obj = cand_obj
                        self._best_nv = cand_nv  # update best NV for features

                        self._stagnation = 0  # reset stagnation if improved
                    else:
                        self._stagnation += 1  # otherwise increase stagnation
                else:
                    self._stagnation += 1  # increase stagnation if rejected

                self.dest_scores[dest_index] += reward  # update destroy operator score
                self.rep_scores[rep_index]   += reward  # update repair operator score

                self.update_weight_net(reward)  # update neural network

            self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)  # update destroy weights
            self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)  # update repair weights

        return best_sol  # return best solution found
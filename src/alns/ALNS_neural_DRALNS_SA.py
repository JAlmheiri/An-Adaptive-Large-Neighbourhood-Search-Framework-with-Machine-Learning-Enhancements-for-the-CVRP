class ALNS_plus_SA_neural1(ALNS_plus_SA_LINEAR):
    
    #extends the simulated annealing version of alns by adding a neural operator selector.

    #so this version keeps all the things already present in the parent class:
    #simulated annealing acceptance, reheating,automatic temperature calibration, all destroy/repair operators

    #on top of that, it learns which operators to choose using neural networks.

    def __init__(self, inst: CVRPInstance, selector='neural', cooling='exponential', lr=0.001):
        # call the parent constructor first so all inherited alns + sa machinery is initialized properly
        # this includes things like temperature settings, operator lists, weights, and base logic
        super().__init__(inst, selector=selector, cooling=cooling)  # ← pass cooling through

        # neural network for selecting destroy operators
        # input = 12 state features describing the current search state and current solution structure
        # output = one raw score for each destroy operator
        self.destroy_net = OperatorNet(n_features=12, hidden=64, n_operators=len(self.D))

        # neural network for selecting repair operators
        # same input feature vector, but output size matches the number of repair operators instead
        self.repair_net  = OperatorNet(n_features=12, hidden=64, n_operators=len(self.R))

        # optimizer for the destroy network
        # adam is used here to update the neural net parameters during online training
        self.destroy_optimizer = torch.optim.Adam(self.destroy_net.parameters(), lr=lr)

        # optimizer for the repair network
        # this is separate because destroy and repair are learned independently
        self.repair_optimizer  = torch.optim.Adam(self.repair_net.parameters(),  lr=lr)

        # this will store the log-probability of the most recently sampled destroy action
        # needed later for the reinforce policy gradient update
        self._last_destroy_log_prob = None

        # same idea, but for the most recently sampled repair action
        self._last_repair_log_prob  = None

        # this will store the current feature vector representing the search state
        # each time we choose operators, we use these features as neural net input
        self._features              = None

        # this seems intended to store a previous objective value if needed later
        # it is initialized here, although in this specific code path prev_obj is handled locally in solve()
        self._prev_obj              = None

        # binary flag: did we improve the global best solution in the previous step?
        self._best_improved         = 0.0

        # binary flag: was the candidate accepted in the previous step?
        self._current_accepted      = 0.0

        # binary flag: did the accepted candidate improve the previous current solution?
        self._current_improved      = 0.0

    # feature computation
    def get_features(self, current_sol, current_obj, best_obj, stagnation, iteration, total_iter):
      # flag that says whether the current solution is equal to or better than the best solution found so far
      # in practice, this should usually be 1 only when current and best coincide
      is_current_best = 1.0 if current_obj <= best_obj else 0.0

      # normalized gap between current solution and best solution
      # if current is worse than best, this is positive
      # dividing by best_obj makes the feature less dependent on raw instance scale
      cost_diff_best  = (current_obj - best_obj) / max(best_obj, 1.0)

      # clip the value into [-1, 1] so extreme values do not destabilize learning
      # this keeps the feature bounded and easier for the network to handle
      cost_diff_best  = max(-1.0, min(1.0, cost_diff_best))

      # normalize stagnation by total search budget
      # this captures how stuck the search currently is, on a roughly 0 to 1 scale
      stagnation_norm = min(stagnation / max(total_iter, 1), 1.0)

      # fraction of the total search process that has already passed
      # early in the run this is near 0, late in the run it moves toward 1
      search_budget   = iteration / max(total_iter, 1)

      #new problem-specific features 
      # compute route loads as a fraction of vehicle capacity
      # for each route, route_demand(...) gives total customer demand on that route,
      # and dividing by capacity turns it into a normalized fill ratio
      loads         = [route_demand(r, self.inst.demands) / self.inst.capacity
                      for r in current_sol]

      # average route load across all current routes
      # this tells the model whether routes are generally lightly or heavily filled
      avg_load      = sum(loads) / len(loads)

      # same value stored again under another name to make the variance formula clearer
      mean_l        = avg_load

      # variance of route loads
      # this measures imbalance between routes:
      # low variance = routes are similarly loaded
      # high variance = some routes are much fuller than others
      variance      = sum((l - mean_l)**2 for l in loads) / len(loads)

      # fraction of routes that are almost full
      # here "almost full" means more than 90% utilized
      # this can help the model sense how tight capacity is in the current solution
      pct_full      = sum(1 for l in loads if l > 0.9) / len(loads)

      # average number of customers per route
      # this is a rough structural descriptor of the current solution
      avg_route_len = sum(len(r) for r in current_sol) / len(current_sol)

      # normalize average route length by problem size
      # this makes the feature more comparable across instances of different sizes
      avg_route_len_norm = avg_route_len / max(self.inst.n, 1)

      # normalized instance size
      # dividing by 400.0 is a hand-chosen scaling factor so the feature stays in a moderate range
      size_norm     = self.inst.n / 400.0

      # return the full 12-dimensional feature vector used by the neural selector
      return [
          # search dynamics (7)
          # these describe what has been happening in the search recently
          self._best_improved,
          self._current_accepted,
          self._current_improved,
          is_current_best,
          cost_diff_best,
          stagnation_norm,
          search_budget,

          # solution structure (5)
          # these describe what the current solution itself looks like
          avg_load,
          variance,
          pct_full,
          avg_route_len_norm,
          size_norm,
      ]

    #neural operator selection 
    def neural(self, op_weights, op_type):
        # choose which network to use depending on whether we are selecting
        # a destroy operator or a repair operator
        net = self.destroy_net if op_type == 'destroy' else self.repair_net

        # convert the current feature list into a pytorch tensor
        # unsqueeze(0) adds a batch dimension, so shape becomes [1, num_features]
        x        = torch.tensor(self._features, dtype=torch.float32).unsqueeze(0)

        # pass the features through the selected network
        # output is one raw score per operator
        # squeeze(0) removes the batch dimension again
        logits   = net(x).squeeze(0)

        # convert raw scores into a probability distribution over operators
        probs    = F.softmax(logits, dim=0)

        # create a categorical distribution so we can sample an operator index
        # this means the method is stochastic, not greedy
        dist     = torch.distributions.Categorical(probs)

        # sample one operator according to the learned probabilities
        action   = dist.sample()

        # store the log probability of the selected action
        # this is what reinforce uses later to push up/down the probability of this choice
        log_prob = dist.log_prob(action)

        # save the log probability in the correct slot so the later learning update
        # knows which network/action pair to train
        if op_type == 'destroy':
            self._last_destroy_log_prob = log_prob
        else:
            self._last_repair_log_prob  = log_prob

        # return the selected operator as a normal python integer
        return action.item()

#policy gradient update 
    def update_neural(self, reward):
        # if reward is zero, there is no learning signal worth using here
        # so we skip the update entirely
        if reward == 0:
            return

        # loop over both operator-selection networks:
        # first destroy, then repair
        for log_prob, optimizer, net in [
            (self._last_destroy_log_prob, self.destroy_optimizer, self.destroy_net),
            (self._last_repair_log_prob,  self.repair_optimizer,  self.repair_net),
        ]:
            # if for some reason no action log-probability was stored, skip this network
            if log_prob is None:
                continue

            # rebuild the feature tensor for the current state
            # this is the same state representation used when the action was chosen
            x       = torch.tensor(self._features, dtype=torch.float32).unsqueeze(0)

            # forward pass through the relevant network to get fresh logits
            logits  = net(x).squeeze(0)

            # turn logits into probabilities
            probs   = F.softmax(logits, dim=0)

            # compute entropy of the probability distribution
            # higher entropy means more spread-out choices
            # this term encourages exploration and prevents the policy from collapsing too early
            entropy = -(probs * torch.log(probs + 1e-8)).sum()

            # reinforce loss:
            # -log_prob * reward means:
            # if reward is good, increase probability of chosen action
            # if reward is bad/negative, decrease probability of chosen action
            #
            # the entropy term is subtracted so that maximizing entropy slightly lowers the loss,
            # encouraging broader exploration
            loss    = -log_prob * reward - 0.01 * entropy

            # clear old gradients before computing new ones
            optimizer.zero_grad()

            # backpropagate through the network
            loss.backward()

            # apply one optimization step
            optimizer.step()

    # solve(): sa + reheating + neural updates 
    def solve(self, initial_solution, n_iter=1000, n_seg=100):
        # start from a copy of the initial solution so the original is untouched
        current_sol = copy_solution(initial_solution)

        # best solution seen so far also starts as the initial solution
        best_sol    = copy_solution(initial_solution)

        # compute objective of the initial solution
        best_obj    = objective(best_sol, self.inst.dist)

        # current objective starts equal to best objective
        current_obj = best_obj

        # total number of inner iterations across the whole run
        total_iter   = n_iter * n_seg

        # automatically calibrate a starting temperature based on the initial solution/problem
        self.T_start = self.calibrate_T_start(initial_solution)

        # current temperature begins at the calibrated starting temperature
        self.T       = self.T_start

        # precompute exponential cooling factor
        # after repeated multiplication, temperature should decay from T_start toward T_end
        self.c       = (self.T_end / self.T_start) ** (1.0 / total_iter)

        # if linear cooling is being used instead of exponential,
        # compute how much temperature should decrease per iteration
        if self.cooling == 'linear':
            cooling_step = (self.T_start - self.T_end) / total_iter

        # stagnation counter tracks how long we've gone without improving the best solution
        stagnation = 0

        # prev_obj stores the previous current objective
        # later used to define whether the current solution improved relative to the prior one
        prev_obj   = current_obj

        # initialize the state flags before the search starts
        self._best_improved    = 0.0
        self._current_accepted = 0.0
        self._current_improved = 0.0

        # outer loop: segments
        # in alns, weights are often updated after batches/segments of iterations
        for i in range(n_iter):
            # reset accumulated operator scores and usage counts for this segment
            self.reset_scoresNusage()

            # choose how many customers to remove during destroy
            # q is randomized each segment here
            q = random.randint(5, 15)

            # inner loop: actual destroy-repair iterations inside the segment
            for j in range(n_seg):
                # flattened iteration counter across the entire run
                iteration = i * n_seg + j

                # build the current feature vector before selecting operators
                # this state is what the neural policy sees when making its choice
                self._features = self.get_features(
                    current_sol, current_obj, best_obj, stagnation, iteration, total_iter
                )

                # choose destroy operator
                dest_index = self.select_operator(self.dest_weights, op_type='destroy')

                # choose repair operator
                rep_index  = self.select_operator(self.rep_weights,  op_type='repair')

                # count how many times each operator was used
                self.dest_usage[dest_index] += 1
                self.rep_usage[rep_index]   += 1

                # apply destroy operator to current solution
                # this produces a partial solution and a list of removed customers
                temp_dest, removed = self.destroy(current_sol, dest_index, q)

                # apply repair operator to reinsert removed customers
                candidate_sol      = self.repair(temp_dest, rep_index, removed)

                # compute objective value of the resulting candidate solution
                candidate_obj      = objective(candidate_sol, self.inst.dist)

                # decide whether to accept the candidate under simulated annealing logic
                # accepted can return labels like 'improved', 'sa_accepted', or rejection
                accepted = self.accept(current_obj, candidate_obj)

                # compute reward for the operator choices
                # this reward is later used both for operator scoring and neural learning
                reward   = self.get_reward(candidate_obj, current_obj, best_obj, accepted)

                # update state flags so the next iteration's feature vector can reflect
                # what just happened in this iteration
                self._current_accepted = 1.0 if accepted in ('improved', 'sa_accepted') else 0.0

                # this flag says whether the accepted candidate improved on the previous current solution
                # note: prev_obj is the previous current objective, not necessarily the global best
                self._current_improved = 1.0 if (accepted in ('improved', 'sa_accepted') and candidate_obj < prev_obj) else 0.0

                # reset this flag first; it will be turned back to 1 only if best improves below
                self._best_improved    = 0.0

                # if candidate was accepted either because it improved or because sa allowed it,
                # then current solution moves forward to that candidate
                if accepted in ('improved', 'sa_accepted'):
                    # save old current objective into prev_obj before overwriting current_obj
                    prev_obj    = current_obj

                    # update current solution and objective
                    current_sol = candidate_sol
                    current_obj = candidate_obj

                    # if the new current is also a new global best, store it
                    if current_obj < best_obj:
                        best_sol            = copy_solution(current_sol)
                        best_obj            = current_obj

                        # improvement resets stagnation
                        stagnation          = 0

                        # mark that the best solution was improved this iteration
                        self._best_improved = 1.0
                    else:
                        # accepted, but did not improve global best
                        stagnation += 1
                else:
                    # rejected candidate also counts as stagnation
                    stagnation += 1

                # reheating logic:
                # if we have been stagnant for a long time and the temperature has cooled too much,
                # bump the temperature back up a bit to reintroduce exploration
                if stagnation > 2000 and self.T < self.T_start * 0.01:
                    self.T     = self.T_start * 0.1
                    stagnation = 0

                # accumulate reward into operator scores
                # these are used by the classical alns weight adaptation mechanism
                self.dest_scores[dest_index] += reward
                self.rep_scores[rep_index]   += reward

                # cooling step after each iteration
                if self.cooling == 'exponential':
                    # exponential cooling multiplies by a constant factor each time
                    self.T *= self.c
                else:
                    # linear cooling subtracts a fixed amount but never goes below T_end
                    self.T = max(self.T_end, self.T - cooling_step)

                # update neural policies online after receiving the reward
                if self.selector == 'neural':
                    self.update_neural(reward)

            # after one segment finishes, update classical alns operator weights
            # based on accumulated score and usage over that segment
            self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)
            self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)

        # return the best solution found during the whole run
        return best_sol
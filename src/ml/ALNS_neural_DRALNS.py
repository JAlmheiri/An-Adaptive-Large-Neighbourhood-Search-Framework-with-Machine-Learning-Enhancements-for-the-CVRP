import torch                    # main PyTorch library (tensors, autograd, etc.)
import torch.nn as nn          # neural network module (layers, models)
import torch.nn.functional as F  # functional utilities like activations (ReLU, softmax)

class OperatorNet(nn.Module):
    #this is a simple neural network (MLP) that takes in the current ALNS state
    #and outputs scores (logits) for choosing an operator.
    
    #it has: input layer (state features), two hidden layers, output layer (one value per operator)
    #it outputs just raw scores (logits).
    def __init__(self, n_features=7, hidden=64, n_operators=3):
        super().__init__()  # initialize nn.Module properly

        # first linear layer
        # takes 7 features -> transforms into a 64-dimensional hidden representation
        self.fc1 = nn.Linear(n_features, hidden)

        # second hidden layer
        # keeps same size (64 -> 64), allows deeper representation learning
        self.fc2 = nn.Linear(hidden, hidden)

        #final layer:
        # maps hidden features to number of operators
        # each output corresponds to one operator
        self.fc3 = nn.Linear(hidden, n_operators)

    def forward(self, x):
        # pass input through first layer, then apply ReLU activation
        # ReLU = max(0, x), introduces non-linearity
        x = F.relu(self.fc1(x))

        # pass through second layer + ReLU again
        x = F.relu(self.fc2(x))

        # Final layer gives raw scores (logits)
        # IMPORTANT: no softmax here (we apply it later manually)
        return self.fc3(x)  # raw logits


class ALNS_plus_neural1(ALNS_plus):

    #this class extends ALNS_plus solver by adding a neural network
    #to decide WHICH operators to use.
    #training is ONLINE (during solving).

    #instead of manually using weights, we learn a policy using reinforcement learning (REINFORCE).

    #Key idea:The neural net outputs probabilities over operators
	    #We sample an operator
	    #if it leads to improvement -> reward it
	    #Iif not -> no reward

    #We use 2 networks:
      #destroy_net -> chooses destroy operator
      #repair_net  -> chooses repair operator


    def __init__(self, inst: CVRPInstance, selector='neural', lr=0.01):
        # initialize parent ALNS class
        super().__init__(inst, selector=selector)

        #neural network for destroy operators
        # output size = number of destroy operators
        self.destroy_net = OperatorNet(n_features=7, hidden=64, n_operators=len(self.D))

        # neural network for repair operators
        self.repair_net  = OperatorNet(n_features=7, hidden=64, n_operators=len(self.R))

        # Adam optimizer for destory network
        # this updates its params using gradients
        self.destroy_optimizer = torch.optim.Adam(self.destroy_net.parameters(), lr=lr)

        #adam optimizer for repair network
        self.repair_optimizer  = torch.optim.Adam(self.repair_net.parameters(),  lr=lr)

        #these store log probabilities of the last chosen actions
        #which are needed for policy gradient update later
        self._last_destroy_log_prob = None
        self._last_repair_log_prob  = None

        #stores current state features (input to network)
        self._features = None

        #binary flags that describe what happened in last iteration
        # they are part of the state representation
        self._best_improved    = 0.0  # did we improve global best?
        self._current_accepted = 0.0  # was candidate accepted?
        self._current_improved = 0.0  # did current improve?

    #feature construction 
    def get_features(self, current_obj, best_obj, stagnation, iteration, total_iter):
        #check if current solution is equal or better than best
        is_current_best = 1.0 if current_obj <= best_obj else 0.0

        # compute relative difference between current and best
        # normalized so it's scale-independent
        cost_diff_best  = (current_obj - best_obj) / max(best_obj, 1.0)

        #clamp this value between -1 and 1 (stability)
        cost_diff_best  = max(-1.0, min(1.0, cost_diff_best))

        #normalize stagnation (how long we've been stuck)
        stagnation_norm = min(stagnation / max(total_iter, 1), 1.0)

        #how far we are in the search process (0 -> 1)
        search_budget   = iteration / max(total_iter, 1)

        # Return full feature vector (7 values)
        return [
            self._best_improved,
            self._current_accepted,
            self._current_improved,
            is_current_best,
            cost_diff_best,
            stagnation_norm,
            search_budget,
        ]

    #neural operator selection (core decision step)
    def neural(self, op_weights, op_type):
        #choose correct network depending on operator type
        net = self.destroy_net if op_type == 'destroy' else self.repair_net

        #convert features -> tensor (batch size = 1)
        x = torch.tensor(self._features, dtype=torch.float32).unsqueeze(0)

        #pass through network to get logits (raw scores)
        logits = net(x).squeeze(0)

        #convert logits to probs using softmax
        probs = F.softmax(logits, dim=0)

        # create categorical distribution from probabilities
        dist = torch.distributions.Categorical(probs)

        # sample an operator index
        action = dist.sample()

        #compute the log probability of chosen action (needed for REINFORCE)
        log_prob = dist.log_prob(action)

        #store log_prob depending on operator type
        if op_type == 'destroy':
            self._last_destroy_log_prob = log_prob
        else:
            self._last_repair_log_prob  = log_prob

        #return integer index of selected operator
        return action.item()

    # Policy gradient update (learning step)
    def update_neural(self, reward):
        #if reward is zero -> no learning signal -> skip update
        if reward == 0:
            return

        #loop over both networks (destroy + repair)
        for log_prob, optimizer, net in [
            (self._last_destroy_log_prob, self.destroy_optimizer, self.destroy_net),
            (self._last_repair_log_prob,  self.repair_optimizer,  self.repair_net),
        ]:
            #if no action was taken, skip
            if log_prob is None:
                continue

            #recompute forward pass (needed for entropy)
            x = torch.tensor(self._features, dtype=torch.float32).unsqueeze(0)
            logits = net(x).squeeze(0)

            #convert to probabilities again
            probs = F.softmax(logits, dim=0)

            #compute entropy:
            # this encourages exploration (avoid collapsing to one operator)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()

            # REINFORCE loss:
            # maximize log_prob * reward -> we minimize negative
            # also subtract entropy bonus (encourage diversity)
            loss = -log_prob * reward - 0.01 * entropy

            #reset gradients
            optimizer.zero_grad()

            #backpropagate loss -> compute gradients
            loss.backward()

            #update parameters
            optimizer.step()

    # Main solve loop (ALNS + neural learning)
    def solve(self, initial_solution, n_iter=1000, n_seg=100):

        #copy initial solution to avoid modifying original
        current_sol = copy_solution(initial_solution)
        best_sol    = copy_solution(initial_solution)

        #compute initial objective value
        best_obj    = objective(best_sol, self.inst.dist)
        current_obj = best_obj

        #total number of iterations (outer * inner loop)
        total_iter  = n_iter * n_seg

        # stagnation counter (how long no improvement)
        stagnation  = 0

        # store previous objective (for improvement check)
        prev_obj    = current_obj

        # initialize state flags
        self._best_improved    = 0.0
        self._current_accepted = 0.0
        self._current_improved = 0.0

        #outer loop (ALNS segments)
        for i in range(n_iter):

            #reset operator scores and usage counters
            self.reset_scoresNusage()

            #random destruction size
            q = random.randint(5, 15)

            #iunner loop (actual iterations)
            for j in range(n_seg):

                #global iteration index
                iteration = i * n_seg + j

                #compute state features BEFORE choosing operators
                self._features = self.get_features(
                    current_obj, best_obj, stagnation, iteration, total_iter
                )

                #select destroy and repair operators (via neural)
                dest_index = self.select_operator(self.dest_weights, op_type='destroy')
                rep_index  = self.select_operator(self.rep_weights,  op_type='repair')

                #track how often each operator is used
                self.dest_usage[dest_index] += 1
                self.rep_usage[rep_index]   += 1

                #apply destroy operator to remove nodes
                temp_dest, removed = self.destroy(current_sol, dest_index, q)

                #apply repair operator to rebuild solution
                candidate_sol = self.repair(temp_dest, rep_index, removed)

                #compute objective value of candidate
                candidate_obj = objective(candidate_sol, self.inst.dist)

                #decide whether to accept candidate solution
                accepted = self.accept(current_obj, candidate_obj)

                #compute reward (based on improvement and acceptance)
                reward = self.get_reward(candidate_obj, current_obj, best_obj, accepted)

                #update state flags (used in NEXT iteration)
                self._current_accepted = 1.0 if accepted else 0.0
                self._current_improved = 1.0 if (accepted and candidate_obj < prev_obj) else 0.0
                self._best_improved    = 0.0

                if accepted:
                    prev_obj    = current_obj
                    current_sol = candidate_sol
                    current_obj = candidate_obj

                    #check if this is a NEW global best
                    if current_obj < best_obj:
                        best_sol            = copy_solution(current_sol)
                        best_obj            = current_obj
                        stagnation          = 0
                        self._best_improved = 1.0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1

                #update operator scores (for classical ALNS weighting)
                self.dest_scores[dest_index] += reward
                self.rep_scores[rep_index]   += reward

                #perform neural update (policy gradient step)
                if self.selector == 'neural':
                    self.update_neural(reward)

            #update operator weights (ALNS mechanism)
            self.update_weights(self.dest_weights, self.dest_scores, self.dest_usage)
            self.update_weights(self.rep_weights,  self.rep_scores,  self.rep_usage)

        #return best solution found
        return best_sol

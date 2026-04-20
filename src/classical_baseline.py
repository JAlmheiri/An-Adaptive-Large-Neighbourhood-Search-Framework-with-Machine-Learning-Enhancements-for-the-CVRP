
from typing import List, Tuple

#defining clarke-wright algorithm, which starts with one route per customer, then computes savings for all sets of nodes
#using the formula S(i,j) = d(0,i) + d(0,j) - d(i,j)
#it then merges routes per customer savings order if no constraints violated

def clarke_wright(inst: CVRPInstance):
    #exctracting data: distance matrix, demand of each customer, vehicle capacity and number of customers respectively
    dist    = inst.dist
    demands = inst.demands
    cap     = inst.capacity
    n       = inst.n

    routes   = [[c] for c in range(1, n + 1)] #initializing routes. each customer has their own route [[1], [2], ...]
    route_of = {c: c - 1 for c in range(1, n + 1)} #keeping track of which customer has which route ex. customer 1 has route 0 {1:0, 2:1, ...}

#computing all savings over all pairs (i,j)
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True) #sort by descending order

  #now we loop over savings
    for s, i, j in savings:
        ri = route_of[i] #first determine which route each customer belongs to (to check constraints)
        rj = route_of[j]

        if ri == rj: #if they belong to the same route, no merging
            continue
        if routes[ri] is None or routes[rj] is None: #skip if the route is deleted (after merging)
            continue

        #get the routes themselves
        r_i = routes[ri]
        r_j = routes[rj]

        #run a capacity check. if total demand exceeds vehicle capacity in the event of merge, don't merge.
        if (sum(demands[c] for c in r_i) + sum(demands[c] for c in r_j)) > cap:
            continue

      #merging logic:
        merged = None
        if r_i[-1] == i and r_j[0] == j:#if i is at the end of route i and j is at the start of route j, merge them
            merged = r_i + r_j
        elif r_j[-1] == i and r_i[0] == j: #if i is at the end of route j and j is at the start of route i, merge them
            merged = r_j + r_i
        elif r_i[0] == i and r_j[-1] == j: #if i is at the start of route i, and j is at the end j, reverse route i and merge.
            merged = r_j + r_i
        elif r_j[0] == i and r_i[-1] == j: #if i is at start of route j, and j is at the end of i, reverse route j and merge
            merged = r_i + r_j

        #if none of the above satisfied, no merging.
        if merged is None:
            continue

        routes[ri] = merged #apply the merging changes
        routes[rj] = None #and delete the other route
        for c in merged: #now add all customers in merge to route i
            route_of[c] = ri

    return [r for r in routes if r is not None]

#adding a standard 2-opt improving algorithm which would attempt reversing every sub-segment and seeing
#if this reduces distance
def two_opt_route(route: List[int], dist):
    route = route[:] #make a copy of the route
    n = len(route) #get customers in the route
    if n < 4: #if they're less than 4, keep it as it is. (it's not useful to reverse a route with 1, 2 or 3 customers only)
        return route

    improved = True
    while improved:
        improved = False
        for i in range(n - 1): #scan every possible subsegment of the route where i is the start index of segment and j is the end index
            for j in range(i + 2, n): #ensure the segment length >=3
                prev_i = route[i - 1] if i > 0 else 0 #make sure prev_i is not the depot
                next_j = route[j + 1] if j + 1 < n else 0 #find the node just after the segment-make sure it's not the depot
                before = dist[prev_i][route[i]] + dist[route[j]][next_j] #this is the cost of the two boundary edges before reversing th segment
                after  = dist[prev_i][route[j]] + dist[route[i]][next_j] #and this is after
                if after < before - 1e-6: #if performance is positive, update the route and continue the loop.
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True
    return route

#applies the 2-opt to every route independently
def two_opt(solution: Solution, inst: CVRPInstance):
    return [two_opt_route(route, inst.dist) for route in solution]

#another improving algorithm is the or_opt, which relocates segments of consecutive customer to
#the cheapest feasible option in any route.
def or_opt(solution: Solution, inst: CVRPInstance,
           segment_sizes: Tuple[int, ...] = (1, 2, 3)):
    #exracting data
    dist    = inst.dist
    demands = inst.demands
    cap     = inst.capacity
    #work on a copy of the solution
    solution = copy_solution(solution)

    #loop over all segment lengths (moving 1 customer, then 2 consecutive customers then 3)
    for seg_len in segment_sizes:
        improved = True
        while improved:
            improved = False
            found    = False

            #loop over the source routes. if it's too short, skip it.
            for ri in range(len(solution)):
                if found:
                    break
                route = solution[ri]
                if len(route) <= seg_len:
                    continue
                #pick where each segment starts inside the source route
                for pos in range(len(route) - seg_len + 1):
                    if found:
                        break

                    segment = route[pos: pos + seg_len] #extract the segment
                    seg_dem = sum(demands[c] for c in segment) #and get total demand of the segment

                    #get the cost of removing segment
                    prev = route[pos - 1] if pos > 0 else 0 #node just before the segment
                    nxt  = route[pos + seg_len] if pos + seg_len < len(route) else 0 #node just after the segment
                    #get the total benefit of removing the segment from the source route
                    rem_gain = (dist[prev][segment[0]]+ dist[segment[-1]][nxt]- dist[prev][nxt])

                    best_gain = 1e-6 #best gain is how much the total objective improves, initiallized to smallest num
                    best_rj   = None #the destination route index
                    best_ins  = None #insertion position in that route

                    #try inserting the segment into every route
                    for rj in range(len(solution)):
                        target = solution[rj]
                        if rj != ri: #if the move would exceed capacity, skip.
                            if route_demand(target, demands) + seg_dem > cap:
                                continue

                        #try every possible spot in the target route
                        for ins in range(len(target) + 1):
                            #avoid putting the segment where it was
                            if rj == ri and pos <= ins <= pos + seg_len:
                                continue

                            #check nodes around the insertion point to get cost and gain
                            ins_prev = target[ins - 1] if ins > 0 else 0
                            ins_next = target[ins] if ins < len(target) else 0
                            ins_cost = (dist[ins_prev][segment[0]]
                                        + dist[segment[-1]][ins_next]
                                        - dist[ins_prev][ins_next])
                            gain = rem_gain - ins_cost

                            #if this move is the best so far, store it
                            if gain > best_gain:
                                best_gain = gain
                                best_rj   = rj
                                best_ins  = ins

                    #if some improving location is found, then perform it
                    if best_rj is not None:
                        #remove segment from source route
                        solution[ri] = route[:pos] + route[pos + seg_len:]

                        #if we're moving it inside the same route, insert with shifting logic
                        if best_rj == ri:
                            adj = best_ins - seg_len if best_ins > pos else best_ins
                            adj = max(0, min(adj, len(solution[ri])))
                            r = solution[ri]
                            solution[ri] = r[:adj] + segment + r[adj:]
                        else: #otherwise just insert it at chosen position
                            r   = solution[best_rj]
                            ins = min(best_ins, len(r))
                            solution[best_rj] = r[:ins] + segment + r[ins:]

                        solution = [r for r in solution if r]
                        improved = True
                        found    = True

    return solution


import math, os, re, json, subprocess, random, tempfile, shutil
from typing import List, Tuple, Dict, Optional


#classical_solve combines CW, 2-opt and OR-opt
def classical_solve(inst: CVRPInstance, verbose: bool = True, or_opt_segments: Tuple[int,...] = (1, 2, 3)) -> Solution:

    #log prints the current quality of a solution after each stage
    def log(sol, label):
        if not verbose: return #only print if logging is enabeled
        nv, td = solution_cost(sol, inst.dist) #get the num of vehicles and total distance from solution_cost()
        print(f"  [{label}]  NV={nv:3d}  TD={td:9,}  Obj={1000*nv+td:12,}") #and print them along with other details

   #checks that the solution is valid after each stage
    def check(sol, label):
        ok, r = is_feasible(sol, inst)
        assert ok, f"{label} infeasible: {r}"

    if verbose:
        print(f"\n{'='*55}")
        print(f"Variant A: {inst.name}  (n={inst.n}, cap={inst.capacity})")
        print(f"{'='*55}")
#build an initial solution with Clarke-Wright, then add 2-opt, then add or-opt
    sol = clarke_wright(inst);          log(sol, "CW    "); check(sol, "CW")
    sol = two_opt(sol, inst);           log(sol, "2-opt "); check(sol, "2-opt")
    sol = or_opt(sol, inst,
                 or_opt_segments);     log(sol, "Or-opt"); check(sol, "Or-opt")

    if verbose: print(f"{'='*55}")
    return sol


#evaluates the final solution
def evaluate(inst: CVRPInstance, sol: Solution, repo_root: str) -> Dict:
    sol_path = os.path.join(repo_root, 'Solutions', 'cvrp',f'{inst.name}.txt')
    os.makedirs(os.path.dirname(sol_path), exist_ok=True)
    write_solution(sol, sol_path) #saves the solution to the save path

    ok, reason = is_feasible(sol, inst) #check solution validity
    nv, td = solution_cost(sol, inst.dist) #get nv, td
    print(f"\nLocal check:  {'feasible' if ok else 'NOT feasible ' + reason}")
    print(f"Our score:    NV={nv}  TD={td:,}  Obj={1000*nv+td:,}")

    #check solution validity using evaluator.py
    result = run_evaluator(inst.name, sol_path, repo_root)
    if result and result['feasible']:
        print(f"Evaluator:     Obj={result['objective']:,}")
    elif result:
        print(f"Evaluator:    infeasible")
        print(result['raw'])
    return result



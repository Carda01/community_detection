from tqdm.auto import tqdm
import numpy as np
from joblib import Parallel, delayed
import networkx as nx

def __run_single_simulation(G, seeds, p):
    newly_activated, all_infected = seeds[:], set(seeds)
    while newly_activated:
        potential_infections = []
        for node in newly_activated:
            neighbors = list(G.successors(node))
            if not neighbors: continue

            success = np.random.random(len(neighbors)) < p
            successful_neighbors = np.array(neighbors)[success]
            potential_infections.extend(successful_neighbors)

        current_new_batch = set(potential_infections)
        newly_activated = current_new_batch - all_infected
        all_infected.update(newly_activated)

    return all_infected


def _run_single_simulation(G, seeds, p):
    return len(__run_single_simulation(G, seeds, p))


def get_spread(G, seeds, p=0.01, mc=1000, n_jobs=-1):
    spread = Parallel(n_jobs=n_jobs)(delayed(_run_single_simulation)(G, seeds, p) for _ in range(mc))
    return np.mean(spread)


def celf(G, k, p=0.01, mc=1000):
    print(f"\n[Starting CELF] Target seeds: {k} | Simulations: {mc} | p: {p}")

    nodes = list(G.nodes())

    print("  > Round 1: Calculating initial spread for all nodes...")


    with Parallel(n_jobs=-1) as parallel:
        results = parallel(delayed(get_spread)(G, [node], p, mc, n_jobs=1) for node in tqdm(nodes))

    margins = [[nodes[i], results[i], 0] for i in range(len(nodes))]

    margins.sort(key=lambda x: x[1], reverse=True)

    S = [margins[0][0]]
    spread = margins[0][1]

    print(f"  > Seed 1 found: Node {S[0]} (Spread: {spread:.2f})")

    margins = margins[1:]

    while len(S) < k:
        current_seed_round = len(S)
        
        while True:
            current_node, old_gain, last_update = margins[0]

            if last_update == current_seed_round:
                S.append(current_node)
                spread += old_gain
                print(f"  > Seed {len(S)} found: Node {current_node} (Marginal Gain: {old_gain:.2f})")
                margins.pop(0)
                break

            new_spread = get_spread(G, S + [current_node], p, mc)
            new_gain = max(0, new_spread - spread)

            margins[0] = [current_node, new_gain, current_seed_round]

            margins.sort(key=lambda x: x[1], reverse=True)

    return S, spread


def greedy(G, k, p=0.01, mc=1000, n_jobs=-1):
    """
    Finds the k most influential nodes in a graph G using the standard greedy algorithm.

    Args:
        G (nx.DiGraph): The graph.
        k (int): The number of seed nodes to find.
        p (float, optional): The propagation probability. Defaults to 0.01.
        mc (int, optional): The number of Monte Carlo simulations to run for spread calculation. Defaults to 1000.
        n_jobs (int, optional): The number of parallel jobs to run for spread calculation. Defaults to -1.

    Returns:
        tuple: A tuple containing the list of seed nodes and the final spread.
    """
    print(f"\n[Starting Greedy] Target seeds: {k} | Simulations: {mc} | p: {p}")

    S = []
    spread = 0
    all_nodes = list(G.nodes())

    for i in range(k):
        best_marginal_gain = -1
        best_node = None

        candidate_nodes = [node for node in all_nodes if node not in S]
        
        print(f"  > Finding seed {i+1}/{k} (evaluating {len(candidate_nodes)} nodes)...")

        gains = Parallel(n_jobs=n_jobs)(delayed(get_spread)(G, S + [node], p, mc, 1) for node in tqdm(candidate_nodes))

        for j, node in enumerate(candidate_nodes):
            marginal_gain = gains[j] - spread
            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_node = node
        
        if best_node is not None and best_marginal_gain > 0:
            S.append(best_node)
            spread += best_marginal_gain
            print(f"  > Seed {len(S)} found: Node {best_node} (Marginal Gain: {best_marginal_gain:.2f}) | Total Spread: {spread:.2f}")
        else:
            print("  > No further node could improve the spread.")
            break

    return S, spread


def simulate_and_tag_infection(G, seeds, p=0.01):
    H = G.copy()

    nx.set_node_attributes(H, values=False, name='infected')

    all_infected = set(seeds)
    newly_activated = seeds[:]

    while newly_activated:
        new_infections = __run_single_simulation(H, newly_activated, p) - all_infected
        newly_activated = list(new_infections)
        all_infected.update(newly_activated)

    nx.set_node_attributes(H, {node: True for node in all_infected}, 'infected')

    return H

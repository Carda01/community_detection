import networkx as nx
import pandas as pd
import altair as alt
import nx_altair as nxa
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
alt.data_transformers.enable("vegafusion")
from itertools import groupby


def generic_show(graph, node_color, node_size, node_tooltip, k_core=3, layout_func=nx.spring_layout, width=400, height=400):
    G = nx.k_core(graph, k=k_core)
    pos = layout_func(G)
    chart = nxa.draw_networkx(
        G=G,
        pos=pos,
        width=0.1,
        alpha=0.8,
        node_color=node_color,
        node_size=node_size,
        node_tooltip=node_tooltip
    )

    chart.properties(width=width, height=height).interactive().show()


def show_mail_graph(G, k_core=3):
    enrich_graph_with_centrality(G, [nx.degree])
    generic_show(G, 'ground_truth', 'degree', ['ground_truth'], k_core=k_core)


def load_email(directed=False):
    edge_list_path = 'data/email-Eu-core.txt'
    nodes_labels_path = 'data/email-Eu-core-department-labels.txt'

    G = nx.read_edgelist(edge_list_path,
                         delimiter=' ',
                         nodetype=int,
                         create_using=nx.DiGraph)

    if not directed:
        G = G.to_undirected()

    df_labels = pd.read_csv(
        nodes_labels_path,
        sep=' ',
        names=['node', 'department_id']
    )

    ground_truth_dict = pd.Series(
        df_labels.department_id.values,
        index=df_labels.node
    ).to_dict()

    nx.set_node_attributes(G, ground_truth_dict, 'ground_truth')
    G.remove_edges_from(nx.selfloop_edges(G))

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G

def run_leiden(nx_graph, resolution=1.0, seed=42):
    """
    Runs Leiden with adjustable resolution.

    Params:
    - resolution:
        1.0 = Standard Modularity (default)
        < 1.0 = Favors LARGER (fewer) communities
        > 1.0 = Favors SMALLER (more) communities
    """
    # convert NetworkX to iGraph
    hg = ig.Graph.from_networkx(nx_graph)

    # Run Leiden using RBConfiguration (supports resolution)
    # We use RBConfigurationVertexPartition because standard Modularity doesn't accept resolution
    partition = leidenalg.find_partition(
        hg,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed
    )

    # map back to NetworkX nodes
    communities = []
    for i in range(len(partition)):
        indices = partition[i]
        if '_nx_name' in hg.vs.attributes():
            original_nodes = set(hg.vs[indices]['_nx_name'])
        else:
            original_nodes = set(indices)
        communities.append(original_nodes)

    return communities


def hist_degrees_cliques(graph):
    degrees = list(nx.get_node_attributes(graph, 'degree').values())
    clique_sizes = [len(c) for c in nx.find_cliques(graph)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    sns.histplot(degrees, kde=True, ax=ax1, color='skyblue', bins='auto')
    ax1.set_title('Degree Distribution', fontsize=16)
    ax1.set_xlabel('Degree', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)

    sns.histplot(clique_sizes, kde=True, ax=ax2, color='salmon', discrete=True, kde_kws={'bw_adjust': 2})
    ax2.set_title('Clique Size (k) Distribution', fontsize=16)
    ax2.set_xlabel('Clique Size (k)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


def spy_plot_adjacency(graph, ordering_key='ground_truth', min_label_spacing=50):
    ordered_nodes = sorted(graph.nodes(), key=lambda n: graph.nodes[n][ordering_key])

    community_labels = [graph.nodes[n][ordering_key] for n in ordered_nodes]
    adj_matrix = nx.to_scipy_sparse_array(graph, nodelist=ordered_nodes)

    community_ticks = []
    unique_communities = []
    community_sizes = []
    start_index = 0
    for comm_id, group in groupby(community_labels):
        unique_communities.append(comm_id)
        group_size = len(list(group))
        community_sizes.append(group_size)
        community_ticks.append(start_index)
        start_index += group_size

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.spy(adj_matrix, markersize=0.1)
    ax.set_title(f"Adjacency Matrix Spy Plot (Ordered by '{ordering_key}')", fontsize=12)

    colors = ['#FFFFFF', '#F0F0F0']
    for i, (pos, size) in enumerate(zip(community_ticks, community_sizes)):
        ax.axhspan(pos, pos + size, facecolor=colors[i % 2], alpha=0.7, zorder=-100)

    display_labels = []
    last_label_pos = -min_label_spacing
    for i, tick_pos in enumerate(community_ticks):
        if tick_pos - last_label_pos >= min_label_spacing:
            display_labels.append(unique_communities[i])
            last_label_pos = tick_pos
        else:
            display_labels.append('')

    ax.set_xticks(community_ticks)
    ax.set_xticklabels(display_labels, rotation=90, fontsize=10)
    ax.set_yticks(community_ticks)
    ax.set_yticklabels(display_labels, fontsize=10)

    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray')
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.set_xlabel("Community ID", labelpad=15)
    ax.set_ylabel("Community ID", labelpad=15)
    plt.show()


def summary_stats(graph):
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    
    degrees = [d for n, d in graph.degree()]
    avg_degree = sum(degrees) / graph.number_of_nodes()
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Density: {nx.density(graph):.4f}")

    avg_clustering = nx.average_clustering(graph)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")

    if nx.is_connected(graph):
        print("Graph is connected.")
        print(f"Radius: {nx.radius(graph)}")
        print(f"Diameter: {nx.diameter(graph)}")
        print(f"Average shortest path length: {nx.average_shortest_path_length(graph):.2f}")
    else:
        num_components = nx.number_connected_components(graph)
        print(f"Graph is not connected. It has {num_components} connected components.")
        
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        print(f"Stats for the largest connected component ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges):")
        print(f"  - Radius: {nx.radius(subgraph)}")
        print(f"  - Diameter: {nx.diameter(subgraph)}")
        print(f"  - Average shortest path length: {nx.average_shortest_path_length(subgraph):.2f}")


def enrich_graph_with_centrality(graph, centrality_funcs):
    if not graph:
        print("Graph is empty, returning.")
        return graph

    first_node = next(iter(graph.nodes()))

    for func in centrality_funcs:
        attribute_name = func.__name__

        if attribute_name in graph.nodes[first_node]:
            print(f"Skipping '{attribute_name}': attribute already exists.")
            continue

        print(f"Calculating and adding node attribute '{attribute_name}'...")
        scores = func(graph)
        nx.set_node_attributes(graph, dict(scores), attribute_name)

    return graph


def calculate_centrality_metrics(graph, centrality_funcs):
    enriched_graph = enrich_graph_with_centrality(graph, centrality_funcs)

    centrality_names = [func.__name__ for func in centrality_funcs]
    columns_to_include = ['ground_truth'] + centrality_names

    data = []
    for node, attrs in enriched_graph.nodes(data=True):
        node_data = {'Node': node}
        for col in columns_to_include:
            node_data[col] = attrs.get(col)
        data.append(node_data)

    df = pd.DataFrame(data)

    final_columns = ['ground_truth'] + centrality_names
    return df[final_columns].rename(columns={'ground_truth': 'Ground Truth'})

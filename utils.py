import networkx as nx
import pandas as pd
import altair as alt
import nx_altair as nxa
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np
alt.data_transformers.enable("vegafusion")

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
    for n in G.nodes():
        G.nodes[n]['degree'] = G.degree[n]

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


def plot_graph_summary(graph, ordering_key='ground_truth'):
    degrees = list(nx.get_node_attributes(graph, 'degree').values())
    clique_sizes = [len(c) for c in nx.find_cliques(graph)]
    ordered_nodes = sorted(graph.nodes(), key=lambda n: graph.nodes[n][ordering_key])
    adj_matrix = nx.to_scipy_sparse_array(graph, nodelist=ordered_nodes)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 8))
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

    ax3.spy(adj_matrix, markersize=0.1)
    ax3.set_title(f"Adjacency Matrix (Ordered by '{ordering_key}')", fontsize=16)
    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()
    plt.show()


def summary_stats(graph):
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    
    degrees = [d for n, d in graph.degree()]
    avg_degree = sum(degrees) / graph.number_of_nodes()
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Density: {nx.density(graph):.4f}")

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
    print(f"Avg clustering: {nx.average_clustering(subgraph):.2f}")

def load_twitch_user_attributes(G):
    df = pd.read_csv("data/musae_PTBR_target.csv")
    attr_dict = {
        row["new_id"]: {
            "id": row["new_id"],
            "days": row["days"],
            "mature": row["mature"],
            "views": row["views"],
            "partner": row["partner"],
        }
        for _, row in df.iterrows()}

    df = df[["days", "mature", "views", "partner", "new_id"]].rename(columns={"new_id": "id"}).copy()
    nx.set_node_attributes(G, attr_dict)
    return G, df

def visualize_static_entire_graph(G, node_size=15):
    pos = nx.spring_layout(
    G,
    k=2,
    iterations=400,   
    seed=42)
    plt.figure(figsize=(14, 14))
    nx.draw(G, pos,
            node_size=node_size,
            edge_color="gray",
            alpha=0.3,
            with_labels=False)
    plt.show()

def twitch_user_exploratory_analysis(df):    
    sns.set(style="whitegrid")
    
    # Distribution of Account Age
    plt.figure(figsize=(8,5))
    sns.histplot(df['days'], bins=30, kde=True)
    plt.title("Distribution of Account Age (days)")
    plt.show()

    # Distribution of Views
    plt.figure(figsize=(8,5))
    sns.histplot(df['views'], bins=40)
    plt.title("Distribution of Channel Views")
    plt.show()

    # Distribution of Views log scale
    plt.figure(figsize=(8,5))
    sns.histplot(df['views'], bins=40, log_scale=True)
    plt.title("Distribution of Channel Views (log scale)")
    plt.show()

    # Mature Content Proportion
    plt.figure(figsize=(6,4))
    sns.countplot(x='mature', data=df)
    plt.title("Mature vs Non-Mature Channels")
    plt.show()

    # Percentage of mature users
    mature_pct = df['mature'].mean() * 100
    print(f"Percentage of mature channels: {mature_pct:.2f}%\n")

    # Partner Status Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='partner', data=df)
    plt.title("Twitch Partner Status Distribution")
    plt.show()

    partner_pct = df['partner'].mean() * 100
    print(f"Percentage of Twitch Partners: {partner_pct:.2f}%\n")

    # Views by Partner Status
    plt.figure(figsize=(7,5))
    sns.boxplot(x='partner', y='views', data=df)
    plt.title("Views by Partner Status")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(6,4))
    corr = df[['days', 'views']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Account Age and Views")
    plt.show()

    # Summary Statistics
    print("Summary statistics:")
    print(df[['days', 'views']].describe(), "\n")

    # Group Comparisons
    print("Average views by partner status:")
    print(df.groupby('partner')['views'].mean(), "\n")

    print("Average account age by mature flag:")
    print(df.groupby('mature')['days'].mean(), "\n")

    # Outliers
    print("Top 10 most viewed accounts:")
    print(df.nlargest(10, 'views')[['id','views','days','partner','mature']], "\n")

    print("Oldest 10 accounts:")
    print(df.nlargest(10, 'days')[['id','views','days','partner','mature']], "\n")


def visualize_centrality(G, measure="degree"):
    # --- Compute centrality ---
    if measure == "degree":
        centrality = nx.degree_centrality(G)
    elif measure == "betweenness":
        centrality = nx.betweenness_centrality(G, normalized=True)
    elif measure == "closeness":
        centrality = nx.closeness_centrality(G)
    elif measure == "eigenvector":
        centrality = nx.eigenvector_centrality(G, max_iter=500)
    elif measure == "pagerank":
        centrality = nx.pagerank(G)
    else:
        raise ValueError("Unknown centrality measure provided.")
    
    print(f"Computed {measure} centrality for {len(G.nodes())} nodes.\n")

    # --- Convert centrality to array ---
    values = np.array(list(centrality.values()))

    # Normalize for coloring and sizing
    norm_values = (values - values.min()) / (values.max() - values.min() + 1e-12)

    # --- Node colors: white → blue palette ---
    # sns.light_palette gives a sequential color map (light to dark)
    cmap = sns.light_palette("blue", as_cmap=True)

    # --- Node sizes: amplify the effect ---
    node_sizes = 200 + (norm_values * 1500)

    # --- Graph layout ---
    pos = nx.spring_layout(G, seed=42)

    # --- Draw graph ---
    plt.figure(figsize=(10, 10))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=norm_values,
        cmap=cmap,
        node_size=node_sizes
    )

    nx.draw_networkx_edges(G, pos, alpha=0.3)

    nodes.set_edgecolor("black")
    plt.title(f"{measure.capitalize()} Centrality Visualization", fontsize=14)
    plt.colorbar(nodes, label=f"{measure.capitalize()} Centrality")
    plt.axis("off")
    plt.show()

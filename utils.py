import networkx as nx
import pandas as pd
import altair as alt
import nx_altair as nxa
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from IPython.display import Markdown, display
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import community as community_louvain
from sklearn.cluster import SpectralClustering
from statistics import median




alt.data_transformers.enable("vegafusion")
from itertools import groupby, zip_longest


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


def visualize_static_entire_graph(G, node_size=15):
    pos = nx.spring_layout(G, k=2, iterations=400, seed=42)
    plt.figure(figsize=(14, 14))
    nx.draw(G, pos,
            node_size=node_size,
            edge_color="gray",
            alpha=0.3,
            with_labels=False)
    plt.show()



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


def visualize_top_n_centrality(df, centrality_measure, top_n=20, centralities_to_show=None):
    if centrality_measure not in df.columns:
        print(f"Error: '{centrality_measure}' not found in DataFrame columns.")
        return

    if centralities_to_show:
        centrality_cols = list(centralities_to_show)
    else:
        centrality_cols = [col for col in df.columns if col != 'Ground Truth']

    rank_df = pd.DataFrame(index=df.index)
    for col in centrality_cols:
        rank_df[col] = df[col].rank(ascending=False, method='min').astype(int)

    top_n_ranked_df = rank_df.sort_values(by=centrality_measure, ascending=True).head(top_n)

    top_n_ground_truth = df.loc[top_n_ranked_df.index]['Ground Truth']

    y_labels = [f"Node {node} (Dept {gt})"
                for i, (node, gt) in enumerate(top_n_ground_truth.items())]

    diff_df = pd.DataFrame(index=top_n_ranked_df.index)
    for col in centrality_cols:
        diff_df[col] = top_n_ranked_df[col] - top_n_ranked_df[centrality_measure]
    
    x_labels = [col.replace('_', '\n') for col in centrality_cols]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data=diff_df[centrality_cols],
        annot=top_n_ranked_df[centrality_cols],
        cmap='coolwarm',
        yticklabels=y_labels,
        fmt='d',
        xticklabels=x_labels,
        linewidths=.5,
        center = 0
    )
    plt.title(
        f'Centrality Measures Rank Comparison for Top {top_n} Nodes, ordered by {centrality_measure.capitalize()}\n'
        f'(Color shows rank difference from {centrality_measure.capitalize()})',
        fontsize=16
    )
    plt.show()


def show_top_nodes_by_centrality(df, centrality_measure, top_n=20):
    if centrality_measure not in df.columns:
        print(f"Error: '{centrality_measure}' not found in DataFrame columns.")
        return

    display_df = df.reset_index().rename(columns={'index': 'Node'})

    top_nodes = display_df.sort_values(by=centrality_measure, ascending=False).head(top_n)

    columns_to_show = ['Node', 'Ground Truth', centrality_measure]
    top_nodes_display = top_nodes[columns_to_show]

    return top_nodes_display.style.set_caption(
        f"Top {top_n} Nodes by {centrality_measure.replace('_', ' ').capitalize()}"
    ).background_gradient(cmap='viridis', subset=[centrality_measure])


# ---------------------------Twitch Specific Functions ---------------------------

def load_twitch_user_attributes(G):
    df = pd.read_csv("data/musae_PTBR_target.csv")
    attr_dict = {
        row["new_id"]: {
            "id": row["new_id"],
            "account_age_days": row["days"],
            "is_mature": row["mature"],
            "total_views": row["views"],
            "is_partner": row["partner"],
        }
        for _, row in df.iterrows()}

    df = df[["days", "mature", "views", "partner", "new_id"]].rename(columns={"new_id": "id",
                                            "days": "account_age_days", "mature": "is_mature", 
                                            "views": "total_views", "partner": "is_partner"}).copy()
    nx.set_node_attributes(G, attr_dict)
    return G, df


# Compare centrality measures with twitch user attributes
def compare_centralities_with_attributes(G):
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, normalized=True)
    clo_cent = nx.closeness_centrality(G)
    eig_cent = nx.eigenvector_centrality(G, max_iter=500)

    rows = []
    for n in G.nodes():
        attrs = G.nodes[n]
        rows.append({
            "node": n,
            "degree": deg_cent[n],
            "betweenness": bet_cent[n],
            "closeness": clo_cent[n],
            "eigenvector": eig_cent[n],
            "total_views": attrs.get("total_views", None),
            "account_age_days": attrs.get("account_age_days", None),
            "is_mature": int(attrs.get("is_mature", False)),
            "is_partner": int(attrs.get("is_partner", False)),
        })

    df = pd.DataFrame(rows).dropna()
    corr = df.drop(columns=["node"]).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="Blues", linewidths=0.5)
    plt.title("Correlation Analysis", fontsize=12)
    plt.show()


def visualize_centrality(G, measure="degree", k_core=3, layout_func=nx.spring_layout):
    H = G.copy()
    if measure == "degree":
        centrality = nx.degree_centrality(H)
        raw_degree = dict(H.degree())
    elif measure == "betweenness":
        centrality = nx.betweenness_centrality(H, normalized=True)
        raw_degree = None
    elif measure == "closeness":
        centrality = nx.closeness_centrality(H)
        raw_degree = None
    elif measure == "eigenvector":
        centrality = nx.eigenvector_centrality(H, max_iter=500)
        raw_degree = None
    elif measure == "pagerank":
        centrality = nx.pagerank(H)
        raw_degree = None
    else:
        raise ValueError("Unknown centrality measure")

    nx.set_node_attributes(H, centrality, name=measure)
    max_val = max(centrality.values())
    scaled_size = {n: (centrality[n] / max_val) * 40 for n in H.nodes()}
    nx.set_node_attributes(H, scaled_size, name="centrality_size")

    print(f"\nTop 10 nodes by {measure} centrality:\n")

    top10 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    rows = []

    for node, val in top10:
        row = {"node": node, measure: round(val, 5)}
        if measure == "degree":
            row["raw_degree"] = raw_degree[node]
        for attr, attr_val in H.nodes[node].items():
            if attr not in ["centrality_size"]:
                row[attr] = attr_val
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    for node in H.nodes():
        if measure == "degree":
            H.nodes[node]["_tooltip"] = (
                f"id={node} | degree_cent={centrality[node]:.5f} | degree={raw_degree[node]}"
            )
        else:
            H.nodes[node]["_tooltip"] = f"id={node} | {measure}={centrality[node]:.5f}"

    generic_show(
        graph=H,
        node_color=measure,
        node_size="centrality_size",
        node_tooltip=["_tooltip"],
        k_core=k_core,
        layout_func=layout_func
    )


def twitch_user_exploratory_analysis(df):    
    sns.set(style="whitegrid")
    
    # Distribution of Account Age
    plt.figure(figsize=(8,5))
    sns.histplot(df['account_age_days'], bins=30, kde=True)
    plt.title("Distribution of Account Age (days)")
    plt.show()
    display(Markdown("The distribution of account age shows a broad spread with two noticeable peaks around ~600 and ~1600 days, " \
    "indicating both a large group of relatively new users and a substantial cluster of " \
    "long-standing accounts, while very old accounts (3000+ days) are rare."))

    # Distribution of Views
    plt.figure(figsize=(8,5))
    sns.histplot(df['total_views'], bins=40)
    plt.title("Distribution of Channel Views")
    plt.show()

    # Distribution of Views log scale
    plt.figure(figsize=(8,5))
    sns.histplot(df['total_views'], bins=40, log_scale=True)
    plt.title("Distribution of Channel Views (log scale)")
    plt.show()
    display(Markdown("The distribution of channel views is heavily right-skewed, with most streamers clustered between 1,000 and" \
    "100,000 views while a small number of outliers reach into the millions, highlighting a strong popularity imbalance" \
    " on the platform."))

    # Mature Content Proportion
    plt.figure(figsize=(6,4))
    sns.countplot(x='is_mature', data=df)
    plt.title("Mature vs Non-Mature Channels")
    plt.show()

    # Percentage of mature users
    mature_pct = df['is_mature'].mean() * 100
    display(Markdown(f"Percentage of mature indicated channels: {mature_pct:.2f}%\n"))

    # Partner Status Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='is_partner', data=df)
    plt.title("Twitch Partner Status Distribution")
    plt.show()

    partner_pct = df['is_partner'].mean() * 100
    display(Markdown(f"Percentage of Twitch Partners: {partner_pct:.2f}%\n"))

    # Views by Partner Status
    plt.figure(figsize=(7,5))
    sns.boxplot(x='is_partner', y='total_views', data=df)
    plt.title("Views by Partner Status")
    plt.show()
    display(Markdown("Partnered streamers have dramatically higher view counts than non-partners," \
    "with most of the extreme outliers concentrated in the partner group, highlighting a" \
    "strong divide in visibility and reach between the groups."))

    # Correlation Heatmap
    plt.figure(figsize=(6,4))
    corr = df[['account_age_days', 'total_views']].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Account Age and Views")
    plt.show()
    display(Markdown("There is only a very weak positive correlation between account age and total views," \
    "suggesting that simply having an older Twitch account does not correspond to how many views a" \
    "channel gets."))

    # Summary Statistics
    print("Summary statistics:")
    print(df[['account_age_days', 'total_views']].describe().applymap(lambda x: f"{x:.0f}"), "\n")

    # Group Comparisons
    print("Average views by partner status:")
    print(df.groupby('is_partner')['total_views'].mean().apply(lambda x: f"{x:.0f}"), "\n")

    print("Average account age by mature flag:")
    print(df.groupby('is_mature')['account_age_days'].mean().apply(lambda x: f"{x:.0f}"), "\n")

    print("Top 10 most viewed accounts:")
    print(df.nlargest(10, 'total_views')[['id','total_views','account_age_days','is_partner','is_mature']]
        .applymap(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else x), "\n")

    print("Oldest 10 accounts:")
    print(df.nlargest(10, 'account_age_days')[['id','total_views','account_age_days','is_partner','is_mature']]
      .applymap(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else x), "\n")
    

# Helper function to create a layout based on community structure
def community_layout(G, partition, scale=3.0, seed=42):
    """
    Creates a layout where communities are positioned in separate clusters.
    
    `partition` is a dict: node -> community_id
    """
    comms = list(set(partition.values()))
    num_comms = len(comms)

    angle_step = 2 * np.pi / num_comms
    community_centers = {
        c: np.array([scale * np.cos(i*angle_step),
                     scale * np.sin(i*angle_step)])
        for i, c in enumerate(comms)
    }

    pos = {}
    for c in comms:
        nodes_in_c = [n for n in G.nodes() if partition[n] == c]
        subG = G.subgraph(nodes_in_c)
        # Base layout for internal structure
        internal_pos = nx.spring_layout(subG, seed=seed)
        
        # Shift community to its center
        center = community_centers[c]
        for n in nodes_in_c:
            pos[n] = internal_pos[n] + center
    return pos


def print_community_statistics(G, communities):
    """
    Print a clean table of per-community statistics.
    Includes: num_nodes, % mature, % partner, median views, avg views,
              median age_days, avg age_days
    """

    rows = []
    for cid, nodes in enumerate(communities):
        node_list = list(nodes)

        # Extract attributes
        mature_flags  = [G.nodes[n].get("is_mature", False) for n in node_list]
        partner_flags = [G.nodes[n].get("is_partner", False) for n in node_list]
        views         = [G.nodes[n].get("total_views", 0) for n in node_list]
        ages          = [G.nodes[n].get("account_age_days", 0) for n in node_list]

        num_nodes = len(node_list)

        pct_mature  = 100 * (sum(mature_flags)  / num_nodes if num_nodes else 0)
        pct_partner = 100 * (sum(partner_flags) / num_nodes if num_nodes else 0)

        med_views = median(views) if views else 0
        avg_views = sum(views) / num_nodes if num_nodes else 0

        med_age   = median(ages) if ages else 0
        avg_age   = sum(ages) / num_nodes if num_nodes else 0

        rows.append([
            cid,
            num_nodes,
            f"{pct_mature:.1f}%",
            f"{pct_partner:.1f}%",
            med_views,
            f"{avg_views:.1f}",
            med_age,
            f"{avg_age:.1f}",
        ])

    headers = [
        "community",
        "num_nodes",
        "% mature",
        "% partner",
        "median_views",
        "avg_views",
        "median_age_days",
        "avg_age_days"
    ]

    print("\n=== COMMUNITY STATISTICS TABLE ===")
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
    except ImportError:
        # Fallback text table
        col_widths = [max(len(str(val)) for val in col) for col in zip(*([headers] + rows))]
        fmt = " | ".join("{:<" + str(w) + "}" for w in col_widths)
        print(fmt.format(*headers))
        print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
        for r in rows:
            print(fmt.format(*r))


def draw_static_community_plot(G, pos):
    plt.figure(figsize=(10, 8))

    # Draw nodes by community
    communities = {}
    for n, d in G.nodes(data=True):
        cid = d["community"]
        communities.setdefault(cid, []).append(n)

    for cid, nodes in communities.items():
        color = G.nodes[nodes[0]]["color"]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=color,
            label=f"Community {cid}",
            node_size=40,
            edgecolors="black",
            linewidths=0.5
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.3)
    plt.axis("off")
    plt.legend(
        title="Communities",
        loc="lower right",
        bbox_to_anchor=(1, 0),
        frameon=True,
        fontsize=8,         # smaller text
        title_fontsize=9,   # slightly larger title
        markerscale=0.6,    # shrink node markers in legend
        handlelength=1.0,   # shrink spacing
    )
    plt.show()



def visualize_communities(
    G,
    method="leiden",
    resolution=1.0,
    k=4,
    seed=42,
    static=True):

    if method.lower() == "leiden":
        communities = run_leiden(G, resolution=resolution, seed=seed)
        print(f"Detected {len(communities)} communities using Leiden with resolution {resolution}.")

    elif method.lower() == "louvain":
        partition = community_louvain.best_partition(G, resolution=resolution, random_state=seed)
        print(f"Detected {len(set(partition.values()))} communities using Louvain with resolution {resolution}.")
        communities = []
        for comm_id in set(partition.values()):
            communities.append({n for n, c in partition.items() if c == comm_id})

    elif method.lower() == "spectral":
        A = nx.to_numpy_array(G)
        sc = SpectralClustering(n_clusters=k, assign_labels="kmeans", random_state=seed)
        labels = sc.fit_predict(A)
        nodes = list(G.nodes())
        communities = []
        for cid in range(k):
            communities.append({nodes[i] for i, lab in enumerate(labels) if lab == cid})

    else:
        raise ValueError("method must be: 'leiden', 'louvain', or 'spectral'")


    # Community lookup build
    node_to_comm = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            node_to_comm[n] = cid

    num_comms = len(communities)
    cmap = cm.get_cmap("tab20", num_comms)

    def get_color(cid):
        return colors.to_hex(cmap(cid))

    # Assign node attributes (color, label, tooltip)
    for n in G.nodes():
        cid = node_to_comm[n]
        color_hex = get_color(cid)

        G.nodes[n]["community"]   = str(cid)
        G.nodes[n]["color"]       = color_hex
        G.nodes[n]["color_label"] = f"Community {cid}"
        G.nodes[n]["tooltip"]     = f"Node {n}, Community {cid}"

    print_community_statistics(G, communities)

    pos = community_layout(G, node_to_comm)
    if static:
        draw_static_community_plot(G, pos)
    else:
        generic_show(
            graph=G,
            node_color="color_label",
            node_size=25,
            node_tooltip="tooltip",
            k_core=1,
            layout_func=lambda graph: pos
        )


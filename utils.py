import networkx as nx
import pandas as pd
import nx_altair as nxa
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

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


def spy_plot_adjacency(graph, ordering_key='ground_truth'):
    ordered_nodes = sorted(graph.nodes(), key=lambda n: graph.nodes[n][ordering_key])

    adj_matrix = nx.to_scipy_sparse_array(graph, nodelist=ordered_nodes)

    plt.figure(figsize=(8, 8))
    plt.spy(adj_matrix, markersize=0.1)
    plt.title(f"Adjacency Matrix Spy Plot (Ordered by '{ordering_key}')", fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.show()

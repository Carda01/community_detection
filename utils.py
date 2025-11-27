import networkx as nx
import pandas as pd
import nx_altair as nxa

def show_mail_graph(G, k_core=3):
    G = nx.k_core(G, k=k_core)
    pos = nx.spring_layout(G)
    chart = nxa.draw_networkx(
        G=G,
        pos=pos,
        width=0.1,
        alpha=0.8,
        node_color='ground_truth',
        node_size='degree',
        node_tooltip=['ground_truth']
    )

    chart.properties(width=400, height=400).interactive().show()



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
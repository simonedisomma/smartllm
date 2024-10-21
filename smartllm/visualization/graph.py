import networkx as nx
import matplotlib.pyplot as plt

def generate_flowchart(function_calls: dict, output_file: str = 'function_flowchart.png'):
    G = nx.DiGraph()
    
    for caller, called_functions in function_calls.items():
        G.add_node(caller)
        for func in called_functions:
            G.add_node(func)
            G.add_edge(caller, func)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold')
    
    edge_labels = {(u, v): '' for (u, v) in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Function Call Flowchart")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

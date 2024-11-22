import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz as pgv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm



# Helper function to print the path
def print_path(node):
    path = node.get_path()
    print("Path from root to current node:")
    for step, n in enumerate(path):
        if n.fragment is not None:
            action = f"Add fragment '{n.fragment}'"
        else:
            action = "Root node"
        print(f"Step {step}: {action} -> SMILES: {n.smiles} | Score: {n.score:.4f}")

def build_tree_graph():
    G = nx.DiGraph()  # Create a directed graph

    # Add all nodes
    for node in Node.all_nodes:
        G.add_node(node.id, smiles=node.smiles, score=node.score)

    # Add all edges
    for node in Node.all_nodes:
        if node.parent is not None:
            G.add_edge(node.parent.id, node.id)

    return G


def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    """
    If there is a cycle, then this will produce a partial layout.
    G: the graph (must be a tree)
    root: the root node of current branch
    width: horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    pos: a dict saying where all nodes go if they have been assigned
    parent: parent of this branch.
    """

    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if not isinstance(G, nx.DiGraph):
        children = list(G.neighbors(root))
        if parent is not None:
            children.remove(parent)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos

def visualize_tree(G, root_ids):
    plt.figure(figsize=(16, 10))  # Increase figure size for better viewing

    # For multiple roots, create positions for each
    pos = {}
    for idx, root_id in enumerate(root_ids):
        sub_pos = hierarchy_pos(G, root_id, xcenter=(idx + 1) / (len(root_ids) + 1), width=2.0, vert_gap=0.5)  # Increase width and vertical gap
        pos.update(sub_pos)

    # Prepare labels
    node_labels = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        if 'smiles' in node_data and 'score' in node_data:
            node_labels[node] = f"{node_data['smiles']}\nScore: {node_data['score']:.2f}"
        else:
            print(f"Node {node} is missing attributes: {node_data}")
            node_labels[node] = f"Node {node}"

    # Get scores for coloring
    scores = [node_data['score'] if 'score' in node_data else 0.0 for node_data in G.nodes.values()]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0

    # Normalize scores between 0 and 1
    norm = plt.Normalize(vmin=min_score, vmax=max_score)
    norm_scores = [norm(s) for s in scores]

    # Choose a color map
    cmap = plt.cm.viridis

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # For matplotlib versions < 3.1

    # Get the current Axes object
    ax = plt.gca()
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
    # Draw nodes with color mapping, specifying the ax
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=norm_scores, cmap=cmap, ax=ax)  # Increase node size

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=15, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax)  # Increase font size for better readability

    plt.title("MCTS Tree Visualization")
    plt.axis('off')

    # Add colorbar to the figure, specifying the ax
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Score')

    plt.show()
    plt.savefig("mcts_tree.png")



def build_tree_graph(root_nodes,Node):
    G = nx.DiGraph()
    visited = set()
    all_scores = []

    # Collect all scores for normalization
    for node in Node.all_nodes:
        all_scores.append(node.score)

    min_score = min(all_scores)
    max_score = max(all_scores)

    # Normalize function to map scores between 0 and 1
    def normalize_score(score):
        if max_score - min_score == 0:
            return 0.5  # Avoid division by zero; assign middle value
        return (score - min_score) / (max_score - min_score)

    for root in root_nodes:
        build_subgraph(G, root, visited, normalize_score)

    return G

def build_subgraph(G, node, visited, normalize_score):
    if node.id in visited:
        return
    visited.add(node.id)
    node_label = get_node_label(node)

    # Normalize the node's score and map it to a color
    normalized_score = normalize_score(node.score)
    color = score_to_color(normalized_score)

    # Add the node with the 'fillcolor' attribute set
    G.add_node(node.id, label=node_label, fillcolor=color, style='filled', fontcolor='black')

    for child in node.children:
        build_subgraph(G, child, visited, normalize_score)
        G.add_edge(node.id, child.id)

def get_node_label(node):
    return f"{node.smiles}\nScore: {node.score:.2f}"

def score_to_color(normalized_score):
    # Use a color map (e.g., 'viridis' or 'coolwarm')
    cmap = cm.get_cmap('viridis')
    rgba = cmap(normalized_score)
    # Convert RGBA to hexadecimal color code
    return mcolors.rgb2hex(rgba)

def visualize_tree(G,filename):
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(
        rankdir='TB',     # Top-Bottom direction
        size='10,10!',    # Width,Height in inches
        dpi='300',        # Dots per inch
        margin='0.0',     # Remove margins
        nodesep='0.5',    # Increase spacing between nodes
        ranksep='0.5',    # Increase spacing between ranks
        fontsize='12'     # Default font size
    )
    A.node_attr.update(
        shape='box',
        style='filled',
        fillcolor='lightblue',  # Default color (will be overridden by individual nodes)
        fontsize='10',
        fontcolor='black'
    )
    A.edge_attr.update(
        arrowsize='0.7',
        fontsize='8'
    )

    # Since we've already assigned 'fillcolor' to nodes individually, no need to adjust here

    # Draw the graph
    A.layout('dot')
    A.draw('mcts_tree.png')           # Save as PNG
    A.draw(f'{filename}.pdf')           # Save as PDF
    A.draw('mcts_tree.svg')           # Save as SVG

    # Display the image if possible
    try:
        from IPython.display import Image, display
        display(Image(filename='mcts_tree.png'))
    except ImportError:
        print("Graph images saved as 'mcts_tree.png', 'mcts_tree.pdf', and 'mcts_tree.svg'.")

import matplotlib.pyplot as plt

def get_tree_positions(tree, x=0.5, y=1.0, dx=0.5, dy=0.05, positions=None, labels=None, feature_names=None):
    if positions is None:
        positions, labels = {}, {}
    node_id = id(tree)
    if tree.get('leaf', False):
        text = f"Leaf\npred={tree['pred']}"
    else:
        feat_name = feature_names[tree['feat']]
        text = f"{feat_name} ≤ {tree['thresh']:.2f}"
    positions[node_id] = (x, y)
    labels[node_id]    = text
    if not tree.get('leaf', False):
        get_tree_positions(tree['left'],  x - dx, y - dy, dx/2, dy, positions, labels, feature_names)
        get_tree_positions(tree['right'], x + dx, y - dy, dx/2, dy, positions, labels, feature_names)
    return positions, labels

def find_node_by_id(tree, target_id):
    if id(tree) == target_id:
        return tree
    if not tree.get('leaf', False):
        left = find_node_by_id(tree['left'], target_id)
        if left is not None:
            return left
        return find_node_by_id(tree['right'], target_id)
    return None

def draw_manual_tree(tree, feature_names, dx=0.5, dy=0.05, figsize=(20,20), fontsize=8):
    positions, labels = get_tree_positions(tree, dx=dx, dy=dy, feature_names=feature_names)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    # draw nodes
    for node_id, (x, y) in positions.items():
        ax.text(x, y, labels[node_id],
                ha='center', va='center',
                fontsize=fontsize,
                bbox=dict(boxstyle='round,pad=0.2', fc='lightblue', ec='black', lw=0.5))
    # draw arrows
    for node_id, (x0, y0) in positions.items():
        node = find_node_by_id(tree, node_id)
        if node and not node.get('leaf', False):
            for side in ('left', 'right'):
                child = node[side]
                cx, cy = positions[id(child)]
                ax.annotate("", xy=(cx, cy + dy*0.3), xytext=(x0, y0 - dy*0.3),
                            arrowprops=dict(arrowstyle='->', lw=0.5))
    plt.tight_layout()
    plt.show()

# --- Usage example (assumes `tree` and `features` are already defined) ---
# feature_names = features  # e.g. ['Sex','SibSp','Parch','Embarked','TicketNumeric']
# tree = build_tree(X, y, max_depth=15)
draw_manual_tree(tree, feature_names)

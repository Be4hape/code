import graphviz

def build_graphviz(tree, feature_names, dot=None, parent_id=None, edge_label=""):
    """
    tree dict를 순회하며 graphviz.Digraph 객체에 노드와 엣지를 추가합니다.
    """
    if dot is None:
        dot = graphviz.Digraph(format='png')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontsize='6')

    # 고유한 노드 ID
    node_id = str(id(tree))
    # 노드 라벨
    if tree.get('leaf', False):
        label = f"Leaf\npred={tree['pred']}"
    else:
        feat = feature_names[tree['feat']]
        thresh = tree['thresh']
        label = f"{feat} ≤ {thresh:.2f}"

    dot.node(node_id, label)

    # 부모가 있으면 엣지 추가
    if parent_id is not None:
        dot.edge(parent_id, node_id, label=edge_label, fontsize='5')

    # 내부 노드면 자식 재귀 호출
    if not tree.get('leaf', False):
        build_graphviz(tree['left'],  feature_names, dot, node_id, edge_label="True")
        build_graphviz(tree['right'], feature_names, dot, node_id, edge_label="False")

    return dot

# 사용 예
feature_names = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']
dot = build_graphviz(tree, feature_names)
dot.render('decision_tree_depth15', view=True)  # PNG로 저장 후 자동으로 열기

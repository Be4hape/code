import matplotlib.pyplot as plt

def get_tree_positions(tree, x=0.5, y=1.0, dx=0.25, dy=0.15, positions=None, labels=None, feature_names=None):
    """
    트리를 재귀 순회하며 각 노드의 (x,y) 위치와 텍스트 라벨을 계산합니다.
    """
    if positions is None:
        positions = {}
        labels    = {}
    # 현재 노드 고유 ID (메모리 주소 사용)
    node_id = id(tree)
    # 라벨 만들기
    if tree['leaf']:
        text = f"Leaf\npred={tree['pred']}"
    else:
        feat = feature_names[tree['feat']]
        thresh = tree['thresh']
        text = f"{feat} ≤ {thresh:.2f}"
    positions[node_id] = (x, y)
    labels[node_id]    = text

    # 내부 노드면 자식 위치 계산
    if not tree['leaf']:
        # 왼쪽 자식
        get_tree_positions(tree['left'],
                           x - dx, y - dy,
                           dx/2, dy,
                           positions, labels,
                           feature_names)
        # 오른쪽 자식
        get_tree_positions(tree['right'],
                           x + dx, y - dy,
                           dx/2, dy,
                           positions, labels,
                           feature_names)
    return positions, labels

def draw_manual_tree(tree, feature_names):
    """
    build_tree로 만든 tree dict를 입력받아
    matplotlib으로 노드와 화살표를 그립니다.
    """
    positions, labels = get_tree_positions(tree, feature_names=feature_names)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # 노드 텍스트
    for node_id, (x, y) in positions.items():
        ax.text(x, y, labels[node_id],
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', ec='black'))

    # 화살표 그리기
    for node_id, (x, y) in positions.items():
        # 내부 노드만
        # original tree reference
        # we need to find the tree dict from id, so we traverse again:
        def find_node_by_id(t):
            if id(t) == node_id:
                return t
            if not t['leaf']:
                left = find_node_by_id(t['left'])
                if left is not None: return left
                return find_node_by_id(t['right'])
            return None
        node = find_node_by_id(tree)
        if not node['leaf']:
            left_id  = id(node['left'])
            right_id = id(node['right'])
            x0, y0 = positions[node_id]
            x1, y1 = positions[left_id]
            x2, y2 = positions[right_id]
            ax.annotate("", xy=(x1, y1+0.02), xytext=(x0, y0-0.02),
                        arrowprops=dict(arrowstyle='->'))
            ax.annotate("", xy=(x2, y2+0.02), xytext=(x0, y0-0.02),
                        arrowprops=dict(arrowstyle='->'))

    plt.show()

# --- 사용 예 ---
# 1) 수작업 트리 생성
# from your_module import build_tree, predict
# tree = build_tree(X, y, max_depth=3)

# 2) 피처 명 리스트
feature_names = ['Sex', 'SibSp', 'Parch', 'Embarked', 'TicketNumeric']

# 3) 시각화
draw_manual_tree(tree, feature_names)

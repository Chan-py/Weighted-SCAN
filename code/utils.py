def load_ground_truth(labels_path):
    """label.dat 포맷: node true_label (공백 구분)"""
    gt = {}
    with open(labels_path, 'r') as f:
        for line in f:
            node, label = line.strip().split()
            gt[int(node)] = label  # label이 int면 int(label)
    return gt
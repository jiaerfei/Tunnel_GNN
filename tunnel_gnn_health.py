import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

# 固定随机种子，结果可复现
torch.manual_seed(42)
np.random.seed(42)


# ==============================================
# 1. 数据加载与异质图构建
# ==============================================
def load_hetero_data(excel_path="tunnel_health_gnn_binary_structure.xlsx"):
    # 读取三个Sheet
    defect_df = pd.read_excel(excel_path, sheet_name="Defect_Node")
    structure_df = pd.read_excel(excel_path, sheet_name="Structure_Node")
    edge_df = pd.read_excel(excel_path, sheet_name="Edge_Relation")

    # ---------- 1.1 Defect节点特征构造（11维）----------
    type_onehot = pd.get_dummies(defect_df["defect_type"]).reindex(columns=["crack", "leakage", "spall"],
                                                                   fill_value=0).values
    pos_onehot = pd.get_dummies(defect_df["circum_pos"]).reindex(
        columns=["arch_top", "left_wall", "right_wall", "invert"], fill_value=0).values
    # 量化值归一化
    w_max, w_min = 2.0, 0.1
    defect_df["width_norm"] = (defect_df["width_mm"].fillna(0) - w_min) / (w_max - w_min)
    a_max, a_min = 20000, 400
    defect_df["area_norm"] = (defect_df["area_mm2"].fillna(0) - a_min) / (a_max - a_min)
    main_quant = defect_df.apply(lambda x: x["width_norm"] if x["defect_type"] == "crack" else x["area_norm"],
                                 axis=1).values.reshape(-1, 1)
    l_max, l_min = 400, 20
    defect_df["length_norm"] = (defect_df["length_mm"].fillna(0) - l_min) / (l_max - l_min)
    sub_quant = defect_df.apply(lambda x: x["length_norm"] if x["defect_type"] == "crack" else 0.5,
                                axis=1).values.reshape(-1, 1)
    # 规范等级归一化
    level_norm = ((defect_df["norm_level"] - 1) / 4).values.reshape(-1, 1)
    defect_x = np.concatenate([type_onehot, main_quant, sub_quant, pos_onehot, level_norm], axis=1).astype(np.float32)
    defect_y = (defect_df["norm_level"] - 1).values.astype(np.int64)  # 标签转0~4

    # ---------- 1.2 Structure节点特征构造（13维）----------
    struct_cols = ["granularity", "length_km", "burial_depth", "geology_level", "service_years",
                   "defect_count", "defect_density", "crack_ratio", "leakage_ratio", "spall_ratio",
                   "max_level", "avg_level", "high_risk_ratio"]
    struct_features = []
    for col in struct_cols:
        col_min, col_max = structure_df[col].min(), structure_df[col].max()
        if col_max == col_min:
            norm_val = np.zeros_like(structure_df[col].values)
        else:
            norm_val = (structure_df[col].values - col_min) / (col_max - col_min)
        struct_features.append(norm_val.reshape(-1, 1))
    struct_x = np.concatenate(struct_features, axis=1).astype(np.float32)
    # Structure节点标签
    struct_y = (structure_df["expert_level"] - 1).values.astype(np.int64)

    # ---------- 1.3 节点ID映射（字符串ID -> 连续整数索引）----------
    defect_id2idx = {did: i for i, did in enumerate(defect_df["defect_id"])}
    struct_id2idx = {sid: i for i, sid in enumerate(structure_df["node_id"])}

    # ---------- 1.4 构造异质图 ----------
    data = HeteroData()
    data["defect"].x = torch.tensor(defect_x)
    data["defect"].y = torch.tensor(defect_y)
    data["structure"].x = torch.tensor(struct_x)
    data["structure"].y = torch.tensor(struct_y)

    # 筛选三类边
    # 边1：defect -> structure
    d2s = edge_df[(edge_df["src_type"] == "defect") & (edge_df["tgt_type"] == "structure")]
    src_idx = [defect_id2idx[sid] for sid in d2s["src_id"]]
    tgt_idx = [struct_id2idx[tid] for tid in d2s["tgt_id"]]
    data["defect", "belongs_to", "structure"].edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)

    # 边2：structure -> defect
    s2d = edge_df[(edge_df["src_type"] == "structure") & (edge_df["tgt_type"] == "defect")]
    src_idx = [struct_id2idx[sid] for sid in s2d["src_id"]]
    tgt_idx = [defect_id2idx[tid] for tid in s2d["tgt_id"]]
    data["structure", "has", "defect"].edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)

    # 边3：structure -> structure（纵向隶属）
    s2s = edge_df[(edge_df["src_type"] == "structure") & (edge_df["tgt_type"] == "structure")]
    src_idx = [struct_id2idx[sid] for sid in s2s["src_id"]]
    tgt_idx = [struct_id2idx[tid] for tid in s2s["tgt_id"]]
    data["structure", "connects_to", "structure"].edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)

    # 划分训练/测试集
    # 病害节点划分
    n_defect = len(defect_df)
    perm = torch.randperm(n_defect)
    test_size = int(0.2 * n_defect)
    test_defect_idx = perm[:test_size]
    train_defect_mask = torch.zeros(n_defect, dtype=torch.bool)
    train_defect_mask[perm[test_size:]] = True

    # Structure节点划分，只用区段的节点，它们和病害节点直接相连，更有意义，半监督学习
    struct_train_mask = torch.zeros(data['structure'].num_nodes, dtype=torch.bool)
    struct_test_idx = []
    for i, row in structure_df.iterrows():
        gran = row['granularity']
        if gran == 0.2:
            # 更改数字即可更改比例
            if np.random.rand() < 0.6:
                struct_train_mask[i] = True
            else:
                struct_test_idx.append(i)
        else:
            struct_test_idx.append(i)

    return data, train_defect_mask, test_defect_idx, struct_train_mask, struct_test_idx, structure_df


# ==============================================
# 2. 异质GNN模型定义
# ==============================================
class BinaryStructureGNN(torch.nn.Module):
    def __init__(self, hidden_dim=32, num_classes=4):
        super().__init__()
        # 两层异质卷积
        self.conv1 = HeteroConv({
            ("defect", "belongs_to", "structure"): SAGEConv(-1, hidden_dim),
            ("structure", "has", "defect"): SAGEConv(-1, hidden_dim),
            ("structure", "connects_to", "structure"): SAGEConv(-1, hidden_dim)
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("defect", "belongs_to", "structure"): SAGEConv(hidden_dim, hidden_dim),
            ("structure", "has", "defect"): SAGEConv(hidden_dim, hidden_dim),
            ("structure", "connects_to", "structure"): SAGEConv(hidden_dim, hidden_dim)
        }, aggr="mean")

        # 两个独立分类头：病害一个，结构一个
        self.head_defect = Linear(hidden_dim, num_classes)
        self.head_structure = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        # x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        out = {
            "defect": self.head_defect(x_dict["defect"]),
            "structure": self.head_structure(x_dict["structure"])
        }
        return out


# ==============================================
# 3. 训练与测试流程
# ==============================================
def main():
    # 加载数据
    data, train_defect_mask, test_defect_idx, struct_train_mask, struct_test_idx, structure_df = load_hetero_data()
    print("异质图构建完成：")
    print(f"Defect节点：{data['defect'].num_nodes}个，Structure节点：{data['structure'].num_nodes}个")

    # 初始化模型
    model = BinaryStructureGNN(hidden_dim=32, num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_acc = 0
    best_state = None
    epochs = 80

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        # 多任务联合损失
        loss_defect = F.cross_entropy(out["defect"][train_defect_mask], data["defect"].y[train_defect_mask])
        loss_struct = F.cross_entropy(out["structure"][struct_train_mask], data["structure"].y[struct_train_mask])
        total_loss = loss_defect + 0.5 * loss_struct

        total_loss.backward()
        optimizer.step()

        # 测试评估
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred_defect = out["defect"][test_defect_idx].argmax(dim=1).numpy()
            true_defect = data["defect"].y[test_defect_idx].numpy()
            test_acc = accuracy_score(true_defect, pred_defect)

            # 结构节点小样本指标
            pred_struct = out["structure"].argmax(dim=1).numpy()
            true_struct = data["structure"].y.numpy()
            pred_std = np.std(pred_struct)  # 预测标准差，检测模型塌缩
            struct_mae = np.mean(np.abs(pred_struct - true_struct))  # 等级MAE
            corr = spearmanr(pred_struct, true_struct)  # 等级相关性

            if test_acc > best_acc:
                best_acc = test_acc
                best_state = model.state_dict()
                torch.save(best_state, "best_binary_gnn_state.pt")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | 训练损失：{total_loss.item():.4f} | "
                  f"测试缺陷准确率：{test_acc:.4f} | 结构预测标准差：{pred_std:.3f} | "
                  f"结构MAE：{struct_mae:.3f} | 结构相关性：{corr.statistic:.3f}")

    # 加载最优模型输出结果
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred_struct = out["structure"].argmax(dim=1).numpy() + 1
        true_struct = data["structure"].y.numpy() + 1

    print("\n===== Structure节点预测结果 =====")
    for i, row in structure_df.iterrows():
        print(f"节点{row['node_id']}（粒度：{row['granularity']}）：真实等级{true_struct[i]}，预测等级{pred_struct[i]}")
    print(f"\n测试集最佳缺陷分类准确率：{best_acc:.4f}")


if __name__ == '__main__':
    main()

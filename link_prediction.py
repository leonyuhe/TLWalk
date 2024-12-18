import os
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from karateclub import (
    DeepWalk, Node2Vec, Role2Vec, Comm2Vec,
    Walklets, GraphWave, FeatherNode,
    FirstOrderLINE, TENE, GLEE,
    Diff2Vec, NetMF, RoleComm2Vec,GraphRole2Vec,CommunityAwareDeepWalk
    ,TwoLayerCommunityWalk,MNMF,CommunityRoleWalk,EPSBE,CommunityEPSBE,EnhancedEPSBE
)
import random

random.seed(10)
np.random.seed(10)

# Step 1: 加载图
def load_graph(graph_file):
    """
    读取图文件。
    """
    return nx.read_edgelist(graph_file, delimiter=",", nodetype=int, create_using=nx.Graph())


# Step 2: 加载正负样本
def load_samples(lp_splits_folder, dataset_name):
    """
    加载正样本和负样本的训练集与测试集。
    """
    # 文件路径
    train_positive_edges_file = os.path.join(lp_splits_folder, dataset_name, "trE_0.csv")
    test_positive_edges_file = os.path.join(lp_splits_folder, dataset_name, "teE_0.csv")
    train_negative_edges_file = os.path.join(lp_splits_folder, dataset_name, "negTrE_0.csv")
    test_negative_edges_file = os.path.join(lp_splits_folder, dataset_name, "negTeE_0.csv")

    # 加载正负样本
    train_positive_edges = np.loadtxt(train_positive_edges_file, delimiter=",", dtype=int)
    test_positive_edges = np.loadtxt(test_positive_edges_file, delimiter=",", dtype=int)
    train_negative_edges = np.loadtxt(train_negative_edges_file, delimiter=",", dtype=int)
    test_negative_edges = np.loadtxt(test_negative_edges_file, delimiter=",", dtype=int)

    # 合并正负样本及其标签
    train_edges = np.vstack([train_positive_edges, train_negative_edges])
    test_edges = np.vstack([test_positive_edges, test_negative_edges])
    train_labels = np.concatenate([
        np.ones(len(train_positive_edges)),
        np.zeros(len(train_negative_edges))
    ])
    test_labels = np.concatenate([
        np.ones(len(test_positive_edges)),
        np.zeros(len(test_negative_edges))
    ])

    return train_edges, test_edges, train_labels, test_labels


# Step 3: 节点到边的嵌入转换
def edge_embedding(node_embeddings, edge_list, method="average"):
    """
    将节点嵌入转换为边嵌入。
    """
    if method == "average":
        return (node_embeddings[edge_list[:, 0]] + node_embeddings[edge_list[:, 1]]) / 2
    elif method == "hadamard":
        return node_embeddings[edge_list[:, 0]] * node_embeddings[edge_list[:, 1]]
    else:
        raise ValueError("Unsupported edge embedding method. Choose 'average' or 'hadamard'.")


# Step 4: 嵌入方法列表
def get_embedding_methods():
    """
    定义所有需要测试的嵌入方法。
    """
    return {
        #"DeepWalk": DeepWalk(),
        #"EPSBE": EPSBE(2,1,3),
        #"EnhancedEPSBE": EnhancedEPSBE(2,1,3),
        #"CommunityEPSBE": CommunityEPSBE(eps_0=2, delta=1.0, max_eps=3.0),
        #"CommunityAwareDeepWalk": CommunityAwareDeepWalk(),
        #"TwoLayerCommunityWalk": TwoLayerCommunityWalk(),
        "CommunityRoleWalk": CommunityRoleWalk(),
        #"MNMF": MNMF(),
        # "Role2Vec": Role2Vec(),
        # "Comm2Vec": Comm2Vec(),
        # "Walklets": Walklets(),
        # "RoleComm2Vec": RoleComm2Vec(),
        # "GraphRole2vec": GraphRole2Vec(),
        # "GraphWave": GraphWave(),
        #"FeatherNode": FeatherNode(),
        #"FirstOrderLINE": FirstOrderLINE(),
        #"TENE": TENE(),
        #"GLEE": GLEE(),
        #"Diff2Vec": Diff2Vec(),
        # "NetMF": NetMF()
    }


# Step 5: 主函数
def main(graph_folder, lp_splits_folder, datasets):
    """
    主函数：遍历数据集和嵌入方法，计算 AUC 分数。
    """
    results = {}
    embedding_methods = get_embedding_methods()

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # 加载图
        graph_file = os.path.join(graph_folder, f"{dataset}.train.edgelist")
        G = load_graph(graph_file)

        # 加载正负样本
        train_edges, test_edges, train_labels, test_labels = load_samples(lp_splits_folder, dataset)

        for method_name, model in embedding_methods.items():
            print(f"Testing {method_name} on {dataset}...")

            # 嵌入训练
            model.fit(G)
            node_embeddings = model.get_embedding()

            # 边嵌入
            train_edge_embeddings = edge_embedding(node_embeddings, train_edges, method="hadamard")
            test_edge_embeddings = edge_embedding(node_embeddings, test_edges, method="hadamard")

            # 下游模型训练
            downstream_model = LogisticRegression(max_iter=2000, random_state=42, penalty="l2")
            downstream_model.fit(train_edge_embeddings, train_labels)

            # 验证
            test_predictions = downstream_model.predict_proba(test_edge_embeddings)[:, 1]
            auc = roc_auc_score(test_labels, test_predictions)

            # 存储结果
            results[(dataset, method_name)] = auc
            print(f"{method_name} AUC: {auc:.4f}")

    # 总结结果
    print("\nSummary of AUC scores:")
    for (dataset, method_name), auc in results.items():
        print(f"Dataset: {dataset}, Method: {method_name}, AUC: {auc:.4f}")


# 运行
if __name__ == "__main__":
    graph_folder = "./output/"
    lp_splits_folder = "./output/lp_train_test_splits/"
    datasets = ["aves-weaver-social",
                "bio-CE-LC",
                "bio-DM-LC",
                "bio-CE-HT",
                "bio-celegans-dir",
                "bio-WormNet-v3",
                "bn-cat-mixed-species_brain_1",
                "soc-wiki-Vote",
                "fb-pages-food",
                "soc-hamsterster" ,
                "ego-Facebook"
                ]  # 添加你的数据集名称
    main(graph_folder, lp_splits_folder, datasets)
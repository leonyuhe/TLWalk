import networkx as nx
import numpy as np
import random
from gensim.models.word2vec import Word2Vec
from community import community_louvain
from networkx.algorithms.community import modularity

class TwoLayerCommunityWalk:
    def __init__(self, walk_length=80, walk_number=15, dimensions=128, window_size=5, epochs=2, workers=4, seed=42, beta = 0.8):
        self.walk_length = walk_length
        self.walk_number = walk_number
        self.dimensions = dimensions
        self.window_size = window_size
        self.epochs = epochs
        self.workers = workers
        self.beta = beta
        self.seed = seed
        self.communities = {}
        self.graph = None
        self.community_graph = None
        self.intra_community_graphs = {}
        self.embedding = None
        self.community_jump_prob = None

    def fit(self, graph):
        self.graph = nx.relabel_nodes(graph, lambda x: str(x))  # 确保节点是字符串
        self.analyze_community_structure(self.graph)
        self._detect_communities()

        # 构建社区间和社区内网络
        self._build_community_graphs()

        # 计算社区跳跃概率
        #self.community_jump_prob = max(0.8,len(self.community_graph.nodes()) / len(self.graph.nodes()))
        self.community_jump_prob = self.beta
        print(f"Community jump probability: {self.community_jump_prob}")

        # 生成随机游走序列
        self.walks = self.generate_walks()
        print(f"Generated {len(self.walks)} walks.")

        # 训练 Word2Vec 模型
        model = Word2Vec(
            sentences=self.walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=1,
            sg=1,
            workers=self.workers,
            epochs=self.epochs
        )
        num_of_nodes = graph.number_of_nodes()
        self._embedding = [model.wv[str(n)] for n in range(num_of_nodes)]

    def analyze_community_structure(self, graph):
        """
        Analyze the community structure of the graph and print metrics.

        Parameters:
        graph (networkx.Graph): Input graph.

        Returns:
        None
        """
        # 检测社区划分（使用 Louvain 算法）
        partition = community_louvain.best_partition(graph)

        # 将 partition 转换为社区列表
        communities = []
        for community_id in set(partition.values()):
            community_nodes = [node for node in partition.keys() if partition[node] == community_id]
            communities.append(community_nodes)

        # 计算社区间节点
        inter_community_nodes = set()
        for u, v in graph.edges():
            if partition[u] != partition[v]:
                inter_community_nodes.add(u)
                inter_community_nodes.add(v)

        # 统计数据
        total_nodes = graph.number_of_nodes()
        inter_community_node_ratio = len(inter_community_nodes) / total_nodes

        # 计算社区间边数
        inter_community_edges = sum(1 for u, v in graph.edges() if partition[u] != partition[v])
        total_edges = graph.number_of_edges()
        inter_community_edge_ratio = inter_community_edges / total_edges

        # 计算模块度
        modularity_score = modularity(graph, communities)

        # 打印结果
        print(f"Total nodes: {total_nodes}")
        print(f"Inter-community nodes: {len(inter_community_nodes)}")
        print(f"Inter-community node ratio: {inter_community_node_ratio:.4f}")
        print(f"Total edges: {total_edges}")
        print(f"Inter-community edges: {inter_community_edges}")
        print(f"Inter-community edge ratio: {inter_community_edge_ratio:.4f}")
        print(f"Number of communities: {len(communities)}")
        print(f"Modularity: {modularity_score:.4f}")
    def _detect_communities(self):
        """
        使用 Louvain 算法检测社区
        """
        partition = community_louvain.best_partition(self.graph)
        self.communities = {}
        for node, community in partition.items():
            self.communities.setdefault(community, []).append(node)

        print(f"Detected {len(self.communities)} communities.")

    def _build_community_graphs(self):
        """
        构建社区间网络和社区内网络
        """
        self.community_graph = nx.Graph()

        # 添加所有社区间的边
        for u, v in self.graph.edges():
            community_u = self._get_community(u)
            community_v = self._get_community(v)
            if community_u != community_v:  # 确保是跨社区边
                self.community_graph.add_edge(u, v)

        # 构建每个社区内的子图
        for community_id, nodes in self.communities.items():
            subgraph = self.graph.subgraph(nodes).copy()
            self.intra_community_graphs[community_id] = subgraph

    def _get_community(self, node):
        """
        获取节点所属社区
        """
        for community_id, nodes in self.communities.items():
            if node in nodes:
                return community_id
        return None

    def generate_walks(self):
        """
        为每个节点生成随机游走
        """
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.walk_number):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._single_walk(node))
        return walks

    def _single_walk(self, start_node):
        """
        单次随机游走
        """
        walk = [start_node]

        # 检查节点在哪些图中存在
        community_id = self._get_community(start_node)
        in_community_graph = start_node in self.community_graph.nodes()
        in_intra_community_graph = (
                community_id in self.intra_community_graphs and
                start_node in self.intra_community_graphs[community_id]
        )

        # 如果节点不属于任何图，则跳过游走
        if not in_community_graph and not in_intra_community_graph:
            print(f"Node {start_node} is not in any graph. Skipping walk.")
            return []

        # 根据条件选择游走的网络
        if in_community_graph and in_intra_community_graph:
            # 节点同时属于社区内和社区间网络，按概率选择
            current_graph = (
                self.community_graph if random.random() < self.community_jump_prob
                else self.intra_community_graphs[community_id]
            )
        elif in_community_graph:
            current_graph = self.community_graph
        else:
            current_graph = self.intra_community_graphs[community_id]

        # 开始游走
        for _ in range(self.walk_length - 1):
            current_node = walk[-1]

            # 获取当前节点的邻居
            neighbors = list(current_graph.neighbors(current_node))
            if not neighbors:
                break  # 如果没有邻居，则结束游走

            # 随机选择下一个节点
            next_node = random.choice(neighbors)
            walk.append(next_node)

        return walk

    def get_embedding(self) -> np.array:
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return np.array(self._embedding)

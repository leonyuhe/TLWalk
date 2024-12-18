
# Two Layer Community Walk (TLWalk)

**TLWalk** is a community-aware graph embedding method integrated into the [Karate Club library](https://github.com/benedekrozemberczki/karateclub). TLWalk enhances node representations by explicitly considering **intra-community** and **inter-community** relationships using a two-layer hierarchical random walk strategy.

---

## Features

- **Community-Aware Embedding**: Explicitly integrates hierarchical community structures.
- **Intra-Community and Inter-Community Dynamics**: Models both dense local relationships and sparse global structures.
- **Seamless Integration**: TLWalk is compatible with the Karate Club graph embedding library and can be used directly for various graph tasks.
- **Scalability**: Efficient on large-scale networks with a clear modular design.

---

## Installation

1. Install the Karate Club library using `pip`:

   ```bash
   pip install karateclub
   ```

2. Clone the TLWalk repository to access the relevant scripts:

   ```bash
   git clone https://github.com/yourusername/TLWalk.git
   cd TLWalk
   ```

---

## Input Data

Edge files must follow the standard `.edges` format:

- Each line represents an edge: `<node_1> <node_2>` (space-separated).
- Example:
   ```
   0 1
   1 2
   2 3
   0 3
   ```

Supported datasets are stored in the `input/` directory.

---

## Usage

### 1. Preprocessing

Preprocess your graph data to ensure compatibility:

```bash
python preprocess.py 
```

---

### 2. Run TLWalk for Embedding

Run TLWalk to generate graph embeddings:

```bash
python _TwoLayerCommunityWalk.py
```

---

### 3. Link Prediction Task

Evaluate TLWalk embeddings for link prediction:

```bash
python link_prediction.py 
```

---

### 4. Node Classification Task

Evaluate node classification performance:

```bash
python node_classification_c.py 
```

---

## Example Integration with Karate Club

Once TLWalk is integrated into the Karate Club library, it can be used as follows:

```python
from karateclub import TwoLayerCommunityWalk
import networkx as nx

# Load a graph
G = nx.karate_club_graph()

# Initialize the model
model = TwoLayerCommunityWalk(dimensions=128, walk_length=10, num_walks=80)

# Fit the model
model.fit(G)

# Generate embeddings
embeddings = model.get_embedding()
print(embeddings)
```

---

## Repository Structure

```plaintext
TLWalk/
│-- input/                # Input graph datasets (.edges files)
│   │-- bio-WormNet-v3.edges
│   │-- ego-Facebook.edges
│   │-- ...
│
│-- _TwoLayerCommunityWalk.py   # Main TLWalk embedding implementation
│-- preprocess.py               # Preprocessing script for input graphs
│-- link_prediction.py          # Script for link prediction
│-- node_classification_c.py    # Script for node classification
│
│-- output/               # Generated embeddings and results
│-- results/              # Task results (accuracy, AUC, etc.)
```

---

## Citation

If you use **TLWalk** in your research, please cite our work:

```bibtex
@article{yourarticle2024,
  title={Two Layer Walk: A Community-Aware Graph Embedding},
  author={He Yu and Jing Liu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## License

This project is released under the **MIT License**.

---

## Contact

For questions or contributions, please open an issue or contact **He Yu** at [yuhehome@gmail.com](mailto:yuhehome@gmail.com).


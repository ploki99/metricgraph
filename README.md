# MetricGraph

**metricgraph** is a Python package to generate, manipulate, and visualize metric graphs in 2D and 3D.
It supports directed/undirected graphs, self-loops, connectivity constraints, and interactive plotting.

---

## 📚 Documentation

Full documentation is available at: [https://metricgraph.readthedocs.io/en/latest/](https://metricgraph.readthedocs.io/en/latest/)

---

## ⚡ Installation

You can install the package via pip:

```bash
pip install metricgraph
```

Or install from source:

```bash
git clone https://github.com/yourusername/metricgraph.git
cd metricgraph
pip install -e .
```

---

## 🚀 Examples

```python
import numpy as np
import random

from metricgraph import MetricGraph, WeightedGraph

# Constructor test + graph plot
vertices = [[0, 0, 0.1],
            [0, 0.8, -0.1],
            [1, 0, 0.4],
            [0.7, 0.4, 0.3]]

edges = [[0, 1], [0, 2], [1, 1], [2, 3], [3, 2]]
graph = MetricGraph(vertices, edges)
print(graph)
print(graph.get_properties())
graph.plot(directed=True)

# Generate weighted graph test + weight plots
random.seed(42)
graph2 = WeightedGraph.generate_graph(n_vertices=10, n_dim=2)
graph2.set_edges_weight('test', np.arange(9))
graph2.set_vertices_weight('test2', np.arange(10))
graph2.plot_weight('test2', title='test vertices weight')
graph2.plot_weight('test', title='test edges weight')
```

---

## 🛠 Features

* Generate graphs with specified number of vertices, dimensions, and connectivity constraints
* Compute graph metrics and boundary nodes
* Interactive 2D/3D plotting
* Weighted graphs support
* Fully documented API

---

## 📄 License

[MIT License](LICENSE)

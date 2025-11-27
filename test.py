from metricgraph import MetricGraph, WeightedGraph

# constructor test
vertices = [[0,0,0],
            [0,0.8,0],
            [1,0,0],
            [0.7,0.4,0]]

edges = [[0,1], [0,2], [1,1], [2,3], [3,2]]
graph = WeightedGraph(vertices, edges)
print(graph)
print(graph.get_properties())
graph.plot()
graph.set_edges_weight('test', [1,2,3,4,5])
graph.set_vertices_weight('test2', [1,2,3,4])
graph.plot_weight('test2', title='test vertices weight')
graph.plot_weight('test', title='test edges weight')

# generate test
graph2 = MetricGraph.generate_graph(n_vertices=10, n_dim=2)
print(graph2)
print(graph2.get_properties())
graph2.plot(directed=True, show_axis=True)

# read test
graph3 = MetricGraph.load('test')
print(graph3)
print(graph3.get_properties())
graph3.plot(vertices_label=False)
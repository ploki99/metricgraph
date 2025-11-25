from metricgraph import MetricGraph

# constructor test
vertices = [[0,0],
            [0,0.8],
            [1,0],
            [0.7,0.4]]
edges = [[0,1], [1,1], [0,2], [2,3], [3,2]]
graph = MetricGraph(vertices, edges)
print(graph)
print(graph.get_properties())
inv = lambda x: 1/x
print(f"Length of the edges are: {graph.get_edges_length()}")
print(f"Graph laplacian is \n{graph.get_graph_laplacian(inv, inv)}")
graph.plot(directed=True, plot_boundary=True)

# generate test
graph2 = MetricGraph.generate_graph(n_vertices=10, n_dim=2)
print(graph2)
print(graph2.get_properties())
graph2.plot(plot_boundary=True)

# read test
graph3 = MetricGraph.read('test')
print(graph3)
print(graph3.get_properties())
graph3.plot(plot_boundary=True)
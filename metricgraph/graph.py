import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from .arrow3d import *
from .graphProperties import GraphProperties

class MetricGraph:

    def __init__(self, vertices, edges):
        """
        Build a MetricGraph object
        :param vertices: vertices coordinates of the graph
        :type vertices: list | np.ndarray
        :param edges: connectivity matrix of the graph
        :type edges: list | np.ndarray
        """
        # Save edges and vertices
        self.vertices = np.array(vertices)
        self.edges = np.array(edges)
        self.properties = None

    def __str__(self):
        return f"Vertices of the graph are:\n {self.vertices}\n" \
               f"Edges of the graph are:\n {self.edges}\n" \
               f"Number of vertices: {self.vertices.shape[0]}\n" \
               f"Number of edges: {self.edges.shape[0]}\n"

    # Private interface
    def __plot2d(self, color='blue', r=0.2, interactive=False, show_axis=True, plot_boundary=False, b_color='green',
             title='', directed=False):
        """
        Plot a 2D image with graph representation.
        :param color: color used to plot the graph
        :param r: scaling factor
        :param interactive: if True enable interactive mode
        :param show_axis: if False doesn't show axis
        :param plot_boundary: if True plot the boundary vertexes using b_color
        :param b_color: color used to plot the boundary vertexes
        :param title: optional title to plot on the figure
        :param directed: if True plot the graph in directed mode
        """
        fig, ax = plt.subplots()

        # Plot edges
        for e in self.edges:
            # Get coordinates
            x0 = self.vertices[e[0]][0]
            y0 = self.vertices[e[0]][1]
            x1 = self.vertices[e[1]][0]
            y1 = self.vertices[e[1]][1]

            # Plot circle if we have a loop
            if e[0] == e[1]:
                theta = np.linspace(0, 1.75 * np.pi, 200)
                x = r * np.cos(theta) + x0 - r / np.sqrt(2)
                y = r * np.sin(theta) + y0 + r / np.sqrt(2)
                ax.plot(x, y, c=color)
                dx1, dy1 = get_dx_dy(x[0], y[0], x1, y1, r / 2)
                ax.arrow(x[0], y[0], x1 + dx1 - x[0], y1 + dy1 - y[0], length_includes_head=True,
                         head_width=0.05, head_length=0.05, fill=True, color=color)

            # Plot line
            else:
                if directed:
                    dx1, dy1 = get_dx_dy(x0, y0, x1, y1, r / 2)
                    dx0, dy0 = 0, 0
                    if is_edge_present(self.edges, [e[1], e[0]], directed):
                        dx0, dy0 = dx1, dy1
                    ax.arrow(x0 - dx0, y0 - dy0, x1 + dx1 - x0 + dx0, y1 + dy1 - y0 + dy0, length_includes_head=True,
                             head_width=0.05, head_length=0.05, fill=True, color=color)
                else:
                    ax.plot([x0, x1], [y0, y1], c=color)

        # Plot vertexes
        mask = []
        if plot_boundary:
            mask = self.get_boundary_mask()
        for i, v in enumerate(self.vertices):
            vert_col = color
            if plot_boundary and mask[i]:
                vert_col = b_color
            ax.scatter(v[0], v[1], s=r * 3000, zorder=2, facecolors='white', edgecolors=vert_col)
            ax.text(v[0], v[1], i, fontsize=12, color=vert_col, horizontalalignment='center', verticalalignment='center')

        # Show plot
        if interactive:
            plt.ion()
        else:
            plt.ioff()
        if show_axis:
            plt.axis('on')
        else:
            plt.axis('off')
        plt.title(title)
        plt.show()

    def __plot3d(self, color='blue', r=0.2, interactive=False, show_axis=True, plot_boundary=False, b_color='green',
             title='', directed=False):
        """
        Plot a 3D image with graph representation.
        :param color: color used to plot the graph
        :param r: scaling factor
        :param interactive: if True enable interactive mode
        :param show_axis: if False doesn't show axis
        :param plot_boundary: if True plot the boundary vertexes using b_color
        :param b_color: color used to plot the boundary vertexes
        :param title: optional title to plot on the figure
        :param directed: if True plot the graph in directed mode
        """

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Plot edges
        for e in self.edges:
            # Get coordinates
            x0 = self.vertices[e[0]][0]
            y0 = self.vertices[e[0]][1]
            z0 = self.vertices[e[0]][2]
            x1 = self.vertices[e[1]][0]
            y1 = self.vertices[e[1]][1]
            z1 = self.vertices[e[1]][2]

            # Plot circle if we have a loop
            if e[0] == e[1]:
                theta = np.linspace(0, 1.7 * np.pi, 200)
                x = r * np.cos(theta) + x0 - r / np.sqrt(2)
                y = r * np.sin(theta) + y0 + r / np.sqrt(2)
                z = np.zeros_like(theta) + z0
                ax.plot(x, y, z, c=color)
                # Plot arrow
                dx, dy = get_dx_dy(x[0], y[0], x1, y1, r / 2)
                ax.arrow3D(x[0], y[0], z0, x1 + dx - x[0], y1 + dy - y[0], 0,
                           mutation_scale=15, arrowstyle="-|>", ec=color, fc=color)
            # Plot line
            else:
                if directed:
                    # Plot arrow
                    dx1, dy1, dz1 = get_dx_dy_dz(x0, y0, z0, x1, y1, z1, r / 2)
                    dx0, dy0, dz0 = 0, 0, 0
                    if is_edge_present(self.edges, [e[1], e[0]], directed):
                        dx0, dy0, dz0 = dx1, dy1, dz1
                    ax.arrow3D(x0 - dx0, y0 - dy0, z0 - dz0,
                               x1 + dx1 - x0 + dx0, y1 + dy1 - y0 + dy0, z1 + dz1 - z0 + dz0,
                               mutation_scale=15, arrowstyle="-|>", ec=color, fc=color)
                else:
                    ax.plot([x0, x1], [y0, y1], [z0, z1], c=color)

        # Plot vertices
        mask = []
        if plot_boundary:
            mask = self.get_boundary_mask()
        for i, v in enumerate(self.vertices):
            vert_col = color
            if plot_boundary and mask[i]:
                vert_col = b_color
            ax.scatter(v[0], v[1], v[2], s=r * 2000, facecolors='white', edgecolors=vert_col)
            ax.text(v[0], v[1], v[2], i, fontsize=12, color=vert_col, zorder=self.vertices.shape[0] + 2,
                    horizontalalignment='center', verticalalignment='center')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Show plot
        if interactive:
            plt.ion()
        else:
            plt.ioff()
        if show_axis:
            plt.axis('on')
        else:
            plt.axis('off')
        plt.title(title)
        plt.show()

    # Public interface of the class

    def get_edges_length(self):
        """
        Compute length of edges between vertices.
        :return: Edge length
        :rtype: np.ndarray
        """
        edges_length = np.zeros(self.edges.shape[0])
        for i, e in enumerate(self.edges):
            x1 = self.vertices[e[0], :]
            x2 = self.vertices[e[1], :]
            edges_length[i] = np.linalg.norm(x1 - x2)
        return edges_length

    def get_graph_laplacian(self, w1=None, w2=None):
        """
        Compute graph laplacian. Consider the edge as undirected. Ignore self loops.
        :param w1: optional weight function, if used L_jj = L_jj * w1(edges_length[i])
        :type w1: function
        :param w2: optional weight function, if used L_jk = L_jk * w2(edges_length[i]), j != k
        :type w2: function
        :return: numpy array containing the graph laplacian
        :rtype: np.ndarray
        """
        # Define matrices
        L = np.zeros((self.vertices.shape[0], self.vertices.shape[0]),
                     dtype=np.int64 if w1 is None and w2 is None else np.float64)
        mask = np.zeros((self.vertices.shape[0], self.vertices.shape[0]), dtype=bool)
        if w1 is None:
            w1 = lambda x: 1
        if w2 is None:
            w2 = lambda x: 1
        # Get edges length
        edges_length = self.get_edges_length()
        # Compute weighted laplacian matrix
        for i, (v1, v2) in enumerate(self.edges):
            if v1 != v2 and not mask[v1, v2]:
                L[v1, v1] += 1 * w1(edges_length[i])
                L[v2, v2] += 1 * w1(edges_length[i])
                L[v1, v2] = -1 * w2(edges_length[i])
                L[v2, v1] = -1 * w2(edges_length[i])
                mask[v1, v2] = True
                mask[v2, v1] = True
        return L

    def get_boundary_mask(self):
        """
        Compute naturally boundary vertices (nodes with at maximum one connection, incoming or outgoing).
        :return: boolean numpy array
        :rtype: np.ndarray[bool]
        """
        n_connections = np.zeros((self.vertices.shape[0]), dtype=int)
        mask = np.zeros((self.vertices.shape[0], self.vertices.shape[0]), dtype=bool)
        for v1, v2 in self.edges:
            if v1 != v2 and not mask[v1, v2]:
                n_connections[v1] += 1
                n_connections[v2] += 1
                mask[v1, v2] = True
                mask[v2, v1] = True
        return n_connections < 2

    def get_boundary_coordinates(self):
        """
        Compute naturally boundary vertices coordinates (nodes with at maximum one connection, incoming or outgoing).
        :return: numpy array of coordinates
        :rtype: np.ndarray
        """
        mask = self.get_boundary_mask()
        coord = np.zeros((np.sum(mask), self.vertices.shape[1]))
        j = 0
        for i, v in enumerate(self.vertices):
            if mask[i]:
                coord[j] = v
                j += 1
        return coord

    def get_inlets(self):
        """
        Compute inlet vertices (nodes with no incoming connections).
        :return: numpy array of inlet vertices
        :rtype: np.ndarray[int]
        """
        n_connections = np.zeros((self.vertices.shape[0]), dtype=int)
        for v1, v2 in self.edges:
            n_connections[v2] += 1
        return np.where(n_connections == 0)[0]

    def get_outlets(self):
        """
        Compute outlet vertices (nodes with no outgoing connections).
        :return: numpy array of outlet vertices
        :rtype: np.ndarray[int]
        """
        n_connections = np.zeros((self.vertices.shape[0]), dtype=int)
        for v1, v2 in self.edges:
            n_connections[v1] += 1
        return np.where(n_connections == 0)[0]

    def get_properties(self):
        """
        Compute and return properties of the graph.
        :rtype: GraphProperties
        """
        if self.properties is None:
            self.properties = GraphProperties.get_properties_from_graph(self.vertices, self.edges)
        return self.properties

    @classmethod
    def generate_graph(cls, n_vertices, n_dim=3, domain_limits=None, minimum_edges=None, directed=False, loop=False,
                       strong_connected=False, weak_connected=True, acyclic=True):
        """
        Generate MetricGraph object, given its properties.
       :param n_vertices: number of vertices of the graph
       :type n_vertices: int
       :param n_dim: dimension of the space
       :param domain_limits: limits of the bounding box containing the graph
       :type domain_limits: list | np.ndarray
       :param minimum_edges: number of minimum edges tried to be added (it may happen that the maximum number
                             of possible edges is smaller)
       :type minimum_edges: int
       :param directed: if True directed graph are considered
       :param loop: if True and directed=True self loop are allowed
       :param strong_connected: if True the graph has to be strongly connected (notice that the definition makes sense
                                only for directed graph)
       :param weak_connected: if True the graph has to be weakly connected in case of directed graph or just connected
                              in case of undirected graph
       :param acyclic: if True the graph has to be acyclic
       """
        properties = GraphProperties(n_vertices=n_vertices, n_dim=n_dim, limits=domain_limits,
                                     minimum_edges=minimum_edges, directed=directed,
                                     loop=loop, strong_connected=strong_connected,
                                     weak_connected=weak_connected, acyclic=acyclic)
        # Create graph
        graph = cls(get_random_vertices(properties), get_random_edges(properties))
        # Save graph properties
        graph.properties = properties
        graph.properties.update_n_cycles(graph.vertices, graph.edges)
        # Return graph
        return graph

    def plot(self, color='blue', r=0.2, interactive=False, show_axis=True, plot_boundary=False, b_color='green',
             title='', directed=False):
        """
        Plot image with graph representation.
        :param color: color used to plot the graph
        :param r: scaling factor
        :param interactive: if True enable interactive mode
        :param show_axis: if False doesn't show axis
        :param plot_boundary: if True plot the boundary vertexes using b_color
        :param b_color: color used to plot the boundary vertexes
        :param title: optional title to plot on the figure
        :param directed: if True plot the graph in directed mode
        """
        if self.vertices.shape[1] == 2:
            self.__plot2d(color, r, interactive, show_axis, plot_boundary, b_color, title, directed)
        elif self.vertices.shape[1] == 3:
            self.__plot3d(color, r, interactive, show_axis, plot_boundary, b_color, title, directed)
        else:
            print("Cannot plot graph in dimension greater than 3")

    def distance(self, point):
        """
        Compute distance between point and the graph --> d(p, G). It returns: \n
        - ret['distance'] = distance between point and the graph, \n
        - ret['space_prj'] = projection on the graph in the nd space, \n
        - ret['abscissa'] = abscissa of the projection, \n
        - ret['nearest_edge'] = nearest edge
        :param point: point to compute distance from
        :type point: list[float] | np.ndarray[float]
        :return: dictionary of computed values
        :rtype: dict
        """
        p = np.array(point)
        # Define points a and b
        a = self.vertices[self.edges[:, 0]]
        b = self.vertices[self.edges[:, 1]]
        # Define segments
        ab = b - a
        ap = p - a
        # Compute projection coefficient
        ab_norm2 = np.sum(ab * ab, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.sum(ab * ap, axis=1) / ab_norm2
        t = np.where(ab_norm2 == 0.0, 0.0, t)  # for segment with length = 0
        # Limit coefficient on the segment: min(1, max(0, t))
        t_s = np.clip(t, 0, 1)
        # Compute projection
        projection = a + t_s[:, None] * ab
        # Compute distances
        dist = np.linalg.norm(p - projection, axis=1)
        # Get minimum distance
        idx = np.argmin(dist)
        # Return distance, projection, abscissa, and edge index
        edge_len = np.linalg.norm(self.vertices[self.edges[idx, 0]] - self.vertices[self.edges[idx, 1]])
        ret = {
            'distance': dist[idx],
            'space_prj': projection[idx],
            'abscissa': t_s[idx] * edge_len,
            'nearest_edge': idx
        }
        return ret

    def refine_graph(self, n_refinements):
        """
        Refine the graph adding internal vertices, return the refined graph.
        :param n_refinements: number of internal vertices to add for each edge
        :type n_refinements: int
        :return: refined graph
        """
        new_nodes = self.vertices.tolist()
        new_edges = []
        # For each edge, add the new vertices
        for (i, j) in self.edges:
            new_i = i
            for new_node in np.linspace(self.vertices[i], self.vertices[j], n_refinements + 2)[1:-1]:
                # Insert node
                new_nodes.append(new_node)
                new_j = len(new_nodes) - 1
                # Connect nodes
                new_edges.append([new_i, new_j])
                new_i = new_j
            # Connect last node
            new_edges.append([new_i, j])

        return self.__class__(new_nodes, new_edges)

    def save(self, name):
        """
        Save graph object to file.
        :param name: path and name of the file
        :type name: str
        """
        with open(name + '.pkl', 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def read(cls, name):
        """
        Read graph object from file.
        :param name: path and name of the file
        :type name: str
        :return: Graph read from file
        """
        with open(name + '.pkl', 'rb') as inp:
            obj = pickle.load(inp)
            if not isinstance(obj, cls):
                raise Exception("Only MetricGraph type allowed")
            return obj

# Static functions
def get_random_vertices(properties):
    """
    Generate random vertex coordinates given certain properties.
    :param properties: properties of the graph
    :type properties: GraphProperties
    """
    vertices = np.zeros((properties.n_vertices, properties.n_dim))
    for k in range(properties.n_vertices):
        for i in range(properties.n_dim):
            vertices[k][i] = random.uniform(properties.limits[i][0], properties.limits[i][1])
    return vertices


def get_random_edges(properties):
    """
    Generate random edges given certain properties.
    :param properties: properties of the graph
    :type properties: GraphProperties
    """
    # Maximum number of edges addable
    max_edges = properties.get_max_edges()
    # Number of edges to add that avoids cycles
    acyclic_edges = 0
    if properties.weak_connected:
        acyclic_edges = properties.n_vertices - 1
        if properties.strong_connected:
            acyclic_edges += 1
    elif properties.acyclic:
        acyclic_edges = min(properties.n_vertices - 1, properties.minimum_edges)
    # Number of random edges to add
    random_edges = max(min(max_edges, properties.minimum_edges) - acyclic_edges, 0)
    # Initialize and fill edges matrix
    edges = np.zeros(shape=(acyclic_edges + random_edges, 2), dtype=np.int64)
    add_acyclic_edges(edges, acyclic_edges, properties.n_vertices, properties.strong_connected)
    for k in range(random_edges):
        add_random_edge(edges, acyclic_edges + k, properties, properties.directed and properties.acyclic)

    return edges


def add_acyclic_edges(edges, n_edges, n_vertices, strong_connect=False):
    """
    Add n_edges to the graph in such a way we don't have cycles.
    Note that the n_edges must be the first edges added in the graph.
    :param edges: edges of the graph
    :type edges: np.ndarray
    :param n_edges: number of edges to add
    :type n_edges: int
    :param n_vertices: number of vertices of the graph
    :type n_vertices: int
    :param strong_connect: if True then it creates a strong connected graph with one cycle
    """
    # In case of strong connected graph, we add the last edges after the loop
    if strong_connect:
        n_edges -= 1
    unselected_v = [i for i in range(n_vertices)]
    i = random.randrange(0, len(unselected_v))
    selected_v = [unselected_v[i]]
    unselected_v.pop(i)
    for k in range(n_edges):
        i = random.randrange(0, len(unselected_v))
        if strong_connect:
            # Take the index of the last element connected to the graph, since we will connect the new vertex to it
            j = len(selected_v) - 1
        else:
            j = random.randrange(0, len(selected_v))
        # Add edge
        edges[k] = [unselected_v[i], selected_v[j]]
        selected_v.append(unselected_v[i])
        unselected_v.pop(i)
    # Add last edge
    if strong_connect:
        edges[n_edges] = [selected_v[0], selected_v[-1]]


def add_random_edge(edges, idx, properties, keep_acyclicity=False):
    """
    Add a random edge to the graph.
    :param edges: edges of the graph
    :type edges: np.ndarray
    :param idx: index of the row of the edge matrix where to add the edge
    :type idx: int
    :param properties: properties of the graph
    :type properties: GraphProperties
    :param keep_acyclicity: if True add an edge keeping the graph acyclic
    """
    while True:
        a = random.randrange(0, properties.n_vertices)
        while True:
            b = random.randrange(0, properties.n_vertices)
            if properties.directed and properties.loop:
                break
            if a != b:
                break
        if not is_edge_present(edges, [a, b], properties.directed):
            if keep_acyclicity and is_edge_present(edges, [b, a], properties.directed):
                continue
            edges[idx] = [a, b]
            if keep_acyclicity and has_dgraph_cycle(edges, properties.n_vertices):
                edges[idx] = [b, a]
            break


def is_edge_present(edges, e, directed):
    """
    Verify if an edge is present in the graph.
    :param edges: edges of the graph
    :type edges: np.ndarray
    :param e: edge to verify in the form [a,b]
    :type e: list
    :param directed: if True the edge are directed
    :type directed: bool
    :return: True if the edge is present
    """
    if directed:
        return any(np.equal(edges, e).all(1))
    return any(np.equal(edges, e).all(1)) or any(np.equal(edges, [e[1], e[0]]).all(1))


def has_dgraph_cycle(edges, n_vertices):
    """
    Implement a topological sort to verify if the graph has a cycle.
    :param edges: edges of the graph
    :type edges: np.ndarray
    :param n_vertices: number of vertices of the graph
    :type n_vertices: int
    :return: True if directed graph has at least a cycle
    """
    # Create a copy of list self.edges
    graph = edges.tolist()

    def has_no_incoming_edge(node):
        for e in graph:
            if e[1] == node:
                return False
        return True

    # Initialize empty list L
    L = []
    # Create set of nodes with no incoming edges
    Q = []
    for i in range(n_vertices):
        if has_no_incoming_edge(i):
            Q.append(i)
    # While Q is not empty
    while len(Q) > 0:
        n = Q.pop(0)
        L.append(n)
        # For each node m with an edge e from n to m
        for m in range(n_vertices):
            if [n, m] in graph:
                # Remove edge e from the graph
                graph.remove([n, m])
                # If m has no other incoming edges
                if has_no_incoming_edge(m):
                    Q.append(m)
    if len(graph) > 0:
        return True
    # Proposed topologically sorted order: L
    return False

# Additional plot functions
def get_dx_dy(x0, y0, x1, y1, rho):
    """
    Only for visualization purposes of the arrows in case of directed graph
    """
    delta = 0
    if x1 == x0:
        theta = np.pi / 2
        delta = np.sign(y0 - y1)
    else:
        theta = np.arctan((y1 - y0) / (x1 - x0))
    dx1 = rho * np.cos(theta) * np.sign(x0 - x1)
    dy1 = rho * np.sin(theta) * np.sign(x0 - x1 + delta) + delta * rho/2
    return dx1, dy1

def get_dx_dy_dz(x0, y0, z0, x1, y1, z1, rho):
    """
    Only for visualization purposes of the arrows in case of directed graph
    """
    dx1, dy1 = get_dx_dy(x0, y0, x1, y1, rho)
    dx2, dz1 = get_dx_dy(x0, z0, x1, z1, rho)
    dy2, dz2 = get_dx_dy(y0, z0, y1, z1, rho)
    dx = (dx1 + dx2) / 2
    dy = (dy1 + dy2) / 2
    dz = (dz1 + dz2) / 2
    return dx, dy, dz
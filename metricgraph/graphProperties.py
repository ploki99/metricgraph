import numpy as np


class GraphProperties:
    """
    Class that contains the properties of the graph in Graph object.
    """

    def __init__(self, n_vertices, n_dim=3, limits=None, minimum_edges=None, directed=True, loop=False,
                 strong_connected=False, weak_connected=True, acyclic=True):
        """
        :param n_vertices: number of vertices of the graph
        :type n_vertices: int
        :param n_dim: dimension of the space
        :param limits: limits of the bounding box containing the graph
        :type limits: list | np.ndarray
        :param minimum_edges: number of minimum edges tried to be added (it may happen that the maximum number
                              of possible edges is smaller). Default is n_vertices - 1.
        :type minimum_edges: int
        :param directed: if True directed graph are considered
        :param loop: if True and directed=True self loop are allowed
        :param strong_connected: if True the graph has to be strongly connected (notice that the definition makes sense
                                 only for directed graph)
        :param weak_connected: if True the graph has to be weakly connected in case of directed graph or just connected
                               in case of undirected graph
        :param acyclic: if True the graph has to be acyclic
        """
        # Number of vertices
        self.n_vertices = int(n_vertices)
        if self.n_vertices < 1:
            self.n_vertices = 1
        # Number of edges
        if minimum_edges is None:
            self.minimum_edges = self.n_vertices - 1
        else:
            self.minimum_edges = int(minimum_edges)
        # Dimension of the space
        self.n_dim = int(n_dim)
        if self.n_dim < 2:
            self.n_dim = 2
        # Limits of the domain
        if limits is None:
            self.limits = np.zeros((self.n_dim, 2))
            self.limits[:, 0], self.limits[:, 1] = -1, 1
        else:
            self.limits = np.array(limits)

        self.directed = directed
        self.loop = loop
        self.strong_connected = strong_connected
        self.weak_connected = weak_connected
        self.acyclic = acyclic
        if self.acyclic:
            # We can't have loop in acyclic graph
            self.loop = False
        if not self.directed:
            # For undirected graph strong connection doesn't make sense:
            self.strong_connected = False
        if self.strong_connected:
            # Strong connection implies weak connection:
            self.weak_connected = True
            # Strong connection implies cyclic graph:
            self.acyclic = False
        # Number of cycle in the graph
        self.n_cycles = -1

    def __str__(self):
        return f"Dimension of the space: {self.n_dim}\n" \
               f"Limits of the bounding box are \n {self.limits}\n" \
               f"Is the graph directed: {self.directed}\n" \
               f"Does the graph allow self loop: {self.loop}\n" \
               f"Has the graph to be strongly connected: {self.strong_connected}\n" \
               f"Has the graph to be weakly connected: {self.weak_connected}\n" \
               f"Has the graph to be acyclic: {self.acyclic}\n" \
               f"Number of cycles in the graph: {self.n_cycles}"


    # Public interface

    def get_max_edges(self):
        """
        Compute the maximum edges the graph can have given its topology
        :return: maximum number of edges
        """
        max_edges = self.n_vertices ** 2
        if self.directed:
            if self.acyclic:
                max_edges = (max_edges - self.n_vertices) / 2
            elif not self.loop:
                max_edges -= self.n_vertices
        else:
            if self.acyclic:
                max_edges = self.n_vertices - 1
            else:
                max_edges = (max_edges - self.n_vertices) / 2
        return int(max_edges)

    def update_n_cycles(self, vertices, edges):
        """
        Update the number of cycles of the graph (after random generate the graph)
        :param vertices: vertices of the graph of shape (n_vertices, n_dim)
        :type vertices: np.ndarray
        :param edges: edges of the graph of shape (n_edges, 2)
        :type edges: np.ndarray
        """
        if self.acyclic:
            self.n_cycles = 0
        else:
            if self.directed:
                self.n_cycles = count_d_cycles(vertices, edges)
            else:
                edges = duplicate_edges(edges)
                self.n_cycles = count_nd_cycles(vertices, edges)

    @classmethod
    def get_properties_from_graph(cls, vertices, edges):
        """
        Compute graph properties given vertices and edges.
        Notice that to be an undirected graph, for each (i,j) in edges also (j,i) must be present
        :param vertices: vertices of the graph of shape (n_vertices, n_dim)
        :type vertices: list | np.ndarray
        :param edges: edges of the graph of shape (n_edges, 2)
        :type edges: list | np.ndarray
        :return: GraphProperties instance
        """
        vertices = np.array(vertices)
        edges = np.array(edges)
        # Find graph properties
        weak_connected, strong_connected = False, False
        self_loop = False
        directed = check_directed(edges)
        if directed:
            weak_connected, strong_connected = check_d_connected(vertices, edges)
            n_self_loop = count_self_loop(edges)
            self_loop = n_self_loop > 0
            n_cycles = count_d_cycles(vertices, edges)
        else:
            weak_connected = check_nd_connected(vertices, edges)
            n_cycles = count_nd_cycles(vertices, edges)

        # Create GraphProperties object
        p = cls(n_vertices=vertices.shape[0],
                n_dim=vertices.shape[1],
                limits=get_bounding_box(vertices),
                directed=directed,
                loop=self_loop,
                strong_connected=strong_connected,
                weak_connected=weak_connected,
                acyclic=n_cycles == 0)
        # Add number of cycles
        p.n_cycles = n_cycles
        # Return object
        return p

# Static functions

def get_bounding_box(vertices):
    """
    Compute bounding box of the graph
    :param vertices: vertices of the graph
    :type vertices: np.ndarray
    :return: numpy array of shape (n_dim, 2)
    """
    limits = np.zeros((vertices.shape[1], 2))
    for i in range(vertices.shape[1]):
        limits[i, 0] = min(vertices[:, i])
        limits[i, 1] = max(vertices[:, i])
    return limits


def check_directed(edges):
    """
    Check if the graph is directed
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: True if the graph is directed, False otherwise
    """
    if count_self_loop(edges) != 0:
        return True
    for e in edges:
        if not any(np.equal(edges, [e[1], e[0]]).all(1)):
            return True
    return False


def check_nd_connected(vertices, edges):
    """
    Check if an undirected graph is connected
    :param vertices: vertices of the graph
    :type vertices: np.ndarray
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: True if the graph is directed, False otherwise
    """
    n_vert = vertices.shape[0]
    is_connected = np.zeros(n_vert)
    if n_vert - 1 > edges.shape[0]:
        return False
    for i in range(n_vert):
        for e in edges:
            if e[0] == i:
                is_connected[e[1]] += 1
    for i in is_connected:
        if i == 0:
            return False
    return True


def check_d_connected(vertices, edges):
    """
    Check if a directed graph is weak connected and strong connected
    :param vertices: vertices of the graph
    :type vertices: np.ndarray
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: first bool for weak connected, second bool for strong connected
    """
    # Remove self loops
    ed = []
    for e in edges:
        if e[0] != e[1]:
            ed.append(e)
    ed = np.array(ed)
    # Check strong connection
    is_strong_connected = check_nd_connected(vertices, ed)
    if is_strong_connected:
        return True, True
    # Check weak connection
    nd_edges = duplicate_edges(ed)
    is_weak_connected = check_nd_connected(vertices, nd_edges)
    if is_weak_connected:
        return True, False
    return False, False


def duplicate_edges(edges):
    """
    It ensures that if (i,j) belongs to edges, also (j,i) will be present
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: numpy array with duplicate edges
    """
    add = []
    for e in edges:
        if not any(np.equal(edges, [e[1], e[0]]).all(1)):
            add.append([e[1], e[0]])
    if not add:
        return edges
    add = np.array(add)
    return np.vstack((edges, add))


def count_self_loop(edges):
    """
    Count the number of self loops in the graph
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: number of self loops
    """
    count = 0
    for e in edges:
        if e[0] == e[1]:
            count += 1
    return count


def count_simple_cycles(vertices, edges):
    """
    Count the number of simple cycles in the graph
    :param vertices: vertices of the graph
    :type vertices: np.ndarray
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: number of simple cycles
    """
    n_vert = vertices.shape[0]
    visited = [False] * (n_vert + 1)
    stack = []
    edges = convert_edges(n_vert, edges)
    count = 0

    for v in range(0, n_vert):
        count = dfs(v, visited, stack, edges, count)
        visited[v] = True

    return count


def convert_edges(n_vert, edges):
    """
    Change representation of edges matrix
    :param n_vert: number of vertices in the graph
    :type n_vert: int
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: new representation
    """
    ed = [[] for i in range(n_vert)]
    for e in edges:
        if e[0] != e[1]:
            ed[e[0]].append(e[1])
    return ed


def dfs(vertex, visited, stack, edges, count):
    """
    Implement depth-first search algorithm
    :param vertex: root vertex
    :type vertex: int
    :param visited: list of visited vertices
    :type visited: list
    :param stack: stack used
    :type stack: list
    :param edges: alternative representation of edges matrix
    :type edges: list[list]
    :param count: current number of cycles found
    :type count: int
    :return: number of cycles found
    """
    visited[vertex] = True
    stack.append(vertex)

    for neighbor in edges[vertex]:
        if not visited[neighbor]:
            count = dfs(neighbor, visited, stack, edges, count)
        elif neighbor in stack:
            if neighbor != stack[-2] and neighbor == stack[0]:
                # print("Cycle found:", stack + [neighbor])
                count += 1

    stack.pop()
    visited[vertex] = False
    return count


def count_d_cycles(vertices, edges):
    """
    Count the total number of cycles in a directed graph (including self loop and two edges cycles)
    :param vertices: vertices of the graph
    :type vertices: np.ndarray
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: number of cycles
    """
    simple_cycles = count_simple_cycles(vertices, edges)
    self_loops = count_self_loop(edges)
    two_cycles = 0
    for e in edges:
        if any(np.equal(edges, [e[1], e[0]]).all(1)):
            two_cycles += 1
    two_cycles = (two_cycles - self_loops) // 2

    return simple_cycles + self_loops + two_cycles


def count_nd_cycles(vertices, edges):
    """
    Count the total number of cycles in an undirected graph
    :param vertices: vertices of the graph
    :type vertices: np.ndarray
    :param edges: edges of the graph
    :type edges: np.ndarray
    :return: number of cycles
    """
    return count_simple_cycles(vertices, edges) // 2

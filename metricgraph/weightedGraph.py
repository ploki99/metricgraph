import numpy as np
import vtk

from .graph import MetricGraph


class WeightedGraph(MetricGraph):
    """
    Extend MetricGraph adding weights on vertices and edges
    """

    def __init__(self, vertices, edges):
        """
        Build a WeightedGraph object
        :param vertices: vertices coordinates of the graph
        :type vertices: list | np.ndarray
        :param edges: connectivity matrix of the graph
        :type edges: list | np.ndarray
        """
        super().__init__(vertices, edges)
        self.__weights = dict()
        self.__types = dict()

    def __set_weight(self, name, value, nodal_weight):
        """
        Add a value that represent a weight
        :param name: name of the weight
        :type name: str
        :param value: value of the weight
        :type value: list | np.ndarray
        :param nodal_weight: if True, it's a nodal weight, else it's an edge weight
        """
        value = np.array(value).ravel()
        length = self.vertices.shape[0] if nodal_weight else self.edges.shape[0]
        if value.shape[0] != length:
            print(f"Warning: value shape should be ({length}, ) ")
            return
        self.__weights[name] = np.array(value)
        self.__types[name] = nodal_weight

    def set_edges_weight(self, name, value):
        """
        Add a weight on the edges
        :param name: name of the weight
        :type name: str
        :param value: value of the weight
        :type value: list | np.ndarray
        """
        self.__set_weight(name, value, False)

    def set_vertices_weight(self, name, value):
        """
        Add a weight on the edges vertices
        :param name: name of the weight
        :type name: str
        :param value: value of the weight
        :type value: list | np.ndarray
        """
        self.__set_weight(name, value, True)

    def get_weight(self, name):
        """
        Get a value that represent a weight, given its name
        :param name: name of the weight
        :type name: str
        :return: value of the weight
        :rtype: np.ndarray
        """
        return self.__weights.get(name)

    def get_weights_name(self):
        """
        Get all the weights' name
        :return: list of names
        :rtype: list[str]
        """
        return list(self.__weights)

    def is_nodal_weight(self, name):
        """
        Check if a value represent a nodal weight
        :param name: name of the weight
        :type name: str
        :return: True if a value represent a nodal weight, else False
        :rtype: bool
        """
        return self.__types.get(name)

    def plot_weight(self, name, dimension=None, interactive=False, legend_label="", max_value=None, min_value=None,
                    palette='viridis', show_axis=False, title=""):
        """
        Plot the graph with weights on vertices or edges
        :param name: name of the weight
        :type name: str
        :param dimension: dimension of plotted weight. For edges is the linewidth (default 3), for vertices is the nodal size (default 100)
        :type dimension: int | float
        :param interactive: if True, display the plot in interactive mode
        :param legend_label: label of the colormap
        :param max_value: maximum value to consider for the colormap, if None it's the actual maximum
        :type max_value: float
        :param min_value: minimum value to consider for the colormap, if None it's the actual minimum
        :type min_value: float
        :param palette: colormap palette
        :param show_axis: if True, display the axis of the figure
        :param title: title of the plot
        """
        if self.plotter is None:
            print("Cannot plot graph in dimension greater than 3")
        elif self.__weights.get(name) is None:
            print(f"Warning: weight {name} does not exist")
        elif self.__types.get(name):
            if dimension is None:
                dimension = 100
            self.plotter.plot_vertices_weight(self.__weights[name], interactive, legend_label, max_value, min_value,
                                              palette, show_axis, title, dimension)
        else:
            if dimension is None:
                dimension = 3
            self.plotter.plot_edges_weight(self.__weights[name], dimension, interactive, legend_label, max_value,
                                           min_value, palette, show_axis, title)

    def write_vtk(self, filename):
        """
        Write the WeightedGraph on a vtk file
        :param filename: name of the file (without extension)
        :type filename: str
        """
        # Create nodes
        points = vtk.vtkPoints()
        for p in self.vertices:
            points.InsertNextPoint(p)
        # Create edges
        lines = vtk.vtkCellArray()
        for e in self.edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, e[0])
            line.GetPointIds().SetId(1, e[1])
            lines.InsertNextCell(line)
        # Create vtkPolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        # Add vertices/edge quantities
        for name in self.get_weights_name():
            quantities = vtk.vtkFloatArray()
            quantities.SetName(name)
            quantities.SetNumberOfComponents(1)
            # Insert values
            for v in self.get_weight(name):
                quantities.InsertNextValue(v)
            # Check if it's a nodal weight
            if self.is_nodal_weight(name):
                polydata.GetPointData().AddArray(quantities)
            else:
                polydata.GetCellData().AddArray(quantities)
        # Write on file
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polydata)
        writer.SetFileName(f"{filename}.vtk")
        writer.Write()

    @classmethod
    def read_vtk(cls, filename):
        """
        Read the WeightedGraph from a vtk file
        :param filename: name of the file (without extension)
        :type filename: str
        :return: WeightedGraph
        """
        # Read vtk
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(f"{filename}.vtk")
        reader.Update()
        # Extract vertices and edges
        polydata = reader.GetOutput()
        vertices = np.array(polydata.GetPoints().GetData())
        edges = np.zeros((polydata.GetNumberOfCells(), 2), dtype=np.int64)
        for i in range(edges.shape[0]):
            edges[i] = [polydata.GetCell(i).GetPointId(0), polydata.GetCell(i).GetPointId(1)]
        # Build weighted graph
        graph = cls(vertices, edges)
        # Add node values
        for i in range(polydata.GetPointData().GetNumberOfArrays()):
            arr = polydata.GetPointData().GetArray(i)
            graph.set_vertices_weight(arr.GetName(), np.array(arr))
        # Add edge values
        for i in range(polydata.GetCellData().GetNumberOfArrays()):
            arr = polydata.GetCellData().GetArray(i)
            graph.set_edges_weight(arr.GetName(), np.array(arr))
        # Return object
        return graph

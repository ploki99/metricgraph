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
        self.__values = dict()
        self.__types = dict()

    def add_value(self, name, value, nodal_value=True):
        """
        Add a value that represent a weight
        :param name: name of the weight
        :type name: str
        :param value: value of the weight
        :type value: list | np.ndarray
        :param nodal_value: if True, it's a nodal weight, else it's an edge weight
        """
        self.__values[name] = np.array(value)
        self.__types[name] = nodal_value

    def get_value(self, name):
        """
        Get a value that represent a weight, given its name
        :param name: name of the weight
        :type name: str
        :return: value of the weight
        :rtype: np.ndarray
        """
        return self.__values.get(name)

    def get_values_name(self):
        """
        Get all the weights' name
        :return: list of names
        :rtype: list[str]
        """
        return list(self.__values)

    def is_nodal_value(self, name):
        """
        Check if a value represent a nodal weight
        :param name: name of the weight
        :type name: str
        :return: True if a value represent a nodal weight, else False
        :rtype: bool
        """
        return self.__types.get(name)

    def write_vtk(self, filename):
        """
        Write the WeightedGraph on a vtk file
        :param filename: name of the file
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
        for name in self.get_values_name():
            quantities = vtk.vtkFloatArray()
            quantities.SetName(name)
            quantities.SetNumberOfComponents(1)
            # Insert values
            for v in self.get_value(name):
                quantities.InsertNextValue(v)
            # Check if it's a nodal weight
            if self.is_nodal_value(name):
                polydata.GetPointData().AddArray(quantities)
            else:
                polydata.GetCellData().AddArray(quantities)
        # Write on file
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polydata)
        writer.SetFileName(filename)
        writer.Write()

    @classmethod
    def read_vtk(cls, filename):
        """
        Read the WeightedGraph from a vtk file
        :param filename: name of the file
        :type filename: str
        :return: WeightedGraph
        """
        # Read vtk
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
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
            graph.add_value(arr.GetName(), np.array(arr))
        # Add edge values
        for i in range(polydata.GetCellData().GetNumberOfArrays()):
            arr = polydata.GetCellData().GetArray(i)
            graph.add_value(arr.GetName(), np.array(arr), False)
        # Return object
        return graph

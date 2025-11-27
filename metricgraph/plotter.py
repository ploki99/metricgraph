from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


class Plotter(ABC):
    def __init__(self, vertices, edges):
        """
        Constructor for Plotter class: collection of methods to easily plot domain and solution
        :type vertices: np.ndarray
        :type edges: np.ndarray
        """
        self.vertices = vertices
        # Save edges without self loops
        tmp = np.array(edges)
        mask = tmp[:, 0] != tmp[:, 1]
        self.edges = tmp[mask]
        self.loops = tmp[~mask]
        # Bounding box
        self.x_min, self.x_max = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        self.y_min, self.y_max = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        # Minimum space for self loops
        self.x_min -= 0.12
        self.x_max += 0.12
        self.y_min -= 0.12
        self.y_max += 0.12

    def _finalize_figure(self, fig, ax, title, interactive, show_axis, collection=None, legend_label=""):
        """
        Perform the final configuration needed to display the figure
        """
        # Set labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # Set limits
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        # Title
        plt.title(title)
        # Add color bar
        if collection is not None:
            cbar = fig.colorbar(collection, ax=ax, pad=0.1)
            cbar.set_label(legend_label)
        # Show plot
        if interactive:
            plt.ion()
        else:
            plt.ioff()
        # Show axis
        if show_axis:
            plt.axis('on')
        else:
            plt.axis('off')
        plt.show()

    # Abstract methods

    @abstractmethod
    def _initialize_figure(self):
        pass

    @abstractmethod
    def _plot_arrows(self, ax, color, linewidths, vertices_label):
        pass

    @abstractmethod
    def _plot_vertices(self, ax, vert_label=False, **kwargs):
        pass

    @abstractmethod
    def _plot_edges(self, ax, **kwargs):
        pass

    @abstractmethod
    def _plot_self_loops(self, ax, color, linewidths, vertices_label):
        pass

    # Public interface

    def plot_graph(self, color="blue", directed=False, interactive=False, line_widths=2, show_axis=False, title="",
                   vertices_label=True):
        """
        Plot image with graph representation.
        :param color: color to use for plotting the edges
        :param directed: if True, plot the graph using arrows
        :param interactive: if True, display the plot in interactive mode
        :param line_widths: width of the lines
        :param show_axis: if True, display the axis of the figure
        :param title: title of the plot
        :param vertices_label: if True, label the vertices of the graph
        """
        # Initialize figure
        fig, ax = self._initialize_figure()
        # Plot edges
        if directed:
            self._plot_arrows(ax, color, line_widths, vertices_label)
        else:
            self._plot_edges(ax, color=color, linewidths=line_widths)
        # Plot self loops
        self._plot_self_loops(ax, color, line_widths, vertices_label)
        # Plot vertices
        self._plot_vertices(ax, vertices_label, c=color, alpha=1)
        # Display figure
        self._finalize_figure(fig, ax, title, interactive, show_axis)

    def plot_edges_weight(self, quantity, edge_width=3, interactive=False, legend_label="", max_value=None,
                          min_value=None, palette='viridis', show_axis=False, title=""):
        """
        Plot edges weight
        :param quantity: vector of weights to plot on edges
        :type quantity: np.ndarray
        :param edge_width: width to use for the edges
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
        if min_value is None:
            min_value = np.min(quantity)
        if max_value is None:
            max_value = np.max(quantity)
        # Initialize figure
        fig, ax = self._initialize_figure()
        # Add edges
        lc = self._plot_edges(ax, cmap=palette, linewidths=edge_width, array=quantity)
        lc.set_clim(vmin=min_value, vmax=max_value)
        # Display figure
        self._finalize_figure(fig, ax, title, interactive, show_axis, lc, legend_label)

    def plot_vertices_weight(self, quantity, interactive=False, legend_label="", max_value=None, min_value=None,
                             palette='viridis', show_axis=False, title="", vertices_size=100):
        """
        Plot vertices weight
        :param quantity: vector of weights to plot on edges
        :type quantity: np.ndarray
        :param interactive: if True, display the plot in interactive mode
        :param legend_label: label of the colormap
        :param max_value: maximum value to consider for the colormap, if None it's the actual maximum
        :type max_value: float
        :param min_value: minimum value to consider for the colormap, if None it's the actual minimum
        :type min_value: float
        :param palette: colormap palette
        :param show_axis: if True, display the axis of the figure
        :param title: title of the plot
        :param vertices_size: size of the vertices in the plot
        """
        if min_value is None:
            min_value = np.min(quantity)
        if max_value is None:
            max_value = np.max(quantity)
        # Initialize figure
        fig, ax = self._initialize_figure()
        # Plot edges
        self._plot_edges(ax, color='dimgray', linewidths=0.5)
        # Plot points
        points = self._plot_vertices(ax, c=quantity, cmap=palette, alpha=1, vmin=min_value, vmax=max_value,
                                     s=vertices_size, zorder=2)
        # Display figure
        self._finalize_figure(fig, ax, title, interactive, show_axis, points, legend_label)


class Plotter2d(Plotter):
    def __init__(self, vertices, edges):
        super().__init__(vertices, edges)

    def _initialize_figure(self):
        """
        Initialize the figure in 2D
        """
        return plt.subplots()

    def _plot_arrows(self, ax, color, linewidths, vertices_label):
        """
        Plot directed edges as arrows (2D)
        """
        # Distance coefficient
        alpha = 0.08 if vertices_label else 0.01
        # Get values
        x0, y0, dx, dy = [], [], [], []
        for e in self.edges:
            x_s, y_s = self.vertices[e[0]]
            x_t, y_t = self.vertices[e[1]]
            d1, d2 = x_t - x_s, y_t - y_s
            x0.append(x_s + d1 * alpha)
            y0.append(y_s + d2 * alpha)
            dx.append(d1 * (1 - 2 * alpha))
            dy.append(d2 * (1 - 2 * alpha))
        # Plot arrows
        ax.quiver(x0, y0, dx, dy, angles='xy', scale_units='xy', scale=1, width=0.0015 * linewidths,
                  headwidth=4, headlength=4, color=color)

    def _plot_vertices(self, ax, vert_label=False, **kwargs):
        """
        Plot vertices in 2D
        """
        if vert_label:
            ret = ax.scatter(self.vertices[:, 0], self.vertices[:, 1], s=600, zorder=2, facecolors='white',
                             edgecolors=kwargs['c'])
            for i, v in enumerate(self.vertices):
                ax.text(v[0], v[1], i, fontsize=12, color=kwargs['c'], ha='center', va='center')
        else:
            ret = ax.scatter(self.vertices[:, 0], self.vertices[:, 1], **kwargs)
        return ret

    def _plot_edges(self, ax, **kwargs):
        """
        Plot edges in an efficient way using LineCollection (2D)
        """
        # Extract line segments
        xs, ys = [], []
        for e in self.edges:
            x0, y0 = self.vertices[e[0]]
            x1, y1 = self.vertices[e[1]]
            xs.append([x0, x1])
            ys.append([y0, y1])
        segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
        # Create LineCollection (2D)
        line_segments = LineCollection(segments, **kwargs)
        # Add to axes
        ax.add_collection(line_segments)
        return line_segments

    def _plot_self_loops(self, ax, color, linewidths, vertices_label):
        """
        Draw self loops in (2D).
        """
        # Circle radius
        radius = 0.05
        for e in self.loops:
            x0, y0 = self.vertices[e[0]]
            # Compute circle
            theta = np.linspace(0, np.pi * (1.7 if vertices_label else 1.8), 50)
            x = x0 + radius * np.cos(theta) - radius
            y = y0 + radius * np.sin(theta)
            # Plot circle
            ax.plot(x, y, color=color, lw=linewidths)
            # Final points as arrow
            ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5 * linewidths))


class Plotter3d(Plotter):
    def __init__(self, vertices, edges):
        super().__init__(vertices, edges)
        # Bounding box (adding z coordinate)
        self.z_min, self.z_max = min(self.vertices[:, 2]), max(self.vertices[:, 2])
        # Minimum space for self loops
        self.z_min -= 0.12
        self.z_max += 0.12

    def _finalize_figure(self, fig, ax, title, interactive, show_axis, collection=None, legend_label=""):
        """
        Perform the final configuration needed to display the figure in 3d (add zlabel and zlim)
        """
        # Add z label and limits
        ax.set_zlabel("z")
        ax.set_zlim(self.z_min, self.z_max)
        super()._finalize_figure(fig, ax, title, interactive, show_axis, collection, legend_label)

    def _initialize_figure(self):
        """
        Initialize the figure in 3D
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        return fig, ax

    def _plot_arrows(self, ax, color, linewidths, vertices_label):
        """
        Plot directed edges as arrows (3D)
        """
        # Distance coefficient
        alpha = 0.1 if vertices_label else 0.01
        # Get values
        x0, y0, z0, dx, dy, dz = [], [], [], [], [], []
        for e in self.edges:
            x_s, y_s, z_s = self.vertices[e[0]]
            x_t, y_t, z_t = self.vertices[e[1]]
            d1, d2, d3 = x_t - x_s, y_t - y_s, z_s - z_t
            x0.append(x_s + d1 * alpha)
            y0.append(y_s + d2 * alpha)
            z0.append(z_s + d3 * alpha)
            dx.append(d1 * (1 - 2 * alpha))
            dy.append(d2 * (1 - 2 * alpha))
            dz.append(d3 * (1 - 2 * alpha))
        # Plot arrows
        ax.quiver(x0, y0, z0, dx, dy, dz, arrow_length_ratio=0.05, length=1,
                  color=color, linewidth=linewidths)

    def _plot_vertices(self, ax, vert_label=False, **kwargs):
        """
        Plot vertices in 3D
        """
        if vert_label:
            ret = ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], s=500, zorder=2, alpha=1,
                             facecolors='white', edgecolors=kwargs['c'])
            for i, v in enumerate(self.vertices):
                ax.text(v[0], v[1], v[2], i, fontsize=12, color=kwargs['c'], zorder=1e6, ha='center', va='center')
        else:
            ret = ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], **kwargs)
        return ret

    def _plot_edges(self, ax, **kwargs):
        """
        Plot edges in an efficient way, with the possibility of creating a colormap
        """
        radius = 0.05
        # Extract line segments
        xs, ys, zs = [], [], []
        for e in self.edges:
            x0, y0, z0 = self.vertices[e[0]]
            x1, y1, z1 = self.vertices[e[1]]
            xs.append([x0, x1])
            ys.append([y0, y1])
            zs.append([z0, z1])
        segments = [np.column_stack([x, y, z]) for x, y, z in zip(xs, ys, zs)]
        # Create LineCollection (3D)
        line_segments = Line3DCollection(segments, **kwargs)
        # Add collection
        ax.add_collection(line_segments)
        # Return collection
        return line_segments

    def _plot_self_loops(self, ax, color, linewidths, vertices_label):
        """
        Draw self loops in (3D).
        """
        # Circle radius
        radius = 0.1
        for e in self.loops:
            x0, y0, z0 = self.vertices[e[0]]
            # Compute circle
            theta = np.linspace(0, np.pi * (1.7 if vertices_label else 1.8), 50)
            x = x0 + radius * np.cos(theta) - radius
            y = y0 + radius * np.sin(theta)
            z = np.full_like(theta, z0)
            # Plot circle
            ax.plot(x, y, z, color=color, lw=linewidths)
            # Final arrow with quiver
            xs, ys, zs = x[-2], y[-2], z[-2]
            xt, yt, zt = x[-1], y[-1], z[-1]
            ax.quiver(xs, ys, zs, xt - xs, yt - ys, zt - zs, arrow_length_ratio=2, color=color, linewidth=linewidths)

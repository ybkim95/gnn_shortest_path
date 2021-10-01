import numpy as np
import networkx as nx
import collections
import matplotlib.pyplot as plt

""" Graph Plotter
"""
class GraphPlotter(object):
    def __init__(self, ax, graph, pos):
        self._ax = ax
        self._graph = graph
        self._pos = pos
        self._base_draw_kwargs = dict(G=self._graph, pos=self._pos, ax=self._ax)
        self._solution_length = None
        self._nodes = None
        self._edges = None
        self._start_nodes = None
        self._end_nodes = None
        self._solution_nodes = None
        self._intermediate_solution_nodes = None
        self._solution_edges = None
        self._non_solution_nodes = None
        self._non_solution_edges = None
        self._ax.set_axis_off()

    @property
    def solution_length(self):
        if self._solution_length is None:
            self._solution_length = len(self._solution_edges)
        return self._solution_length

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._graph.nodes()
        return self._nodes

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._graph.edges()
        return self._edges

    @property
    def start_nodes(self):
        if self._start_nodes is None:
            self._start_nodes = [n for n in self.nodes if self._graph.nodes[n].get("start", False)]
        return self._start_nodes

    @property
    def end_nodes(self):
        if self._end_nodes is None:
            self._end_nodes = [n for n in self.nodes if self._graph.nodes[n].get("end", False)]
        return self._end_nodes

    @property
    def solution_nodes(self):
        if self._solution_nodes is None:
            self._solution_nodes = [n for n in self.nodes if self._graph.nodes[n].get("solution", False)]
        return self._solution_nodes

    @property
    def intermediate_solution_nodes(self):
        if self._intermediate_solution_nodes is None:
            self._intermediate_solution_nodes = [n for n in self.nodes if self._graph.nodes[n].get("solution", False) and not self._graph.nodes[n].get("start", False) and 
                                                 not self._graph.nodes[n].get("end", False)]
        return self._intermediate_solution_nodes

    @property
    def solution_edges(self):
        if self._solution_edges is None:
            self._solution_edges = [e for e in self.edges if self._graph.get_edge_data(e[0], e[1]).get("solution", False)]
        return self._solution_edges

    @property
    def non_solution_nodes(self):
        if self._non_solution_nodes is None:
            self._non_solution_nodes = [n for n in self.nodes if not self._graph.nodes[n].get("solution", False)]
        return self._non_solution_nodes

    @property
    def non_solution_edges(self):
        if self._non_solution_edges is None:
            self._non_solution_edges = [e for e in self.edges if not self._graph.get_edge_data(e[0], e[1]).get("solution", False)]
        return self._non_solution_edges

    def _make_draw_kwargs(self, **kwargs):
        kwargs.update(self._base_draw_kwargs)
        return kwargs

    def _draw(self, draw_function, zorder=None, **kwargs):
        draw_kwargs = self._make_draw_kwargs(**kwargs)
        collection = draw_function(**draw_kwargs)
        if collection is not None and zorder is not None:
            try:
                # This is for compatibility with older matplotlib.
                collection.set_zorder(zorder)
            except AttributeError:
                # This is for compatibility with newer matplotlib.
                collection[0].set_zorder(zorder)
        return collection


    def draw_nodes(self, **kwargs):
        if ("node_color" in kwargs and isinstance(kwargs["node_color"], collections.Sequence) and len(kwargs["node_color"]) in {3, 4} and
            not isinstance(kwargs["node_color"][0],(collections.Sequence, np.ndarray))):
            
            num_nodes = len(kwargs.get("nodelist", self.nodes))
            kwargs["node_color"] = np.tile(np.array(kwargs["node_color"])[None], [num_nodes, 1])

        return self._draw(nx.draw_networkx_nodes, **kwargs)


    def draw_edges(self, **kwargs):
        """Useful kwargs: edgelist, width."""
        
        return self._draw(nx.draw_networkx_edges, **kwargs)


    def draw_graph(self, node_size=200, node_color=(1.0, 1.0, 1.0), node_linewidth=1.0, edge_width=1.0):
        # 1) 정점 그리기 
        self.draw_nodes(nodelist=self.nodes, node_size=node_size, node_color=node_color, linewidths=node_linewidth, zorder=20)
        # 2) 간선 그리기.
        self.draw_edges(edgelist=self.edges, width=edge_width, zorder=10)


    def draw_graph_with_solution(self, node_size=200, node_color=(1.0, 1.0, 1.0), node_linewidth=1.0, edge_width=1.0, start_color="g", end_color="r", solution_node_linewidth=3.0, solution_edge_width=3.0):
        node_border_color = (0.0, 0.0, 0.0, 1.0)
        node_collections = {}
        
        # END 노드 plot
        node_collections["start nodes"] = self.draw_nodes(nodelist=self.start_nodes,
                                                            node_size=node_size,
                                                            node_color=start_color,
                                                            linewidths=solution_node_linewidth,
                                                            edgecolors=node_border_color,
                                                            zorder=100)                                      
        # END 라벨 
        x1,y1=self._pos[self.end_nodes[0]] # TODO
        plt.text(x1+0.16,y1-0.03, s='END', bbox=dict(facecolor='red', alpha=0.5),horizontalalignment='center')
        # nx.draw_networkx_labels(self._graph, self._pos, {n: "START" for n in self.start_nodes}, font_size=15) 
        
        # START 노드 plot
        node_collections["end nodes"] = self.draw_nodes(nodelist=self.end_nodes,
                                                            node_size=node_size,
                                                            node_color=end_color,
                                                            linewidths=solution_node_linewidth,
                                                            edgecolors=node_border_color,
                                                            zorder=90)
        # START 라벨                                              
        x2,y2=self._pos[self.start_nodes[0]] # TODO 
        plt.text(x2+0.08,y2-0.03, s='START', bbox=dict(facecolor='green', alpha=0.5),horizontalalignment='center')
        # nx.draw_networkx_labels(self._graph, self._pos, {n: "END" for n in self.end_nodes}, font_size=15) 
        
        # Plot intermediate solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.intermediate_solution_nodes]
        else:
            c = node_color
            node_collections["intermediate solution nodes"] = self.draw_nodes(nodelist=self.intermediate_solution_nodes,
                                                                                node_size=node_size,
                                                                                node_color=c,
                                                                                linewidths=solution_node_linewidth,
                                                                                edgecolors=(node_border_color),
                                                                                zorder=80
                                                                              )
        # Plot solution edges.
        node_collections["solution edges"] = self.draw_edges(edgelist=self.solution_edges, width=solution_edge_width, edge_color = 'r', zorder=70)
        
        # Plot non-solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.non_solution_nodes]
        else:
            c = node_color
            node_collections["non-solution nodes"] = self.draw_nodes(nodelist=self.non_solution_nodes,
                                                                        node_size=node_size,
                                                                        node_color=c,
                                                                        linewidths=node_linewidth,
                                                                        edgecolors=node_border_color,
                                                                        zorder=20
                                                                     )
        # Plot non-solution edges.
        node_collections["non-solution edges"] = self.draw_edges(edgelist=self.non_solution_edges, width=edge_width, zorder=10)
        
        # Set title as solution length.
        self._ax.set_title("Solution length: {}".format(self.solution_length))
        
        return node_collections
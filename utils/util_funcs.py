import itertools
import networkx as nx
import numpy as np
import collections

from .a_star import *

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def set_diff(seq0, seq1):
    return list(set(seq0) - set(seq1))

def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot

def get_node_dict(graph, attr):
    return {k: v[attr] for k, v in graph.nodes.items()}
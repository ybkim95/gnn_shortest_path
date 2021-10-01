from heapq import heappush, heappop
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function # TODO
import networkx as nx
import collections

from .util_funcs import *

DISTANCE_WEIGHT_NAME = "distance"

""" Shortest paths and path lengths using the A* ("A star") algorithm.
    중요한건 Connected & Directed Graph임이 반영되야한다.
"""

def a_star_shortest_path(DG, start, end, heuristic = None, weight = "weight"):
    """ 최단 경로 탐색 알고리즘 중 A* 알고리즘은 시작 노드와 목적지 노드를 분명하게 지정해 두 노드 간의 최단 경로를 파악한다. 
        (*시작 노드를 지정후 다른 모든 노드에 대한 최대나 경로를 파악하는 Dijkstra 알고리즘과는 다름) 
        * DG: Directed Graph
        * start: start node
        * end: end node 
    """
    # 시작점과 도착점이 그래프 안에 존재하는지 체크
    # [중요] 여기에 추가적으로, Directed Graph인 상황이기 때문에 최소한 1개의 path가 존재하는지 살펴봐야 함
    if start not in DG or end not in DG:
        message = "Either start node {} or end node {} is not in Graph G".format(start, end)
        raise nx.NodeNotFound(message)

    """ A* 알고리즘은 휴리스틱 추정값을 통해 알고리즘을 개선할 수 있는데, 이러한 추정값을 어떤 방식으로 
        제공하는지에 따라 얼마나 빠르게 최단 경로를 파악할 수 있는 지 결정된다고 함.
    """
    # H: Heuristic (F = G + H)
    if heuristic is None:
        # The default heuristic is h=0 -> same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(DG, weight) # TODO

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    #  attempting to compare the nodes themselves. The hash breaks ties in the
    #  priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), start, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    #  computed heuristics to target. We avoid computing the heuristics
    #  more than once and inserting the node into the queue too many times.
    open_list = {}
    # Maps explored nodes to parent closest to the source.
    close_list = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, cur_node, dist, parent = pop(queue)

        # 경유지들을 거쳐 end node에 마침내 도달했을 때
        if cur_node == end:
            path = [cur_node] # 이제 여기서부터 parent를 찾아 나가는 것
            node = parent # 나의 부모를 역추적해 나가서 path에 추가
            while node is not None:
                path.append(node)
                node = close_list[node]
            path.reverse() # end node에서부터 부모를 찾아가므로 reverse를 해줘야 start -> end 방향의 path가 완성됨
            return path


        if cur_node in close_list:
            # Do not override the parent of starting node
            if close_list[cur_node] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, H_Score = open_list[cur_node]
            if qcost < dist:
                continue

        close_list[cur_node] = parent # 다시 돌아가기 

        # 현재 node를 중심으로 이어진 이웃 node들에 대해서 F, G, H Score를 계산하여 비교하는 과정
        for neighbor, w in DG[cur_node].items():
            # DG는 바로 직전 노드에 소요되는 비용 + 직전 노드에서 현 노드에 도달하기 까지의 비용을 추가해주면됨
            DG_Score = dist + weight(cur_node, neighbor, w)
            if neighbor in open_list:
                prev_DG_Score, H_Score = open_list[neighbor]
                # if qcost <= ncost, a less costly path from the
                #  neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                #  to the queue
                # 이미 저장되어있던 것이 더 작은 값을 가지고 있다면! (어차피 H 값은 같을 것이므로, G_Score만 비교하면 된다)
                if prev_DG_Score <= DG_Score:
                    continue
            else:
                H_Score = heuristic(neighbor, end) # H = 0
            
            open_list[neighbor] = DG_Score, H_Score
            F_Score = DG_Score + H_Score
            push(queue, (F_Score, next(c), neighbor, DG_Score, cur_node)) # 이렇게 하겠다는 것은 cur_node가 이제 부모 역할을 하고 neighbor로 이동하겠다는 뜻이므로

    raise nx.NetworkXNoPath("There exist no path between from node {} to node {}".format(start, end))

def a_star_shortest_path_length(Graph, start, end, heuristic=None, weight="weight"):
    if start not in Graph or end not in Graph:
        msg = "Either source {} or target {} is not in G".format(start, end)
        raise nx.NodeNotFound(msg)

    weight = _weight_function(Graph, weight)
    path = a_star_shortest_path(Graph, start, end, heuristic, weight)
    return sum(weight(u, v, Graph[u][v]) for u, v in zip(path[:-1], path[1:]))


def add_shortest_path(rand, graph, min_length=1):
    pair_to_length_dict = {}
    # Map from node pairs to the length of their shortest path.

    # a star로부터 기인
    lengths = []
    for start in graph.nodes:
        temp = {}
        for end in graph.nodes:
            dis = a_star_shortest_path_length(graph, start, end)
            temp[end] = dis
        lengths.append((start, temp))

    # try:
    #     # This is for compatibility with older networkx.
    #     lengths = nx.all_pairs_shortest_path_length(graph).items()    # TODO -> 여기를 a_star로 대체해야 하는 것!
    # except AttributeError:
    #     # This is for compatibility with newer networkx.
    #     lengths = list(nx.all_pairs_shortest_path_length(graph))

    for x, yy in lengths:
        for y, l in yy.items():
            if l >= min_length:
                pair_to_length_dict[x, y] = l
    if max(pair_to_length_dict.values()) < min_length:
        raise ValueError("All shortest paths are below the minimum length")
    # The node pairs which exceed the minimum length.
    node_pairs = list(pair_to_length_dict)

    # Computes probabilities per pair, to enforce uniform sampling of each
    # shortest path lengths.
    # The counts of pairs per length.
    counts = collections.Counter(pair_to_length_dict.values())
    prob_per_length = 1.0 / len(counts)
    probabilities = [prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs]

    # Choose the start and end points.
    i = rand.choice(len(node_pairs), p=probabilities)
    start, end = node_pairs[i]
    # path = nx.shortest_path(graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

    # A* 알고리즘으로부터 나온 path
    shortest_path = a_star_shortest_path(graph, start, end, heuristic=None, weight=DISTANCE_WEIGHT_NAME)

    # Creates a directed graph, to store the directed path from start to end.
    digraph = graph.to_directed()

    # Add the "start", "end", and "solution" attributes to the nodes and edges.
    digraph.add_node(start, start=True)
    digraph.add_node(end, end=True)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), shortest_path), solution=False)
    digraph.add_nodes_from(shortest_path, solution=True)
    path_edges = list(pairwise(shortest_path))
    digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
    digraph.add_edges_from(path_edges, solution=True)

    return digraph
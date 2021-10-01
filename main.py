import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from termcolor import colored

import networkx as nx
import sonnet as snt

from graph_nets import graphs, utils_np, utils_tf
from graph_nets.demos_tf2 import models
from scipy import spatial

from utils.a_star import *
from utils.util_funcs import *
from utils.graph_plotter import *

DISTANCE_WEIGHT_NAME = "distance"

""" Generate Graph
"""
# Directed & Connected Graph 생성
def generate_graph(rand, num_nodes_min_max, dimensions=2, theta=1000.0, rate=1.0):
    # node 개수를 random하게 결정
    num_nodes = rand.randint(*num_nodes_min_max)

    # Create GTG. 이 부분이 핵심 -> networkx 문서를 참고하면 사용법 나와있음
    pos_array = rand.uniform(size=(num_nodes, dimensions))
    pos = dict(enumerate(pos_array))
    weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
    geo_graph = nx.geographical_threshold_graph(num_nodes, theta, pos=pos, weight=weight) # 여기서도 역시 networkx를 기반으로 하였다. 

    # Create MST across geo_graph's nodes.
    # scipy.spatial.distance -> distance computations에 대한 function reference를 담고 있음
    # scipy.spatial.pdist -> pairwise distance인데 euclidian이 default임
    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array)) # adj matrix (n x n matrix)
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel())) # 이제 i,j 노드별 distance를 다 random하게 지정함
    
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
    mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME) # MST 생성
    
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
        mst_graph.nodes[i].update(geo_graph.nodes[i])
        # print(geo_graph.nodes[i])

    # Compose the graphs.
    combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
    
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
        combined_graph.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME, distances[i, j])
    
    return combined_graph, mst_graph, geo_graph


def graph_to_input_target(graph):
    """ train을 하기 위해 완성된 graph를 input과 target으로 나누는 과정
    """
    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("pos", "weight", "start", "end")
    input_edge_fields = ("distance",)
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    solution_length = 0
    # Graph의 nodes
    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(node_index, features=create_feature(node_feature, input_node_fields))
        target_node = to_one_hot(create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.add_node(node_index, features=target_node)
        solution_length += int(node_feature["solution"])
    solution_length /= graph.number_of_nodes()
    # Graph의 edges
    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(sender, receiver, features=create_feature(features, input_edge_fields))
        target_edge = to_one_hot(create_feature(features, target_edge_fields).astype(int), 2)[0]
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([solution_length], dtype=float)

    return input_graph, target_graph


# Training을 위한 graph 생성 함수
def generate_networkx_graphs(rand, num_examples, num_nodes_min_max, theta):
    input_graphs = []
    target_graphs = []
    graphs = []
    for _ in range(num_examples):
        graph = generate_graph(rand, num_nodes_min_max, theta=theta)[0] # 바로 위에 정의된 함수 이용
        graph = add_shortest_path(rand, graph)
        
        input_graph, target_graph = graph_to_input_target(graph) # input과 target을 생성
        input_graphs.append(input_graph)
        target_graphs.append(target_graph)
        graphs.append(graph)
    
    return input_graphs, target_graphs, graphs


def create_graph_dicts_tf(num_examples, num_elements_min_max):
    num_elements = tf.random.uniform(
        [num_examples],
        minval=num_elements_min_max[0],
        maxval=num_elements_min_max[1],
        dtype=tf.int32)
    inputs_graphs = []
    sort_indices_graphs = []
    ranks_graphs = []
    for i in range(num_examples):
        values = tf.random.uniform(shape=[num_elements[i]])
        sort_indices = tf.cast(
            tf.argsort(values, axis=-1), tf.float32)
        ranks = tf.cast(
            tf.argsort(sort_indices, axis=-1), tf.float32) / (
                tf.cast(num_elements[i], tf.float32) - 1.0)
        inputs_graphs.append({"nodes": values[:, None]})
        sort_indices_graphs.append({"nodes": sort_indices[:, None]})
        ranks_graphs.append({"nodes": ranks[:, None]})
    return inputs_graphs, sort_indices_graphs, ranks_graphs


def create_linked_list_target(batch_size, input_graphs):
    target_graphs = []
    for i in range(batch_size):
        input_graph = utils_tf.get_graph(input_graphs, i)
        num_elements = tf.shape(input_graph.nodes)[0]
        si = tf.cast(tf.squeeze(input_graph.nodes), tf.int32)
        nodes = tf.reshape(tf.one_hot(si[:1], num_elements), (-1, 1))
        x = tf.stack((si[:-1], si[1:]))[None]
        y = tf.stack(
            (input_graph.senders, input_graph.receivers), axis=1)[:, :, None]
        edges = tf.reshape(
            tf.cast(
                tf.reduce_any(tf.reduce_all(tf.equal(x, y), axis=1), axis=1),
                tf.float32), (-1, 1))
        target_graphs.append(input_graph._replace(nodes=nodes, edges=edges))
    return utils_tf.concat(target_graphs, axis=0)


def compute_accuracy(target, output):
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        num_elements = td["nodes"].shape[0]
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)

        xe = np.reshape(np.argmax(np.reshape(td["edges"], (num_elements, num_elements, 2)), axis=-1), (-1,))
        ye = np.reshape(np.argmax(np.reshape(od["edges"], (num_elements, num_elements, 2)), axis=-1), (-1,))
        c = np.concatenate((xn == yn, xe == ye), axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved


def create_data(batch_size, num_elements_min_max):
    inputs, sort_indices, ranks = create_graph_dicts_tf(batch_size, num_elements_min_max)
    inputs = utils_tf.data_dicts_to_graphs_tuple(inputs)
    sort_indices = utils_tf.data_dicts_to_graphs_tuple(sort_indices)
    ranks = utils_tf.data_dicts_to_graphs_tuple(ranks)

    inputs = utils_tf.fully_connect_graph_dynamic(inputs)
    sort_indices = utils_tf.fully_connect_graph_dynamic(sort_indices)
    ranks = utils_tf.fully_connect_graph_dynamic(ranks)

    targets = create_linked_list_target(batch_size, sort_indices)
    nodes = tf.concat((targets.nodes, 1.0 - targets.nodes), axis=1)
    edges = tf.concat((targets.edges, 1.0 - targets.edges), axis=1)
    targets = targets._replace(nodes=nodes, edges=edges)

    return inputs, targets, sort_indices, ranks

""" Loss Function: Softmax Cross-Entropy(Categorical Cross-Entropy Loss)
    Loss가 정의된 방식을 잘 봐보자.
"""
def create_loss(target, outputs):
    losss = [
        tf.compat.v1.losses.softmax_cross_entropy(target.nodes, output.nodes) + tf.compat.v1.losses.softmax_cross_entropy(target.edges, output.edges)
        for output in outputs
    ]

    return tf.stack(losss)


def plot_linked_list(ax, graph, sort_indices):
    nd = len(graph.nodes())
    probs = np.zeros((nd, nd))
    for edge in graph.edges(data=True):
        probs[edge[0], edge[1]] = edge[2]["features"][0]
    ax.matshow(probs[sort_indices][:, sort_indices], cmap="viridis")
    ax.grid(False)

# Data.
@tf.function
def get_data():
    inputs_tr, targets_tr, sort_indices_tr, _ = create_data(batch_size_tr, num_elements_min_max_tr)
    inputs_tr = utils_tf.set_zero_edge_features(inputs_tr, 1)
    inputs_tr = utils_tf.set_zero_global_features(inputs_tr, 1)
    
    # Test/generalization.
    inputs_ge, targets_ge, sort_indices_ge, _ = create_data(batch_size_ge, num_elements_min_max_ge)
    inputs_ge = utils_tf.set_zero_edge_features(inputs_ge, 1)
    inputs_ge = utils_tf.set_zero_global_features(inputs_ge, 1)

    targets_tr = utils_tf.set_zero_global_features(targets_tr, 1)
    targets_ge = utils_tf.set_zero_global_features(targets_ge, 1)

    return inputs_tr, targets_tr, sort_indices_tr, inputs_ge, targets_ge, sort_indices_ge


# Train
def update_step(inputs_tr, targets_tr):
    with tf.GradientTape() as tape:
        outputs_tr = model(inputs_tr, num_processing_steps_tr)
        # Loss.
        loss_tr = create_loss(targets_tr, outputs_tr)
        loss_tr = tf.math.reduce_sum(loss_tr) / num_processing_steps_tr

    gradients = tape.gradient(loss_tr, model.trainable_variables)
    optimizer.apply(gradients, model.trainable_variables)
    
    return outputs_tr, loss_tr


def softmax_prob_last_dim(x):  # pylint: disable=redefined-outer-name
    e = np.exp(x)
    return e[:, -1] / np.sum(e, axis=-1)


def create_directory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("[ERROR] Creating directory", dir)


if __name__ == "__main__":
    SEED = 123  
    rand = np.random.RandomState(seed=SEED)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    timestr = 'results/' + timestr
    create_directory(timestr)

    """ Sample Graph Plots
    """
    num_examples = 20 # 만들고 싶은 graph case의 개수

    # Large values (1000+) make trees. Try 20-60 for good non-trees.
    theta = 20 
    num_nodes_min_max = (7, 15)

    input_graphs, target_graphs, graphs = generate_networkx_graphs(rand, num_examples, num_nodes_min_max, theta)

    num = min(num_examples, 16)
    w = 5
    h = int(np.ceil(num / w))
    fig = plt.figure(40, figsize=(w * 4, h * 4))
    fig.clf()
    for j, graph in enumerate(graphs):
        ax = fig.add_subplot(h, w, j + 1)
        pos = get_node_dict(graph, "pos")
        plotter = GraphPlotter(ax, graph, pos)
        plotter.draw_graph_with_solution()
    
    fig.savefig(timestr + '/sample.png')
    
    SEED = 1234
    np.random.RandomState(SEED)
    tf.random.set_seed(SEED)

    theta = 20  # Large values (1000+) make trees. Try 20-60 for good non-trees.
    num_nodes_min_max = (10, 15)

    # input, target data 생성
    inputs, targets, sort_indices, ranks = create_data(1, num_nodes_min_max)

    inputs_nodes = inputs.nodes.numpy()
    targets = utils_tf.nest_to_numpy(targets)
    sort_indices_nodes = sort_indices.nodes.numpy()
    ranks_nodes = ranks.nodes.numpy()
    sort_indices = np.squeeze(sort_indices_nodes).astype(int)

    # Model parameters.
    # Number of processing (message-passing) steps.
    num_processing_steps_tr = 10
    num_processing_steps_ge = 10

    """ Data / training parameters.
    """
    num_training_iterations = 50000
    batch_size_tr = 32
    batch_size_ge = 100

    # Number of elements in each list is sampled uniformly from this range.
    num_elements_min_max_tr = (8, 17)
    num_elements_min_max_ge = (16, 33)

    # Optimizer.
    learning_rate = 1e-3
    optimizer = snt.optimizers.Adam(learning_rate)

    ## EncodeProcessDecode
    # https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos_tf2/models.py
    model = models.EncodeProcessDecode(edge_output_size=2, node_output_size=2)
    last_iteration = 0
    logged_iterations = []
    losses_tr = []
    corrects_tr = []
    solveds_tr = []
    losses_ge = []
    corrects_ge = []
    solveds_ge = []

    # Dummy Code
    # Get some example data that resembles the tensors that will be fed
    # into update_step():
    example_input_data, example_target_data = get_data()[:2]

    # Get the input signature for that function by obtaining the specs
    input_signature = [
    utils_tf.specs_from_graphs_tuple(example_input_data),
    utils_tf.specs_from_graphs_tuple(example_target_data)
    ]

    # Compile the update function using the input signature for speedy code.
    compiled_update_step = tf.function(update_step, input_signature=input_signature)

    import warnings
    warnings.filterwarnings('ignore')

    # How much time between logging and printing the current results.
    LOG_INTERVALS = 20

    start_time = time.time()
    last_log_time = start_time
    for iteration in range(last_iteration, num_training_iterations):
        last_iteration = iteration
        (inputs_tr, targets_tr, sort_indices_tr, inputs_ge, targets_ge, sort_indices_ge) = get_data()

        outputs_tr, loss_tr = compiled_update_step(inputs_tr, targets_tr)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        if elapsed_since_last_log > LOG_INTERVALS:
            last_log_time = the_time
            outputs_ge = model(inputs_ge, num_processing_steps_ge)
            losss_ge = create_loss(targets_ge, outputs_ge)
            loss_ge = losss_ge[-1]

            # Replace the globals again to prevent exceptions.
            outputs_tr[-1] = outputs_tr[-1].replace(globals=None)
            targets_tr = targets_tr.replace(globals=None)

            correct_tr, solved_tr = compute_accuracy(utils_tf.nest_to_numpy(targets_tr), utils_tf.nest_to_numpy(outputs_tr[-1]))
            correct_ge, solved_ge = compute_accuracy(utils_tf.nest_to_numpy(targets_ge), utils_tf.nest_to_numpy(outputs_ge[-1]))
            
            elapsed = time.time() - start_time
            
            losses_tr.append(loss_tr.numpy())
            corrects_tr.append(correct_tr)
            solveds_tr.append(solved_tr)
            
            losses_ge.append(loss_ge.numpy())
            corrects_ge.append(correct_ge)
            solveds_ge.append(solved_ge)
            
            logged_iterations.append(iteration)
            
            print("[num_iter]: {}".format(iteration))
            print("*[elapsed_sec]: {:.1f}s\n*[train_loss]: {:.2f}\n*[test_loss]: {:.2f}%\n*[train nodes/edges labeled correctly]: {:.2f}%\n*[train examples solved correctly]: {:.2f}%\n*[test nodes/edges labeled correctly]: {:.2f}%\n*[test examples solved correctly]: {:.2f}%".format(elapsed, loss_tr.numpy(), loss_ge.numpy(), correct_tr*100, solved_tr*100, correct_ge*100, solved_ge*100))
            print()

    """ Result Plotting
    """
    fig = plt.figure(1, figsize=(18, 3))
    fig.clf()
    x = np.array(logged_iterations)
    y_tr = losses_tr
    y_ge = losses_ge
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(x, y_tr, "k", label="Training")
    ax.plot(x, y_ge, "k--", label="Test/generalization")
    ax.set_title("Loss across training")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss (binary cross-entropy)")
    ax.legend()

    # Correct.
    y_tr = corrects_tr
    y_ge = corrects_ge
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(x, y_tr, "k", label="Training")
    ax.plot(x, y_ge, "k--", label="Test/generalization")
    ax.set_title("Fraction examples labeled correctly")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Fraction nodes/edges correct")

    # Solved.
    y_tr = solveds_tr
    y_ge = solveds_ge
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(x, y_tr, "k", label="Training")
    ax.plot(x, y_ge, "k--", label="Test/generalization")
    ax.set_title("Fraction examples solved correctly")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Fraction examples solved")
    fig.savefig(timestr + '/result1.png')

    # Plot sort linked lists for test/generalization.
    # The matrix plots show each element from the sorted list (rows), and which
    # element they link to as next largest (columns). Ground truth is a diagonal
    # offset toward the upper-right by one.
    outputs = utils_np.graphs_tuple_to_networkxs(outputs_tr[-1])
    targets = utils_np.graphs_tuple_to_networkxs(targets_tr)
    inputs = utils_np.graphs_tuple_to_networkxs(inputs_tr)
    batch_element = 0
    fig = plt.figure(12, figsize=(8, 4.5))
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sort_indices = np.squeeze(
        utils_np.get_graph(sort_indices_tr,
                        batch_element).nodes).astype(int)
    fig.suptitle("Element-to-element link predictions for sorted elements")
    plot_linked_list(ax1, targets[batch_element], sort_indices)
    ax1.set_title("Ground truth")
    ax1.set_axis_off()
    plot_linked_list(ax2, outputs[batch_element], sort_indices)
    ax2.set_title("Predicted")
    ax2.set_axis_off()

    # save results
    fig.savefig(timestr + '/result2.png')
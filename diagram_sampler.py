import numpy as np
from collections import deque


ZERO=[1, 0, 0, 0, 0]
PI=[0, 1, 0, 0, 0]
PIHALF=[0, 0, 1, 0, 0]
ARBITRARY=[0, 0, 0, 1, 0]

NO_ANGLE=[0, 0, 0, 0, 1]

RED=[1, 0, 0, 0, 0]
GREEN=[0, 1, 0, 0, 0]
HADAMARD=[0, 0, 1, 0, 0]
INPUT=[0, 0, 0, 1, 0]
OUTPUT=[0, 0, 0, 0, 1]

ANGLE_LIST = [ZERO, PI, PIHALF, ARBITRARY]


class ZX_diag_sampler():
    def __init__(self,
                 n_in_min:int,
                 n_in_max:int,
                 min_spiders:int,
                 max_spiders:int,
                 pi_fac:float,
                 pi_half_fac:float,
                 arb_fac:float,
                 p_hada:float,
                 min_mean_neighbours:int,
                 max_mean_neighbours:int,
                 remove_disconnected:bool,
                 rng:np.random.Generator):
        """n_in_min: minimum number of input/output nodes,
        n_in_max: maximum number of input/output nodes,
        min_spiders: minimum number of spiders in total,
        max_spiders: maximum number of spiders in total,
        pi_fac: factor by which to reduce probability of pi angle,
        pi_half_fac: factor by which to reduce probability of pi/2 angle,
        arb_fac: factor by which to reduce probability of arbitrary angle,
        p_hada: p_hada*n_spiders is maximal number of hadamard nodes,
        min_mean_neighbours: minimum expected number of neighbours per node,
        max_mean_neighbours: maximum expected number of neighbours per node,
        remove_disconnected: whether to remove nodes not connected to any input/output
        rng: numpy random generator"""
        self.n_in_min = n_in_min
        self.n_in_max = n_in_max
        self.min_spiders = min_spiders
        self.max_spiders = max_spiders
        self.pi_fac = pi_fac
        self.pi_half_fac = pi_half_fac
        self.arb_fac = arb_fac
        self.p_hada = p_hada
        self.min_mean_neighbours = min_mean_neighbours
        self.max_mean_neighbours = max_mean_neighbours
        self.remove_disconnected = remove_disconnected
        self.rng = rng

    def sample(self)->tuple:
        """returns (colors, angles, source, target)
        colors/angles: lists of node features, one-hot encoded (each index is one node),
        source/target: lists specifying source and target of an edge (each index is one edge)
        Builds random ZX diagrams"""
        # Sample inout and output number unifomrly
        n_input = self.rng.integers(low=self.n_in_min, high=self.n_in_max+1)
        n_output = self.rng.integers(low=self.n_in_min, high=self.n_in_max+1)
        # Sample number of spiders uniformly
        n_init_spiders = self.rng.integers(low=self.min_spiders, high=self.max_spiders+1)
        # Sample number of hadamards
        n_hada  = self.rng.integers(low=0, high=(n_init_spiders * self.p_hada))
        # Make sure there is at least as many spiders as input and outputs
        n_init_spiders = np.max([n_init_spiders, n_input + n_output])

        # Sample neighbour number uniformly
        mean_neighbours = self.rng.uniform(low=self.min_mean_neighbours, high=self.max_mean_neighbours+1)
        # Calculate probability of each edge such that self.mean_neighbours 
        # is expected value of neighbours per node
        p_edge = (mean_neighbours - (n_input + n_output) / n_init_spiders) / (n_init_spiders+1)
        if p_edge < 0:
            p_edge = 0

        # Sample probabilities for angles uniformly, reduce, and normalize
        p_zero, p_pi, p_pi_half, p_arb = self.rng.uniform(size=len(ANGLE_LIST))
        p_pi *= self.pi_fac
        p_arb *= self.arb_fac
        p_pi_half *= self.pi_half_fac

        norm = p_zero + p_pi + p_arb + p_pi_half
        p_zero /= norm
        p_pi /= norm
        p_arb /= norm
        p_pi_half /= norm

        ps_angle = np.zeros(len(ANGLE_LIST), dtype=np.float32)
        ps_angle[ANGLE_LIST.index(ZERO)] = p_zero
        ps_angle[ANGLE_LIST.index(PIHALF)] = p_pi_half
        ps_angle[ANGLE_LIST.index(PI)] = p_pi
        ps_angle[ANGLE_LIST.index(ARBITRARY)] = p_arb

        # Sample color of each spider randomly
        red_spiders= self.rng.binomial(n_init_spiders, 0.5)
        # Hackky way to make 1d numpy array out of angle list
        # angle_arr = np.empty(len(ANGLE_LIST)+1, dtype="O")
        # angle_arr[:] = (ANGLE_LIST+[ARBITRARY])[:]
        angle_arr = np.array(ANGLE_LIST, dtype=np.int32)

        # Create angle list for all spiders
        node_angle_list = self.rng.choice(angle_arr, size=n_init_spiders, p=ps_angle) 

        # Create color list for all spiders
        node_color_list = np.array([RED] * red_spiders + [GREEN] * (n_init_spiders - red_spiders), dtype=np.int32)
        self.rng.shuffle(node_color_list, axis=0)

        # Add input and output nodes
        colors = np.row_stack((np.array([INPUT] * n_input + [OUTPUT] * n_output, dtype=np.int32), node_color_list))
        angles = np.row_stack((np.array([NO_ANGLE] * (n_input + n_output), dtype=np.int32), node_angle_list))

        # Create edges by sampling each possible edge with probability p_edge
        edge_source, edge_target = np.where(np.triu(np.reshape(
            self.rng.choice(2, size=int(n_init_spiders**2), p=(1-p_edge, p_edge)), (n_init_spiders, n_init_spiders)), k=1))
        
        source = np.concatenate((np.arange(n_input + n_output), edge_source + n_input + n_output))


        target = np.concatenate((np.arange(n_input + n_output, 2 * n_input + n_output), 
                                np.arange(2 * n_input + n_output, 2 * n_input + 2 * n_output), 
                                edge_target + n_input + n_output))
        
        # Add hadamards
        # Indices of all spiders
        idcs_to_connect = np.arange(n_input + n_output, len(colors))

        if len(idcs_to_connect) >= 2:

            for _ in range(n_hada):
                    # Create hadamard node
                    idx_new_node = len(colors)
                    colors = np.row_stack((colors, HADAMARD))
                    angles = np.row_stack((angles, NO_ANGLE))

                    # Connect to two random spiders
                    connected_idx1 = self.rng.choice(idcs_to_connect, 1)[0]
                    new_to_connect = np.delete(idcs_to_connect, np.where(idcs_to_connect==connected_idx1)[0])
                    connected_idx2 = self.rng.choice(new_to_connect, 1)[0]

                    source = np.append(source, connected_idx1)
                    target = np.append(target, idx_new_node)

                    source = np.append(source, connected_idx2)
                    target = np.append(target, idx_new_node)

        # Remove nodes not connected to any input/output
        if self.remove_disconnected:
            n_in_out = n_input + n_output
            colors, angles, source, target = remove_disconnected_nodes(
                colors, angles, source, target, n_in_out)
        

        return colors, angles, source, target


def remove_disconnected_nodes(colors, angles, source, target, n_in_out):
    """Removes nodes that are not connected to inputs or outputs"""
    # Input and output nodes are in the beginning
    input_idcs = np.where(np.all(colors == INPUT, axis=1))
    output_idcs = np.where(np.all(colors == OUTPUT, axis=1))
    search_from_idcs = np.append(input_idcs, output_idcs)

    connected_nodes = []
    # Start searching from each input/ouput node
    for src_idx in search_from_idcs:
        # If not already reached this node:
        if src_idx not in connected_nodes:
            # Get all connected nodes
            conn_nodes = breadth_first_search(src_idx, source, target, len(colors))
            connected_nodes += conn_nodes
    connected_nodes = np.array(connected_nodes, dtype=np.int32)
    
    # Get disconnected node indices
    all_nodes = np.arange(len(colors), dtype=np.int32)
    not_connected = all_nodes[np.in1d(all_nodes, connected_nodes, invert=True)]
    # Sort in reverse order to not mess up indices when removing
    not_connected[::-1].sort()

    # Remove all unconnected
    for idx in not_connected:
        colors, angles, source, target = remove_node_with_edges(
            idx, colors, angles, source, target)
        
    return colors, angles, source, target


def breadth_first_search(source_idx, source, target, n_nodes)->list:
    """Returns indices of all nodes connected to source_idx"""
    # Save if node already visited
    visited = np.zeros(n_nodes, dtype=np.int32)
    visited[source_idx] = 1

    queue = deque()
    queue.append(source_idx)

    reachable_nodes = []

    while(len(queue) > 0):
        # Dequeue a vertex from queue
        u = queue.popleft()
 
        reachable_nodes.append(u)
        # Get all adjacent vertices of the dequeued
        # vertex u. If a adjacent has not been visited,
        # then mark it visited and enqueue it
        for itr in get_neighbours(u, source, target):
            if (visited[itr] == 0):
                visited[itr] = 1
                queue.append(itr)
 
    return reachable_nodes

def remove_node_with_edges(node_idx, colors, angles, source, target):
    """Removes node at node_idx and all connected edges"""
    # Remove node color
    colors = np.delete(colors, node_idx, axis=0)
    # Remove node angle
    angles = np.delete(angles, node_idx, axis=0)

    # Important: This need to be done before the edge indices are decreased
    connected_edge_indices = np.append(np.where(source==node_idx), np.where(target==node_idx))
    # Sort in descending order
    connected_edge_indices[::-1].sort()

    for edge_idx in connected_edge_indices:
        source = np.delete(source, edge_idx, axis=0)
        target = np.delete(target, edge_idx, axis=0)

    # Decrease edge indcs over deleted node
    source[source > node_idx] = source[source > node_idx] - 1
    target[target > node_idx] = target[target > node_idx] - 1

    
    return colors, angles, source, target



def get_neighbours(index, source, target):
    """Returns indices of all nodes connected to index"""
    idcs = target[np.where(source==index)[0]]
    idcs = np.append(idcs, source[np.where(target==index)[0]])
    return idcs



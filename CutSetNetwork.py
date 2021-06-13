import numpy as np
import pandas as pd

from numba import prange, njit
from collections import namedtuple

from nodes import TreeNode, LeafNode
from cltree import create_cltree

Network = namedtuple('Network', ['leaf', 'min_instances','max_depth', 'train_data'])


@njit()
def findEntropy(X: np.ndarray) -> float:
    "Calculate the entropy of the split"

    # Get frequency count for the class (outcome variable)
    counts = np.bincount(X)

    # If the split only has 1 class, ignore class with 0 instances
    counts = counts[np.nonzero(counts)]

    # Divide by total instances to get probability per class
    probabilty = counts/X.shape[0]

    # Calculate entropy of the split
    entropy = np.sum(-1 * probabilty * np.log(probabilty))

    return entropy

@njit(parallel=True)
def findCut(X: np.ndarray, y: np.ndarray, dataset_entropy: float) -> tuple:
    """
    Find the optimal splitting point for a variable when splitting into 2 bins.
    Based on entropy of the dataset (information gain).

    X: variable data
    y: class data
    Both X and y should be arrays of the same length
    """

    unique_values = np.unique(X)   # Returns a sorted list of unique values

    iteration_length = len(unique_values)-1

    if iteration_length == 0:
        return (-1, 0)

    information_gain = np.zeros(iteration_length)
    
    """
    This range function exists because of a problem with the numba library. Currently, if this code
    doesn't have this function, numba tries to fuse the prange for-loop with the loop that creates
    the information_gain array. This results in undefined behaviour and the for-loop gives unreliable
    values.
    """
    range(iteration_length)

    # Only iterate over the unique values in the variable
    for i in prange(iteration_length):

        # Choose splitting value as the middle of 2 sequential values
        # TODO: Implement better splitting heuristic
        split_value = (unique_values[i] + unique_values[i+1])/2
        bins = [-np.inf, split_value, np.inf]
        bin_index = np.digitize(X, bins)

        #TODO: Check if both sections have entries

        counts = np.bincount(bin_index)

        entropy = 0
        for index, value in enumerate(counts):

            if value == 0:
                pass
            else:
                entropy += (value * findEntropy(y[bin_index==index]))

        total_entropy = entropy/len(X)
        temp_inf_gain = dataset_entropy - total_entropy

        information_gain[i] = temp_inf_gain
    
    max_inf_gain = information_gain.max()
    max_entropy = dataset_entropy + max_inf_gain

    gain_index = information_gain.argmax()
    split_value = (unique_values[gain_index] + unique_values[gain_index+1])/2

    # return information_gain
    return (split_value, max_inf_gain)

@njit()
def selectVariable(data: np.ndarray, entropy: float, variable_mask: np.ndarray) -> tuple:
    
    y = data[:, -1]
    y = y.astype(np.int8)
    variables = data.shape[1] -1 

    final_variable = None
    final_index = None
    split_value = None

    storage = np.zeros((variables, 2))

    for variable in range(variables):
        if variable_mask[variable]:
            continue
        X = data[:, variable].astype(np.number)
        temp_inf = findCut(X, y, entropy)
        storage[variable] = temp_inf

    final_variable = storage[:, 1].argmax()
    inf_gain = storage[final_variable, 1]
    split_value = storage[final_variable, 0]

    tmp_bins = [-np.inf, split_value, np.inf]
    tmp_variable_data = data[:, final_variable]
    final_index = np.digitize(tmp_variable_data, tmp_bins)

    return (final_variable, final_index, inf_gain, split_value)

# @njit(parallel=True)
def predict(data: np.ndarray, node: TreeNode):

    rows = data.shape[0]
    prediction = np.zeros(rows)
    
    for i in range(rows):

        row = rows[i]
        model = node.findLeaf(row)

        predicted = model.predict(row)
        prediction[i] = predicted

    return prediction


class CutSetNetwork:

    def __init__(self, data, leaf_type=create_cltree, min_instances_leaf=50, max_depth=20) -> None:
        """
        Create the cutset network using the data provided. Uses the TreeNode class to store the nodes in the network.
        The network terminates with a Chow-Liu Tree at each leaf.

        data - A numpy ndarray. Uses ndarray for calculation.
        """

        self.min_leaf = min_instances_leaf
        self.max_depth = max_depth
        self.leaf_type = leaf_type
        self.nodes = []
        self.DEBUG = 0
        self.columns = None
        
        self.description = Network(str(leaf_type), min_instances_leaf, max_depth, len(data))

        # Get numeric columns from the data
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include=np.number)
            self.columns = data.columns
            data = data.to_numpy()
        else:
            data = self.findNumeric(data).astype(np.number)
        

        variables = np.full(data.shape[1]-1, False)
        self.tree = self.learnNetwork(data, variables)

    def __str__(self) -> str:
        return str(self.description)

    def learnNetwork(self, data, variable_mask, depth=0) -> TreeNode:
        """
        Create the network of nodes using the provided dataset
        """
        if (len(data) <= self.min_leaf) or (depth == self.max_depth) or np.all(variable_mask):
            
            leaf_node = self.createLeaf(data, depth)
            #TODO: Import Chow-Liu Tree and return it here
            return leaf_node

        dataset_entropy = findEntropy(data[:, -1].astype(np.int8))

        if dataset_entropy == 0:
            leaf_node = self.createLeaf(data, depth)
            return leaf_node
        
        variable, index, inf_gain, split_value = selectVariable(data, dataset_entropy, variable_mask)

        if inf_gain == 0:
            leaf_node = self.createLeaf(data, depth)
            # leaf_node = LeafNode(self.leaf_type, len(data), depth)
            return leaf_node

        variable_mask_new = variable_mask.copy()
        variable_mask_new[variable] = True


        _, counts = np.unique(index, return_counts=True)
        probability = counts / len(data)

        # Create TreeNode 

        node = TreeNode(*probability, variable, split_value, len(data), inf_gain, depth)

        # for i, value in zip(_, counts):

        #     if i == 1:
        #         if value == 0:
        #             node.left_child = None
        #         else:
        node.left_child = self.learnNetwork(data[index==1], variable_mask_new, depth=depth+1)
            # else:
            #     if value == 0:
            #         node.right_child = None
            #     else:
        node.right_child = self.learnNetwork(data[index==2], variable_mask_new, depth=depth+1)

        self.nodes.append(node)

        return node
  
    def findNumeric(self, data: np.ndarray) -> np.ndarray:

        variables = data.shape[1]-1
        numeric_variables = []

        for i in range(variables):
            try:
                tmp = data[:, i]
                tmp.astype(np.number)
                numeric_variables.append(i)
            except ValueError:
                pass
        
        numeric_variables.append(-1)
        return data[:, numeric_variables]

    def predict(self, X: np.ndarray) -> np.ndarray:
        
        prediction = predict(X, self.tree)
        
        return prediction

    def createLeaf(self, data: np.ndarray, depth: int) -> LeafNode:

        model = self.leaf_type(data)
        leaf_node = LeafNode(model, len(data), depth)

        return leaf_node

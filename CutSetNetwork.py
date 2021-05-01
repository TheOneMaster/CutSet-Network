import numpy as np
import pandas as pd

from numba import prange, njit
from collections import namedtuple

import nodes

Network = namedtuple('Network', ['leaf', 'min_instances','max_depth'])


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
    # dataset_entropy = findEntropy(y)
    # print(dataset_entropy)

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

        # entropy_less_than_split = findEntropy(y[bin_index==1])
        # entropy_greater_than_split = findEntropy(y[bin_index==2])

        total_entropy = entropy/len(X)
        # print(total_entropy)

        # total_entropy = ((counts[0] * entropy_less_than_split) + (counts[1]* entropy_greater_than_split))/len(X)
        temp_inf_gain = dataset_entropy - total_entropy

        information_gain[i] = temp_inf_gain
    
    max_inf_gain = information_gain.max()
    max_entropy = dataset_entropy + max_inf_gain

    gain_index = information_gain.argmax()
    split_value = (unique_values[gain_index] + unique_values[gain_index+1])/2

    # return information_gain
    return (split_value, max_inf_gain)

@njit()
def selectVariable(data: np.ndarray, entropy: float) -> tuple:
    
    y = data[:, -1]
    y = y.astype(np.int8)
    variables = data.shape[1] -1 

    final_variable = None
    final_index = None
    split_value = None

    storage = np.zeros((variables, 2))

    for variable in range(variables):
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

@njit(parallel=True)
def predict(data: np.ndarray, node: nodes.Node):

    rows = data.shape[0]
    prediction = np.zeros(rows)
    
    for i in prange(rows):

        row = rows[i]
        model = node.findLeaf(row)

        predicted = model.predict(row)
        prediction[i] = predicted

    return prediction


class CutSetNetwork:

    def __init__(self, data, leaf_type='chow-liu', min_instances_leaf=50, max_depth=20) -> None:
        """
        Create the cutset network using the data provided. Uses the Node class to store the nodes in the network.
        The network terminates with a Chow-Liu Tree at each leaf.

        data - A numpy ndarray. Uses ndarray for calculation.
        """

        if isinstance(data, pd.DataFrame):
            self.columns = data.columns
            data = data.values
        
        self.min_leaf = min_instances_leaf
        self.max_depth = max_depth
        self.nodes = []
        self.leaf_type = leaf_type
        self.DEBUG = 0

        # print(data.shape)
        data = self.findNumeric(data).astype(np.number)
        # print(data.shape)  # Check whether the shape of the array changed after removing object columns

        self.tree = self.learnNetwork(data)

    def __str__(self) -> str:
        tree = Network(self.leaf_type, self.min_leaf, self.max_depth)
        return str(tree)

    def learnNetwork(self, data, depth=0) -> nodes.Node:
        """
        Create the network of nodes using the provided dataset
        """

        min_instances = max([self.min_leaf, len(data)/100])   # Simple termination condition for when leaves are created
        dataset_entropy = findEntropy(data[:, -1].astype(np.int8))

        if (len(data) < min_instances) or (dataset_entropy == 0) or (depth == self.max_depth):
            #TODO: Import Chow-Liu Tree and return it here
            return self.leaf_type
        
        # self.DEBUG += 1
        # print(self.DEBUG)

        # if self.DEBUG == 11:
            # print("test")
            # np.savetxt('test.out', data, delimiter=',')
        
        variable, index, inf_gain, split_value = selectVariable(data, dataset_entropy)

        _, counts = np.unique(index, return_counts=True)
        probability = counts / len(data)

        # Create Node 

        node = nodes.Node(*probability, variable, split_value, len(data), inf_gain, depth)

        for i, value in zip(_, counts):

            if i == 1:
                if value == 0:
                    node.left_child = None
                else:
                    node.left_child = self.learnNetwork(data[index==i], depth=depth+1)
            else:
                if value == 0:
                    node.right_child = None
                else:
                    node.right_child = self.learnNetwork(data[index==i], depth=depth+1)

        # node.left_child = self.learnNetwork(data[index==1])
        # node.right_child = self.learnNetwork(data[index==2])

        self.nodes.append(node)

        return node
  
    def findNumeric(self, data) -> np.ndarray:

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

if __name__ == "__main__":

    # df = pd.read_csv(r"data\diabetes.csv")
    # df['class'] = df['class'].map({"tested_positive": 1, "tested_negative": 0})

    # network = CutSetNetwork(df.values)
    # print(network)

    df = pd.read_csv(r"data\bank-additional-full.csv", sep=';')
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    network = CutSetNetwork(df, min_instances_leaf=1000)
    print(network)

    # cut = findCut(df['age'].values, df['class'].values, findEntropy(df['class'].values))
    # print(cut)
    

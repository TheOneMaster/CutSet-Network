import numpy as np
import nodes
from numba import jit, prange
from numba.experimental import jitclass

@jit(nopython=True)
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

@jit(nopython=True)
def findCut(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the optimal splitting point for a variable when splitting into 2 bins.
    Based on entropy of the dataset (information gain).

    X: variable data
    y: class data
    Both X and y should be arrays of the same length
    """
    dataset_entropy = findEntropy(y)
    information_gain = -np.inf
    final_split = 0
    final_index = None

    unique_values = np.unique(X)   # Returns a sorted list of unique values

    iteration_length = len(unique_values)-1
    
    # Only iterate over the unique values in the variable
    for i in range(iteration_length):

        # Choose splitting value as the middle of 2 sequential values
        # TODO: Implement better splitting heuristic
        split_value = (unique_values[i] + unique_values[i+1])/2
        bins = [-np.inf, split_value, np.inf]
        bin_index = np.digitize(X, bins)

        entropy_less_than_split = findEntropy(y[bin_index==1])
        entropy_greater_than_split = findEntropy(y[bin_index==2])

        counts = np.bincount(bin_index)
        counts = counts[1:]

        total_entropy = ((counts[0] * entropy_less_than_split) + (counts[1]* entropy_greater_than_split))/len(X)
        temp_inf_gain = dataset_entropy - total_entropy

        if temp_inf_gain > information_gain:
            information_gain = temp_inf_gain
            final_split = split_value
            final_index = bin_index

    return (final_split, information_gain, final_index)

class CutSetNetwork:

    def __init__(self, data) -> None:
        """
        Create the cutset network using the data provided. Uses the Node class to store the nodes in the network.
        The network terminates with a Chow-Liu Tree at each leaf.

        data - Either a pandas dataframe or a numpy ndarray. Uses ndarray for calculation.
        """

        self.nodes = []

        data = self.findNumeric(data)
        self.tree = self.learnNetwork(data)

    def __str__(self) -> str:
        return str(self.tree)

    def learnNetwork(self, data) -> nodes.Node:
        """
        Create the network of nodes using the provided dataset
        """

        termination_condition = 50   # Simple termination condition for when leaves are created

        if len(data) < termination_condition:
            #TODO: Import Chow-Liu Tree and return it here
            return 'Chow-Liu Tree'
        
        variable, index, inf_gain, split_value = self.selectVariable(data)

        _, counts = np.unique(index, return_counts=True)
        probability = counts / len(data)

        # Create Node 

        node = nodes.Node()
        node.left_prob, node.right_prob = probability
        node.variable = variable
        node.split_value = split_value
        node.data_points = len(data)

        node.left_child = self.learnNetwork(data[index==1])
        node.right_child = self.learnNetwork(data[index==2])

        node.information_gain = inf_gain

        self.nodes.append(node)

        return node

    @staticmethod
    @jit(nopython=True, parallel=True)
    def selectVariable(data) -> tuple:
        
        y = data[:, -1]
        y = y.astype(np.int8)
        variables = data.shape[1] -1 

        inf_gain = -np.inf
        final_variable = None
        final_index = None
        split_value = None

        storage = np.zeros((variables, 2))

        for variable in prange(variables):
            X = data[:, variable].astype(np.number)
            temp_inf = findCut(X, y)
            storage[variable] = temp_inf[:-1]

        final_variable = storage[:, 1].argmax()
        inf_gain = storage[final_variable, 1]
        split_value = storage[final_variable, 0]

        tmp_bins = [-np.inf, split_value, np.inf]
        tmp_variable_data = data[:, final_variable]
        final_index = np.digitize(tmp_variable_data, tmp_bins)

        return (final_variable, final_index, inf_gain, split_value)
  
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

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(r"data\diabetes.csv")
    df['class'] = df['class'].map({"tested_positive": 1, "tested_negative": 0})

    network = CutSetNetwork(df.values)
    print(network)
    # findCut.parallel_diagnostics(level=4)
    # CutSetNetwork.selectVariable.parallel_diagnostics(level=4)

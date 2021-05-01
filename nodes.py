import numpy as np
from collections import namedtuple

from numpy.core.fromnumeric import var

Name = namedtuple('Node', ['variable', 'split', 'l_prob', 'r_prob', 'data', 'inf_gain'])


class Node:

    def __init__(self, left_prob=0, right_prob=0, variable=0, split_value=0, data_points=0, information_gain=0) -> None:
        
        self.left_prob = left_prob
        self.right_prob = right_prob
        self.left_child = None
        self.right_child = None

        self.variable = variable
        self.split_value = split_value
        self.data_points = data_points
        self.information_gain = information_gain

    def __str__(self) -> str:
        n = Name(self.variable, self.split_value, self.left_prob, self.right_prob, self.data_points, self.information_gain)
        return str(n)

    def printTree(self, depth=0) -> None:
        
        print(f"{depth*'-'}[{self.variable} < {self.split_value}] [{self.data_points}]")

        children = [self.left_child, self.right_child]

        for child in children:

            if isinstance(child, Node):
                child.printTree(depth=depth+1)
            else:
                print(f"{depth*'-'} [Chow-Liu Tree]")
        
    def findLeaf(self, row: np.ndarray) -> str:

        value = row[self.variable]

        if value < self.split_value:

            if isinstance(self.left_child, Node):
                return self.left_child.findLeaf(row)
            else:
                return self.left_child
        else:
            
            if isinstance(self.right_child, Node):
                return self.right_child.findLeaf(row)
            else:
                return self.right_child
        

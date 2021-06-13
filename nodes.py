import numpy as np
from collections import namedtuple

Name = namedtuple('TreeNode', ['variable', 'split', 'l_prob', 'r_prob', 'data', 'inf_gain'])

class BaseNode:

    def __init__(self, data_points, depth) -> None:
        
        self.data_points = data_points
        self.depth = depth


class TreeNode(BaseNode):

    def __init__(self, left_prob=0, right_prob=0, variable=0, split_value=0, data_points=0, information_gain=0, depth=0) -> None:
        super().__init__(data_points, depth)
        
        self.left_prob = left_prob
        self.right_prob = right_prob
        self.variable = variable
        self.split_value = split_value
        self.information_gain = information_gain

        self.left_child = None
        self.right_child = None

    def __str__(self) -> str:
        n = Name(self.variable, self.split_value, self.left_prob, self.right_prob, self.data_points, self.information_gain)
        return str(n)

    def printTree(self) -> None:
        
        print(f"{self.depth*'-'}[{self.variable} < {self.split_value}] [{self.data_points}]")

        children = [self.left_child, self.right_child]

        for child in children:

            if isinstance(child, TreeNode):
                child.printTree()
            else:
                temp_num = self.depth + 1
                print(f"{temp_num*'-'}[{str(child)}][{child.data_points}]")
            
    def findLeaf(self, row: np.ndarray) -> str:

        value = row[self.variable]

        if value < self.split_value:

            if isinstance(self.left_child, TreeNode):
                return self.left_child.findLeaf(row)
            else:
                return self.left_child
        else:
            
            if isinstance(self.right_child, TreeNode):
                return self.right_child.findLeaf(row)
            else:
                return self.right_child
        

class LeafNode(BaseNode):

    def __init__(self, model, data_points=0, depth=0) -> None:
        super().__init__(data_points, depth)
        self.model = model

    def __str__(self):
        return str(self.model)


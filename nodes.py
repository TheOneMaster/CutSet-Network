import numpy as np
from collections import namedtuple

Name = namedtuple('Node', ['variable', 'split', 'l_prob', 'r_prob', 'data', 'inf_gain'])

class Node:

    def __init__(self) -> None:
        
        self.left_prob = 0
        self.right_prob = 0
        self.left_child = None
        self.right_child = None

        self.variable = None
        self.split_value = None
        self.data_points = None
        self.information_gain = None

    def __str__(self) -> str:
        n = Name(self.variable, self.split_value, self.left_prob, self.right_prob, self.data_points, self.information_gain)
        return str(n)
        

        

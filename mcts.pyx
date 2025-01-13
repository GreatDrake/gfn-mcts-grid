# distutils:language=c++
# cython:language_level=3

from libcpp.vector cimport vector
from libcpp.utility cimport pair

from mcts_cpp cimport Node, NodePtrContainer, TreeRoots

cdef class NodePtrContainerWrapper:
    cdef NodePtrContainer nodes
    
    def __init__(self):
        pass
    
    def size(self):
        return self.nodes.size()


cdef class TreeRootsWrapper:
    cdef TreeRoots croots
    
    def __init__(self, int dim, int side, double eps, int batch_size, int tree_max_size,
                 vector[vector[int]] states):
        self.croots = TreeRoots(dim, side, eps, batch_size, tree_max_size)
        self.croots.initialize_roots(states)
        
    def size(self):
        return self.croots.roots.size()
        
    def set_root_q_values(self, vector[vector[double]] predictions):
        self.croots.set_root_q_values(predictions)
        
    def get_root_q_values(self):
        return self.croots.get_root_q_values()
    
    def move_to_children(self, vector[int] actions, vector[char] dones, 
                         vector[vector[int]] states, 
                         vector[vector[double]] predictions):
        self.croots.move_to_children(actions, dones, states, predictions)
        
    def expand(self, vector[char] dones):
        leaves_ohes = self.croots.expand(dones)
        leaf_container = NodePtrContainerWrapper()
        leaf_container.nodes = leaves_ohes.first
        ohes = leaves_ohes.second
        
        return leaf_container, ohes
    
    def backup(self, NodePtrContainerWrapper leaves, vector[vector[double]] predictions):
        self.croots.backup(leaves.nodes, predictions)
        
        
        
    
        
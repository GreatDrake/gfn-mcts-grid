# distutils:language=c++
# cython:language_level=3

from libcpp cimport bool

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.random cimport mt19937

cdef extern from "mcts_cpp.cpp":
    pass

cdef extern from "mcts_cpp.h" namespace "mcts_tree":
    cdef cppclass Node:
        Node(int, int, bool, vector[int], pair[int, Node*], double) except +
        
        int dim
        int side
        double eps
        bool terminal
        vector[int] state
        vector[char] mask
        int size
        pair[int, Node*] parent
        vector[Node*] children
        vector[double] q_values
        double reward
        
        void set_q_values(vector[double] predictions)
        pair[int, Node*] expand()
        void backup()
        vector[int] ohe()
        
    cdef cppclass NodePtrContainer:
        NodePtrContainer() except +
        vector[Node*] nodes
        int size()
        
    cdef cppclass TreeRoots:
        TreeRoots()
        TreeRoots(int, int, double, int, int)
        
        int dim
        int side
        double eps
        int batch_size
        int tree_max_size
        vector[Node*] roots
        
        void initialize_roots(vector[vector[int]] states)
        void set_root_q_values(vector[vector[double]] predictions)
        vector[vector[double]] get_root_q_values()
        
        void move_to_children(vector[int] actions, vector[char] dones, 
                              vector[vector[int]] states, 
                              vector[vector[double]] predictions)
        
        pair[NodePtrContainer, vector[vector[int]]] expand(vector[char] dones)
        void backup(NodePtrContainer leaves, vector[vector[double]] predictions)
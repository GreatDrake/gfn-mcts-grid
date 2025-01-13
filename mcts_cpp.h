#ifndef MCTSTREE_H
#define MCTSTREE_H

#include <vector>
#include <random>
#include <utility>
#include <random>

namespace mcts_tree {
    class Node {
    public:
        std::mt19937 gen;
        int dim;
        int side;
        double eps;
        bool terminal;
        std::vector<int> state;
        std::vector<char> mask;
        int size;

        std::pair<int, Node*> parent;
        std::vector<Node*> children;
        std::vector<double> q_values;
        double reward;

        Node() = delete;
        Node(int dim, int side, bool terminal, std::vector<int> state,
             std::pair<int, Node*> parent, double eps);
        ~Node();

        void set_q_values(std::vector<double> predictions);
        std::pair<int, Node*> expand();
        void backup();
        std::vector<int> ohe();
    };

    // helper class
    class NodePtrContainer {
    public:
        std::vector<Node*> nodes;

        NodePtrContainer();
        ~NodePtrContainer();
        int size();
    };

    class TreeRoots {
    public:
        int dim;
        int side;
        double eps;
        int batch_size;
        int tree_max_size;
        std::vector<Node*> roots;

        TreeRoots();
        TreeRoots(int dim, int side, double eps, int batch_size, int tree_max_size);
        ~TreeRoots();

        void initialize_roots(std::vector< std::vector<int> > states);

        void set_root_q_values(std::vector< std::vector<double> > predictions);
        std::vector< std::vector<double> > get_root_q_values();

        void move_to_children(std::vector<int> actions, std::vector<char> dones,
                              std::vector< std::vector<int> > states,
                              std::vector< std::vector<double> > predictions);

        std::pair<NodePtrContainer, std::vector< std::vector<int> > > expand(std::vector<char> dones);
        void backup(NodePtrContainer leaves, std::vector< std::vector<double> > predictions);
    };
}

#endif
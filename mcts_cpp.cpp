#include <iostream>
#include <algorithm>
#include <utility>
#include <vector>
#include <random>
#include <cmath>
#include "mcts_cpp.h"

using namespace std;

namespace mcts_tree {

    void my_assert(bool flag) {
        if (!flag) {
            cout << "ERROR" << endl;
            exit(1);
        }
    }

    double get_max_logit(const vector<double> &logits, const vector<char> &mask) {
        my_assert((logits.size() == mask.size()));
        bool found = false;
        double result = 0.0;
        for (int i = 0; i < logits.size(); ++i) {
            if (!mask[i]) {
                if (!found) {
                    result = logits[i];
                    found = true;
                } else {
                    result = max(result, logits[i]);
                }
            }
        }
        my_assert(found);
        return result;
    }

    // changes logits vector inplace
    void calculate_tree_policy(vector<double> &logits, const vector<char> &mask, double lmbd, double uniform_prob) {
        double max_logit = get_max_logit(logits, mask);
        double denom = 0.0;
        for (int i = 0; i < logits.size(); ++i) {
            if (!mask[i]) {
                denom += exp(logits[i] - max_logit);
            }
        }
        for (int i = 0; i < logits.size(); ++i) {
            if (!mask[i]) {
                logits[i] = (1 - lmbd) * exp(logits[i] - max_logit) / denom + lmbd * uniform_prob;
            } else {
                logits[i] = 0;
            }
        }
    }

    double logsumexp(const vector<double> &logits, const vector<char> &mask) {
        double max_logit = get_max_logit(logits, mask);
        double sum_exp = 0.0;
        for (int i = 0; i < logits.size(); ++i) {
            if (!mask[i]) {
                sum_exp += exp(logits[i] - max_logit);
            }
        }
        return max_logit + log(sum_exp);
    }

    pair<int, Node*> null_parent() {
        return make_pair(-1, nullptr);
    }

    Node::Node(int dim, int side, bool terminal, vector<int> state, std::pair<int, Node*> parent, double eps) {
        this->gen = std::mt19937((int)(reinterpret_cast<long long>(this) % 123456789));
        this->dim = dim;
        this->side = side;
        this->terminal = terminal;
        this->state = move(state);
        this->parent = parent;
        this->eps = eps;
        this->size = 1;

        if (this->terminal) {
            this->reward = 0.0; // placeholder, use std::optional here?
        } else {
            int n_parents = 0;
            for (int val : this->state) {
                n_parents += (val != 0);
            }
            if (n_parents > 0) {
                this->reward = log(1.0 / n_parents);
            } else {
                this->reward = 0.0; // placeholder, use std::optional here?
            }
        }

        this->mask = vector<char>(this->dim + 1, 0);
        for (int i = 0; i < this->dim; ++i) {
            if (this->state[i] == this->side - 1) {
                this->mask[i] = 1;
            }
        }

        this->q_values = vector<double>(this->dim + 1, 0.0);
        this->children = vector<Node*>(this->dim + 1, nullptr);
    }

    Node::~Node() {
        //cout << reinterpret_cast<long long>(this) << " node destructor" << endl;
        for (int i = 0; i < this->children.size(); ++i) {
            if (this->children[i]) {
                delete this->children[i];
            }
        }
    }

    void Node::set_q_values(vector<double> predictions) {
        //my_assert((this->q_values.size() == predictions.size()));
        this->q_values = move(predictions);
    }

    vector<int> Node::ohe() {
        vector<int> state_ohe(this->dim * this->side, 0);
        for (int i = 0; i < this->state.size(); ++i) {
            state_ohe[i * this->side + this->state[i]] = 1;
        }
        return state_ohe;
    }

    pair<int, Node*> Node::expand() {
        if (this->terminal) {
            return make_pair(0, this);
        }

        vector<double> probs(this->q_values);
        double lmbd = this->eps * (this->dim + 1) / log(this->size + 1.0);
        double uniform_prob = 1.0 / (probs.size() - accumulate(mask.begin(), mask.end(), 0));
        calculate_tree_policy(probs, this->mask, lmbd, uniform_prob);
        //my_assert(abs(accumulate(probs.begin(), probs.end(), 0.0) - 1.0) < 1e-3);

        discrete_distribution<int> d(probs.begin(), probs.end());
        int action = d(this->gen);

        if (this->children[action]) {
            auto cnt_leaf = this->children[action]->expand();
            this->size += cnt_leaf.first;
            return cnt_leaf;
        }

        this->size += 1;
        vector<int> next_state(this->state);
        if (action == this->dim) {
            this->children[action] = new Node(this->dim, this->side, true, move(next_state),
                                              make_pair(action, this), this->eps);
            this->children[action]->reward = this->q_values[action];
            return make_pair(1, this->children[action]);
        } else {
            next_state[action] += 1;
            this->children[action] = new Node(this->dim, this->side, false, move(next_state),
                                              make_pair(action, this), this->eps);
            return make_pair(1, this->children[action]);
        }
    }

    void Node::backup() {
        if (!this->parent.second) {
            return;
        }

        Node* parent_node = this->parent.second;
        int parent_action = this->parent.first;

        if (this->terminal) {
            parent_node->q_values[parent_action] = this->reward;
        } else {
            parent_node->q_values[parent_action] = this->reward + logsumexp(this->q_values, this->mask);
        }

        parent_node->backup();
    }

    NodePtrContainer::NodePtrContainer() {}
    NodePtrContainer::~NodePtrContainer() {}
    int NodePtrContainer::size() {
        return this->nodes.size();
    }

    TreeRoots::TreeRoots() { }

    TreeRoots::TreeRoots(int dim, int side, double eps, int batch_size, int tree_max_size) {
        this->dim = dim;
        this->side = side;
        this->eps = eps;
        this->batch_size = batch_size;
        this->tree_max_size = tree_max_size;
    }

    void TreeRoots::initialize_roots(vector< vector<int> > states) {
        //my_assert(this->roots.empty());
        this->roots = vector<Node*>(this->batch_size, nullptr);
        //my_assert((this->batch_size == states.size()));

        for (int i = 0; i < this->batch_size; ++i) {
            this->roots[i] = new Node(this->dim, this->side, false,
                                      move(states[i]), null_parent(), this->eps);
        }
    }

    TreeRoots::~TreeRoots() {
        //cout << reinterpret_cast<long long>(this) << " roots destructor, size: " << this->roots.size() << endl;
        for (Node* root : this->roots) {
            delete root;
        }
    }


    void TreeRoots::set_root_q_values(vector< vector<double> > predictions) {
        for (int i = 0; i < this->batch_size; ++i) {
            this->roots[i]->set_q_values(move(predictions[i]));
        }
    }

    vector< vector<double> > TreeRoots::get_root_q_values() {
        vector< vector<double> > result;
        for (auto root : this->roots) {
            result.push_back(root->q_values);
        }
        return result;
    }

    void TreeRoots::move_to_children(vector<int> actions, vector<char> dones,
                                     vector< vector<int> > states,
                                     vector< vector<double> > predictions) {
        for (int i = 0; i < this->batch_size; ++i) {
            if (!dones[i]) {
                Node* old_root = this->roots[i];
                if (this->roots[i]->children[actions[i]]) {
                    Node* new_root = old_root->children[actions[i]];
                    old_root->children[actions[i]] = nullptr;
                    new_root->parent = null_parent();
                    this->roots[i] = new_root;
                } else {
                    this->roots[i] = new Node(this->dim, this->side, false,
                                              move(states[i]), null_parent(), this->eps);
                    this->roots[i]->set_q_values(move(predictions[i]));
                }
                delete old_root;
            }
        }
    }

    pair<NodePtrContainer, vector< vector<int> > > TreeRoots::expand(vector<char> dones) {
        NodePtrContainer leaves;

        for (int i = 0; i < this->batch_size; ++i) {
            if (dones[i] || this->roots[i]->size > this->tree_max_size) {
                continue;
            }
            auto cnt_leaf = this->roots[i]->expand();
            if (cnt_leaf.first == 1) {
                leaves.nodes.push_back(cnt_leaf.second);
            }
        }
        vector< vector<int> > ohes;
        for (auto leaf : leaves.nodes) {
            ohes.push_back(move(leaf->ohe()));
        }
        return make_pair(leaves, ohes);
    }

    void TreeRoots::backup(NodePtrContainer leaves, vector< vector<double> > predictions) {
        //my_assert((leaves.size() == predictions.size()));
        for (int i = 0; i < leaves.size(); ++i) {
            leaves.nodes[i]->set_q_values(move(predictions[i]));
            leaves.nodes[i]->backup();
        }
    }
}
#ifndef COBWEB_CONTINUOUS_NODE_H
#define COBWEB_CONTINUOUS_NODE_H

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <iostream>
#include "helper.h"



class CobwebContinuousTree;

class CobwebContinuousNode {
public:
    CobwebContinuousTree *tree;
    CobwebContinuousNode *parent;
    std::vector<CobwebContinuousNode *> children;

    double count;
    Eigen::VectorXd mean;
    Eigen::VectorXd sum_sq;

    CobwebContinuousNode(int size);
    CobwebContinuousNode(CobwebContinuousNode *otherNode);
    void increment_counts(const Eigen::VectorXd &instance);
    void update_counts_from_node(CobwebContinuousNode *node);
    bool is_exact_match(const Eigen::VectorXd &instance);
    size_t _hash();
    std::string __str__();

    std::tuple<Eigen::VectorXd, Eigen::VectorXd> mean_var();
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> mean_var_new(const Eigen::VectorXd &instance);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> mean_var_insert(const Eigen::VectorXd &instance);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> mean_var_merge(CobwebContinuousNode *other, const Eigen::VectorXd &instance);

    double pu_for_insert(CobwebContinuousNode *child, const Eigen::VectorXd &instance);
    double pu_for_new(const Eigen::VectorXd &instance);
    double pu_for_merge(CobwebContinuousNode *best1, CobwebContinuousNode *best2, const Eigen::VectorXd &instance);
    double pu_for_split(CobwebContinuousNode *best);

    std::tuple<double, int> get_best_operation(const Eigen::VectorXd &instance, CobwebContinuousNode *best1, CobwebContinuousNode *best2, double best1_pu);
    std::tuple<double, CobwebContinuousNode *, CobwebContinuousNode *> two_best_children(const Eigen::VectorXd &instance);

    // std::string concept_hash();
    // std::string pretty_print(int depth = 0);
    // int depth();
    // bool is_parent(CobwebContinuousNode *otherConcept);
    // int num_concepts();

    // std::string avcounts_to_json();
    // std::string ser_avcounts();
    // std::string a_count_to_json();
    // std::string sum_n_logn_to_json();
    // std::string dump_json();
    std::string output_json();
};

#endif // COBWEB_CONTINUOUS_NODE_H

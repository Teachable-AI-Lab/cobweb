#ifndef COBWEB_CONTINUOUS_NODE_H
#define COBWEB_CONTINUOUS_NODE_H

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include "helper.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;




class CobwebContinuousTree;

class CobwebContinuousNode {
public:
    CobwebContinuousTree *tree;
    CobwebContinuousNode *parent;
    std::vector<CobwebContinuousNode *> children;

    float count;
    Eigen::VectorXf mean;
    Eigen::VectorXf sum_sq;

    CobwebContinuousNode(int size);
    CobwebContinuousNode(CobwebContinuousNode *otherNode);
    void increment_counts(const Eigen::VectorXf &instance);
    int depth();
    void update_counts_from_node(CobwebContinuousNode *node);
    bool is_exact_match(const Eigen::VectorXf &instance);
    size_t _hash();
    std::string __str__();

    std::tuple<Eigen::VectorXf, Eigen::VectorXf> mean_var();
    std::tuple<Eigen::VectorXf, Eigen::VectorXf> mean_var_new(const Eigen::VectorXf &instance);
    std::tuple<Eigen::VectorXf, Eigen::VectorXf> mean_var_insert(const Eigen::VectorXf &instance);
    std::tuple<Eigen::VectorXf, Eigen::VectorXf> mean_var_merge(CobwebContinuousNode *other, const Eigen::VectorXf &instance);

    float pu_for_insert(CobwebContinuousNode *child, const Eigen::VectorXf &instance);
    float pu_for_new(const Eigen::VectorXf &instance);
    float pu_for_merge(CobwebContinuousNode *best1, CobwebContinuousNode *best2, const Eigen::VectorXf &instance);
    float pu_for_split(CobwebContinuousNode *best);

    std::tuple<float, int> get_best_operation(const Eigen::VectorXf &instance, CobwebContinuousNode *best1, CobwebContinuousNode *best2, float best1_pu);
    std::tuple<float, CobwebContinuousNode *, CobwebContinuousNode *> two_best_children(const Eigen::VectorXf &instance);

    float log_prob(const Eigen::VectorXf &instance);
    float log_prob_class_given_instance(const Eigen::VectorXf &instance);

    std::vector<float> log_prob_children_given_instance(const Eigen::VectorXf &instance);
    const Eigen::VectorXf& predict_mean(const Eigen::VectorXf &instance);

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
    nb::dict to_map();
    std::string export_tree_json();
    void save_tree_to_file(const std::string &filename);
};

#endif // COBWEB_CONTINUOUS_NODE_H

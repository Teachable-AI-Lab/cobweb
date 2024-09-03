#ifndef COBWEB_CONTINUOUS_TREE_H
#define COBWEB_CONTINUOUS_TREE_H

#include <nanobind/eigen/dense.h>
#include <Eigen/Dense>
#include "cobweb_continuous_node.h"

namespace nb = nanobind;

#define BEST 0
#define NEW 1
#define MERGE 2
#define SPLIT 3

class CobwebContinuousTree {
public:
    int size;
    Eigen::VectorXf prior_var;
    CobwebContinuousNode *root;

    CobwebContinuousTree(int size);

    CobwebContinuousNode* ifit(const Eigen::VectorXf &instance);
    CobwebContinuousNode* ifit_helper(const Eigen::VectorXf &instance);
    CobwebContinuousNode* cobweb(const Eigen::VectorXf &instance);

    std::string __str__();
    // std::string dump_json()
    // std::string load_json()
    void clear();

    Eigen::VectorXf compute_var(const Eigen::VectorXf& meanSq, const float count);
    float compute_score(const Eigen::VectorXf& child_mean,
            const Eigen::VectorXf& child_var, const Eigen::VectorXf& parent_mean,
            const Eigen::VectorXf& parent_var);

};

#endif // COBWEB_CONTINUOUS_TREE_H

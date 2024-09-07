#ifndef COBWEB_CONTINUOUS_TREE_H
#define COBWEB_CONTINUOUS_TREE_H

#include <queue>
#include <nanobind/eigen/dense.h>
#include <Eigen/Dense>
#include "cobweb_continuous_node.h"
#include "helper.h"

namespace nb = nanobind;

#define BEST 0
#define NEW 1
#define MERGE 2
#define SPLIT 3

class CobwebContinuousTree {
public:
    int size;
    int covar_type;
    int covar_from;
    Eigen::VectorXf prior_var;
    CobwebContinuousNode *root;

    // covar_type: 1=diag
    // covar_from: 1=self, 2=parent
    CobwebContinuousTree(int size, int covar_type, int covar_from);

    CobwebContinuousNode* ifit(const Eigen::VectorXf &instance);
    CobwebContinuousNode* ifit_helper(const Eigen::VectorXf &instance);
    CobwebContinuousNode* cobweb(const Eigen::VectorXf &instance);
    Eigen::VectorXf predict(const Eigen::VectorXf &instance, int max_nodes, bool greedy);
    Eigen::VectorXf predict_helper(const Eigen::VectorXf &instance, int max_nodes, bool greedy);
    float log_prob(const Eigen::VectorXf &instance, int max_nodes, bool greedy);

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

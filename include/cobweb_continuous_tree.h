#ifndef COBWEB_CONTINUOUS_TREE_H
#define COBWEB_CONTINUOUS_TREE_H

#include <Eigen/Dense>
#include "cobweb_continuous_node.h"

#define BEST 0
#define NEW 1
#define MERGE 2
#define SPLIT 3

class CobwebContinuousTree {
public:
    int size;
    Eigen::VectorXd prior_var;
    CobwebContinuousNode *root;

    CobwebContinuousTree(int size);

    CobwebContinuousNode* ifit(Eigen::VectorXd instance);
    CobwebContinuousNode* ifit_helper(const Eigen::VectorXd &instance);
    CobwebContinuousNode* cobweb(const Eigen::VectorXd &instance);

    std::string __str__();
    // std::string dump_json()
    // std::string load_json()
    void clear();

    Eigen::VectorXd compute_var(const Eigen::VectorXd& meanSq, const double count);
    double compute_score(const Eigen::VectorXd& child_mean,
            const Eigen::VectorXd& child_var, const Eigen::VectorXd& parent_mean,
            const Eigen::VectorXd& parent_var);

};

#endif // COBWEB_CONTINUOUS_TREE_H

#include "cobweb_continuous_node.h"
#include "cobweb_continuous_tree.h"

CobwebContinuousTree::CobwebContinuousTree(int size)
    : root(nullptr),
      size(size)
{
    this->prior_var = Eigen::VectorXf::Constant(size, 0.05854983152);
    this->clear();
}


CobwebContinuousNode* CobwebContinuousTree::ifit(const Eigen::VectorXf &instance){
    return this->ifit_helper(instance);
}

CobwebContinuousNode* CobwebContinuousTree::ifit_helper(const Eigen::VectorXf &instance){
    return this->cobweb(instance);
}

CobwebContinuousNode* CobwebContinuousTree::cobweb(const Eigen::VectorXf &instance){

    CobwebContinuousNode* current = this->root;

    while (true) {
        if (current->children.empty() &&
                (current->count == 0 || current->is_exact_match(instance))) {
            // std::cout << "empty / exact match" << std::endl;
            current->increment_counts(instance);
            break;

        } else if (current->children.empty()) {
            // std::cout << "fringe split" << std::endl;
            CobwebContinuousNode* new_node = new CobwebContinuousNode(current);
            current->parent = new_node;
            new_node->children.push_back(current);

            if (new_node->parent == nullptr) {
                root = new_node;
            }
            else{
                new_node->parent->children.erase(remove(new_node->parent->children.begin(),
                            new_node->parent->children.end(), current), new_node->parent->children.end());
                new_node->parent->children.push_back(new_node);
            }
            new_node->increment_counts(instance);

            current = new CobwebContinuousNode(this->size);
            current->parent = new_node;
            current->tree = this;
            current->increment_counts(instance);
            new_node->children.push_back(current);
            break;

        } else {
            auto[best1_mi, best1, best2] = current->two_best_children(instance);
            auto[_, bestAction] = current->get_best_operation(instance, best1, best2, best1_mi);

            if (bestAction == BEST) {
                // std::cout << "best" << std::endl;
                current->increment_counts(instance);
                current = best1;

            } else if (bestAction == NEW) {
                // std::cout << "new" << std::endl;
                current->increment_counts(instance);

                // current = current->create_new_child(instance);
                CobwebContinuousNode *new_child = new CobwebContinuousNode(this->size);
                new_child->parent = current;
                new_child->tree = this;
                new_child->increment_counts(instance);
                current->children.push_back(new_child);
                current = new_child;
                break;

            } else if (bestAction == MERGE) {
                // std::cout << "merge" << std::endl;
                current->increment_counts(instance);
                // CobwebContinuousNode* new_child = current->merge(best1, best2);

                CobwebContinuousNode *new_child = new CobwebContinuousNode(this->size);
                new_child->parent = current;
                new_child->tree = this;

                new_child->update_counts_from_node(best1);
                new_child->update_counts_from_node(best2);
                best1->parent = new_child;
                best2->parent = new_child;
                new_child->children.push_back(best1);
                new_child->children.push_back(best2);
                current->children.erase(remove(current->children.begin(),
                            current->children.end(), best1), current->children.end());
                current->children.erase(remove(current->children.begin(),
                            current->children.end(), best2), current->children.end());
                current->children.push_back(new_child);
                current = new_child;

            } else if (bestAction == SPLIT) {
                // std::cout << "split" << std::endl;
                current->children.erase(remove(current->children.begin(),
                            current->children.end(), best1), current->children.end());
                for (auto &c: best1->children) {
                    c->parent = current;
                    c->tree = this;
                    current->children.push_back(c);
                }
                delete best1;

            } else {
                throw "Best action choice \"" + std::to_string(bestAction) +
                    "\" (best=0, new=1, merge=2, split=3) not a recognized option. This should be impossible...";
            }
        }
    }
    return current;
}

std::string CobwebContinuousTree::__str__()
{
    return this->root->__str__();
}

void CobwebContinuousTree::clear()
{
    delete this->root;
    this->root = new CobwebContinuousNode(this->size);
    this->root->tree = this;
}

Eigen::VectorXf CobwebContinuousTree::compute_var(const Eigen::VectorXf& sum_sq, const float count){
    return sum_sq / count + this->prior_var;
}

float CobwebContinuousTree::compute_score(const Eigen::VectorXf& child_mean,
        const Eigen::VectorXf& child_var, const Eigen::VectorXf& parent_mean,
        const Eigen::VectorXf& parent_var){

    // float score = 0.5 * (1 - (child_mean).cwiseProduct(parent_mean).array().sum() / (child_mean.norm() * parent_mean.norm()));

    // Using linked covar based on parent
    float score = 0.5 * (child_mean - parent_mean).cwiseProduct(child_mean - parent_mean).cwiseQuotient(parent_var).array().sum();

    // Typical info CU (using own diag covar)
    // float score = 0.5 * (parent_var.array().log() - child_var.array().log()).sum();

    return score;
}

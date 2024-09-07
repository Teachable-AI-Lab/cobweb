#include "cobweb_continuous_node.h"
#include "cobweb_continuous_tree.h"

CobwebContinuousTree::CobwebContinuousTree(int size, int covar_type, int covar_from)
    : root(nullptr),
      size(size),
      covar_type(covar_type),
      covar_from(covar_from)
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

Eigen::VectorXf CobwebContinuousTree::predict(const Eigen::VectorXf &instance, int max_nodes, bool greedy){
    // this wrapper is useful if we want to do anything before calling predict.
    return this->predict_helper(instance, max_nodes, greedy);
}

Eigen::VectorXf CobwebContinuousTree::predict_helper(const Eigen::VectorXf &instance, int max_nodes, bool greedy){

    float total_weight = 0.0;
    Eigen::VectorXf out = Eigen::VectorXf::Zero(this->size);

    int nodes_expanded = 0;

    float root_ll_inst = this->root->log_prob(instance);

    auto queue = std::priority_queue<std::tuple<float, float, CobwebContinuousNode *>>();

    // std::cout << "root score: " << std::to_string(root_ll_inst) << std::endl;
    queue.push(std::make_tuple(root_ll_inst, 0.0, this->root));

    while (queue.size() > 0){
        auto node = queue.top();
        queue.pop();
        nodes_expanded += 1;

        if (greedy){
            queue = std::priority_queue<
                std::tuple<float, float, CobwebContinuousNode*>>();
        }

        float curr_score = std::get<0>(node);
        float curr_ll = std::get<1>(node);
        CobwebContinuousNode* curr = std::get<2>(node);

        // total_weight += curr_score;
        // std::cout << "weight = logsumexp(" << std::to_string(total_weight) << ", " << std::to_string(curr_score) << ")" << std::endl;
        // std::cout << "weight += " << std::to_string(curr_score) << " (" << std::to_string(exp(curr_score)) << ")" << std::endl;

        if (total_weight == 0){
            total_weight = curr_score;
        } else{
            total_weight = logsumexp_f(total_weight, curr_score);
        }

        // auto curr_preds = curr->predict_probs();
        auto curr_predicted_mean = curr->predict_mean(instance);

        // std::cout << "curr_score: " << std::to_string(curr_score) << std::endl;
        // std::cout << "total weight: " << std::to_string(total_weight) << std::endl;
        // std::cout << "weight ratio: " << std::to_string(exp(curr_score - total_weight)) << std::endl;
        // std::cout << "predicted mean: " << std::endl << curr_predicted_mean << std::endl << std::endl;

        // Eigen::VectorXf delta = curr_predicted_mean - out;
        // out += exp(curr_score - total_weight) * delta;
        out += exp(curr_score - total_weight) * (curr_predicted_mean - out);
        // std::cout << "out: " << std::endl << out << std::endl << std::endl;
        // std::cout << std::endl;

        if (nodes_expanded >= max_nodes) break;

        // TODO look at missing in computing prob children given instance
        //std::vector<double> children_probs = curr->prob_children_given_instance(instance);
        std::vector<float> log_children_probs = curr->log_prob_children_given_instance(instance);

        for (size_t i = 0; i < curr->children.size(); ++i) {
            auto child = curr->children[i];
            float child_ll_inst = child->log_prob(instance);
            float child_ll_given_parent = log_children_probs[i];
            float child_ll = child_ll_given_parent + curr_ll;

            // std::cout << "ll_node: " << child_ll << ", ll_inst: " << child_ll_inst << std::endl;
            queue.push(std::make_tuple(child_ll_inst + child_ll, child_ll, child));
        }
    }

    return out;
}

float CobwebContinuousTree::log_prob(const Eigen::VectorXf &instance, int max_nodes, bool greedy){

    float total_weight = 0.0;
    float out = 0.0;
    int nodes_expanded = 0;

    float root_ll_inst = this->root->log_prob(instance);
    auto queue = std::priority_queue<std::tuple<float, float, CobwebContinuousNode *>>();

    // std::cout << "root score: " << std::to_string(root_ll_inst) << std::endl;
    queue.push(std::make_tuple(root_ll_inst, 0.0, this->root));

    while (queue.size() > 0){
        auto node = queue.top();
        queue.pop();
        nodes_expanded += 1;

        if (greedy){
            queue = std::priority_queue<
                std::tuple<float, float, CobwebContinuousNode*>>();
        }

        float curr_score = std::get<0>(node);
        float curr_ll = std::get<1>(node);
        CobwebContinuousNode* curr = std::get<2>(node);

        if (total_weight == 0){
            total_weight = curr_score;
        } else{
            total_weight = logsumexp_f(total_weight, curr_score);
        }

        // auto curr_preds = curr->predict_probs();
        auto curr_predicted_log_prob = curr->log_prob(instance);

        // std::cout << "curr_score: " << std::to_string(curr_score) << std::endl;
        // std::cout << "total weight: " << std::to_string(total_weight) << std::endl;
        // std::cout << "weight ratio: " << std::to_string(exp(curr_score - total_weight)) << std::endl;
        // std::cout << "predicted mean: " << std::endl << curr_predicted_mean << std::endl << std::endl;

        // Eigen::VectorXf delta = curr_predicted_mean - out;
        // out += exp(curr_score - total_weight) * delta;
        out += exp(curr_score - total_weight) * (curr_predicted_log_prob - out);
        // std::cout << "out: " << std::endl << out << std::endl << std::endl;
        // std::cout << std::endl;

        if (nodes_expanded >= max_nodes) break;

        // TODO look at missing in computing prob children given instance
        //std::vector<double> children_probs = curr->prob_children_given_instance(instance);
        std::vector<float> log_children_probs = curr->log_prob_children_given_instance(instance);

        for (size_t i = 0; i < curr->children.size(); ++i) {
            auto child = curr->children[i];
            float child_ll_inst = child->log_prob(instance);
            float child_ll_given_parent = log_children_probs[i];
            float child_ll = child_ll_given_parent + curr_ll;

            // std::cout << "ll_node: " << child_ll << ", ll_inst: " << child_ll_inst << std::endl;
            queue.push(std::make_tuple(child_ll_inst + child_ll, child_ll, child));
            // queue.push(std::make_tuple(child_ll_inst, child_ll, child));
        }
    }

    return out;
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

    // something like cosine angle?
    // float score = 0.5 * (1 - (child_mean).cwiseProduct(parent_mean).array().sum() / (child_mean.norm() * parent_mean.norm()));

    float score;

    // use own covar
    if (this->covar_from==1){
        // Typical info CU (using own diag covar)
        score = 0.5 * (parent_var.array().log() - child_var.array().log()).sum();
    }
    // use parent covar
    else if (this->covar_from==2){
        score = 0.5 * (child_mean - parent_mean).cwiseProduct(child_mean - parent_mean).cwiseQuotient(parent_var).array().sum();
    }

    return score;
}

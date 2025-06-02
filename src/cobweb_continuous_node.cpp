#include "cobweb_continuous_node.h"
#include "cobweb_continuous_tree.h"

CobwebContinuousNode::CobwebContinuousNode(int size)
    : tree(nullptr),
      parent(nullptr),
      children(),  // This initializes the vector as empty
      mean(Eigen::VectorXf::Zero(size)),
      sum_sq(Eigen::VectorXf::Zero(size)),
      count(0){}

CobwebContinuousNode::CobwebContinuousNode(CobwebContinuousNode *other)
    : tree(other->tree),
      parent(other->parent),
      children(),  // Initialize as an empty vector
      mean(other->mean),  // Copy mean from the other node
      sum_sq(other->sum_sq),  // Copy sum_sq from the other node
      count(other->count) // copy count from the other node
{
    // Recursively clone the children
    for (auto child : other->children)
    {
        children.push_back(new CobwebContinuousNode(child));
    }
}

void CobwebContinuousNode::increment_counts(const Eigen::VectorXf &instance)
{
    this->count += 1;
    Eigen::VectorXf delta = instance - this->mean;
    this->mean += delta / this->count;
    this->sum_sq += delta.cwiseProduct(instance - this->mean);
    // std::cout << std::endl;
    // std::cout << "increment" << std::endl;
    // std::cout << "Mean sum: " << this->mean.array().sum() << std::endl;
    // std::cout << "SumSq sum: " << this->sum_sq.array().sum() << std::endl;
}

int CobwebContinuousNode::depth()
{
    if (this->parent == nullptr)
    {
        return 0;
    }
    else
    {
        return 1 + this->parent->depth();
    }
}

void CobwebContinuousNode::update_counts_from_node(CobwebContinuousNode *other)
{
    Eigen::VectorXf delta = other->mean - this->mean;
    this->sum_sq = (this->sum_sq + other->sum_sq + delta.cwiseProduct(delta) *
                    ((this->count * other->count) / (this->count + other->count)));
    this->mean = ((this->count * this->mean + other->count * other->mean) /
                  (this->count + other->count));
    this->count += other->count;
}

float CobwebContinuousNode::pu_for_insert(CobwebContinuousNode *child,
        const Eigen::VectorXf &instance){

    float score = 0.0;
    auto parent_mean_var = this->mean_var_insert(instance);

    for (auto &c : this->children) {
        std::tuple<Eigen::VectorXf, Eigen::VectorXf> child_mean_var;
        float p_of_child;

        if (c == child){
            p_of_child = (c->count + 1) / (this->count + 1);
            child_mean_var = c->mean_var_insert(instance);
        }
        else{
            p_of_child = c->count / (this->count + 1);
            child_mean_var = c->mean_var();
        }

        // std::cout << "Child var sum: " << std::get<1>(child_mean_var).array().sum() << std::endl;

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    return score / this->children.size();

}

float CobwebContinuousNode::pu_for_new(const Eigen::VectorXf &instance){

    float score = 0.0;
    auto parent_mean_var = this->mean_var_insert(instance);

    for (auto &c : this->children) {
        float p_of_child = c->count / (this->count + 1);
        auto child_mean_var = c->mean_var();

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    float p_of_child = 1.0 / (this->count + 1);
    auto child_mean_var = this->mean_var_new(instance);
    score += p_of_child * this->tree->compute_score(
            std::get<0>(child_mean_var), std::get<1>(child_mean_var),
            std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));

    return score / (this->children.size() + 1);

}

float CobwebContinuousNode::pu_for_merge(CobwebContinuousNode *best1, CobwebContinuousNode *best2, const Eigen::VectorXf &instance){

    float score = 0.0;
    auto parent_mean_var = this->mean_var_insert(instance);

    for (auto &c : this->children) {
        if (c == best1 || c == best2){
            continue;
        }
        float p_of_child = c->count / (this->count + 1);
        auto child_mean_var = c->mean_var();

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    float p_of_child = (best1->count + best2->count + 1) / (this->count + 1);
    auto child_mean_var = best1->mean_var_merge(best2, instance);
    score += p_of_child * this->tree->compute_score(
            std::get<0>(child_mean_var), std::get<1>(child_mean_var),
            std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));

    return score / (this->children.size() - 1);

}

float CobwebContinuousNode::pu_for_split(CobwebContinuousNode *child){

    float score = 0.0;
    auto parent_mean_var = this->mean_var();

    for (auto &c : this->children) {
        if (c == child){
            continue;
        }

        float p_of_child = c->count / this->count;
        auto child_mean_var = c->mean_var();

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    for (auto &c : child->children) {

        float p_of_child = c->count / this->count;
        auto child_mean_var = c->mean_var();

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    return score / (this->children.size() - 1 + child->children.size());

}


bool CobwebContinuousNode::is_exact_match(const Eigen::VectorXf &instance){

    Eigen::VectorXf var = this->sum_sq / this->count;

    if (!is_close_to_zero(var, 1e-5)){
        return false;
    }

    Eigen::VectorXf delta = instance - this->mean;

    if (!is_close_to_zero(delta, 1e-5)){
        return false;
    }

    return true;

}

size_t CobwebContinuousNode::_hash()
{
    return std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(this));
}

std::string CobwebContinuousNode::__str__()
{
    return std::to_string(std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(this)));
    // return "Not Implemented";
    // return this->pretty_print(0);
}

// std::string CobwebContinuousNode::pretty_print(int depth)
// {
//     std::string ret = repeat("\t", depth) + "|-" + avcounts_to_json() + "\n";
//
//     for (auto &c : children)
//     {
//         ret += c->pretty_print(depth + 1);
//     }
//
//     return ret;
// }

std::tuple<float, int> CobwebContinuousNode::get_best_operation(const Eigen::VectorXf &instance, CobwebContinuousNode *best1, CobwebContinuousNode *best2, float best1_pu){
    if (best1 == nullptr)
    {
        throw "Need at least one best child.";
    }

    // std::cout << std::endl;
    std::vector<std::tuple<float, float, int>> operations;
    operations.push_back(std::make_tuple(best1_pu,
                                         custom_rand(),
                                         BEST));
    // std::cout << "BEST: " << best1_pu << std::endl;
    operations.push_back(std::make_tuple(this->pu_for_new(instance),
                                         custom_rand(),
                                         NEW));
    // std::cout << "NEW: " << this->pu_for_new(instance) << std::endl;
    if (children.size() > 2 && best2 != nullptr)
    {
        operations.push_back(std::make_tuple(pu_for_merge(best1, best2,
                                                          instance),
                                             custom_rand(),
                                             MERGE));
        // std::cout << "MERGE: " << this->pu_for_merge(best1, best2, instance) << std::endl;
    }

    if (best1->children.size() > 0)
    {
        operations.push_back(std::make_tuple(pu_for_split(best1),
                                             custom_rand(),
                                             SPLIT));
        // std::cout << "SPLIT: " << this->pu_for_split(best1) << std::endl;
    }

    sort(operations.rbegin(), operations.rend());
    std::pair<float, int> bestOp = std::make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));

    return bestOp;
}

std::tuple<float, CobwebContinuousNode *, CobwebContinuousNode *> CobwebContinuousNode::two_best_children(const Eigen::VectorXf &instance){

    if (this->children.empty())
    {
        throw "No children!";
    }

    auto parent_mean_var = this->mean_var_insert(instance);
    std::tuple<Eigen::VectorXf, Eigen::VectorXf> child_mean_var;

    std::vector<std::tuple<float, float, float, CobwebContinuousNode *>> relative_pu;
    for (auto &child : this->children)
    {
        float p_of_c = (child->count + 1) / (this->count + 1);
        child_mean_var = child->mean_var_insert(instance);
        float score_gain = p_of_c * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));

        // std::cout << "SCORE GAIN (first): " << child->count << " " << this->count << " " << p_of_c << " " << score_gain << std::endl;

        p_of_c = child->count / (this->count + 1);
        child_mean_var = child->mean_var();
        score_gain -= p_of_c * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));

        // std::cout << "SCORE GAIN: " << score_gain << std::endl;

        relative_pu.push_back(
            std::make_tuple(
                score_gain,
                child->count,
                custom_rand(),
                child));
    }

    sort(relative_pu.rbegin(), relative_pu.rend());
    CobwebContinuousNode *best1 = std::get<3>(relative_pu[0]);
    float best1_pu = pu_for_insert(best1, instance);
    // float best1_pu = 0.0;
    CobwebContinuousNode *best2 = relative_pu.size() > 1 ? std::get<3>(relative_pu[1]) : nullptr;
    return std::make_tuple(best1_pu, best1, best2);

}

float CobwebContinuousNode::log_prob_class_given_instance(const Eigen::VectorXf &instance){
    return this->log_prob(instance) + log(this->count) - log(this->tree->root->count);
}

std::vector<float> CobwebContinuousNode::log_prob_children_given_instance(const Eigen::VectorXf &instance){
    std::vector<float> raw_log_probs = std::vector<float>();
    std::vector<float> norm_log_probs = std::vector<float>();

    for (auto &child: this->children){
        raw_log_probs.push_back(child->log_prob_class_given_instance(instance));
    }

    float log_p_of_x = logsumexp(raw_log_probs);

    for (auto log_p: raw_log_probs){
        norm_log_probs.push_back(log_p - log_p_of_x);
    }

    return norm_log_probs;
}

float CobwebContinuousNode::log_prob(const Eigen::VectorXf &instance){
    // Linked var with parent
    Eigen::ArrayXf var;

    // use own covar
    if (this->tree->covar_from==1){
        var = (this->sum_sq / this->count + this->tree->prior_var).array();
    }

    // use parent's covar
    else if (this->tree->covar_from==2){
        if (this->parent){
            var = (this->parent->sum_sq / this->parent->count + this->tree->prior_var);
        } else {
            var = (this->sum_sq / this->count + this->tree->prior_var);
        }
    }

    return -0.5 * (var.log() + log(2.0f * M_PI) + ((instance - this->mean).array().square() / var)).sum();
}

const Eigen::VectorXf& CobwebContinuousNode::predict_mean(const Eigen::VectorXf &instance){
    return this->mean;
}

std::tuple<Eigen::VectorXf, Eigen::VectorXf> CobwebContinuousNode::mean_var(){
    Eigen::VectorXf var = this->tree->compute_var(this->sum_sq, this->count);
    return std::make_tuple(this->mean, std::move(var));
}

std::tuple<Eigen::VectorXf, Eigen::VectorXf> CobwebContinuousNode::mean_var_new(const Eigen::VectorXf &instance){
    return std::make_tuple(instance, this->tree->prior_var);
}

std::tuple<Eigen::VectorXf, Eigen::VectorXf> CobwebContinuousNode::mean_var_insert(const Eigen::VectorXf &instance){

    float count = this->count + 1;
    Eigen::VectorXf delta = instance - this->mean;
    Eigen::VectorXf mean = this->mean + delta / count;
    Eigen::VectorXf sum_sq = this->sum_sq + delta.cwiseProduct(instance - mean);
    Eigen::VectorXf var = this->tree->compute_var(sum_sq, count);
    return std::make_tuple(std::move(mean), std::move(var));

}

std::tuple<Eigen::VectorXf, Eigen::VectorXf> CobwebContinuousNode::mean_var_merge(CobwebContinuousNode *other, const Eigen::VectorXf &instance){

    Eigen::VectorXf delta = other->mean - this->mean;
    Eigen::VectorXf sum_sq = (this->sum_sq + other->sum_sq + delta.cwiseProduct(delta) *
            ((this->count * other->count) / (this->count + other->count)));
    Eigen::VectorXf mean = ((this->count * this->mean + other->count * other->mean) /
            (this->count + other->count));
    float count = this->count + other->count;

    count += 1;
    delta = instance - mean;
    mean += delta / count;
    sum_sq += delta.cwiseProduct(instance - mean);

    Eigen::VectorXf var = this->tree->compute_var(sum_sq, count);
    return std::make_tuple(std::move(mean), std::move(var));

}

std::string CobwebContinuousNode::output_json()
{
    std::string output = "{";

    output += "\"name\": \"Concept" + std::to_string(this->_hash()) + "\",\n";
    output += "\"size\": " + std::to_string(this->count) + ",\n";
    output += "\"children\": [\n";
    bool first = true;
    for (auto &c : children)
    {
        if (!first)
            output += ",";
        else
            first = false;
        output += c->output_json();
    }
    output += "],\n";

    output += "\"counts\": {},\n";
    // output += "\"counts\": " + this->avcounts_to_json() + ",\n";
    // output += "\"attr_counts\": " + this->a_count_to_json() + "\n";

    output += "}\n";

    return output;
}

nb::dict CobwebContinuousNode::to_map() {
    nanobind::gil_scoped_acquire gil;
    nb::dict node;
    node["node_id"] = std::to_string(this->_hash());

    nb::list mean_list;
    for (int i = 0; i < this->mean.size(); ++i) {
        mean_list.append(this->mean(i));
    }
    node["mean"] = mean_list;

    nb::list sum_sq_list;
    for (int i = 0; i < this->sum_sq.size(); ++i) {
        // float value = (this->count > 0) ? (this->sum_sq(i) / this->count) : 0.0f;
        sum_sq_list.append(this->sum_sq(i));
    }
    node["sum_sq"] = sum_sq_list;

    node["count"] = this->count;

    nb::list logvar_list;
    const float epsilon = 1e-8f;
    for (int i = 0; i < this->sum_sq.size(); ++i) {
        float value = (this->count > 0) ? (this->sum_sq(i) / this->count) : 0.0f;
        logvar_list.append(std::log(value + epsilon));
    }
    node["logvar"] = logvar_list;

    nb::list children_list;
    for (auto *child : this->children) {
        children_list.append(child->to_map());
    }
    node["children"] = children_list;

    return node;
}


std::string CobwebContinuousNode::export_tree_json() {
    std::stringstream ss;
    ss << "{";
    ss << "\"node_id\": \"" << this->_hash() << "\",";
    ss << "\"mean\": [";
    for (int i = 0; i < this->mean.size(); i++) {
        ss << this->mean(i);
        if (i < this->mean.size() - 1)
            ss << ", ";
    }
    ss << "],";
    ss << "\"count\": " << this->count << ",";
    ss << "\"sum_sq\": [";
    for (int i = 0; i < this->sum_sq.size(); i++) {
        ss << this->sum_sq(i);
        if (i < this->sum_sq.size() - 1)
            ss << ", ";
    }
    ss << "]";
    if (!this->children.empty()) {
        ss << ", \"children\": [";
        for (size_t i = 0; i < this->children.size(); i++) {
            ss << this->children[i]->export_tree_json();
            if (i < this->children.size() - 1)
                ss << ", ";
        }
        ss << "]";
    }
    ss << "}";
    return ss.str();
}
#include <fstream>
#include <iostream>

void CobwebContinuousNode::save_tree_to_file(const std::string &filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return;
    }
    file << export_tree_json();
    file.close();
}
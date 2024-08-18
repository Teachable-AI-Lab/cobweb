#include "cobweb_continuous_node.h"
#include "cobweb_continuous_tree.h"

CobwebContinuousNode::CobwebContinuousNode(int size)
    : tree(nullptr),
      parent(nullptr),
      children(),  // This initializes the vector as empty
      mean(Eigen::VectorXd::Zero(size)),
      sum_sq(Eigen::VectorXd::Zero(size)),
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

void CobwebContinuousNode::increment_counts(const Eigen::VectorXd &instance)
{
    this->count += 1;
    Eigen::VectorXd delta = instance - this->mean;
    this->mean += delta / this->count;
    this->sum_sq += delta.cwiseProduct(instance - this->mean);
    // std::cout << std::endl;
    // std::cout << "increment" << std::endl;
    // std::cout << "Mean sum: " << this->mean.array().sum() << std::endl;
    // std::cout << "SumSq sum: " << this->sum_sq.array().sum() << std::endl;
}

void CobwebContinuousNode::update_counts_from_node(CobwebContinuousNode *other)
{
    Eigen::VectorXd delta = other->mean - this->mean;
    this->sum_sq = (this->sum_sq + other->sum_sq + delta.cwiseProduct(delta) *
                    ((this->count * other->count) / (this->count + other->count)));
    this->mean = ((this->count * this->mean + other->count * other->mean) /
                  (this->count + other->count));
    this->count += other->count;
}

double CobwebContinuousNode::pu_for_insert(CobwebContinuousNode *child,
        const Eigen::VectorXd &instance){

    double score = 0.0;
    auto parent_mean_var = this->mean_var_insert(instance);

    for (auto &c : this->children) {
        std::tuple<Eigen::VectorXd, Eigen::VectorXd> child_mean_var;
        double p_of_child;

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

double CobwebContinuousNode::pu_for_new(const Eigen::VectorXd &instance){

    double score = 0.0;
    auto parent_mean_var = this->mean_var_insert(instance);

    for (auto &c : this->children) {
        double p_of_child = c->count / (this->count + 1);
        auto child_mean_var = c->mean_var();

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    double p_of_child = 1.0 / (this->count + 1);
    auto child_mean_var = this->mean_var_new(instance);
    score += p_of_child * this->tree->compute_score(
            std::get<0>(child_mean_var), std::get<1>(child_mean_var),
            std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));

    return score / (this->children.size() + 1);

}

double CobwebContinuousNode::pu_for_merge(CobwebContinuousNode *best1, CobwebContinuousNode *best2, const Eigen::VectorXd &instance){

    double score = 0.0;
    auto parent_mean_var = this->mean_var_insert(instance);

    for (auto &c : this->children) {
        if (c == best1 || c == best2){
            continue;
        }
        double p_of_child = c->count / (this->count + 1);
        auto child_mean_var = c->mean_var();

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    double p_of_child = (best1->count + best2->count + 1) / (this->count + 1);
    auto child_mean_var = best1->mean_var_merge(best2, instance);
    score += p_of_child * this->tree->compute_score(
            std::get<0>(child_mean_var), std::get<1>(child_mean_var),
            std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));

    return score / (this->children.size() - 1);

}

double CobwebContinuousNode::pu_for_split(CobwebContinuousNode *child){

    double score = 0.0;
    auto parent_mean_var = this->mean_var();

    for (auto &c : this->children) {
        if (c == child){
            continue;
        }

        double p_of_child = c->count / this->count;
        auto child_mean_var = c->mean_var();

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    for (auto &c : child->children) {

        double p_of_child = c->count / this->count;
        auto child_mean_var = c->mean_var();

        score += p_of_child * this->tree->compute_score(
                std::get<0>(child_mean_var), std::get<1>(child_mean_var),
                std::get<0>(parent_mean_var), std::get<1>(parent_mean_var));
    }

    return score / (this->children.size() - 1 + child->children.size());

}


bool CobwebContinuousNode::is_exact_match(const Eigen::VectorXd &instance){

    Eigen::VectorXd var = this->sum_sq / this->count;

    if (!is_close_to_zero(var, 1e-5)){
        return false;
    }

    Eigen::VectorXd delta = instance - this->mean;

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
    return "Not Implemented";
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

std::tuple<double, int> CobwebContinuousNode::get_best_operation(const Eigen::VectorXd &instance, CobwebContinuousNode *best1, CobwebContinuousNode *best2, double best1_pu){
    if (best1 == nullptr)
    {
        throw "Need at least one best child.";
    }

    // std::cout << std::endl;
    std::vector<std::tuple<double, double, int>> operations;
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
    std::pair<double, int> bestOp = std::make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));

    return bestOp;
}

std::tuple<double, CobwebContinuousNode *, CobwebContinuousNode *> CobwebContinuousNode::two_best_children(const Eigen::VectorXd &instance){

    if (this->children.empty())
    {
        throw "No children!";
    }

    auto parent_mean_var = this->mean_var_insert(instance);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> child_mean_var;

    std::vector<std::tuple<double, double, double, CobwebContinuousNode *>> relative_pu;
    for (auto &child : this->children)
    {
        double p_of_c = (child->count + 1) / (this->count + 1);
        child_mean_var = child->mean_var_insert(instance);
        double score_gain = p_of_c * this->tree->compute_score(
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
    double best1_pu = pu_for_insert(best1, instance);
    // double best1_pu = 0.0;
    CobwebContinuousNode *best2 = relative_pu.size() > 1 ? std::get<3>(relative_pu[1]) : nullptr;
    return std::make_tuple(best1_pu, best1, best2);

}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> CobwebContinuousNode::mean_var(){
    Eigen::VectorXd var = this->tree->compute_var(this->sum_sq, this->count);
    return std::make_tuple(this->mean, std::move(var));
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> CobwebContinuousNode::mean_var_new(const Eigen::VectorXd &instance){
    return std::make_tuple(instance, this->tree->prior_var);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> CobwebContinuousNode::mean_var_insert(const Eigen::VectorXd &instance){

    double count = this->count + 1;
    Eigen::VectorXd delta = instance - this->mean;
    Eigen::VectorXd mean = this->mean + delta / count;
    Eigen::VectorXd sum_sq = this->sum_sq + delta.cwiseProduct(instance - mean);
    Eigen::VectorXd var = this->tree->compute_var(sum_sq, count);
    return std::make_tuple(std::move(mean), std::move(var));

}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> CobwebContinuousNode::mean_var_merge(CobwebContinuousNode *other, const Eigen::VectorXd &instance){

    Eigen::VectorXd delta = other->mean - this->mean;
    Eigen::VectorXd sum_sq = (this->sum_sq + other->sum_sq + delta.cwiseProduct(delta) *
            ((this->count * other->count) / (this->count + other->count)));
    Eigen::VectorXd mean = ((this->count * this->mean + other->count * other->mean) /
            (this->count + other->count));
    double count = this->count + other->count;

    count += 1;
    delta = instance - mean;
    mean += delta / count;
    sum_sq += delta.cwiseProduct(instance - mean);

    Eigen::VectorXd var = this->tree->compute_var(sum_sq, count);
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

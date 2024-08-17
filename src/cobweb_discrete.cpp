#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <random>
#include <tuple>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <variant>

#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/filereadstream.h"
#include <stack>

#include "assert.h"
#include "json.hpp"
#include "cached_string.hpp"
#include "BS_thread_pool.hpp"

namespace py = pybind11;

#define NULL_STRING CachedString("\0")
#define BEST 0
#define NEW 1
#define MERGE 2
#define SPLIT 3

// typedef CachedString ATTR_TYPE;
// typedef CachedString VALUE_TYPE;
// typedef double COUNT_TYPE;
// typedef std::unordered_map<std::string, std::unordered_map<std::string, COUNT_TYPE>> INSTANCE_TYPE;
// typedef std::unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
// typedef std::unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
// typedef std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
// typedef std::unordered_map<ATTR_TYPE, COUNT_TYPE> ATTR_COUNT_TYPE;
// typedef std::pair<double, int> OPERATION_TYPE;

typedef int ATTR_TYPE;
typedef int VALUE_TYPE;
typedef double COUNT_TYPE;
typedef std::unordered_map<int, std::unordered_map<int, COUNT_TYPE>> INSTANCE_TYPE;
typedef std::unordered_map<VALUE_TYPE, COUNT_TYPE> VAL_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, VAL_COUNT_TYPE> AV_COUNT_TYPE;
typedef std::unordered_map<ATTR_TYPE, std::unordered_set<VALUE_TYPE>> AV_KEY_TYPE;
typedef std::unordered_map<ATTR_TYPE, COUNT_TYPE> ATTR_COUNT_TYPE;
typedef std::pair<double, int> OPERATION_TYPE;

class CobwebTree;
class CobwebNode;

std::random_device rd;
std::mt19937_64 gen(rd());
std::uniform_real_distribution<double> unif(0, 1);

std::unordered_map<int, double> lgammaCache;
std::unordered_map<int, std::unordered_map<int, int>> binomialCache;
std::unordered_map<int, std::unordered_map<double, double>> entropy_k_cache;

const std::unordered_map<std::string, int> ATTRIBUTE_MAP = {
    {"alpha", 1000000},
    {"weight_attr", 10000001},
    {"objective", 10000002},
    {"children_norm", 10000003},
    {"norm_attributes", 10000004},
    {"root", 10000005},
    {"count", 100000011},
    {"a_count", 100000012},
    {"sum_n_logn", 100000013},
    {"av_count", 100000014},
    {"children", 100000015}};


void displayProgressBar(int width, double progress_percentage, double seconds_elapsed)
{

    int hours = seconds_elapsed / 3600;
    int minutes = (seconds_elapsed - hours * 3600) / 60;
    int secs = seconds_elapsed - hours * 3600 - minutes * 60;

    double estimated = seconds_elapsed / progress_percentage * (1.0 - progress_percentage);

    int hours_left = estimated / 3600;
    int minutes_left = (estimated - hours_left * 3600) / 60;
    int secs_left = estimated - hours_left * 3600 - minutes_left * 60;

    int pos = width * progress_percentage;
    std::cout << "[";
    for (int i = 0; i < width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress_percentage * 100.0) << " %; " << hours << ":" << std::setfill('0') << std::setw(2) << minutes << ":" << std::setfill('0') << std::setw(2) << secs << " elapsed; " << hours_left << ":" << std::setfill('0') << std::setw(2) << minutes_left << ":" << std::setfill('0') << std::setw(2) << secs_left << " left\r";
    std::cout.flush();
}

double lgamma_cached(int n)
{
    auto it = lgammaCache.find(n);
    if (it != lgammaCache.end())
        return it->second;

    double result = std::lgamma(n);
    lgammaCache[n] = result;
    return result;
}

int nChoosek(int n, int k)
{
    if (k > n)
        return 0;
    if (k * 2 > n)
        k = n - k;
    if (k == 0)
        return 1;

    // Check if the value is in the cache
    auto it_n = binomialCache.find(n);
    if (it_n != binomialCache.end())
    {
        auto it_k = it_n->second.find(k);
        if (it_k != it_n->second.end())
            return it_k->second;
    }

    int result = n;
    for (int i = 2; i <= k; ++i)
    {
        result *= (n - i + 1);
        result /= i;
    }

    // Store the computed value in the cache
    binomialCache[n][k] = result;
    // std::cout << n << "!" << k << " : " << result << std::endl;

    return result;
}

double entropy_component_k(int n, double p)
{
    if (p == 0.0 || p == 1.0)
    {
        return 0.0;
    }

    auto it_n = entropy_k_cache.find(n);
    if (it_n != entropy_k_cache.end())
    {
        auto it_p = it_n->second.find(p);
        if (it_p != it_n->second.end())
            return it_p->second;
    }

    double precision = 1e-10;
    double info = -n * p * log(p);

    // This is where we'll see the highest entropy
    int mid = std::ceil(n * p);

    for (int xi = mid; xi > 2; xi--)
    {
        double v = nChoosek(n, xi) * std::pow(p, xi) * std::pow((1 - p), (n - xi)) * lgamma_cached(xi + 1);
        if (v < precision)
            break;
        info += v;
    }

    for (int xi = mid + 1; xi <= n; xi++)
    {
        double v = nChoosek(n, xi) * std::pow(p, xi) * std::pow((1 - p), (n - xi)) * lgamma_cached(xi + 1);
        if (v < precision)
            break;
        info += v;
    }

    entropy_k_cache[n][p] = info;

    return info;
}

double custom_rand()
{
    return unif(gen);
}

std::string repeat(std::string s, int n)
{
    std::string res = "";
    for (int i = 0; i < n; i++)
    {
        res += s;
    }
    return res;
}

VALUE_TYPE most_likely_choice(std::vector<std::tuple<VALUE_TYPE, double>> choices)
{
    std::vector<std::tuple<double, double, VALUE_TYPE>> vals;

    for (auto &[val, prob] : choices)
    {
        if (prob < 0)
        {
            std::cout << "most_likely_choice: all weights must be greater than or equal to 0" << std::endl;
        }
        vals.push_back(std::make_tuple(prob, custom_rand(), val));
    }
    sort(vals.rbegin(), vals.rend());

    return std::get<2>(vals[0]);
}

VALUE_TYPE weighted_choice(std::vector<std::tuple<VALUE_TYPE, double>> choices)
{
    std::cout << "weighted_choice: Not implemented yet" << std::endl;
    return std::get<0>(choices[0]);
}

std::string doubleToString(double cnt)
{
    std::ostringstream stream;
    // Set stream to output floating point numbers with maximum precision
    stream << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10) << cnt;
    return stream.str();
}

double logsumexp(std::vector<double> arr)
{
    if (arr.size() > 0)
    {
        double max_val = arr[0];
        double sum = 0;

        for (auto &v : arr)
        {
            if (v > max_val)
            {
                max_val = v;
            }
        }

        for (auto &v : arr)
        {
            sum += exp(v - max_val);
        }
        return log(sum) + max_val;
    }
    else
    {
        return 0.0;
    }
}

double logsumexp(double n1, double n2)
{
    double max_val = std::max(n1, n2);
    return log(exp(n1 - max_val) + exp(n2 - max_val)) + max_val;
}

class CobwebNode
{

public:
    CobwebTree *tree;
    CobwebNode *parent;
    std::vector<CobwebNode *> children;

    COUNT_TYPE count;
    ATTR_COUNT_TYPE a_count;
    ATTR_COUNT_TYPE sum_n_logn;
    AV_COUNT_TYPE av_count;

    CobwebNode();
    CobwebNode(CobwebNode *otherNode);
    void increment_counts(const AV_COUNT_TYPE &instance);
    void update_counts_from_node(CobwebNode *node);
    double entropy_attr_insert(ATTR_TYPE attr, const AV_COUNT_TYPE &instance);
    double entropy_insert(const AV_COUNT_TYPE &instance);
    double entropy_attr_merge(ATTR_TYPE attr, CobwebNode *other, const AV_COUNT_TYPE &instance);
    double entropy_merge(CobwebNode *other, const AV_COUNT_TYPE
                                                &instance);
    CobwebNode *get_best_level(INSTANCE_TYPE instance);
    CobwebNode *get_basic_level();
    double category_utility();
    double entropy_attr(ATTR_TYPE attr);
    double entropy();
    double partition_utility();
    std::tuple<double, int> get_best_operation(const AV_COUNT_TYPE
                                                   &instance,
                                               CobwebNode *best1, CobwebNode *best2, double best1Cu);
    std::tuple<double, CobwebNode *, CobwebNode *>
    two_best_children(const AV_COUNT_TYPE &instance);
    std::vector<double> log_prob_children_given_instance(const AV_COUNT_TYPE &instance);
    std::vector<double> log_prob_children_given_instance_ext(INSTANCE_TYPE instance);
    std::vector<double> prob_children_given_instance(const AV_COUNT_TYPE &instance);
    std::vector<double> prob_children_given_instance_ext(INSTANCE_TYPE instance);
    double log_prob_instance(const AV_COUNT_TYPE &instance);
    double log_prob_instance_missing(const AV_COUNT_TYPE &instance);
    double log_prob_instance_ext(INSTANCE_TYPE instance);
    double log_prob_instance_missing_ext(INSTANCE_TYPE instance);
    double log_prob_class_given_instance(const AV_COUNT_TYPE &instance,
                                         bool use_root_counts = false);
    double log_prob_class_given_instance_ext(INSTANCE_TYPE instance,
                                             bool use_root_counts = false);
    double pu_for_insert(CobwebNode *child, const AV_COUNT_TYPE
                                                &instance);
    double pu_for_new_child(const AV_COUNT_TYPE &instance);
    double pu_for_merge(CobwebNode *best1, CobwebNode *best2, const AV_COUNT_TYPE &instance);
    double pu_for_split(CobwebNode *best);
    bool is_exact_match(const AV_COUNT_TYPE &instance);
    size_t _hash();
    std::string __str__();
    std::string concept_hash();
    std::string pretty_print(int depth = 0);
    int depth();
    bool is_parent(CobwebNode *otherConcept);
    int num_concepts();
    std::string avcounts_to_json();
    std::string ser_avcounts();
    std::string a_count_to_json();
    std::string sum_n_logn_to_json();
    std::string dump_json();
    std::string output_json();
    std::vector<std::tuple<VALUE_TYPE, double>>
    get_weighted_values(ATTR_TYPE attr, bool allowNone = true);
    std::unordered_map<int, std::unordered_map<int, double>> predict_probs();
    std::unordered_map<int, std::unordered_map<int, double>> predict_log_probs();
    std::unordered_map<int, std::unordered_map<int, double>> predict_weighted_probs(INSTANCE_TYPE instance);
    std::unordered_map<int, std::unordered_map<int, double>> predict_weighted_leaves_probs(INSTANCE_TYPE instance);
    VALUE_TYPE predict(ATTR_TYPE attr, std::string choiceFn = "most likely",
                       bool allowNone = true);
    double probability(ATTR_TYPE attr, VALUE_TYPE val);
};

class CobwebTree
{

public:
    float alpha;
    bool weight_attr;
    int objective;
    bool children_norm;
    bool norm_attributes;
    CobwebNode *root;
    AV_KEY_TYPE attr_vals;

    CobwebTree(float alpha, bool weight_attr, int objective, bool children_norm, bool norm_attributes)
    {
        this->alpha = alpha;
        this->weight_attr = weight_attr;
        this->objective = objective;
        this->children_norm = children_norm;
        this->norm_attributes = norm_attributes;

        this->root = new CobwebNode();
        this->root->tree = this;
        this->attr_vals = AV_KEY_TYPE();
    }

    std::string __str__()
    {
        return this->root->__str__();
    }

    // CobwebNode *load_json_helper(json_object_s *object)
    // {
    //     CobwebNode *new_node = new CobwebNode();
    //     new_node->tree = this;

    //     // // Get concept_id
    //     // struct json_object_element_s* concept_id_obj = object->start;
    //     // unsigned long long concept_id_val = stoull(json_value_as_number(concept_id_obj->value)->number);
    //     // new_node->concept_id = concept_id_val;
    //     // new_node->update_counter(concept_id_val);

    //     // Get count
    //     struct json_object_element_s *count_obj = object->start;
    //     // struct json_object_element_s* count_obj = object->start;
    //     double count_val = atof(json_value_as_number(count_obj->value)->number);
    //     new_node->count = count_val;

    //     // Get a_count
    //     struct json_object_element_s *a_count_obj = count_obj->next;
    //     struct json_object_s *a_count_dict = json_value_as_object(a_count_obj->value);
    //     struct json_object_element_s *a_count_cursor = a_count_dict->start;
    //     while (a_count_cursor != NULL)
    //     {
    //         // Get attr name
    //         std::string attr_name = std::string(a_count_cursor->name->string);

    //         // A count is stored with each attribute
    //         double count_value = atof(json_value_as_number(a_count_cursor->value)->number);
    //         new_node->a_count[attr_name] = count_value;

    //         a_count_cursor = a_count_cursor->next;
    //     }

    //     // Get sum_n_logn
    //     struct json_object_element_s *sum_n_logn_obj = a_count_obj->next;
    //     struct json_object_s *sum_n_logn_dict = json_value_as_object(sum_n_logn_obj->value);
    //     struct json_object_element_s *sum_n_logn_cursor = sum_n_logn_dict->start;
    //     while (sum_n_logn_cursor != NULL)
    //     {
    //         // Get attr name
    //         std::string attr_name = std::string(sum_n_logn_cursor->name->string);

    //         // A count is stored with each attribute
    //         double count_value = atof(json_value_as_number(sum_n_logn_cursor->value)->number);
    //         new_node->sum_n_logn[attr_name] = count_value;
    //         sum_n_logn_cursor = sum_n_logn_cursor->next;
    //     }

    //     // Get av counts
    //     struct json_object_element_s *av_count_obj = sum_n_logn_obj->next;
    //     struct json_object_s *av_count_dict = json_value_as_object(av_count_obj->value);
    //     struct json_object_element_s *av_count_cursor = av_count_dict->start;
    //     while (av_count_cursor != NULL)
    //     {
    //         // Get attr name
    //         std::string attr_name = std::string(av_count_cursor->name->string);

    //         // The attr val is a dict of strings to ints
    //         struct json_object_s *attr_val_dict = json_value_as_object(av_count_cursor->value);
    //         struct json_object_element_s *inner_counts_cursor = attr_val_dict->start;
    //         while (inner_counts_cursor != NULL)
    //         {
    //             // this will be a word
    //             std::string val_name = std::string(inner_counts_cursor->name->string);

    //             // This will always be a number
    //             double attr_val_count = atof(json_value_as_number(inner_counts_cursor->value)->number);
    //             // Update the new node's counts
    //             new_node->av_count[attr_name][val_name] = attr_val_count;

    //             inner_counts_cursor = inner_counts_cursor->next;
    //         }

    //         av_count_cursor = av_count_cursor->next;
    //     }

    //     // At this point in the coding, I am supremely annoyed at
    //     // myself for choosing this approach.

    //     // Get children
    //     struct json_object_element_s *children_obj = av_count_obj->next;
    //     struct json_array_s *children_array = json_value_as_array(children_obj->value);
    //     struct json_array_element_s *child_cursor = children_array->start;
    //     std::vector<CobwebNode *> new_children;
    //     while (child_cursor != NULL)
    //     {
    //         struct json_object_s *json_child = json_value_as_object(child_cursor->value);
    //         CobwebNode *child = load_json_helper(json_child);
    //         child->parent = new_node;
    //         new_children.push_back(child);
    //         child_cursor = child_cursor->next;
    //     }
    //     new_node->children = new_children;

    //     return new_node;

    //     // It's important to me that you know that this code
    //     // worked on the first try.
    // }

    std::string dump_json()
    {
        std::string output = "{";

        output += "\"alpha\": " + doubleToString(this->alpha) + ",\n";
        output += "\"weight_attr\": " + std::to_string(this->weight_attr) + ",\n";
        output += "\"objective\": " + std::to_string(this->objective) + ",\n";
        output += "\"children_norm\": " + std::to_string(this->children_norm) + ",\n";
        output += "\"norm_attributes\": " + std::to_string(this->norm_attributes) + ",\n";
        output += "\"root\": " + this->root->dump_json();
        output += "}\n";

        return output;
        // return this->root->dump_json();
    }

    // void load_json_stream(std::string json_path) {
    //     // Open the file
    //     FILE* fp = fopen(json_path.c_str(), "rb");
    //     if (!fp) {
    //         std::cerr << "Could not open file: " << json_path << std::endl;
    //         return;
    //     }

    //     // Allocate a buffer for file reading
    //     const size_t bufferSize = 65536; // 64KB buffer size
    //     char buffer[bufferSize];

    //     // Create a FileReadStream object
    //     rapidjson::FileReadStream is(fp, buffer, sizeof(buffer));

    //     // Create a JSON reader
    //     rapidjson::Reader reader;

    //     // Create an instance of the handler
    //     MyHandler handler;

    //     // Parse the JSON file using the SAX API
    //     if (!reader.Parse(is, handler)) {
    //         std::cerr << "Error while parsing JSON." << std::endl;
    //     }

    //     // Close the file
    //     fclose(fp);

    //     std::cout << "JSON file parsed successfully." << std::endl;
    // }

    // void load_json_stream(std::string json_path);

    class MyHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, MyHandler>
    {
    public:
        CobwebTree *cobwebTree;
        std::stack<CobwebNode *> nodeStack;
        std::stack<int> keyStack; // Stack to keep track of the nested keys

        MyHandler(CobwebTree *tree) : cobwebTree(tree) {}

        bool Key(const char *str, rapidjson::SizeType length, bool copy)
        {
            std::string key(str, length);
            // check if the key is in the map
            if (ATTRIBUTE_MAP.find(key) != ATTRIBUTE_MAP.end())
            {
                keyStack.push(ATTRIBUTE_MAP.at(key));
            }
            else
            {
                int key = std::stoi(std::string(str, length));
                keyStack.push(key);
            }
            // int key = std::stoi(std::string(str, length));
            // keyStack.push(key);

            // tree address
            // std::cout << "\ntree address at key: " << cobwebTree << std::endl;
            // print key
            // std::cout << "key: " << key << std::endl;
            // print stack length
            // std::cout << "stack length: " << keyStack.size() << std::endl;
            return true;
        }

        bool Bool(bool b)
        {
            if (keyStack.top() == ATTRIBUTE_MAP.at("weight_attr"))
            {
                cobwebTree->weight_attr = b;
                keyStack.pop();
                // print here
                // std::cout << "weight_attr: " << b << std::endl;
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("children_norm"))
            {
                cobwebTree->children_norm = b;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("norm_attributes"))
            {
                cobwebTree->norm_attributes = b;
                keyStack.pop();
            }
            return true;
        }

        bool Uint(unsigned i)
        {
            if (keyStack.top() == ATTRIBUTE_MAP.at("objective"))
            {
                cobwebTree->objective = i;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("weight_attr"))
            {
                // print here
                // std::cout << "weight_attr: " << i << std::endl;
                cobwebTree->weight_attr = i;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("children_norm"))
            {
                cobwebTree->children_norm = i;
                keyStack.pop();
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("norm_attributes"))
            {
                cobwebTree->norm_attributes = i;
                keyStack.pop();
            }
            return true;
        }

        int pop_x(int x)
        {
            // copy the stack, not just the reference
            std::stack<int> tempStack = keyStack;
            if (x > tempStack.size())
            {
                return -1;
            }
            for (int i = 0; i < x; i++)
            {
                tempStack.pop();
            }
            return tempStack.top();
        }

        bool Double(double d)
        {
            // std::cout << "nodeStack length: " << nodeStack.size() << std::endl;
            // std::cout << "keyStack length: " << keyStack.size() << std::endl;
            // // check if the stack is empty
            // // CobwebNode *currentNode = nodeStack.top();
            // std::cout << "double currentNode top" << std::endl;
            int currentKey = keyStack.top();
            // // CobwebNode *currentNode = nodeStack.top();
            // std::cout << "double currentKey: " << std::endl;
            // change it to vector of strings

            if (currentKey == ATTRIBUTE_MAP.at("alpha"))
            {
                // std::cout << "nodeStack top at alpha: " << nodeStack.top() << std::endl;
                std::cout << "alpha: " << d << std::endl;
                std::cout << "currentKey: " << currentKey << std::endl;
                cobwebTree->alpha = d;
                keyStack.pop();
                // std::cout << "alpha: " << d << std::endl;
            }
            else if (currentKey == ATTRIBUTE_MAP.at("count"))
            {
                CobwebNode *currentNode = nodeStack.top();
                // std::cout << "nodeStack top at count: " << nodeStack.top() << std::endl;
                currentNode->count = d;
                keyStack.pop();
                // std::cout << "count: " << d << std::endl;
            }
            else if (pop_x(1) == ATTRIBUTE_MAP.at("a_count"))
            {
                CobwebNode *currentNode = nodeStack.top();
                // std::cout << "nodeStack top at a_count: " << nodeStack.top() << std::endl;
                currentNode->a_count[currentKey] = d;
                keyStack.pop();
            }
            else if (pop_x(1) == ATTRIBUTE_MAP.at("sum_n_logn"))
            {
                CobwebNode *currentNode = nodeStack.top();
                // std::cout << "nodeStack top at sum_n_logn: " << nodeStack.top() << std::endl;
                // print here
                // std::cout << "sum_n_logn, c: " << d << std::endl;
                // output the keys of sum_n_logn
                // std::cout << "sum_n_logn keys: " << currentNode->sum_n_logn.size() << std::endl;

                currentNode->sum_n_logn[currentKey] = d;
                // std::cout << "assigned sum_n_logn" << std::endl;
                keyStack.pop();
            }
            else if (pop_x(2) == ATTRIBUTE_MAP.at("av_count"))
            {
                CobwebNode *currentNode = nodeStack.top();
                // std::cout << "nodeStack top at av_count: " << nodeStack.top() << std::endl;
                currentNode->av_count[pop_x(1)][currentKey] = d;
                keyStack.pop();
            }
            else
            {
                // std::cout << "nodeStack top at other double: " << nodeStack.top() << std::endl;
                // std::cout << "other double key: " << keyStack.top() << std::endl;
                keyStack.pop();
            }
            return true;
        }

        bool StartObject()
        {
            // std::cout << "start object" << std::endl;
            if (keyStack.empty())
            {
                return true;
            }
            if (keyStack.top() == ATTRIBUTE_MAP.at("root"))
            {
                // nodeStack.push(new CobwebNode());
                CobwebNode *rootNode = new CobwebNode();
                nodeStack.push(rootNode);
                cobwebTree->root = rootNode;
                // print out the length of the stack
                // std::cout << "stack length at root start: " << nodeStack.size() << std::endl;
                // std::cout << "root node address at root start: " << nodeStack.top() << std::endl;
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("a_count") || keyStack.top() == ATTRIBUTE_MAP.at("sum_n_logn") || keyStack.top() == ATTRIBUTE_MAP.at("av_count"))
            {
                return true;
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("children") && !nodeStack.empty())
            {
                // added children
                // std::cout << "added children" << std::endl;
                CobwebNode *newNode = new CobwebNode();

                // std::cout << "new node address: " << newNode << std::endl;
                // std::cout << "parent node address: " << nodeStack.top() << std::endl;

                nodeStack.top()->children.push_back(newNode);
                // std::cout << "children size: " << nodeStack.top()->children.size() << std::endl;

                nodeStack.push(newNode);
            }
            return true;
        }

        bool EndObject(rapidjson::SizeType memberCount)
        {
            // std::cout << "again, end object" << std::endl;

            // std::cout << "tree address: " << cobwebTree << std::endl;
            // if (cobwebTree->root == nullptr)
            // {
            //     std::cout << "root is null, for now" << std::endl;
            // }
            // else if (cobwebTree->root != nullptr)
            // {
            //     std::cout << "\nroot node address: " << cobwebTree->root << std::endl;
            //     if (cobwebTree->root->children.size() > 0)
            //     {
            //         std::cout << "root children size: " << cobwebTree->root->children.size() << std::endl;
            //         std::cout << "root children address: " << cobwebTree->root->children[0] << std::endl;
            //     }
            //     if (cobwebTree->root->av_count["c"].size() > 0)
            //     {
            //         std::cout << "root av_count at c size: " << cobwebTree->root->av_count["c"]["3809"] << std::endl;
            //         // if (cobwebTree->root->av_count->_a != nullptr) {
            //         //     std::cout << "root av_count _a address: " << cobwebTree->root->av_count->_a << std::endl;
            //         // }
            //     }
            // }

            if (keyStack.empty() && nodeStack.empty())
            {
                // std::cout << "The End" << std::endl;
                return true;
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("root"))
            {
                // CobwebNode* completedNode = nodeStack.top();
                nodeStack.pop();
                // check if the stack is empty
                // std::cout << "stack length at root end: " << nodeStack.size() << std::endl;

                // call the member function dump_json in CobwebTree
                // std::string cobwebTreeStr = cobwebTree->dump_json();
                // std::cout << "cobwebTreeStr: " << cobwebTreeStr << std::endl;
                // cobwebTree->root = completedNode;

                // print out the length of the children array
                // std::cout << "children array length: " << completedNode->children.size() << std::endl;
                // // print out the length of av_count_c
                // std::cout << "av_count_c length: " << completedNode->av_count.size() << std::endl;
                // // print out the length of the stack
                // std::cout << "node stack length: " << nodeStack.size() << std::endl;

                // std::cout << "key stack length: " << keyStack.size() << std::endl;
                // // newline
                // std::cout << std::endl;

                // else if (keyStack.top() == "a_count" || keyStack.top() == "sum_n_logn" || keyStack.top() == "av_count") {
                //     return true;
            }
            else if (keyStack.top() == ATTRIBUTE_MAP.at("children") && !nodeStack.empty())
            {
                nodeStack.pop();
            }

            if (keyStack.top() == ATTRIBUTE_MAP.at("children") && !nodeStack.empty())
            {
                return true;
            }
            else
            {
                keyStack.pop();
                // std::cout << "key stack length: " << keyStack.size() << std::endl;
            }
            return true;
        }

        bool StartArray()
        {
            return true;
        }

        bool EndArray(rapidjson::SizeType elementCount)
        {
            keyStack.pop();
            return true;
        }

        void display_progress()
        {
            // Placeholder for progress display logic
        }
    };

    void load_json_stream(std::string json_model_path)
    {
        FILE *fp = fopen(json_model_path.c_str(), "rb");
        if (!fp)
        {
            std::cerr << "Could not open file: " << json_model_path << std::endl;
            return;
        }

        const int bufferSize = 65536; // 256KB
        char buffer[bufferSize];
        rapidjson::FileReadStream is(fp, buffer, sizeof(buffer));
        rapidjson::Reader reader;
        MyHandler handler(this);

        // std::cout << "Tree init address: " << this << std::endl;

        // // print if it is nullptr
        // if (this == nullptr)
        // {
        //     std::cout << "Tree is nullptr" << std::endl;
        // }
        // else
        // {
        //     std::cout << this << std::endl;
        // }

        if (!reader.Parse(is, handler))
        {
            std::cout << "Error while parsing JSON." << std::endl;
        }

        // std::cout << "after parsing address: " << this << std::endl;

        // std::string this_str = this->dump_json();
        // std::cout << this_str << std::endl;

        // if (this == nullptr)
        // {
        //     std::cout << "Tree, after is nullptr" << std::endl;
        // }
        // else
        // {
        //     std::cout << "Not a nullptr" << std::endl;

        // }

        // std::cout << "Loaded JSON from: " << json_model_path << std::endl;

        fclose(fp);

        // std::cout << "after close address: " << this << std::endl;
    }
    // void load_json(std::string json)
    // {
    //     struct json_value_s *tree = json_parse(json.c_str(), strlen(json.c_str()));
    //     // print the size of the json_value_s in byte in the memory
    //     struct json_object_s *object = (struct json_object_s *)tree->payload;

    //     // alpha
    //     struct json_object_element_s *alpha_obj = object->start;
    //     double alpha = atof(json_value_as_number(alpha_obj->value)->number);
    //     this->alpha = alpha;

    //     // weight_attr
    //     struct json_object_element_s *weight_attr_obj = alpha_obj->next;
    //     bool weight_attr = bool(atoi(json_value_as_number(weight_attr_obj->value)->number));
    //     this->weight_attr = weight_attr;

    //     // objective
    //     struct json_object_element_s *objective_obj = weight_attr_obj->next;
    //     int objective = atoi(json_value_as_number(objective_obj->value)->number);
    //     this->objective = objective;

    //     // children_norm
    //     struct json_object_element_s *children_norm_obj = objective_obj->next;
    //     bool children_norm = bool(atoi(json_value_as_number(children_norm_obj->value)->number));
    //     this->children_norm = children_norm;

    //     // norm_attributes
    //     struct json_object_element_s *norm_attributes_obj = children_norm_obj->next;
    //     bool norm_attributes = bool(atoi(json_value_as_number(norm_attributes_obj->value)->number));
    //     this->norm_attributes = norm_attributes;

    //     // root
    //     struct json_object_element_s *root_obj = norm_attributes_obj->next;
    //     struct json_object_s *root = json_value_as_object(root_obj->value);

    //     delete this->root;
    //     this->root = this->load_json_helper(root);

    //     for (auto &[attr, val_map] : this->root->av_count)
    //     {
    //         for (auto &[val, cnt] : val_map)
    //         {
    //             this->attr_vals[attr].insert(val);
    //         }
    //     }
    // }

    void clear()
    {
        delete this->root;
        this->root = new CobwebNode();
        this->root->tree = this;
        this->attr_vals = AV_KEY_TYPE();
    }

    std::tuple<CobwebNode *, std::unordered_map<std::string, double>> ifit_helper(const INSTANCE_TYPE &instance, size_t mode)
    {
        // AV_COUNT_TYPE cached_instance;
        // for (auto &[attr, val_map] : instance)
        // {
        //     for (auto &[val, cnt] : val_map)
        //     {
        //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        //     }
        // }
        // return this->cobweb(cached_instance, mode);
        return this->cobweb(instance, mode);
    }

    std::tuple<CobwebNode *, std::unordered_map<std::string, double>> ifit(INSTANCE_TYPE instance, size_t mode)
    {
        return this->ifit_helper(instance, mode);
    }

    void fit(std::vector<INSTANCE_TYPE> instances, size_t mode, int iterations = 1, bool randomizeFirst = true)
    {
        for (int i = 0; i < iterations; i++)
        {
            if (i == 0 && randomizeFirst)
            {
                shuffle(instances.begin(), instances.end(), std::default_random_engine());
            }
            for (auto &instance : instances)
            {
                this->ifit(instance, mode);
            }
            shuffle(instances.begin(), instances.end(), std::default_random_engine());
        }
    }

    int chooseRandomAction()
    {
        std::random_device rd;                            // Seed for the random number engine
        std::mt19937 gen(rd());                           // Mersenne Twister random number engine
        std::uniform_int_distribution<> dis(BEST, SPLIT); // Distribution range: BEST to SPLIT

        return dis(gen); // Generate a random action
    }

    std::tuple<CobwebNode *, std::unordered_map<std::string, double>> cobweb(const AV_COUNT_TYPE &instance, size_t mode)
    {
        // std::cout << "cobweb top level" << std::endl;

        for (auto &[attr, val_map] : instance)
        {
            for (auto &[val, cnt] : val_map)
            {
                attr_vals[attr].insert(val);
            }
        }

        CobwebNode *current = root;

        // create a dictionary that holds counts and time for each operation
        // <string, double>
        std::unordered_map<std::string, double> operation_stats;
        double while_loop_count = 0;
        double while_loop_time = 0;
        double is_match_count = 0;
        double is_match_IC_time = 0;
        double fringe_split_count = 0;
        double fringe_split_time = 0;
        double fringe_split_IC_time = 0;
        double four_ops_count = 0;

        double TBC_count = 0;
        double TBC_time = 0;

        double GBO_count = 0;
        double GBO_time = 0;

        double BEST_count = 0;
        double BEST_IC_time = 0;
        // double BEST_time = 0;

        double NEW_count = 0;
        double NEW_IC_time = 0;
        double NEW_time = 0;

        double MERGE_count = 0;
        double MERGE_IC_time = 0;
        double MERGE_UC_time = 0;
        double MERGE_time = 0;

        double SPLIT_count = 0;
        double SPLIT_time = 0;

        auto start_while = std::chrono::high_resolution_clock::now();
        while (true)
        {
            while_loop_count += 1;
            if (current->children.empty() && (current->count == 0 || current->is_exact_match(instance)))
            {
                is_match_count += 1;
                // std::cout << "empty / exact match" << std::endl;

                auto start = std::chrono::high_resolution_clock::now();
                current->increment_counts(instance);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                is_match_IC_time += elapsed.count();
                break;
            }
            else if (current->children.empty())
            {
                // std::cout << "fringe split" << std::endl;
                fringe_split_count += 1;

                auto start_fs = std::chrono::high_resolution_clock::now();
                CobwebNode *new_node = new CobwebNode(current);
                current->parent = new_node;
                new_node->children.push_back(current);

                if (new_node->parent == nullptr)
                {
                    root = new_node;
                }
                else
                {
                    new_node->parent->children.erase(remove(new_node->parent->children.begin(),
                                                            new_node->parent->children.end(), current),
                                                     new_node->parent->children.end());
                    new_node->parent->children.push_back(new_node);
                }

                auto start2 = std::chrono::high_resolution_clock::now();
                new_node->increment_counts(instance);
                auto end2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed2 = end2 - start2;
                fringe_split_IC_time += elapsed2.count();

                current = new CobwebNode();
                current->parent = new_node;
                current->tree = this;

                auto start3 = std::chrono::high_resolution_clock::now();
                current->increment_counts(instance);
                auto end3 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed3 = end3 - start3;
                fringe_split_IC_time += elapsed3.count();

                new_node->children.push_back(current);
                auto end_fs = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_fs = end_fs - start_fs;
                fringe_split_time += elapsed_fs.count();
                break;
            }
            else
            {
                four_ops_count += 1;

                auto start_tbc = std::chrono::high_resolution_clock::now();
                auto [best1_mi, best1, best2] = current->two_best_children(instance);
                auto end_tbc = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_tbc = end_tbc - start_tbc;
                TBC_time += elapsed_tbc.count();
                TBC_count += 1;
                // std::cout << "two_best_children took " << elapsed.count() << " seconds" << std::endl;
                size_t bestAction = 0;
                auto start_gbo = std::chrono::high_resolution_clock::now();
                if (mode == 0)
                {
                    auto [_, action] = current->get_best_operation(instance, best1, best2, best1_mi);
                    bestAction = action;
                }
                else if (mode == 1)
                {
                    bestAction = BEST;
                }
                else if (mode == 2)
                {
                    bestAction = 0;
                    // check if best1 is null
                    if (best1 == nullptr)
                    {
                        std::cout << "best1 is null" << std::endl;
                        // randomly choose between 0 and 1
                        bestAction = rand() % 2;
                    }
                    else
                    {
                        if (best2 == nullptr)
                        {
                            // randomly choose between 0,1,3
                            bestAction = rand() % 3;
                            if (bestAction == 2)
                            {
                                bestAction = 3;
                            }
                        }
                        else
                        {
                            // randomly choose between 0,1,2,3
                            bestAction = rand() % 4;
                        }
                    }
                }
                else if (mode == 3)
                {
                    // epsilon BEST
                    bestAction = BEST;
                    if (best1 != nullptr && best2 != nullptr)
                    {
                        double epsilon = 1e-2;
                        double p = ((double)rand() / (RAND_MAX));
                        // std::cout << "p: " << p << std::endl;
                        if (p < epsilon)
                        {
                            auto [_, action] = current->get_best_operation(instance, best1, best2, best1_mi);
                            bestAction = action;
                        }
                    }
                }
                else
                {
                    bestAction = 0;
                }
                auto end_gbo = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_gbo = end_gbo - start_gbo;
                GBO_time += elapsed_gbo.count();
                GBO_count += 1;
                // randomly choose the best action from
                // #define BEST 0
                // #define NEW 1
                // #define MERGE 2
                // #define SPLIT 3
                // size_t bestAction = 0;
                // // check if best1 is null
                // if (best1 == nullptr) {
                //     std::cout << "best1 is null" << std::endl;
                //     // randomly choose between 0 and 1
                //     bestAction = rand() % 2;
                // }
                // else {
                //     if (best2 == nullptr) {
                //         // randomly choose between 0,1,3
                //         bestAction = rand() % 3;
                //         if (bestAction == 2) {
                //             bestAction = 3;
                //         }
                //     }
                //     else {
                //         // randomly choose between 0,1,2,3
                //         bestAction = rand() % 4;
                //     }
                // }

                // auto end2 = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> elapsed2 = end2 - start2;
                // std::cout << "get_best_operation took " << elapsed2.count() << " seconds" << std::endl;

                if (bestAction == BEST)
                {
                    // std::cout << "best" << std::endl;
                    BEST_count += 1;
                    auto start_best = std::chrono::high_resolution_clock::now();
                    current->increment_counts(instance);
                    auto end_best = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_best = end_best - start_best;
                    BEST_IC_time += elapsed_best.count();
                    // std::cout << "increment_counts took " << elapsed3.count() << " seconds" << std::endl;

                    current = best1;
                }
                else if (bestAction == NEW)
                {
                    // std::cout << "new" << std::endl;
                    NEW_count += 1;
                    auto start_new = std::chrono::high_resolution_clock::now();

                    auto start_ic1 = std::chrono::high_resolution_clock::now();
                    current->increment_counts(instance);
                    auto end_ic1 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_ic1 = end_ic1 - start_ic1;
                    NEW_IC_time += elapsed_ic1.count();
                    // std::cout << "increment_counts took " << elapsed4.count() << " seconds" << std::endl;

                    // current = current->create_new_child(instance);
                    CobwebNode *new_child = new CobwebNode();
                    new_child->parent = current;
                    new_child->tree = this;

                    auto start_ic2 = std::chrono::high_resolution_clock::now();
                    new_child->increment_counts(instance);
                    auto end_ic2 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_ic2 = end_ic2 - start_ic2;
                    NEW_IC_time += elapsed_ic2.count();
                    // std::cout << "increment_counts took " << elapsed5.count() << " seconds" << std::endl;

                    current->children.push_back(new_child);
                    current = new_child;

                    // clear bestAction from memory
                    bestAction = 0;
                    auto end_new = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_new = end_new - start_new;
                    NEW_time += elapsed_new.count();
                    break;
                }
                else if (bestAction == MERGE)
                {
                    // std::cout << "merge" << std::endl;
                    MERGE_count += 1;
                    auto start_merge = std::chrono::high_resolution_clock::now();

                    auto start_mic1 = std::chrono::high_resolution_clock::now();
                    current->increment_counts(instance);
                    auto end_mic1 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_mic1 = end_mic1 - start_mic1;
                    MERGE_IC_time += elapsed_mic1.count();
                    // std::cout << "increment_counts took " << elapsed6.count() << " seconds" << std::endl;
                    // CobwebNode* new_child = current->merge(best1, best2);

                    CobwebNode *new_child = new CobwebNode();
                    new_child->parent = current;
                    new_child->tree = this;

                    auto start_uc = std::chrono::high_resolution_clock::now();
                    new_child->update_counts_from_node(best1);
                    new_child->update_counts_from_node(best2);
                    auto end_uc = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_uc = end_uc - start_uc;
                    MERGE_UC_time += elapsed_uc.count();

                    best1->parent = new_child;
                    best2->parent = new_child;
                    new_child->children.push_back(best1);
                    new_child->children.push_back(best2);
                    current->children.erase(remove(current->children.begin(),
                                                   current->children.end(), best1),
                                            current->children.end());
                    current->children.erase(remove(current->children.begin(),
                                                   current->children.end(), best2),
                                            current->children.end());
                    current->children.push_back(new_child);
                    current = new_child;

                    auto end_merge = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_merge = end_merge - start_merge;
                    MERGE_time += elapsed_merge.count();
                }
                else if (bestAction == SPLIT)
                {
                    // std::cout << "split" << std::endl;
                    SPLIT_count += 1;
                    auto start_split = std::chrono::high_resolution_clock::now();
                    current->children.erase(remove(current->children.begin(),
                                                   current->children.end(), best1),
                                            current->children.end());
                    for (auto &c : best1->children)
                    {
                        c->parent = current;
                        c->tree = this;
                        current->children.push_back(c);
                    }
                    delete best1;
                    auto end_split = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed_split = end_split - start_split;
                    SPLIT_time += elapsed_split.count();
                }
                else
                {
                    throw "Best action choice \"" + std::to_string(bestAction) +
                        "\" (best=0, new=1, merge=2, split=3) not a recognized option. This should be impossible...";
                }
            }
        }
        auto end_while = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_while = end_while - start_while;
        while_loop_time += elapsed_while.count();
        // record all the stats
        operation_stats["while_loop_count"] = while_loop_count;
        operation_stats["while_loop_time"] = while_loop_time;
        operation_stats["is_match_count"] = is_match_count;
        operation_stats["is_match_IC_time"] = is_match_IC_time;
        operation_stats["fringe_split_count"] = fringe_split_count;
        operation_stats["fringe_split_time"] = fringe_split_time;
        operation_stats["fringe_split_IC_time"] = fringe_split_IC_time;
        operation_stats["four_ops_count"] = four_ops_count;
        operation_stats["TBC_count"] = TBC_count;
        operation_stats["TBC_time"] = TBC_time;
        operation_stats["GBO_count"] = GBO_count;
        operation_stats["GBO_time"] = GBO_time;
        operation_stats["BEST_count"] = BEST_count;
        operation_stats["BEST_IC_time"] = BEST_IC_time;
        operation_stats["BEST_time"] = BEST_IC_time;
        operation_stats["NEW_count"] = NEW_count;
        operation_stats["NEW_IC_time"] = NEW_IC_time;
        operation_stats["NEW_time"] = NEW_time;
        operation_stats["MERGE_count"] = MERGE_count;
        operation_stats["MERGE_IC_time"] = MERGE_IC_time;
        operation_stats["MERGE_UC_time"] = MERGE_UC_time;
        operation_stats["MERGE_time"] = MERGE_time;
        operation_stats["SPLIT_count"] = SPLIT_count;
        operation_stats["SPLIT_time"] = SPLIT_time;

        // make return a tuple
        return std::make_tuple(current, operation_stats);

        // return current;
    }

    CobwebNode *_cobweb_categorize(const AV_COUNT_TYPE &instance)
    {

        auto current = this->root;

        while (true)
        {
            if (current->children.empty())
            {
                return current;
            }

            auto parent = current;
            current = nullptr;
            double best_logp;

            for (auto &child : parent->children)
            {
                double logp = child->log_prob_class_given_instance(instance, false);
                if (current == nullptr || logp > best_logp)
                {
                    best_logp = logp;
                    current = child;
                }
            }
        }
    }

    CobwebNode *categorize_helper(const INSTANCE_TYPE &instance)
    {
        // AV_COUNT_TYPE cached_instance;
        // for (auto &[attr, val_map] : instance)
        // {
        //     for (auto &[val, cnt] : val_map)
        //     {
        //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        //     }
        // }
        return this->_cobweb_categorize(instance);
    }

    CobwebNode *categorize(const INSTANCE_TYPE instance)
    {
        return this->categorize_helper(instance);
    }

    std::tuple<std::unordered_map<std::string, double>, std::unordered_map<int, std::unordered_map<int, double>>> predict_probs_mixture_helper(const AV_COUNT_TYPE &instance, double ll_path, int max_nodes, bool greedy, bool missing)
    {

        std::unordered_map<int, std::unordered_map<int, double>> out;
        std::unordered_map<int, std::unordered_map<int, std::vector<double>>> weighted_pred_probs;

        std::unordered_map<std::string, double> operation_stats;

        int nodes_expanded = 0;
        double total_weight = 0;
        bool first_weight = true;

        double log_prob_missing_time = 0;
        double log_prob_instance_time = 0;
        double while_count = 0;
        double while_time = 0;

        double predict_prob_log_time = 0;

        double lse_loop_time = 0;
        // double lse_loop_count = 0;
        double log_prob_children_time = 0;
        // double log_prob_loop_count = 0;
        double log_prob_loop_time = 0;

        double exp_subtract_time = 0;

        double root_ll_inst = 0;
        if (missing)
        {
            auto log_prob_missing_time_start = std::chrono::high_resolution_clock::now();
            root_ll_inst = this->root->log_prob_instance_missing(instance);
            auto log_prob_missing_time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_log_prob_missing_time = log_prob_missing_time_end - log_prob_missing_time_start;
            log_prob_missing_time += elapsed_log_prob_missing_time.count();
        }
        else
        {
            auto log_prob_instance_time_start = std::chrono::high_resolution_clock::now();
            root_ll_inst = this->root->log_prob_instance(instance);
            auto log_prob_instance_time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_log_prob_instance_time = log_prob_instance_time_end - log_prob_instance_time_start;
            log_prob_instance_time += elapsed_log_prob_instance_time.count();
        }

        // std::cout << "root instance ll " << root_ll_inst << std::endl;

        auto queue = std::priority_queue<
            std::tuple<double, double, CobwebNode *>>();

        // std::cout << "root score: " << score << std::endl;
        queue.push(std::make_tuple(root_ll_inst, 0.0, this->root));

        auto start_while = std::chrono::high_resolution_clock::now();
        while (queue.size() > 0)
        {
            while_count += 1;

            auto node = queue.top();
            queue.pop();
            nodes_expanded += 1;

            if (greedy)
            {
                queue = std::priority_queue<
                    std::tuple<double, double, CobwebNode *>>();
            }

            auto curr_score = std::get<0>(node);
            auto curr_ll = std::get<1>(node);
            auto curr = std::get<2>(node);

            // total_weight += curr_score;
            // std::cout << "weight += " << std::to_string(curr_score) << " (" << std::to_string(exp(curr_score)) << ")" << std::endl;
            if (first_weight)
            {
                total_weight = curr_score;
                first_weight = false;
            }
            else
            {
                total_weight = logsumexp(total_weight, curr_score);
            }

            // auto curr_preds = curr->predict_probs();
            auto start_predict_prob_log_time = std::chrono::high_resolution_clock::now();
            auto curr_log_probs = curr->predict_log_probs();
            auto end_predict_prob_log_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_predict_prob_log_time = end_predict_prob_log_time - start_predict_prob_log_time;
            predict_prob_log_time += elapsed_predict_prob_log_time.count();

            auto start_lse_loop = std::chrono::high_resolution_clock::now();
            for (auto &[attr, val_set] : curr_log_probs)
            {
                for (auto &[val, log_p] : val_set)
                {
                    if (out.count(attr) && out.at(attr).count(val))
                    {
                        out[attr][val] = logsumexp(out[attr][val], curr_score + log_p);
                    }
                    else
                    {
                        out[attr][val] = curr_score + log_p;
                    }
                }
            }
            auto end_lse_loop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_lse_loop = end_lse_loop - start_lse_loop;
            lse_loop_time += elapsed_lse_loop.count();

            if (nodes_expanded >= max_nodes)
                break;

            // TODO look at missing in computing prob children given instance
            // std::vector<double> children_probs = curr->prob_children_given_instance(instance);
            auto start_log_prob_children = std::chrono::high_resolution_clock::now();
            std::vector<double> log_children_probs = curr->log_prob_children_given_instance(instance);
            auto end_log_prob_children = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_log_prob_children = end_log_prob_children - start_log_prob_children;
            log_prob_children_time += elapsed_log_prob_children.count();

            auto start_log_prob_loop = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < curr->children.size(); ++i)
            {
                auto child = curr->children[i];
                double child_ll_inst = 0;
                if (missing)
                {
                    child_ll_inst = child->log_prob_instance_missing(instance);
                }
                else
                {
                    child_ll_inst = child->log_prob_instance(instance);
                }
                auto child_ll_given_parent = log_children_probs[i];
                auto child_ll = child_ll_given_parent + curr_ll;

                // double score = exp(child_ll_inst + child_ll);
                // std::cout << "Node score: " << score << ", ll_node: " << child_ll << ", ll_inst: " << child_ll_inst << std::endl;
                queue.push(std::make_tuple(child_ll_inst + child_ll, child_ll, child));
            }
            auto end_log_prob_loop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_log_prob_loop = end_log_prob_loop - start_log_prob_loop;
            log_prob_loop_time += elapsed_log_prob_loop.count();
        }
        auto end_while = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_while = end_while - start_while;
        while_time += elapsed_while.count();

        auto start_exp_subtract = std::chrono::high_resolution_clock::now();
        for (auto &[attr, val_set] : out)
        {
            for (auto &[val, p] : val_set)
            {
                // out[attr][val] /= total_weight;
                // std::cout << attr << "=" << val << " -> " << out[attr][val] << " - " << total_weight << " = " << exp(out[attr][val] - total_weight) << std::endl;
                out[attr][val] = exp(out[attr][val] - total_weight);
            }
        }
        auto end_exp_subtract = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_exp_subtract = end_exp_subtract - start_exp_subtract;
        exp_subtract_time += elapsed_exp_subtract.count();
        // std::cout << "Total Weight: " << total_weight << std::endl;

        operation_stats["log_prob_missing_time"] = log_prob_missing_time;
        operation_stats["log_prob_instance_time"] = log_prob_instance_time;
        operation_stats["while_count"] = while_count;
        operation_stats["while_time"] = while_time;
        operation_stats["predict_prob_log_time"] = predict_prob_log_time;
        operation_stats["lse_loop_time"] = lse_loop_time;
        operation_stats["log_prob_children_time"] = log_prob_children_time;
        operation_stats["log_prob_loop_time"] = log_prob_loop_time;
        operation_stats["exp_subtract_time"] = exp_subtract_time;

        // return out;
        return std::make_tuple(operation_stats, out);
    }

    std::tuple<std::unordered_map<std::string, double>, std::unordered_map<int, std::unordered_map<int, double>>> predict_probs_mixture(INSTANCE_TYPE instance, int max_nodes, bool greedy, bool missing)
    {
        // AV_COUNT_TYPE cached_instance;
        // for (auto &[attr, val_map] : instance)
        // {
        //     for (auto &[val, cnt] : val_map)
        //     {
        //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
        //     }
        // }
        return this->predict_probs_mixture_helper(instance, 0.0,
                                                  max_nodes, greedy, missing);
    }

    // std::vector<std::tuple<std::unordered_map<std::string, double>, std::unordered_map<int, std::unordered_map<int, double>>>> predict_probs_mixture_parallel(std::vector<INSTANCE_TYPE> instances, int max_nodes, bool greedy, bool missing, int num_threads)
    std::vector<std::unordered_map<int, std::unordered_map<int, double>>> predict_probs_mixture_parallel(std::vector<INSTANCE_TYPE> instances, int max_nodes, bool greedy, bool missing, int num_threads)
    {

        BS::thread_pool pool = BS::thread_pool(num_threads);

        std::vector<std::unordered_map<int, std::unordered_map<int, double>>> out(instances.size());

        auto start = std::chrono::high_resolution_clock::now();

        // pool.detach_loop<unsigned int>(0, instances.size(),
        pool.detach_sequence<unsigned int>(0, instances.size(),
                                           [this, &instances, &out, max_nodes, greedy, missing](const unsigned int i)
                                           {
                                            //    out[i] = this->predict_probs_mixture(instances[i], max_nodes, greedy, missing);
                                            // detuple the tuple
                                               auto [operation_stats, probs] = this->predict_probs_mixture(instances[i], max_nodes, greedy, missing);
                                               out[i] = probs;
                                           });

        while (true)
        {
            if (!pool.wait_for(std::chrono::milliseconds(1000)))
            {
                double progress = (instances.size() - pool.get_tasks_total()) / double(instances.size());
                auto current = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double, std::chrono::seconds> elapsed = current - start;
                std::chrono::duration<double, std::milli> elapsed = current - start;
                displayProgressBar(70, progress, elapsed.count() / 1000.0);
                // std::cout << "There are " << pool.get_tasks_total() << " not done yet." << std::endl;
                // std::cout << pool.get_tasks_total() <<  " tasks total, " << pool.get_tasks_running() << " tasks running, " << pool.get_tasks_queued() <<  " tasks queued." << std::endl;
            }
            else
            {
                break;
            }
        }

        pool.wait();

        return out;
    }
};

inline CobwebNode::CobwebNode()
{
    count = 0;
    sum_n_logn = ATTR_COUNT_TYPE();
    a_count = ATTR_COUNT_TYPE();
    parent = nullptr;
    tree = nullptr;
}

inline CobwebNode::CobwebNode(CobwebNode *otherNode)
{
    count = 0;
    sum_n_logn = ATTR_COUNT_TYPE();
    a_count = ATTR_COUNT_TYPE();

    parent = otherNode->parent;
    tree = otherNode->tree;

    update_counts_from_node(otherNode);

    for (auto child : otherNode->children)
    {
        children.push_back(new CobwebNode(child));
    }
}

inline void CobwebNode::increment_counts(const AV_COUNT_TYPE &instance)
{
    this->count += 1;
    for (auto &[attr, val_map] : instance)
    {
        for (auto &[val, cnt] : val_map)
        {
            this->a_count[attr] += cnt;

            // if (!attr.is_hidden())
            if (attr > 0)
            {
                if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
                {
                    double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                    this->sum_n_logn[attr] -= tf * log(tf);
                }
            }

            this->av_count[attr][val] += cnt;

            // if (!attr.is_hidden())
            if (attr > 0)
            {
                double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                this->sum_n_logn[attr] += tf * log(tf);
                // std::cout << "av_count for [" << attr.get_string() << "] = [" << val.get_string() << "]: " << this->av_count[attr][val] << std::endl;
                // std::cout << "updated sum nlogn for [" << attr.get_string() << "]: " << this->sum_n_logn[attr] << std::endl;
            }
        }
    }
}

inline void CobwebNode::update_counts_from_node(CobwebNode *node)
{
    this->count += node->count;

    for (auto &[attr, val_map] : node->av_count)
    {
        this->a_count[attr] += node->a_count.at(attr);

        for (auto &[val, cnt] : val_map)
        {
            // if (!attr.is_hidden())
            if (attr > 0)
            {
                if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
                {
                    double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                    this->sum_n_logn[attr] -= tf * log(tf);
                }
            }

            this->av_count[attr][val] += cnt;

            // if (!attr.is_hidden())
            if (attr > 0)
            {
                double tf = this->av_count.at(attr).at(val) + this->tree->alpha;
                this->sum_n_logn[attr] += tf * log(tf);
            }
        }
    }
}

inline double CobwebNode::entropy_attr_insert(ATTR_TYPE attr, const AV_COUNT_TYPE &instance)
{
    //if (attr.is_hidden())
    if (attr < 0)
        return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    double ratio = 1.0;
    if (this->tree->weight_attr)
    {
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
        // ratio = (1.0 * attr_count) / this->count;
    }
    // ratio = std::ceil(ratio);

    if (this->av_count.count(attr))
    {
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr))
    {
        sum_n_logn = this->sum_n_logn.at(attr);
    }

    if (instance.count(attr))
    {
        for (auto &[val, cnt] : instance.at(attr))
        {
            attr_count += cnt;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
            {
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + this->tree->alpha;
                sum_n_logn -= tf * log(tf);
            }
            else
            {
                num_vals_in_c += 1;
            }
            COUNT_TYPE tf = prior_av_count + cnt + this->tree->alpha;
            sum_n_logn += (tf)*log(tf);
        }
    }

    int n0 = num_vals_total - num_vals_in_c;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
                                (sum_n_logn + n0 * alpha * log(alpha)) -
                            log(attr_count +
                                num_vals_total * alpha));
    return info;
}

inline double CobwebNode::entropy_insert(const AV_COUNT_TYPE &instance)
{

    double info = 0.0;

    for (auto &[attr, av_inner] : this->av_count)
    {
        //if (attr.is_hidden())
        if (attr < 0)
            continue;
        info += this->entropy_attr_insert(attr, instance);
    }

    // iterate over attr in instance not in av_count
    for (auto &[attr, av_inner] : instance)
    {
        // if (attr.is_hidden())
        if (attr < 0)
            continue;
        if (this->av_count.count(attr))
            continue;
        info += this->entropy_attr_insert(attr, instance);
    }

    return info;
}

inline double CobwebNode::entropy_attr_merge(ATTR_TYPE attr,
                                             CobwebNode *other, const AV_COUNT_TYPE &instance)
{

    // if (attr.is_hidden())
    if (attr < 0)
        return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    double ratio = 1.0;
    if (this->tree->weight_attr)
    {
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
        // ratio = (1.0 * attr_count) / this->count;
    }
    // ratio = std::ceil(ratio);

    if (this->av_count.count(attr))
    {
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr))
    {
        sum_n_logn = this->sum_n_logn.at(attr);
    }

    if (other->av_count.count(attr))
    {
        for (auto &[val, other_av_count] : other->av_count.at(attr))
        {
            COUNT_TYPE instance_av_count = 0.0;

            if (instance.count(attr) && instance.at(attr).count(val))
            {
                instance_av_count = instance.at(attr).at(val);
            }

            attr_count += other_av_count + instance_av_count;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
            {
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + alpha;
                sum_n_logn -= tf * log(tf);
            }
            else
            {
                num_vals_in_c += 1;
            }

            COUNT_TYPE new_tf = prior_av_count + other_av_count + instance_av_count + alpha;
            sum_n_logn += (new_tf)*log(new_tf);
        }
    }

    if (instance.count(attr))
    {
        for (auto &[val, instance_av_count] : instance.at(attr))
        {
            if (other->av_count.count(attr) && other->av_count.at(attr).count(val))
            {
                continue;
            }
            COUNT_TYPE other_av_count = 0.0;

            attr_count += other_av_count + instance_av_count;
            COUNT_TYPE prior_av_count = 0.0;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
            {
                prior_av_count = this->av_count.at(attr).at(val);
                COUNT_TYPE tf = prior_av_count + alpha;
                sum_n_logn -= tf * log(tf);
            }
            else
            {
                num_vals_in_c += 1;
            }

            COUNT_TYPE new_tf = prior_av_count + other_av_count + instance_av_count + alpha;
            sum_n_logn += (new_tf)*log(new_tf);
        }
    }

    int n0 = num_vals_total - num_vals_in_c;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
                                (sum_n_logn + n0 * alpha * log(alpha)) -
                            log(attr_count +
                                num_vals_total * alpha));
    return info;
}

inline double CobwebNode::entropy_merge(CobwebNode *other,
                                        const AV_COUNT_TYPE &instance)
{

    double info = 0.0;

    for (auto &[attr, inner_vals] : this->av_count)
    {
        // if (attr.is_hidden())
        if (attr < 0)
            continue;
        info += this->entropy_attr_merge(attr, other, instance);
    }

    for (auto &[attr, inner_vals] : other->av_count)
    {
        // if (attr.is_hidden())
        if (attr < 0)
            continue;
        if (this->av_count.count(attr))
            continue;
        info += this->entropy_attr_merge(attr, other, instance);
    }

    for (auto &[attr, inner_vals] : instance)
    {
        // if (attr.is_hidden())
        if (attr < 0)
            continue;
        if (this->av_count.count(attr))
            continue;
        if (other->av_count.count(attr))
            continue;
        info += entropy_attr_merge(attr, other, instance);
    }

    return info;
}

inline CobwebNode *CobwebNode::get_best_level(
    INSTANCE_TYPE instance)
{

    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    CobwebNode *curr = this;
    CobwebNode *best = this;
    double best_ll = this->log_prob_class_given_instance(instance, true);

    while (curr->parent != nullptr)
    {
        curr = curr->parent;
        double curr_ll = curr->log_prob_class_given_instance(instance, true);

        if (curr_ll > best_ll)
        {
            best = curr;
            best_ll = curr_ll;
        }
    }

    return best;
}

inline CobwebNode *CobwebNode::get_basic_level()
{
    CobwebNode *curr = this;
    CobwebNode *best = this;
    double best_cu = this->category_utility();

    while (curr->parent != nullptr)
    {
        curr = curr->parent;
        double curr_cu = curr->category_utility();

        if (curr_cu > best_cu)
        {
            best = curr;
            best_cu = curr_cu;
        }
    }

    return best;
}

inline double CobwebNode::entropy_attr(ATTR_TYPE attr)
{
    if (attr < 0)
        return 0.0;

    float alpha = this->tree->alpha;
    int num_vals_total = this->tree->attr_vals.at(attr).size();
    int num_vals_in_c = 0;
    COUNT_TYPE attr_count = 0;

    if (this->av_count.count(attr))
    {
        attr_count = this->a_count.at(attr);
        num_vals_in_c = this->av_count.at(attr).size();
    }

    double ratio = 1.0;
    if (this->tree->weight_attr)
    {
        ratio = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
        // ratio = (1.0 * attr_count) / this->count;
    }
    // ratio = std::ceil(ratio);

    double sum_n_logn = 0.0;
    if (this->sum_n_logn.count(attr))
    {
        sum_n_logn = this->sum_n_logn.at(attr);
    }

    int n0 = num_vals_total - num_vals_in_c;
    // std::cout << "sum n logn: " << sum_n_logn << std::endl;
    // std::cout << "n0: " << n0 << std::endl;
    // std::cout << "alpha: " << alpha << std::endl;
    // std::cout << "attr_count: " << attr_count << std::endl;
    double info = -ratio * ((1 / (attr_count + num_vals_total * alpha)) *
                                (sum_n_logn + n0 * alpha * log(alpha)) -
                            log(attr_count +
                                num_vals_total * alpha));
    return info;

    /*
       int n = std::ceil(ratio);
       info -= lgamma_cached(n+1);

       for (auto &[val, cnt]: inner_av){
       double p = ((cnt + alpha) / (attr_count + num_vals * alpha));
       info += entropy_component_k(n, p);
       }

       COUNT_TYPE num_missing = num_vals - inner_av.size();
       if (num_missing > 0 and alpha > 0){
       double p = (alpha / (attr_count + num_vals * alpha));
       info += num_missing * entropy_component_k(n, p);
       }

       return info;
       */
}

inline double CobwebNode::entropy()
{

    double info = 0.0;
    for (auto &[attr, inner_av] : this->av_count)
    {
        // if (attr.is_hidden())
        if (attr < 0)
            continue;
        info += this->entropy_attr(attr);
    }

    return info;
}

inline std::tuple<double, int> CobwebNode::get_best_operation(
    const AV_COUNT_TYPE &instance, CobwebNode *best1,
    CobwebNode *best2, double best1_pu)
{

    if (best1 == nullptr)
    {
        throw "Need at least one best child.";
    }
    std::vector<std::tuple<double, double, int>> operations;
    operations.push_back(std::make_tuple(best1_pu,
                                         custom_rand(),
                                         BEST));
    operations.push_back(std::make_tuple(pu_for_new_child(instance),
                                         custom_rand(),
                                         NEW));
    if (children.size() > 2 && best2 != nullptr)
    {
        operations.push_back(std::make_tuple(pu_for_merge(best1, best2,
                                                          instance),
                                             custom_rand(),
                                             MERGE));
    }

    if (best1->children.size() > 0)
    {
        operations.push_back(std::make_tuple(pu_for_split(best1),
                                             custom_rand(),
                                             SPLIT));
    }

    sort(operations.rbegin(), operations.rend());

    OPERATION_TYPE bestOp = std::make_pair(std::get<0>(operations[0]), std::get<2>(operations[0]));
    return bestOp;
}

inline std::tuple<double, CobwebNode *, CobwebNode *> CobwebNode::two_best_children(
    const AV_COUNT_TYPE &instance)
{

    if (children.empty())
    {
        throw "No children!";
    }

    if (this->tree->objective == 0)
    {
        // DO RELATIVE PU, requires only B
        std::vector<std::tuple<double, double, double, CobwebNode *>> relative_pu;
        for (auto &child : this->children)
        {
            relative_pu.push_back(
                std::make_tuple(
                    (child->count * child->entropy()) -
                        ((child->count + 1) * child->entropy_insert(instance)),
                    child->count,
                    custom_rand(),
                    child));
        }

        sort(relative_pu.rbegin(), relative_pu.rend());
        CobwebNode *best1 = std::get<3>(relative_pu[0]);
        // double best1_pu = pu_for_insert(best1, instance);
        double best1_pu = 0.0;
        CobwebNode *best2 = relative_pu.size() > 1 ? std::get<3>(relative_pu[1]) : nullptr;
        return std::make_tuple(best1_pu, best1, best2);
    }
    else
    {
        // Evaluate each insert, requires B^2 where B is branching factor
        // However, we need to do this for other objectives because the denominator changes.
        std::vector<std::tuple<double, double, double, CobwebNode *>> pus;
        for (auto &child : this->children)
        {
            pus.push_back(
                std::make_tuple(
                    pu_for_insert(child, instance),
                    child->count,
                    custom_rand(),
                    child));
        }
        sort(pus.rbegin(), pus.rend());
        CobwebNode *best1 = std::get<3>(pus[0]);
        // double best1_pu = std::get<0>(pus[0]);
        double best1_pu = 0.0;
        CobwebNode *best2 = pus.size() > 1 ? std::get<3>(pus[1]) : nullptr;

        return std::make_tuple(best1_pu, best1, best2);
    }
}

inline double CobwebNode::partition_utility()
{
    if (children.empty())
    {
        return 0.0;
    }

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            parent_entropy += this->entropy_attr(attr);
        }

        for (auto &child : children)
        {
            double p_of_child = (1.0 * child->count) / this->count;
            concept_entropy -= p_of_child * log(p_of_child);

            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= this->children.size();
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &child : children)
        {
            double p_of_child = (1.0 * child->count) / this->count;
            children_entropy += p_of_child * child->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr(attr);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= this->children.size();
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / this->children.size();
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline double CobwebNode::pu_for_insert(CobwebNode *child, const AV_COUNT_TYPE &instance)
{

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            parent_entropy += this->entropy_attr_insert(attr, instance);
        }

        for (auto &c : children)
        {
            if (c == child)
            {
                double p_of_child = (c->count + 1.0) / (this->count + 1.0);
                concept_entropy -= p_of_child * log(p_of_child);
                for (auto &[attr, val_set] : this->tree->attr_vals)
                {
                    children_entropy += p_of_child * c->entropy_attr_insert(attr, instance);
                }
            }
            else
            {
                double p_of_child = (1.0 * c->count) / (this->count + 1.0);
                concept_entropy -= p_of_child * log(p_of_child);

                for (auto &[attr, val_set] : this->tree->attr_vals)
                {
                    children_entropy += p_of_child * c->entropy_attr(attr);
                }
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= this->children.size();
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c : this->children)
        {
            if (c == child)
            {
                double p_of_child = (c->count + 1.0) / (this->count + 1.0);
                children_entropy += p_of_child * c->entropy_attr_insert(attr, instance);
                concept_entropy -= p_of_child * log(p_of_child);
            }
            else
            {
                double p_of_child = (1.0 * c->count) / (this->count + 1.0);
                children_entropy += p_of_child * c->entropy_attr(attr);
                concept_entropy -= p_of_child * log(p_of_child);
            }
        }

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= this->children.size();
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / this->children.size();
        // entropy += (parent_entropy - children_entropy) / parent_entropy / this->children.size();
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / this->children.size();
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline double CobwebNode::pu_for_new_child(const AV_COUNT_TYPE &instance)
{

    // TODO maybe modify so that we can evaluate new child without copying
    // instance.
    CobwebNode new_child = CobwebNode();
    new_child.parent = this;
    new_child.tree = this->tree;
    new_child.increment_counts(instance);
    double p_of_new_child = 1.0 / (this->count + 1.0);

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = -p_of_new_child * log(p_of_new_child);

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            children_entropy += p_of_new_child * new_child.entropy_attr(attr);
            parent_entropy += this->entropy_attr_insert(attr, instance);
        }

        for (auto &child : children)
        {
            double p_of_child = (1.0 * child->count) / (this->count + 1.0);
            concept_entropy -= p_of_child * log(p_of_child);

            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= (this->children.size() + 1);
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        double children_entropy = p_of_new_child * new_child.entropy_attr(attr);
        double concept_entropy = -p_of_new_child * log(p_of_new_child);

        for (auto &c : this->children)
        {
            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= (this->children.size() + 1);
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() + 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline double CobwebNode::pu_for_merge(CobwebNode *best1,
                                       CobwebNode *best2, const AV_COUNT_TYPE &instance)
{

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double p_of_merged = (best1->count + best2->count + 1.0) / (this->count + 1.0);
        double concept_entropy = -p_of_merged * log(p_of_merged);

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            parent_entropy += this->entropy_attr_insert(attr, instance);
            children_entropy += p_of_merged * best1->entropy_attr_merge(attr, best2, instance);
        }

        for (auto &child : children)
        {
            if (child == best1 || child == best2)
            {
                continue;
            }
            double p_of_child = (1.0 * child->count) / (this->count + 1.0);
            concept_entropy -= p_of_child * log(p_of_child);

            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= (this->children.size() - 1);
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c : children)
        {
            if (c == best1 || c == best2)
            {
                continue;
            }

            double p_of_child = (1.0 * c->count) / (this->count + 1.0);
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double p_of_child = (best1->count + best2->count + 1.0) / (this->count + 1.0);
        children_entropy += p_of_child * best1->entropy_attr_merge(attr, best2, instance);
        concept_entropy -= p_of_child * log(p_of_child);

        double parent_entropy = this->entropy_attr_insert(attr, instance);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= (this->children.size() - 1);
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() - 1);
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline double CobwebNode::pu_for_split(CobwebNode *best)
{

    // BEGIN INDIVIDUAL
    if (!this->tree->norm_attributes)
    {
        double parent_entropy = 0.0;
        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            parent_entropy += this->entropy_attr(attr);
        }

        for (auto &child : children)
        {
            if (child == best)
                continue;
            double p_of_child = (1.0 * child->count) / this->count;
            concept_entropy -= p_of_child * log(p_of_child);
            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        for (auto &child : best->children)
        {
            double p_of_child = (1.0 * child->count) / this->count;
            concept_entropy -= p_of_child * log(p_of_child);
            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                children_entropy += p_of_child * child->entropy_attr(attr);
            }
        }

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }
        if (this->tree->children_norm)
        {
            obj /= (this->children.size() - 1 + best->children.size());
        }
        return obj;
    }
    // END INDIVIDUAL

    double entropy = 0.0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {

        double children_entropy = 0.0;
        double concept_entropy = 0.0;

        for (auto &c : children)
        {
            if (c == best)
                continue;
            double p_of_child = (1.0 * c->count) / this->count;
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        for (auto &c : best->children)
        {
            double p_of_child = (1.0 * c->count) / this->count;
            children_entropy += p_of_child * c->entropy_attr(attr);
            concept_entropy -= p_of_child * log(p_of_child);
        }

        double parent_entropy = this->entropy_attr(attr);

        double obj = (parent_entropy - children_entropy);
        if (this->tree->objective == 1)
        {
            obj /= parent_entropy;
        }
        else if (this->tree->objective == 2)
        {
            obj /= (children_entropy + concept_entropy);
        }

        if (this->tree->children_norm)
        {
            obj /= (this->children.size() - 1 + best->children.size());
        }
        entropy += obj;

        // entropy += (parent_entropy - children_entropy) / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / parent_entropy / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy) / (this->children.size() - 1 + best->children.size());
        // entropy += (parent_entropy - children_entropy) / parent_entropy;
        // entropy += (parent_entropy - children_entropy) / (children_entropy + concept_entropy);
    }

    return entropy;
}

inline bool CobwebNode::is_exact_match(const AV_COUNT_TYPE &instance)
{
    std::unordered_set<ATTR_TYPE> all_attrs;
    for (auto &[attr, tmp] : instance)
        all_attrs.insert(attr);
    for (auto &[attr, tmp] : this->av_count)
        all_attrs.insert(attr);

    for (auto &attr : all_attrs)
    {
        // if (attr.is_hidden())
        //     continue;
        if (attr < 0)
            continue;
        if (instance.count(attr) && !this->av_count.count(attr))
        {
            return false;
        }
        if (this->av_count.count(attr) && !instance.count(attr))
        {
            return false;
        }
        // if (this->av_count.count(attr) && instance.count(attr)) {
        //     double instance_attr_exp_sum = 0.0;
        //     double concept_attr_exp_sum = 0.0;
        //     std::unordered_set<VALUE_TYPE> all_vals;

        //     // Collect all values and exponentiated sums from instance
        //     for (auto &[val, cnt]: instance.at(attr)) {
        //         all_vals.insert(val);
        //         instance_attr_exp_sum += std::exp(cnt);
        //     }

        //     // Collect all values and exponentiated sums from av_count
        //     for (auto &[val, cnt]: this->av_count.at(attr)) {
        //         all_vals.insert(val);
        //         concept_attr_exp_sum += std::exp(cnt);
        //     }

        //     std::vector<double> instance_probs, concept_probs;
        //     for (auto &val: all_vals) {
        //         double instance_prob = 0.0;
        //         double concept_prob = 0.0;

        //         if (instance.at(attr).count(val) && !this->av_count.at(attr).count(val)) {
        //             return false;
        //         }
        //         if (this->av_count.at(attr).count(val) && !instance.at(attr).count(val)) {
        //             return false;
        //         }

        //         if (instance.at(attr).count(val)) {
        //             instance_prob = std::exp(instance.at(attr).at(val)) / instance_attr_exp_sum;
        //         }
        //         std::cout << "Instance Prob: " << instance_prob << std::endl;
        //         instance_probs.push_back(instance_prob);

        //         if (this->av_count.at(attr).count(val)) {
        //             concept_prob = std::exp(this->av_count.at(attr).at(val)) / concept_attr_exp_sum;
        //         }
        //         // std::cout << "Concept Prob: " << concept_prob << std::endl;
        //         concept_probs.push_back(concept_prob);
        //     }

        //     double kl_divergence = 0.0;
        //     for (size_t i = 0; i < instance_probs.size(); ++i) {
        //         if (instance_probs[i] > 0) {
        //             kl_divergence += instance_probs[i] * std::log(instance_probs[i] / concept_probs[i]);
        //         }
        //     }

        //     // print out the kl divergence
        //     // std::cout << "KL Divergence: " << kl_divergence << std::endl;

        //     if (kl_divergence >= 0.02) {
        //         return false;
        //     }
        // }
        if (this->av_count.count(attr) && instance.count(attr))
        {
            double instance_attr_count = 0.0;
            std::unordered_set<VALUE_TYPE> all_vals;
            for (auto &[val, tmp] : this->av_count.at(attr))
                all_vals.insert(val);
            for (auto &[val, cnt] : instance.at(attr))
            {
                all_vals.insert(val);
                instance_attr_count += cnt;
            }

            for (auto &val : all_vals)
            {
                if (instance.at(attr).count(val) && !this->av_count.at(attr).count(val))
                {
                    return false;
                }
                if (this->av_count.at(attr).count(val) && !instance.at(attr).count(val))
                {
                    return false;
                }

                double instance_prob = (1.0 * instance.at(attr).at(val)) / instance_attr_count;
                double concept_prob = (1.0 * this->av_count.at(attr).at(val)) / this->a_count.at(attr);

                if (abs(instance_prob - concept_prob) > 0.00001)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

inline size_t CobwebNode::_hash()
{
    return std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(this));
}

inline std::string CobwebNode::__str__()
{
    return this->pretty_print();
}

inline std::string CobwebNode::concept_hash()
{
    return std::to_string(this->_hash());
}

inline std::string CobwebNode::pretty_print(int depth)
{
    std::string ret = repeat("\t", depth) + "|-" + avcounts_to_json() + "\n";

    for (auto &c : children)
    {
        ret += c->pretty_print(depth + 1);
    }

    return ret;
}

inline int CobwebNode::depth()
{
    if (this->parent)
    {
        return 1 + this->parent->depth();
    }
    return 0;
}

inline bool CobwebNode::is_parent(CobwebNode *otherConcept)
{
    CobwebNode *temp = otherConcept;
    while (temp != nullptr)
    {
        if (temp == this)
        {
            return true;
        }
        try
        {
            temp = temp->parent;
        }
        catch (std::string e)
        {
            std::cout << temp;
            assert(false);
        }
    }
    return false;
}

inline int CobwebNode::num_concepts()
{
    int childrenCount = 0;
    for (auto &c : children)
    {
        childrenCount += c->num_concepts();
    }
    return 1 + childrenCount;
}

inline std::string CobwebNode::avcounts_to_json()
{
    std::string ret = "{";

    // // ret += "\"_expected_guesses\": {\n";
    // ret += "\"_entropy\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + std::to_string(this->entropy()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

    ret += "\"_category_utility\": {\n";
    ret += "\"#ContinuousValue#\": {\n";
    ret += "\"mean\": " + std::to_string(this->category_utility()) + ",\n";
    ret += "\"std\": 1,\n";
    ret += "\"n\": 1,\n";
    ret += "}},\n";

    // ret += "\"_mutual_info\": {\n";
    // ret += "\"#ContinuousValue#\": {\n";
    // ret += "\"mean\": " + std::to_string(this->mutual_information()) + ",\n";
    // ret += "\"std\": 1,\n";
    // ret += "\"n\": 1,\n";
    // ret += "}},\n";

    int c = 0;
    for (auto &[attr, vAttr] : av_count)
    {
        // ret += "\"" + attr.get_string() + "\": {";
        ret += "\"" + std::to_string(attr) + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt] : vAttr)
        {
            // ret += "\"" + val.get_string() + "\": " + doubleToString(cnt);
            ret += "\"" + std::to_string(val) + "\": " + doubleToString(cnt);
            // std::to_string(cnt);
            if (inner_count != int(vAttr.size()) - 1)
            {
                ret += ", ";
            }
            inner_count++;
        }
        ret += "}";

        if (c != int(av_count.size()) - 1)
        {
            ret += ", ";
        }
        c++;
    }
    ret += "}";
    return ret;
}

inline std::string CobwebNode::ser_avcounts()
{
    std::string ret = "{";

    int c = 0;
    for (auto &[attr, vAttr] : av_count)
    {
        // ret += "\"" + attr.get_string() + "\": {";
        ret += "\"" + std::to_string(attr) + "\": {";
        int inner_count = 0;
        for (auto &[val, cnt] : vAttr)
        {
            // ret += "\"" + val.get_string() + "\": " + doubleToString(cnt);
            ret += "\"" + std::to_string(val) + "\": " + doubleToString(cnt);
            // std::to_string(cnt);
            if (inner_count != int(vAttr.size()) - 1)
            {
                ret += ", ";
            }
            inner_count++;
        }
        ret += "}";

        if (c != int(av_count.size()) - 1)
        {
            ret += ", ";
        }
        c++;
    }
    ret += "}";
    return ret;
}

inline std::string CobwebNode::a_count_to_json()
{
    std::string ret = "{";

    bool first = true;
    for (auto &[attr, cnt] : this->a_count)
    {
        if (!first)
            ret += ",\n";
        else
            first = false;
        // ret += "\"" + attr.get_string() + "\": " + doubleToString(cnt);
        ret += "\"" + std::to_string(attr) + "\": " + doubleToString(cnt);
        // std::to_string(cnt);
    }

    ret += "}";
    return ret;
}

inline std::string CobwebNode::sum_n_logn_to_json()
{
    std::string ret = "{";

    bool first = true;
    for (auto &[attr, cnt] : this->sum_n_logn)
    {
        if (!first)
            ret += ",\n";
        else
            first = false;
        // ret += "\"" + attr.get_string() + "\": " + doubleToString(cnt);
        ret += "\"" + std::to_string(attr) + "\": " + doubleToString(cnt);
        // std::to_string(cnt);
    }

    ret += "}";
    return ret;
}

inline std::string CobwebNode::dump_json()
{
    std::string output = "{";

    // output += "\"concept_id\": " + std::to_string(this->_hash()) + ",\n";
    output += "\"count\": " + doubleToString(this->count) + ",\n";
    output += "\"a_count\": " + this->a_count_to_json() + ",\n";
    output += "\"sum_n_logn\": " + this->sum_n_logn_to_json() + ",\n";
    output += "\"av_count\": " + this->ser_avcounts() + ",\n";

    output += "\"children\": [\n";
    bool first = true;
    for (auto &c : children)
    {
        if (!first)
            output += ",";
        else
            first = false;
        output += c->dump_json();
    }
    output += "]\n";

    output += "}\n";

    return output;
}

inline std::string CobwebNode::output_json()
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

    output += "\"counts\": " + this->avcounts_to_json() + ",\n";
    output += "\"attr_counts\": " + this->a_count_to_json() + "\n";

    output += "}\n";

    return output;
}

// TODO
// TODO This should use the path prob, not the node prob.
// TODO
inline std::unordered_map<int, std::unordered_map<int, double>> CobwebNode::predict_weighted_leaves_probs(INSTANCE_TYPE instance)
{

    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    double concept_weights = 0.0;
    std::unordered_map<int, std::unordered_map<int, double>> out;

    // std::cout << std::endl << "Prob of nodes along path (starting with leaf)" << std::endl;
    auto curr = this;
    while (curr->parent != nullptr)
    {
        auto prev = curr;
        curr = curr->parent;

        for (auto &child : curr->children)
        {
            if (child == prev)
                continue;
            double c_prob = exp(child->log_prob_class_given_instance(instance, true));
            // double c_prob = 1.0;
            // std::cout << c_prob << std::endl;
            concept_weights += c_prob;

            for (auto &[attr, val_set] : this->tree->attr_vals)
            {
                // std::cout << attr << std::endl;
                int num_vals = this->tree->attr_vals.at(attr).size();
                float alpha = this->tree->alpha;
                COUNT_TYPE attr_count = 0;

                if (child->a_count.count(attr))
                {
                    attr_count = child->a_count.at(attr);
                }

                for (auto val : val_set)
                {
                    // std::cout << val << std::endl;
                    COUNT_TYPE av_count = 0;
                    if (child->av_count.count(attr) and child->av_count.at(attr).count(val))
                    {
                        av_count = child->av_count.at(attr).at(val);
                    }

                    double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                    // std::cout << p << std::endl;
                    // if (attr.get_string() == "class"){
                    //     std::cout << val.get_string() << ", " << c_prob << ", " << p << ", " << p * c_prob << " :: ";
                    // }
                    // out[attr.get_string()][val.get_string()] += p * c_prob;
                    out[attr][val] += p * c_prob;
                }
            }
        }
        // std::cout << std::endl;
    }

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        for (auto val : val_set)
        {
            // out[attr.get_string()][val.get_string()] /= concept_weights;
            out[attr][val] /= concept_weights;
        }
    }

    return out;
}

inline std::unordered_map<int, std::unordered_map<int, double>> CobwebNode::predict_weighted_probs(INSTANCE_TYPE instance)
{

    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    double concept_weights = 0.0;
    std::unordered_map<int, std::unordered_map<int, double>> out;

    // std::cout << std::endl << "Prob of nodes along path (starting with leaf)" << std::endl;
    auto curr = this;
    while (curr != nullptr)
    {
        double c_prob = exp(curr->log_prob_class_given_instance(instance, true));
        // double c_prob = 1.0;
        // std::cout << c_prob << std::endl;
        concept_weights += c_prob;

        for (auto &[attr, val_set] : this->tree->attr_vals)
        {
            // std::cout << attr << std::endl;
            int num_vals = this->tree->attr_vals.at(attr).size();
            float alpha = this->tree->alpha;
            COUNT_TYPE attr_count = 0;

            if (curr->a_count.count(attr))
            {
                attr_count = curr->a_count.at(attr);
            }

            for (auto val : val_set)
            {
                // std::cout << val << std::endl;
                COUNT_TYPE av_count = 0;
                if (curr->av_count.count(attr) and curr->av_count.at(attr).count(val))
                {
                    av_count = curr->av_count.at(attr).at(val);
                }

                double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                // std::cout << p << std::endl;
                // if (attr.get_string() == "class"){
                //     std::cout << val.get_string() << ", " << c_prob << ", " << p << ", " << p * c_prob << " :: ";
                // }
                // out[attr.get_string()][val.get_string()] += p * c_prob;
                out[attr][val] += p * c_prob;
            }
        }
        // std::cout << std::endl;

        curr = curr->parent;
    }

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        for (auto val : val_set)
        {
            // out[attr.get_string()][val.get_string()] /= concept_weights;
            out[attr][val] /= concept_weights;
        }
    }

    return out;
}

inline std::unordered_map<int, std::unordered_map<int, double>> CobwebNode::predict_log_probs()
{
    std::unordered_map<int, std::unordered_map<int, double>> out;
    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        // std::cout << attr << std::endl;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha;
        COUNT_TYPE attr_count = 0;

        if (this->a_count.count(attr))
        {
            attr_count = this->a_count.at(attr);
        }

        for (auto val : val_set)
        {
            // std::cout << val << std::endl;
            COUNT_TYPE av_count = 0;
            if (this->av_count.count(attr) and this->av_count.at(attr).count(val))
            {
                av_count = this->av_count.at(attr).at(val);
            }

            // double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            // out[attr.get_string()][val.get_string()] += p;
            // std::cout << p << std::endl;
            // out[attr.get_string()][val.get_string()] = (log(av_count + alpha) - log(attr_count + num_vals * alpha));
            out[attr][val] = (log(av_count + alpha) - log(attr_count + num_vals * alpha));
        }
    }

    return out;
}

inline std::unordered_map<int, std::unordered_map<int, double>> CobwebNode::predict_probs()
{
    std::unordered_map<int, std::unordered_map<int, double>> out;
    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        // std::cout << attr << std::endl;
        int num_vals = this->tree->attr_vals.at(attr).size();
        float alpha = this->tree->alpha;
        COUNT_TYPE attr_count = 0;

        if (this->a_count.count(attr))
        {
            attr_count = this->a_count.at(attr);
        }

        for (auto val : val_set)
        {
            // std::cout << val << std::endl;
            COUNT_TYPE av_count = 0;
            if (this->av_count.count(attr) and this->av_count.at(attr).count(val))
            {
                av_count = this->av_count.at(attr).at(val);
            }

            double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
            // std::cout << p << std::endl;
            // out[attr.get_string()][val.get_string()] += p;
            out[attr][val] += p;
        }
    }

    return out;
}

inline std::vector<std::tuple<VALUE_TYPE, double>> CobwebNode::get_weighted_values(
    ATTR_TYPE attr, bool allowNone)
{

    std::vector<std::tuple<VALUE_TYPE, double>> choices;
    if (!this->av_count.count(attr))
    {
        choices.push_back(std::make_tuple(-1, 1.0));
    }
    double valCount = 0;
    for (auto &[val, tmp] : this->av_count.at(attr))
    {
        COUNT_TYPE count = this->av_count.at(attr).at(val);
        choices.push_back(std::make_tuple(val, (1.0 * count) / this->count));
        valCount += count;
    }
    if (allowNone)
    {
        choices.push_back(std::make_tuple(-1, ((1.0 * (this->count - valCount)) / this->count)));
    }
    return choices;
}

inline VALUE_TYPE CobwebNode::predict(ATTR_TYPE attr, std::string choiceFn, bool allowNone)
{
    std::function<ATTR_TYPE(std::vector<std::tuple<VALUE_TYPE, double>>)> choose;
    if (choiceFn == "most likely" || choiceFn == "m")
    {
        choose = most_likely_choice;
    }
    else if (choiceFn == "sampled" || choiceFn == "s")
    {
        choose = weighted_choice;
    }
    else
        throw "Unknown choice_fn";
    if (!this->av_count.count(attr))
    {
        return -1;
    }
    std::vector<std::tuple<VALUE_TYPE, double>> choices = this->get_weighted_values(attr, allowNone);
    return choose(choices);
}

inline double CobwebNode::probability(ATTR_TYPE attr, VALUE_TYPE val)
{
    if (val == -1)
    {
        double c = 0.0;
        if (this->av_count.count(attr))
        {
            for (auto &[attr, vAttr] : this->av_count)
            {
                for (auto &[val, cnt] : vAttr)
                {
                    c += cnt;
                }
            }
            return (1.0 * (this->count - c)) / this->count;
        }
    }
    if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
    {
        return (1.0 * this->av_count.at(attr).at(val)) / this->count;
    }
    return 0.0;
}

inline double CobwebNode::category_utility()
{
    // double p_of_c = (1.0 * this->count) / this->tree->root->count;
    // return (p_of_c * (this->tree->root->entropy() - this->entropy()));

    double root_entropy = 0.0;
    double child_entropy = 0.0;

    double p_of_child = (1.0 * this->count) / this->tree->root->count;
    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        root_entropy += this->tree->root->entropy_attr(attr);
        child_entropy += this->entropy_attr(attr);
    }

    return p_of_child * (root_entropy - child_entropy);
}

inline std::vector<double> CobwebNode::log_prob_children_given_instance_ext(INSTANCE_TYPE instance)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->log_prob_children_given_instance(instance);
}

inline std::vector<double> CobwebNode::log_prob_children_given_instance(const AV_COUNT_TYPE &instance)
{
    std::vector<double> raw_log_probs = std::vector<double>();
    std::vector<double> norm_log_probs = std::vector<double>();

    for (auto &child : this->children)
    {
        raw_log_probs.push_back(child->log_prob_class_given_instance(instance, false));
    }

    double log_p_of_x = logsumexp(raw_log_probs);

    for (auto log_p : raw_log_probs)
    {
        norm_log_probs.push_back(log_p - log_p_of_x);
    }

    return norm_log_probs;
}

inline std::vector<double> CobwebNode::prob_children_given_instance_ext(INSTANCE_TYPE instance)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->prob_children_given_instance(instance);
}

inline std::vector<double> CobwebNode::prob_children_given_instance(const AV_COUNT_TYPE &instance)
{

    double sum_probs = 0;
    std::vector<double> raw_probs = std::vector<double>();
    std::vector<double> norm_probs = std::vector<double>();

    for (auto &child : this->children)
    {
        double p = exp(child->log_prob_class_given_instance(instance, false));
        sum_probs += p;
        raw_probs.push_back(p);
    }

    for (auto p : raw_probs)
    {
        norm_probs.push_back(p / sum_probs);
    }

    return norm_probs;
}

inline double CobwebNode::log_prob_class_given_instance_ext(INSTANCE_TYPE instance, bool use_root_counts)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->log_prob_class_given_instance(instance, use_root_counts);
}

inline double CobwebNode::log_prob_class_given_instance(const AV_COUNT_TYPE &instance, bool use_root_counts)
{

    double log_prob = log_prob_instance(instance);

    if (use_root_counts)
    {
        log_prob += log((1.0 * this->count) / this->tree->root->count);
    }
    else
    {
        log_prob += log((1.0 * this->count) / this->parent->count);
    }

    // std::cout << "LOB PROB" << std::to_string(log_prob) << std::endl;

    return log_prob;
}

inline double CobwebNode::log_prob_instance_ext(INSTANCE_TYPE instance)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->log_prob_instance(instance);
}

inline double CobwebNode::log_prob_instance(const AV_COUNT_TYPE &instance)
{

    double log_prob = 0;

    for (auto &[attr, vAttr] : instance)
    {
        // bool hidden = attr.is_hidden();
        bool hidden = (attr < 0);
        if (hidden || !this->tree->attr_vals.count(attr))
        {
            continue;
        }

        double num_vals = this->tree->attr_vals.at(attr).size();

        for (auto &[val, cnt] : vAttr)
        {
            if (!this->tree->attr_vals.at(attr).count(val))
            {
                continue;
            }

            double alpha = this->tree->alpha;
            double av_count = alpha;
            if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
            {
                av_count += this->av_count.at(attr).at(val);
            }

            // a_count starts with the alphas over all values (even vals not in
            // current node)
            COUNT_TYPE a_count = num_vals * alpha;
            if (this->a_count.count(attr))
            {
                a_count += this->a_count.at(attr);
            }

            // we use cnt here to weight accuracy by counts in the training
            // instance. Usually this is 1, but in  models, it might
            // be something else.
            log_prob += cnt * (log(av_count) - log(a_count));
        }
    }

    return log_prob;
}

inline double CobwebNode::log_prob_instance_missing_ext(INSTANCE_TYPE instance)
{
    // AV_COUNT_TYPE cached_instance;
    // for (auto &[attr, val_map] : instance)
    // {
    //     for (auto &[val, cnt] : val_map)
    //     {
    //         cached_instance[CachedString(attr)][CachedString(val)] = instance.at(attr).at(val);
    //     }
    // }

    return this->log_prob_instance_missing(instance);
}

inline double CobwebNode::log_prob_instance_missing(const AV_COUNT_TYPE &instance)
{

    double log_prob = 0;

    for (auto &[attr, val_set] : this->tree->attr_vals)
    {
        // for (auto &[attr, vAttr]: instance) {
        // bool hidden = attr.is_hidden();
        bool hidden = (attr < 0);
        if (hidden)
        {
            continue;
        }

        double num_vals = this->tree->attr_vals.at(attr).size();
        double alpha = this->tree->alpha;

        if (instance.count(attr))
        {
            for (auto &[val, cnt] : instance.at(attr))
            {

                // TODO IS THIS RIGHT???
                // we could treat it as just alpha...
                if (!this->tree->attr_vals.at(attr).count(val))
                {
                    std::cout << "VALUE MISSING TREATING AS ALPHA" << std::endl;
                    // continue;
                }

                double av_count = alpha;
                if (this->av_count.count(attr) && this->av_count.at(attr).count(val))
                {
                    av_count += this->av_count.at(attr).at(val);
                }

                // a_count starts with the alphas over all values (even vals not in
                // current node)
                COUNT_TYPE a_count = num_vals * alpha;
                if (this->a_count.count(attr))
                {
                    a_count += this->a_count.at(attr);
                }

                // we use cnt here to weight accuracy by counts in the training
                // instance. Usually this is 1, but in  models, it might
                // be something else.
                log_prob += cnt * (log(av_count) - log(a_count));
            }
        }
        else
        {
            double cnt = 1.0;
            if (this->tree->weight_attr)
            {
                cnt = (1.0 * this->tree->root->a_count.at(attr)) / (this->tree->root->count);
            }

            int num_vals_in_c = 0;
            if (this->av_count.count(attr))
            {
                auto attr_count = this->a_count.at(attr);
                num_vals_in_c = this->av_count.at(attr).size();
                for (auto &[val, av_count] : this->av_count.at(attr))
                {
                    double p = ((av_count + alpha) / (attr_count + num_vals * alpha));
                    log_prob += cnt * p * log(p);
                }
            }

            int n0 = num_vals - num_vals_in_c;
            double p_missing = alpha / (num_vals * alpha);
            log_prob += cnt * n0 * p_missing * log(p_missing);
        }
    }

    return log_prob;
}

int main(int argc, char *argv[])
{
    // std::vector<AV_COUNT_TYPE> instances;
    // std::vector<CobwebNode *> cs;
    // auto tree = CobwebTree(0.01, false, 2, true, true);

    // for (int i = 0; i < 1000; i++)
    // {
    //     INSTANCE_TYPE inst;
    //     inst["anchor"]["word" + std::to_string(i)] = 1;
    //     inst["anchor2"]["word" + std::to_string(i % 10)] = 1;
    //     inst["anchor3"]["word" + std::to_string(i % 20)] = 1;
    //     inst["anchor4"]["word" + std::to_string(i % 100)] = 1;
    //     // unpack the tuple return
    //     auto [current, _] = tree.ifit(inst, 0);
    //     cs.push_back(current);
    // }

    return 0;
}

PYBIND11_MODULE(cobweb_discrete, m)
{
    m.doc() = "cobweb plug-in"; // optional module docstring

    py::class_<CobwebNode>(m, "CobwebNode")
        .def(py::init<>())
        .def("pretty_print", &CobwebNode::pretty_print)
        .def("output_json", &CobwebNode::output_json)
        .def("predict_probs", &CobwebNode::predict_probs)
        .def("predict_log_probs", &CobwebNode::predict_log_probs)
        .def("predict_weighted_probs", &CobwebNode::predict_weighted_probs)
        .def("predict_weighted_leaves_probs", &CobwebNode::predict_weighted_leaves_probs)
        .def("predict", &CobwebNode::predict, py::arg("attr") = "",
             py::arg("choiceFn") = "most likely",
             py::arg("allowNone") = true)
        .def("get_best_level", &CobwebNode::get_best_level, py::return_value_policy::reference)
        .def("get_basic_level", &CobwebNode::get_basic_level, py::return_value_policy::reference)
        .def("log_prob_class_given_instance", &CobwebNode::log_prob_class_given_instance_ext)
        .def("log_prob_instance", &CobwebNode::log_prob_instance_ext)
        .def("log_prob_instance_missing", &CobwebNode::log_prob_instance_missing_ext)
        .def("prob_children_given_instance", &CobwebNode::prob_children_given_instance_ext)
        .def("log_prob_children_given_instance", &CobwebNode::log_prob_children_given_instance_ext)
        .def("entropy", &CobwebNode::entropy)
        .def("category_utility", &CobwebNode::category_utility)
        .def("partition_utility", &CobwebNode::partition_utility)
        .def("__str__", &CobwebNode::__str__)
        .def("concept_hash", &CobwebNode::concept_hash)
        .def_readonly("count", &CobwebNode::count)
        .def_readonly("children", &CobwebNode::children, py::return_value_policy::reference)
        .def_readonly("parent", &CobwebNode::parent, py::return_value_policy::reference)
        .def_readonly("av_count", &CobwebNode::av_count, py::return_value_policy::reference)
        .def_readonly("a_count", &CobwebNode::a_count, py::return_value_policy::reference)
        .def_readonly("tree", &CobwebNode::tree, py::return_value_policy::reference);

    py::class_<CobwebTree>(m, "CobwebTree")
        .def(py::init<float, bool, int, bool, bool>(),
             py::arg("alpha") = 1.0,
             py::arg("weight_attr") = false,
             py::arg("objective") = 0,
             py::arg("children_norm") = true,
             py::arg("norm_attributes") = false)
        .def("ifit", &CobwebTree::ifit,
             py::arg("instance") = std::vector<AV_COUNT_TYPE>(),
             py::arg("mode"),
             py::return_value_policy::reference)
        .def("fit", &CobwebTree::fit,
             py::arg("instances") = std::vector<AV_COUNT_TYPE>(),
             py::arg("mode"),
             py::arg("iterations") = 1,
             py::arg("randomizeFirst") = true)
        .def("categorize", &CobwebTree::categorize,
             py::arg("instance") = std::vector<AV_COUNT_TYPE>(),
             // py::arg("get_best_concept") = false,
             py::return_value_policy::reference)
        .def("predict_probs", &CobwebTree::predict_probs_mixture)
        .def("predict_probs_parallel", &CobwebTree::predict_probs_mixture_parallel)
        .def("clear", &CobwebTree::clear)
        .def("__str__", &CobwebTree::__str__)
        .def("dump_json", &CobwebTree::dump_json)
        // .def("load_json", &CobwebTree::load_json)
        .def("load_json_stream", &CobwebTree::load_json_stream)
        .def_readonly("root", &CobwebTree::root, py::return_value_policy::reference);
}

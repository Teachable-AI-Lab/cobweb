# Very Fast Cobweb
__See ([cobweb/README.md](https://github.com/Teachable-AI-Lab/cobweb/blob/main/README.md))__

# Extra features in Very Fast Cobweb
### Intergrated [Rapidjson](https://github.com/Tencent/rapidjson) for fast model loading `load_json_stream(json_path)`
### Intergrated [Rapidjson](https://github.com/Tencent/rapidjson) for fast model writing `write_json_stream(json_path)`
### Attribute types are now changed to `int` from `CachedString`
Here is the default (arbitrary) key mapping:
`const std::unordered_map<std::string, int> ATTRIBUTE_MAP = {
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
    {"children", 100000015}};`
__You should also need to setup unique integer key for instance representation. Note that all keys (including the keys in instance's attribute value dictionary) should be unique.__
### x7 times faster prediction by disabling the `predict_log_probs` calculation on zero counts in node's attribute value (`av_count`) map
### (slightly) faster implementation of `logsumexp`
### Use wandb to profile the `ifit` function.
### You can set `mode` in `ifit` function
- `mode=0`: evaluating all four tree operations everytime
- `mode=1`: using _insert_ (BEST) operation only
- `mode=2`: randomly select one of the four operations after identifying two best children
- `mode=3`: using _insert_ (BEST) operation, and with a small probability, use `mode=0` to get current tree operation


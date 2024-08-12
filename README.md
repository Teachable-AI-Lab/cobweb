# Very Fast Cobweb
__See ([cobweb/README.md](https://github.com/Teachable-AI-Lab/cobweb/blob/main/README.md))__

# Extra features in Very Fast Cobweb
### Intergrated [Rapidjson](https://github.com/Tencent/rapidjson) for fast model loading
### Attribute types are now changed to `int` from `CachedString`
### Use wandb to profile the `ifit` function.
### You can set `mode` in `ifit` function
- `mode=0`: evaluating all four tree operations everytime
- `mode=1`: using _insert_ (BEST) operation only
- `mode=2`: randomly select one of the four operations after identifying two best children
- `mode=3`: using _insert_ (BEST) operation, and with a small probability, use `mode=0` to get current tree operation

# Todos
[] Support streamline json model saving
[] Investigate high memory usage during inference


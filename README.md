# gfn-mcts-grid

Install Cython:

```sh
pip install Cython
```

Compile MCTS code:

```sh
setup.py build_ext --inplace
```

Run training:

```
python run.py --algo mcts_all --mcts_rollouts 4
```

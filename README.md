# DistLearn.jl
Example of distributed learning in Julia. Note this is **not** a full featured
distributed machine learning library, therefore we are **not** going to register
this in the Julia package system.

Instead, this is a demonstrative project showing how to use Julia's parallel and
distributed computing interfaces to easily implement your own distributed
optimization / learning algorithms.

## Dependencies

This example requires the following packages (with minimum version number specified):

- Julia 0.6.0
- ArgParse.jl 0.5.0

## Usages

The examples should be run with multiple Julia workers --- could be local processes or processes on remote nodes. To run Julia with `N` local worker process, type

```
julia -p N
```

or use `--machinefile <file>` to run workers on remote nodes specified by the hostnames listed in `<file>`. Please refer to [the Julia doc on parallel computing](https://docs.julialang.org/en/stable/manual/parallel-computing/) for more details and other ways to run distributed Julia.

The entry point for the examples are in `run.jl`. The general format is

```
julia -p <N> run.jl <algor-name> --<algor-opts> ...
```

Currently two algorithms are included: `asgd` and `cocoa`. Both algorithms are only implemented for binary classification with +1/-1 labels. ASGD is implemented with the hinge loss, and COCOA with the smoothed hinge loss. The reference for COCOA is

> Jaggi, M., Smith, V., Takác, M., Terhorst, J., Krishnan, S., Hofmann, T. and Jordan, M.I., 2014.
> *Communication-efficient distributed dual coordinate ascent*.
> In Advances in Neural Information Processing Systems (pp. 3068-3076).

For example, to run ASGD on the demo toy data with batch size 100 with 4 local worker processes:

```
julia -p 4 run.jl asgd --batch-size=100
```

Or you can get a list of available options via

```bash
$ julia -p 2 run.jl asgd --help
usage: run.jl [--data DATA] [--n-epoch N-EPOCH]
              [--regu-coef REGU-COEF] [--batch-size BATCH-SIZE]
              [--learning-rate LEARNING-RATE] [-h]

optional arguments:
  --data DATA            (default: "demo")
  --n-epoch N-EPOCH     (type: Int64, default: 100)
  --regu-coef REGU-COEF
                        (type: Float64, default: 0.01)
  --batch-size BATCH-SIZE
                        (type: Int64, default: 100)
  --learning-rate LEARNING-RATE
                        (type: Float64, default: 0.01)
  -h, --help            show this help message and exit
```

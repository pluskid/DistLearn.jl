__precompile__()

"""
We implement the *COCOA* algorithm from the following reference with the
*smoothed hinge loss* function.

> Jaggi, M., Smith, V., Takác, M., Terhorst, J., Krishnan, S., Hofmann, T. and Jordan, M.I., 2014.
> *Communication-efficient distributed dual coordinate ascent*.
> In Advances in Neural Information Processing Systems (pp. 3068-3076).

"""
module COCOA

using ArgParse
using DistLearn

function parse_cmdline(cmdline :: Vector{String}, s :: ArgParseSettings)
  @add_arg_table s begin
    "--beta"
      arg_type = Float64
      default = 1.0
      help = "scaling parameter beta"
    "--smoothness"
      arg_type = Float64
      default = 1.0
      help = "the smoothness of the loss"
  end

  return parse_args(cmdline, s)
end


################################################################################
# Parameters
################################################################################
struct HyperParameters
  β  :: Float64
  λ  :: Float64  # regularizer coefficient
  γ  :: Float64  # smoothness
  R2 :: Float64  # max l2 norm^2 of train samples
  n  :: Int      # (avg) num train per worker
  m  :: Int      # num workers
end
function Base.show(io :: IO, hp :: HyperParameters)
  println(io, "COCOA Hyper Parameters")
  println(io, "----------------------")
  println(io, "    β: ", hp.β)
  println(io, "    λ: ", hp.λ)
  println(io, "    γ: ", hp.γ)
  println(io, "   R2: ", hp.R2)
  println(io, "    n: ", hp.n)
  println(io, "    m: ", hp.m)
end

mutable struct LocalParameters
  α :: Vector{Float64}  # local dual variables
end

mutable struct Parameters
  w :: Vector{Float64}  # primal weights

  Parameters(dim :: Int) = new(zeros(dim))
end



################################################################################
# Worker
################################################################################
"To be run on remote workers"
function make_worker(w_id :: Int, n_workers :: Int, dset_name :: String)
  dset_tr, dset_tt = load_datasets(dset_name, w_id, n_workers)
  worker = Worker{HyperParameters,LocalParameters}()
  worker.dset_tr = dset_tr
  worker.dset_tt = dset_tt

  n_dim, n_smp = get_num_features(dset_tr), get_num_samples(dset_tr)
  worker.lp = LocalParameters(zeros(n_smp))

  return worker
end

"To be run on remote workers"
function gather_local_statistics(worker_ref :: Future)
  worker = fetch(worker_ref)

  stats = Dict(:n_train => get_num_samples(worker.dset_tr),
               :n_test => get_num_samples(worker.dset_tt),
               :fea_dim => get_num_features(worker.dset_tr),
               :max_l2_norm => maximum(norm(worker.dset_tr.data[:, i])
                                       for i = 1:size(worker.dset_tr.data, 2)))
  return stats
end

###############################################################################
# COCOA Building Blocks
################################################################################
"""
    smoothed_hinge_gradient(pred, y)

Compute gradient w.r.t. pred for the smoothed hinge loss defined as:

```
loss(pred, y) =
  let margin = pred * y
    if margin >= 1
      0
    elseif margin <= 0
      0.5 - margin
    else
      0.5 * (1 - margin)^2
```

The smoothed hinge loss is a smoothed version of the hinge loss, with smoothness
parameter 1.0. If needed, one can define general `Loss` type to dispatch gradient
computation automatically for different loss.

Note because the COCOA here compute one example at a time, so we define this
function for scalar prediction and label.
"""
function smoothed_hinge_gradient(pred :: Float64, y :: Int)
  margin = pred * y
  if margin >= 1
    return 0
  elseif margin <= 0
    return -y
  else
    return - (1 - margin) * y
  end
end

"To be run on remote worker"
function sdca(worker_ref :: Future, w :: Vector{Float64})
  worker = fetch(worker_ref)

  hp = worker.hp
  X = worker.dset_tr.data
  y = worker.dset_tr.labels
  Δα = zeros(size(worker.lp.α))
  α = copy(worker.lp.α)
  idx_all = shuffle(1:length(Δα))

  N = hp.n * hp.m
  λNγ = hp.λ * N * hp.γ

  for idx in idx_all
    pred = dot(X[:, idx], w)
    grad = smoothed_hinge_gradient(pred, y[idx])

    Δ = λNγ / (hp.R2 + λNγ) * (-grad - α[idx])

    α[idx] += Δ
    Δα[idx] += Δ
    w += Δ * X[:, idx] / (hp.λ * N)
  end

  worker.lp.α += (hp.β / hp.m) * Δα
  Δw = X * Δα / (hp.λ * N)
  Δw *= (hp.β / hp.m)
  return Δw
end

function cocoa_epoch(workers, params :: Parameters)
  Δw_all = invoke_on_workers(sdca, workers, params.w)
  params.w .+= sum(Δw_all)
end

################################################################################
# COCOA Algorithm
################################################################################
function cocoa(args)
  println("Spawning workers and loading data...")
  workers = spawn_workers(make_worker, args["data"])

  println("Gathering statistics from workers...")
  stats_all = invoke_on_workers(gather_local_statistics, workers)
  for stats in stats_all
    @assert stats[:fea_dim] == stats_all[1][:fea_dim]
  end
  n_tr_tot = sum(stats[:n_train] for stats in stats_all)

  println("Initializing parameters")
  params = Parameters(stats_all[1][:fea_dim])

  println("Initializing hyper-parameters")
  R2 = maximum(stats[:max_l2_norm] for stats in stats_all)^2
  n = round(Int, mean(stats[:n_train] for stats in stats_all))
  h_params = HyperParameters(args["beta"], args["regu-coef"],
                             args["smoothness"], R2, n, length(workers))
  invoke_on_workers(workers, h_params) do worker_ref, hp
    fetch(worker_ref).hp = hp
  end
  println(h_params)

  println("----------------------")
  println("Entering training loop...")
  for epoch = 1:args["n-epoch"]
    met_tr, met_tt = accumulate_metric(workers, Accuracy, params.w)

    # show progress
    print(@sprintf("E%03d WT(%s): ", epoch, Dates.format(now(), "dd.HH:MM:SS.sss")))
    print("TRAIN-", met_tr)
    println(", TEST-", met_tt)

    cocoa_epoch(workers, params)
  end
end

end  # module COCOA

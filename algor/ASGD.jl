__precompile__()

module ASGD
using ArgParse
using DistLearn

function parse_cmdline(cmdline :: Vector{String}, s :: ArgParseSettings)
  @add_arg_table s begin
    "--batch-size"
      arg_type = Int
      default = 100
    "--learning-rate"
      arg_type = Float64
      default = 0.01
  end

  return parse_args(cmdline, s)
end

################################################################################
# Parameters
################################################################################
"Hyper parameters for ASGD."
struct HyperParameters
  η          :: Float64
  batch_size :: Int
  λ          :: Float64
end
function Base.show(io :: IO, hp :: HyperParameters)
  println(io, "ASGD Hyper Parameters")
  println(io, "---------------------")
  println(io, "learning rate: ", hp.η)
  println(io, "   batch size: ", hp.batch_size)
  println(io, "    regu coef: ", hp.λ)
end

"Locally stored states for each worker."
mutable struct LocalParameters
  tr_idx  :: Vector{Int}
  i_batch :: Int
  ηₜ      :: Float64
end

"Global parameters for ASGD."
mutable struct Parameters
  w :: Vector{Float64}

  Parameters(dim :: Int) = new(randn(dim) / sqrt(dim))
end

################################################################################
# Worker
################################################################################
"To be run on remote workers"
function make_worker(w_id :: Int, n_workers :: Int, dset_name :: String)
  dset_tr, dset_tt = load_datasets(dset_name, w_id, n_workers)
  worker = Worker{HyperParameters, LocalParameters}()
  worker.dset_tr = dset_tr
  worker.dset_tt = dset_tt

  n_smp = get_num_samples(dset_tr)
  worker.lp = LocalParameters(collect(1:n_smp), 1, 0)
  return worker
end

"To be run on remote workers"
function gather_local_statistics(worker_ref :: Future)
  worker = try_fetch(worker_ref)

  stats = Dict(:n_train => get_num_samples(worker.dset_tr),
               :n_test => get_num_samples(worker.dset_tt),
               :fea_dim => get_num_features(worker.dset_tr))
  return stats
end


################################################################################
# ASGD Building Blocks
################################################################################
"""
    prepare_for_epoch(workers, epoch)

Ask each worker to prepare for a new epoch. This include resetting counter,
re-initialize random shuffling of the training points, and updating the step
size.
"""
function prepare_for_epoch(workers, epoch :: Int)
  function worker_prepare_for_epoch(worker_ref :: Future, epoch :: Int)
    worker = try_fetch(worker_ref)
    worker.lp.i_batch = 1
    shuffle!(worker.lp.tr_idx)
    worker.lp.ηₜ = worker.hp.η / sqrt(epoch)
    return true
  end

  invoke_on_workers(worker_prepare_for_epoch, workers, epoch)
end

"""
    hinge_gradient(pred, y)

Compute gradient w.r.t. pred for hinge loss `max(1 - pred*y, 0)`.
If needed, one can define general `Loss` type to dispatch gradient
computation automatically for different loss.
"""
function hinge_gradient(pred :: Vector{Float64}, y :: Vector{Int})
  margin = pred .* y
  grad = zeros(size(margin))
  for i = 1:length(margin)
    if 1 - margin[i] > 0
      grad[i] = -y[i]
    end
  end
  return grad
end

"""
    compute_delta(worker_ref, w)

To be run on remote workers. Compute the updates for the weights. We have
applied step sizes locally on each worker.
"""
function compute_delta(worker_ref :: Future, w :: Vector{Float64})
  worker = try_fetch(worker_ref)
  idx1 = (worker.lp.i_batch-1) * worker.hp.batch_size + 1
  if idx1 > length(worker.lp.tr_idx)
    return nothing  # end of epoch
  end

  idx2 = min(worker.lp.i_batch * worker.hp.batch_size,
             length(worker.lp.tr_idx))

  data_idx = worker.lp.tr_idx[idx1:idx2]
  X = worker.dset_tr.data[:, data_idx]
  y = worker.dset_tr.labels[data_idx]

  pred = X' * w
  ∇w = (X * hinge_gradient(pred, y)) / size(X, 2)
  ∇w += worker.hp.λ * w  # gradient from the regularizer

  Δw = -worker.lp.ηₜ * ∇w
  worker.lp.i_batch += 1

  return Δw
end

function run_epoch(workers, params :: Parameters)
  @sync begin
    for (pid, worker_ref) in workers
      @async begin
        while true
          Δw = remotecall_fetch(compute_delta, pid, worker_ref, params.w)
          if Δw === nothing
            # this worker finishes its epoch
            break
          end
          params.w .+= Δw
        end
      end
    end
  end
end


################################################################################
# ASGD Algorithm
################################################################################
function asgd(args)
  println("Spawning workers and loading data...")
  workers = spawn_workers(make_worker, args["data"])

  println("Gathering statistics from workers...")
  stats_all = invoke_on_workers(gather_local_statistics, workers)
  for stats in stats_all
    @assert stats[:fea_dim] == stats_all[1][:fea_dim]
  end
  fea_dim = stats_all[1][:fea_dim]

  println("Initializing parameters...")
  params = Parameters(fea_dim)

  println("Initializing hyper parameters...")
  h_params = HyperParameters(args["learning-rate"], args["batch-size"],
                             args["regu-coef"])
  invoke_on_workers(workers, h_params) do worker_ref, hp
    try_fetch(worker_ref).hp = hp
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

    # optimization
    prepare_for_epoch(workers, epoch)
    run_epoch(workers, params)
  end
end

end  # module ASGD

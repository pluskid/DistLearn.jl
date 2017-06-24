mutable struct Worker{HPType, LPType}
  dset_tr   :: Dataset # training set
  dset_tt   :: Dataset # test / validation set

  hp        :: HPType  # hyper parameters
  lp        :: LPType  # worker local parameters

  Worker{HPType, LPType}() where {HPType, LPType} = new()
end


"""
    spanw_workers(ctor, args...)

Create `Worker` objects on each worker process. Return a list
of (pid, worker future ref) tuples.
"""
function spawn_workers(ctor :: Function, args...)
  return map(1:nworkers()) do wid
    pid = wid + 1  # head node is 1, all other nodes are workers
    (pid, remotecall(ctor, pid, wid, nworkers(), args...))
  end
end


"""
    invoke_on_workers(f, workers, args...)

Invoke `f` on each workers in parallel, with `f(worker_ref, args...)`.
This function blocks until all workers finishes, and return a list
of results from each worker.
"""
function invoke_on_workers(f :: Function, workers, args...)
  rets = Array{Any}(length(workers))
  @sync begin
    for (i, (pid, worker_ref)) in enumerate(workers)
      @async begin
        rets[i] = remotecall_fetch(f, pid, worker_ref, args...)
      end
    end
  end

  # propagate remote exceptions
  for obj in rets
    if obj isa RemoteException
      throw(obj)
    end
  end
  return rets
end

"""
    accumulate_metric(workers, Type{MetricType}, w)

Collect and accumulate metrics from workers. This function assumes linear
predictors and assume only one metric object is needed. One can define this
in more general ways for general predictors and multiple metrics. Also it
might be more efficient to collect the metrics during the forward stage of
the training to avoid duplicated computation.
"""
function accumulate_metric{T <: Metric}(workers, :: Type{T}, w :: Vector{Float64})
  function eval_metric{T <: Metric}(worker_ref :: Future, :: Type{T}, w :: Vector{Float64})
    worker = fetch(worker_ref)
    return map((worker.dset_tr, worker.dset_tt)) do dset
      pred = dset.data' * w
      evaluate(T, pred, dset.labels)
    end
  end
  # rets is [(m_tr1, m_tt1), (m_tr2, m_tt2), ...]
  rets = invoke_on_workers(eval_metric, workers, T, w)
  return merge([x[1] for x in rets]), merge([x[2] for x in rets])
end

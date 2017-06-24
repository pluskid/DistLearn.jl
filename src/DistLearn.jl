__precompile__()

module DistLearn

export try_fetch
include("utils.jl")

export Dataset
export load_datasets, get_num_samples, get_num_features
include("dataset.jl")

export Metric, MetricAccumulator, Accuracy
export evaluate, merge
include("metric.jl")

export Worker
export spawn_workers, invoke_on_workers
export accumulate_metric
include("worker.jl")

export make_arg_parser
include("arg_parse.jl")

end  # module DistLearn

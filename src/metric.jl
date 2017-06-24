abstract type Metric end

struct MetricAccumulator{T <: Metric}
  val :: Float64
  cnt :: Int
end

function merge{T <: Metric}(ms :: Vector{MetricAccumulator{T}})
  val = sum(m.val for m in ms)
  cnt = sum(m.cnt for m in ms)
  return MetricAccumulator{T}(val, cnt)
end

function Base.show{T <: Metric}(io :: IO, m :: MetricAccumulator{T})
  print(io, @sprintf("%s = %8.4f", name(T), m.val / m.cnt))
end


type Accuracy <: Metric
end

"""
`evaluate` takes a real-valued prediction vector and a +1/-1 integer vector of labels.
It returns a metric accumulator.
"""
function evaluate(:: Type{Accuracy}, pred :: Vector{Float64}, y :: Vector{Int})
  cnt = length(y)
  acc = 0.0
  for (a, b) in zip(pred, y)
    acc += (a*b > 0)
  end
  return MetricAccumulator{Accuracy}(acc, cnt)
end

function name(:: Type{Accuracy})
  return "ACC"
end

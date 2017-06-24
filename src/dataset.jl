"""
A simple dataset container. We assume each worker can hold its
full dataset in its local memory here.
"""
struct Dataset
  data   :: AbstractArray{Float64, 2}
  labels :: Vector{Int64}
end

function get_num_samples(dset :: Dataset)
  return size(dset.data, 2)
end

function get_num_features(dset :: Dataset)
  return size(dset.data, 1)
end

"""
    load_dataset(name, i_slice, n_slices)

Load dataset for a worker. We use data parallelism here, so the full
dataset is split into `n_slices` chunks, and the i-th worker only
load its own chunk.
"""
function load_datasets(name :: String, i_slice :: Int, n_slices :: Int)
  if name == "demo"
    # a toy linear model with noise
    INPUT_DIM = 500
    N_TRAIN_TOTAL = 5000
    N_TEST_TOTAL = 1000
    NOISE_LEVEL = 0.05

    # we fix a random seed so that each worker
    # is getting the same synthetic classification
    # ground-truth
    RND_SEED = 1234
    srand(RND_SEED)
    c_pos = randn(INPUT_DIM, 1)
    c_neg = randn(INPUT_DIM, 1)

    N_TRAIN = round(Int, N_TRAIN_TOTAL / n_slices)
    N_TEST = round(Int, N_TEST_TOTAL / n_slices)

    dsets = map((N_TRAIN, N_TEST)) do n
      n_pos = round(Int, n / 2)
      n_neg = n - n_pos

      x_pos = randn((INPUT_DIM, n_pos)) .+ c_pos
      x_neg = randn((INPUT_DIM, n_neg)) .+ c_neg
      x = cat(2, x_pos, x_neg) / sqrt(INPUT_DIM)
      y = cat(1, ones(n_pos), -ones(n_neg))

      noise = rand(n)
      for i = 1:length(y)
        if noise[i] <= NOISE_LEVEL
          y[i] = -y[i]
        end
      end

      return Dataset(x, y)
    end
    return dsets
  else
    error("Unknown dataset: ", name)
  end
end

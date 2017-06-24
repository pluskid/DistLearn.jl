using ArgParse

function make_arg_parser()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--data"
      default = "demo"
    "--n-epoch"
        arg_type = Int
        default = 100
    "--regu-coef"
      arg_type = Float64
      default = 0.01
  end
  return s
end

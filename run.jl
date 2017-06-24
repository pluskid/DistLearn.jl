# Since we are not installing the package, we manually push the
# src dir to load path so that the supporting code can be used.
# Note we use @everywhere to ensure all the worker knows the correct
# load path
@everywhere append!(LOAD_PATH,
                    [joinpath(dirname(@__FILE__), x) for x in ("src", "algor")])

using DistLearn
using ASGD
using COCOA

if length(ARGS) < 1
  println("Usage:")
  println()
  println("  julia -p NWORKER run.jl (asgd|cocoa|...) --options...")
  println()
  println("e.g.  julia -p 2 run.jl asgd --help")
  exit(-1)
end

if nprocs() <= 1
  error("Please start julia with multiple workers.")
end

algor_name = ARGS[1]
cmdline = ARGS[2:end]
s = DistLearn.make_arg_parser()

if algor_name == "asgd"
  args = ASGD.parse_cmdline(cmdline, s)
  ASGD.asgd(args)
elseif algor_name == "cocoa"
  args = COCOA.parse_cmdline(cmdline, s)
  COCOA.cocoa(args)
else
  error("Unknown algorithm: ", algor_name)
end

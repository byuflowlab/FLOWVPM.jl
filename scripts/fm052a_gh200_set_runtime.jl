# 052a Phase A (A1): write the offline GH200 CUDA preferences WITHOUT loading
# CUDA.jl — the x86 login node has no CUDA toolkit or GPU, so `using CUDA`
# cannot precompile there while the env still carries the canonical
# local-toolkit prefs (observed: CUDA_Runtime_Discovery ptxas_path failure).
# Pref names match CUDA.jl's runtime selection ("version"/"local" on
# CUDA_Runtime_jll) and the canonical fm023/fm052 recipe's "local" prefs:
# runtime AND compiler pinned to the FP052_CUDA_PIN artifact (default 12.6;
# unpinned, the compiler hook queries the node driver and picked ptxas 13.3
# against runtime 12.6 in job 13477127 — CUDA.jl requires matching majors),
# driver kept on the node's system libcuda.
using TOML

pin = get(ENV, "FP052_CUDA_PIN", "12.6")
lp = joinpath(dirname(Base.active_project()), "LocalPreferences.toml")
prefs = isfile(lp) ? TOML.parsefile(lp) : Dict{String,Any}()
prefs["CUDA_Runtime_jll"] = merge(get(prefs, "CUDA_Runtime_jll", Dict{String,Any}()),
                                  Dict{String,Any}("version" => pin, "local" => "false"))
prefs["CUDA_Compiler_jll"] = merge(get(prefs, "CUDA_Compiler_jll", Dict{String,Any}()),
                                   Dict{String,Any}("version" => pin, "local" => "false"))
prefs["CUDA_Driver_jll"] = merge(get(prefs, "CUDA_Driver_jll", Dict{String,Any}()),
                                 Dict{String,Any}("local" => "true"))
open(lp, "w") do io
    TOML.print(io, prefs)
end

# Preferences are resolved by UUID→name lookup against the project's
# deps/extras tables; without an [extras] entry the JLL augmentation hooks
# cannot see the pin and fall back to querying the node driver (observed in
# job 13476523: the hook selected the un-staged cuda+13.3 artifact). This is
# the bookkeeping CUDA.set_runtime_version! normally performs.
jlls = ("CUDA_Runtime_jll", "CUDA_Compiler_jll", "CUDA_Driver_jll")
pf = Base.active_project()
mf = joinpath(dirname(pf), "Manifest.toml")
man = TOML.parsefile(mf)
proj = TOML.parsefile(pf)
extras = get!(proj, "extras", Dict{String,Any}())
for name in jlls
    haskey(man["deps"], name) || error("$name not present in $mf")
    extras[name] = man["deps"][name][1]["uuid"]
end
open(pf, "w") do io
    TOML.print(io, proj)
end
println("gh200 CUDA preferences written to $lp: runtime pin=$pin (artifact route), ",
        "compiler local=false, driver local=true; [extras] registered: ",
        join(jlls, ", "))

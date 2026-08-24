# Corrected task-049 H200 harness. Requires all p018 dumps 710:719 as ARGS.
import FLOWVPM, CUDA
using Random: MersenneTwister, randperm
using Statistics: median
using SHA: sha256
const vpm=FLOWVPM; const fm=FLOWVPM.fmm
const STEPS=collect(710:719); const RHOS=(4.211,4.789); const PS=(4,8)
# D14 production selection (2026-08-22): residency A/B, budget, and profile
# stages run at the production operating point; the acceptance matrix above
# (PS x TYPES x RHOS) brackets it unchanged.
const P_PROD=6; const RHO_PROD=4.789
const TYPES=(Float64,Float32); const REPS=parse(Int,get(ENV,"FM049_REPS","5"))
const NSAMPLE=parse(Int,get(ENV,"FM049_N_TARGETS","2000")); const SAMPLE_SEED=49049
const DT=1/3240
CUDA.functional() || error("CUDA unavailable")
length(ARGS)==10 || error("pass exactly p018 steps 710:719")
const INPUT_MANIFEST=get(ENV,"FM049_MANIFEST",joinpath(dirname(first(ARGS)),"manifest.csv"))

function loadbin(path)
    m=match(r"p018_(\d+)_particles\.bin$",basename(path)); m===nothing && error("bad name $path")
    A,n=open(path,"r") do io
        nr=read(io,Int64); n=read(io,Int64); nr==46 || error("$path: rows=$nr")
        A=Matrix{Float64}(undef,46,n); read!(io,A); eof(io)||error("trailing bytes: $path"); A,Int(n)
    end
    filesize(path)==16+46*n*sizeof(Float64)||error("size mismatch: $path")
    all(isfinite,A)||error("nonfinite: $path"); all(>(0),view(A,vpm.SIGMA_INDEX,:))||error("sigma: $path")
    (;step=parse(Int,m.captures[1]),path,A,n,sha=bytes2hex(open(sha256,path)))
end
snaps=sort!(loadbin.(ARGS);by=x->x.step)
getfield.(snaps,:step)==STEPS || error("need steps 710:719")
A0=snaps[1].A; n=snaps[1].n; NSAMPLE<=n||error("sample too large")
isfile(INPUT_MANIFEST)||error("input manifest missing: $INPUT_MANIFEST")
manifest_lines=readlines(INPUT_MANIFEST)
!isempty(manifest_lines)&&manifest_lines[1]=="file,step,np,sha256"||error("bad manifest header")
M=Dict{String,Tuple{Int,Int,String}}(); seen_steps=Set{Int}()
for line in manifest_lines[2:end]
    !isempty(line)||error("blank manifest row")
    x=split(line,','); length(x)==4||error("bad manifest row: $line")
    m=match(r"^p018_(\d+)_particles\.bin$",x[1]);m===nothing&&error("bad manifest filename: $(x[1])")
    step=parse(Int,x[2]);step==parse(Int,m.captures[1])||error("manifest step/filename mismatch: $line")
    np=parse(Int,x[3]);np>0||error("manifest np must be positive: $line")
    hash=lowercase(x[4]);occursin(r"^[0-9a-f]{64}$",hash)||error("bad manifest SHA-256: $line")
    haskey(M,x[1])&&error("duplicate manifest filename: $(x[1])")
    step in seen_steps&&error("duplicate manifest step: $step")
    M[x[1]]=(step,np,hash);push!(seen_steps,step)
end
Set(keys(M))==Set(basename.(ARGS))||error("manifest files differ from supplied snapshots")
seen_steps==Set(STEPS)||error("manifest steps differ from 710:719")
for s in snaps
    get(M,basename(s.path),nothing)==(s.step,s.n,s.sha)||error("manifest mismatch: $(s.path)")
end

opts(P,rho=1.0)=vpm.FMM(;p=P,ncrit=50,theta=.4,autotune_p=false,autotune_ncrit=false,
  autotune_reg_error=false,default_rho_over_sigma=rho)
function field(A,::Type{T},P,rho;device=true,sfs=false,UJ=vpm.UJ_fmm,fmm_rho=1.0) where T
    pf=vpm.ParticleField(size(A,2),T;formulation=vpm.rVPM,kernel=vpm.gaussianerf,
      viscous=vpm.Inviscid(),SFS=sfs ? vpm.SFS_Cs_nobackscatter : vpm.noSFS,
      transposed=true,integration=vpm.rungekutta3,UJ,fmm=opts(P,fmm_rho),
      arraytype=device ? CUDA.CuArray : Matrix)
    copyto!(pf.particles,T.(A)); pf.np=size(A,2)
    device && vpm.radix_fmm_settings!(pf;expansion_order=P,rho_t=rho)
    pf
end
function rr(A,B,inds)
    a=Float64.(Array(view(A,inds,:))); b=Float64.(Array(view(B,inds,:)))
    sqrt(sum(abs2,a.-b)/max(sum(abs2,b),eps()))
end
secs(f)=(CUDA.synchronize();t=time_ns();f();CUDA.synchronize();(time_ns()-t)/1e9)
function ab(f,g)
    a=Float64[];b=Float64[]
    for k=1:REPS
        isodd(k) ? (push!(a,secs(f));push!(b,secs(g))) : (push!(b,secs(g));push!(a,secs(f)))
    end
    a,b
end
function counters(pf)
    c=vpm._radix_fmm_couplings[pf].cache.state.counters
    (;body=c.body_uploads,influence=c.influence_downloads,expansion=c.expansion_host_copies,
      route=c.route_uploads,operator=c.operator_uploads,metadata=c.metadata_downloads)
end
R=NamedTuple[]
failures=String[]
rec(;section,case="p018_710",P="",precision="",rho_t="",metric,value,gate="",verdict="",detail="")=
 push!(R,(;section,case,P,precision,rho_t,metric,value,gate,verdict,detail))
for s=snaps
    rec(;section="inventory",case="p018_$(s.step)",metric="np",value=s.n,
      verdict="PASS",detail="sha256=$(s.sha);file=$(basename(s.path))")
end

idx=sort(randperm(MersenneTwister(SAMPLE_SEED),n)[1:NSAMPLE]); refs=Dict()
for T in TYPES
    ref=field(A0,T,4,RHOS[1]); tr=secs(()->vpm.UJ_direct(ref;sfs=true,reset_sfs=true)); H=Array(ref.particles); refs[T]=H
    src=field(A0,T,4,RHOS[1];device=false,UJ=vpm.UJ_direct)
    tgt=field(A0[:,idx],T,4,RHOS[1];device=false,UJ=vpm.UJ_direct)
    tgt.particles[vpm.U_INDEX,:].=0; tgt.particles[vpm.J_INDEX,:].=0; vpm.UJ_direct(src,tgt)
    gu=T===Float64 ? 1e-9 : 5e-5
    for (m,e) in (("direct_u_integrity",rr(tgt.particles,H[:,idx],vpm.U_INDEX)),
                  ("direct_j_integrity",rr(tgt.particles,H[:,idx],vpm.J_INDEX)))
        rec(;section="reference",precision=string(T),metric=m,value=e,gate=gu,verdict=e<=gu ? "PASS" : "FAIL")
        e<=gu || push!(failures,"$T $m failed: $e")
    end
    rec(;section="reference",precision=string(T),metric="gpu_direct_s",value=tr,verdict="INFO")
    # Independent CPU SFS oracles on the same seeded subset. This exercises
    # both Estr_direct! and the legacy CPU Estr_fmm! path without an O(N^2)
    # CPU solve over all 210k particles.
    cpu_direct=field(A0[:,idx],T,4,RHOS[1];device=false,UJ=vpm.UJ_direct)
    vpm.UJ_direct(cpu_direct;sfs=true,reset_sfs=true)
    gpu_subset=field(A0[:,idx],T,4,RHOS[1]); vpm.UJ_direct(gpu_subset;sfs=true,reset_sfs=true)
    es=rr(gpu_subset.particles,cpu_direct.particles,vpm.SFS_INDEX)
    rec(;section="reference",precision=string(T),metric="gpu_cpu_Estr_direct_rel_rms",value=es,
      gate=gu,verdict=es<=gu ? "PASS" : "FAIL")
    es<=gu||push!(failures,"$T GPU/CPU Estr_direct failed: $es")
    for P in PS,rho in RHOS
      cpu_fmm=field(A0[:,idx],T,P,rho;device=false,UJ=vpm.UJ_fmm,fmm_rho=rho)
      vpm.UJ_fmm(cpu_fmm;sfs=true,reset_sfs=true,autotune=false)
      ef=rr(cpu_fmm.particles,cpu_direct.particles,vpm.SFS_INDEX)
      rec(;section="reference",P,precision=string(T),rho_t=rho,
        metric="cpu_Estr_fmm_vs_direct_rel_rms",value=ef,
        gate=1e-3,verdict=ef<=1e-3 ? "PASS" : "FAIL",
        detail="legacy CPU FMM matched to candidate cutoff")
      ef<=1e-3||push!(failures,"$T P$P rho=$rho CPU Estr_fmm failed: $ef")
    end
end

for T in TYPES,P in PS,rho in RHOS
    pf=field(A0,T,P,rho)
    for _=1:3; vpm.UJ_fmm(pf;sfs=false); vpm.UJ_fmm(pf;sfs=true,reset_sfs=true); end
    vpm.UJ_fmm(pf;sfs=true,reset_sfs=true)
    gate=T===Float64 ? 5e-4 : 1e-3
    for (m,inds) in (("u_rel_rms",vpm.U_INDEX),("j_rel_rms",vpm.J_INDEX),("sfs_rel_rms",vpm.SFS_INDEX))
        e=rr(pf.particles,refs[T],inds); rec(;section="accuracy",P,precision=string(T),rho_t=rho,
          metric=m,value=e,gate,verdict=e<=gate ? "PASS" : "FAIL")
        e<=gate || push!(failures,"accuracy $m failed for $T P$P rho=$rho: $e > $gate")
    end
    a,b=ab(()->vpm.UJ_fmm(pf;sfs=false),()->vpm.UJ_fmm(pf;sfs=true,reset_sfs=true))
    for (m,x) in (("uj_median_s",median(a)),("ujsfs_median_s",median(b)),("sfs_marginal_s",median(b)-median(a)))
        rec(;section="timing",P,precision=string(T),rho_t=rho,metric=m,value=x,verdict="INFO",detail="interleaved synchronized")
    end
    c0=counters(pf); ha=@allocated vpm.UJ_fmm(pf;sfs=false); da=CUDA.@allocated vpm.UJ_fmm(pf;sfs=false)
    he=@allocated vpm.UJ_fmm(pf;sfs=true,reset_sfs=true); de=CUDA.@allocated vpm.UJ_fmm(pf;sfs=true,reset_sfs=true)
    state=vpm._radix_fmm_couplings[pf].cache.state; hctx=state.interaction_list
    eligible=fm._cuda_graph_eligible(state); exec0=hctx.graph_exec; epoch0=hctx.epoch_id
    vpm.UJ_fmm(pf;sfs=true,reset_sfs=true); x=Array(pf.particles[vcat(vpm.U_INDEX,vpm.J_INDEX,vpm.SFS_INDEX),:])
    vpm.UJ_fmm(pf;sfs=true,reset_sfs=true); y=Array(pf.particles[vcat(vpm.U_INDEX,vpm.J_INDEX,vpm.SFS_INDEX),:]); c1=counters(pf)
    checks=(("host_alloc_uj_bytes",ha,ha<=4096),("host_alloc_ujsfs_bytes",he,he<=4096),
      ("device_alloc_uj_bytes",da,da==0),("device_alloc_ujsfs_bytes",de,de==0),
      ("replay_bitwise_equal",isequal(x,y),isequal(x,y)),
      ("graph_eligible",eligible,eligible),("graph_exec_reused",hctx.graph_exec===exec0,exec0!==nothing&&hctx.graph_exec===exec0),
      ("graph_epoch_reused",hctx.graph_epoch==epoch0==hctx.epoch_id,hctx.graph_epoch==epoch0==hctx.epoch_id))
    for (m,x,ok) in checks
        rec(;section="contract",P,precision=string(T),rho_t=rho,metric=m,value=x,verdict=ok ? "PASS" : "FAIL",detail="before=$c0")
        ok||push!(failures,"contract $m failed for $T P$P rho=$rho")
    end
    for name in propertynames(c0)
      before=getproperty(c0,name);after=getproperty(c1,name);delta=after-before
      must_zero=name in (:body,:influence,:expansion,:metadata)
      ok=must_zero ? before==after==0 : before==after
      for (suffix,value) in (("before",before),("after",after),("delta",delta))
        rec(;section="counter",P,precision=string(T),rho_t=rho,
          metric="$(name)_$suffix",value,gate=must_zero ? 0 : "unchanged",
          verdict=ok ? "PASS" : "FAIL")
      end
      ok||push!(failures,"counter $name failed for $T P$P rho=$rho: $before -> $after")
    end
    vpm.clear_radix_fmm_cache!(pf)
    released=!haskey(vpm._radix_fmm_couplings,pf)
    rec(;section="contract",P,precision=string(T),rho_t=rho,metric="cache_registry_released",
      value=released,verdict=released ? "PASS" : "FAIL")
    released||push!(failures,"cache registry release failed for $T P$P rho=$rho")
    pf=nothing; GC.gc(); CUDA.reclaim()
end

function checkpoint(pf)
  st=pf.splitting_state
  (;particles=copy(pf.particles),scratch=copy(pf.scratch),np=pf.np,nt=pf.nt,t=pf.t,
    sigma_0=copy(st.sigma_0),H_chi=copy(st.H_chi),hold=copy(st.hold_counter),
    cooldown=copy(st.cooldown_counter))
end
function restore!(pf,c;particles=true)
  particles&&copyto!(pf.particles,c.particles);copyto!(pf.scratch,c.scratch)
  pf.np=c.np;pf.nt=c.nt;pf.t=c.t
  st=pf.splitting_state
  copyto!(st.sigma_0,c.sigma_0);copyto!(st.H_chi,c.H_chi)
  copyto!(st.hold_counter,c.hold);copyto!(st.cooldown_counter,c.cooldown)
  pf
end
vec_rel(a,b)=sqrt(sum(abs2,Float64.(a).-Float64.(b))/max(sum(abs2,Float64.(b)),eps()))
function clean_stage_seconds(cache,pf,which)
  state=cache.state;targets=(pf,)
  which==:tree_refresh&&return secs(()->fm.update_cuda_radix_state!(cache,targets))
  fm.update_cuda_radix_state!(cache,targets)
  which==:b2m&&return secs(()->fm._launch_cuda_b2m!(state))
  fm._launch_cuda_b2m!(state)
  which==:m2m&&return secs(()->fm._launch_cuda_resident_m2m!(state))
  fm._launch_cuda_resident_m2m!(state)
  which==:m2l&&return secs(()->fm._launch_cuda_resident_m2l!(state))
  fm._launch_cuda_resident_m2l!(state)
  which==:l2l&&return secs(()->fm._launch_cuda_resident_l2l!(state))
  fm._launch_cuda_resident_l2l!(state)
  which==:nearfield_uj&&return secs(()->fm._launch_cuda_nearfield_kernel!(state))
  fm._launch_cuda_nearfield_kernel!(state)
  which==:l2b&&return secs(()->fm._launch_cuda_resident_l2b_only!(state,nothing))
  fm._launch_cuda_resident_l2b_only!(state,nothing)
  which==:sfs&&return secs(()->fm._launch_cuda_sfs!(state))
  error("unknown clean stage $which")
end

B=NamedTuple[]
for snap in snaps
    rho=RHO_PROD
    resident=field(snap.A,Float64,P_PROD,rho;sfs=true); upload=field(snap.A,Float64,P_PROD,rho;sfs=true)
    base_r=checkpoint(resident);base_u=checkpoint(upload)
    vpm.nextstep(resident,DT);vpm.nextstep(upload,DT);CUDA.synchronize()
    hin=copy(snap.A);rout=similar(snap.A);uout=similar(snap.A)
    fr=()->vpm.nextstep(resident,DT)
    fu=()->(copyto!(upload.particles,hin);vpm.nextstep(upload,DT);copyto!(uout,upload.particles))
    ar=Float64[];au=Float64[]
    for k=1:REPS
        restore!(resident,base_r);restore!(upload,base_u;particles=false);CUDA.synchronize()
        isodd(k) ? (push!(ar,secs(fr));push!(au,secs(fu))) :
          (push!(au,secs(fu));push!(ar,secs(fr)))
    end
    copyto!(rout,resident.particles);CUDA.synchronize()
    mr,mu=median(ar),median(au)
    for (m,x,v) in (("resident_step_median_s",mr,"USER_CHECKPOINT"),("upload_step_median_s",mu,"USER_CHECKPOINT"),
                    ("upload_minus_resident_s",mu-mr,"USER_CHECKPOINT"))
        rec(;section="residency_ab",case="p018_$(snap.step)",P=P_PROD,precision="Float64",rho_t=rho,metric=m,value=x,verdict=v,
          detail="true interleaved identical-state A/B; no automatic default")
    end
    channels=(("all_46",1:46),("X",vpm.X_INDEX),("Gamma",vpm.GAMMA_INDEX),
      ("sigma",vpm.SIGMA_INDEX:vpm.SIGMA_INDEX),("U",vpm.U_INDEX),
      ("J",vpm.J_INDEX),("SFS",vpm.SFS_INDEX))
    for (name,inds) in channels
      e=rr(uout,rout,inds); ok=e<=1e-11
      rec(;section="residency_parity",case="p018_$(snap.step)",P=P_PROD,precision="Float64",rho_t=rho,
        metric="$(name)_rel_rms",value=e,gate=1e-11,verdict=ok ? "PASS" : "FAIL",
        detail=name=="all_46" ? "includes physical/history/model/scratch rows" : "independent channel")
      ok||push!(failures,"residency $name parity step=$(snap.step) rho=$rho: $e")
    end
    aux=(
      ("scratch_rel_rms",rr(upload.scratch,resident.scratch,axes(resident.scratch,1)),1e-11),
      ("np_equal",upload.np==resident.np==snap.n,0),
      ("nt_equal",upload.nt==resident.nt==base_r.nt+1,0),
      ("t_equal",upload.t==resident.t==base_r.t+DT,0),
      ("split_sigma0_rel_rms",vec_rel(upload.splitting_state.sigma_0,resident.splitting_state.sigma_0),1e-11),
      ("split_H_chi_rel_rms",vec_rel(upload.splitting_state.H_chi,resident.splitting_state.H_chi),1e-11),
      ("split_hold_equal",upload.splitting_state.hold_counter==resident.splitting_state.hold_counter,0),
      ("split_cooldown_equal",upload.splitting_state.cooldown_counter==resident.splitting_state.cooldown_counter,0))
    for (name,value,gate) in aux
      ok=value isa Bool ? value : value<=gate
      rec(;section="residency_parity",case="p018_$(snap.step)",P=P_PROD,precision="Float64",rho_t=rho,
        metric=name,value,gate,verdict=ok ? "PASS" : "FAIL",detail="non-particle mutable state")
      ok||push!(failures,"residency $name failed step=$(snap.step) rho=$rho: $value")
    end
    h2d,_=ab(()->copyto!(upload.particles,hin),()->copyto!(upload.particles,hin))
    d2h,_=ab(()->copyto!(uout,upload.particles),()->copyto!(uout,upload.particles))
    push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage="full_resident_rk3",median_s=mr,note="end-to-end"))
    push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage="full_upload_rk3",median_s=mu,note="contiguous H2D + RK3 + contiguous D2H"))
    push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage="h2d_46xn",median_s=median(h2d),note="preallocated contiguous transfer"))
    push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage="d2h_46xn",median_s=median(d2h),note="preallocated contiguous transfer"))
    if snap.step==first(STEPS)
      # Profiling owns a disposable field/cache. It never mutates either A/B
      # arm, and every isolated sample rebuilds its prerequisite chain.
      prof=field(snap.A,Float64,P_PROD,rho;sfs=true);profile_base=checkpoint(prof)
      vpm.UJ_fmm(prof;sfs=true,reset_sfs=true);CUDA.synchronize()
      cache=vpm._radix_fmm_couplings[prof].cache
      hctx=cache.state.interaction_list
      profile_update=[Float64[] for _=1:5];profile_m2l=Float64[]
      try
        hctx.profile_stages=true
        for _=1:REPS
          restore!(prof,profile_base);CUDA.synchronize()
          vpm.UJ_fmm(prof;sfs=true,reset_sfs=true);CUDA.synchronize()
          for i=1:5;push!(profile_update[i],hctx.update_stage_ns[i]/1e9);end
          push!(profile_m2l,sum(hctx.m2l_level_ns)/1e9)
        end
      finally
        hctx.profile_stages=false
      end
      for (stage,samples) in zip((:profile_grid,:profile_occupancy,:profile_direct_gen,
          :profile_route_gen,:profile_stage_groups),profile_update)
        push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage=string(stage),
          median_s=median(samples),note="normal lifecycle hctx.profile_stages telemetry"))
      end
      push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage="profile_m2l_levels",
        median_s=median(profile_m2l),note="normal lifecycle hctx.profile_stages per-level sum"))
      stage_names=(:tree_refresh,:b2m,:m2m,:m2l,:l2l,:nearfield_uj,:l2b,:sfs)
      stage_total=0.0
      for stage in stage_names
        samples=[clean_stage_seconds(cache,prof,stage) for _=1:REPS]
        value=median(samples);stage_total+=value
        push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage=string(stage),median_s=value,
          note="disposable profiling cache; clean ordered prerequisite chain per sample"))
      end
      restore!(prof,profile_base);CUDA.synchronize()
      eval_s=median([secs(()->vpm.UJ_fmm(prof;sfs=true,reset_sfs=true)) for _=1:REPS])
      integrator_residual=mr-3eval_s
      push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage="ujsfs_complete",median_s=eval_s,
        note="single complete synchronized UJ+SFS evaluation"))
      push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage="rk3_integrator_residual",median_s=integrator_residual,
        note="full resident RK3 minus 3 complete UJ+SFS evaluations; includes RK3 state updates"))
      push!(B,(;step=snap.step,np=snap.n,rho_t=rho,stage="isolated_stage_sum",median_s=stage_total,
        note="diagnostic only; production nearfield/farfield overlap means this does not sum to ujsfs_complete"))
      vpm.clear_radix_fmm_cache!(prof)
      released=!haskey(vpm._radix_fmm_couplings,prof)
      restore!(prof,profile_base);vpm.UJ_fmm(prof;sfs=true,reset_sfs=true);CUDA.synchronize()
      rebuilt=haskey(vpm._radix_fmm_couplings,prof)
      vpm.clear_radix_fmm_cache!(prof);released_again=!haskey(vpm._radix_fmm_couplings,prof)
      ok_profile_cache=released&&rebuilt&&released_again
      rec(;section="contract",case="p018_$(snap.step)",P=P_PROD,precision="Float64",rho_t=rho,
        metric="profiling_cache_release_rebuild_release",value=ok_profile_cache,
        verdict=ok_profile_cache ? "PASS" : "FAIL")
      ok_profile_cache||push!(failures,"profiling cache lifecycle failed step=$(snap.step) rho=$rho")
      prof=nothing;GC.gc();CUDA.reclaim()
    end
    for pf_release in (resident,upload)
      vpm.clear_radix_fmm_cache!(pf_release)
      released=!haskey(vpm._radix_fmm_couplings,pf_release)
      rec(;section="contract",case="p018_$(snap.step)",P=P_PROD,precision="Float64",rho_t=rho,
        metric="ab_cache_registry_released",value=released,verdict=released ? "PASS" : "FAIL")
      released||push!(failures,"A/B cache release failed step=$(snap.step) rho=$rho")
    end
    resident=nothing;upload=nothing;GC.gc();CUDA.reclaim()
end

q(x)="\""*replace(string(x),'"'=>"\"\"")*"\""
open("fm049_results.csv","w") do io
 println(io,"section,case,p,precision,rho_t,metric,value,gate,verdict,detail")
 for r=R;println(io,join(q.((r.section,r.case,r.P,r.precision,r.rho_t,r.metric,r.value,r.gate,r.verdict,r.detail)),','));end
end
open("fm049_budget.csv","w") do io
 println(io,"step,np,rho_t,stage,median_s,target_fraction,note");for r=B;println(io,"$(r.step),$(r.np),$(r.rho_t),$(r.stage),$(r.median_s),$(r.median_s/3.3),\"$(r.note)\"");end
end
project=Base.active_project(); manifest=joinpath(dirname(project),"Manifest.toml")
source_path=abspath(@__FILE__); source_sha=bytes2hex(open(sha256,source_path))
githead(path)=try readchomp(`git -C $path rev-parse HEAD`) catch; "unavailable" end
open("fm049_report.md","w") do io
 println(io,"# Task 049 corrected rotor verification\n\n- Job: `$(get(ENV,"SLURM_JOB_ID","local"))`; GPU UUID: `$(get(ENV,"FM049_GPU_UUID","unavailable"))`\n- Julia: `$(VERSION)`; threads: `$(Threads.nthreads())`\n- CUDA runtime/driver: `$(CUDA.runtime_version())` / `$(CUDA.driver_version())`; device: `$(CUDA.name(CUDA.device()))`\n- Command: `$(join(Base.ARGS," "))`; seed: `$SAMPLE_SEED`; reps: `$REPS`; sample: `$NSAMPLE`; dt: `$DT`\n- Matrix: P=`$PS`, precision=`$TYPES`, rho_t=`$RHOS`; benchmark snapshot: `710`; residency snapshots: `710:719`; residency/budget/profile at production settings P=`$P_PROD`, rho_t=`$RHO_PROD` (D14, 2026-08-22)\n- Project: `$project`; Manifest SHA-256: `$(isfile(manifest) ? bytes2hex(open(sha256,manifest)) : "missing")`\n- Harness: `$source_path`; SHA-256: `$source_sha`\n- FLOWVPM commit: `$(githead(joinpath(@__DIR__,"..")))`; FastMultipole commit: `$(githead(dirname(dirname(pathof(fm)))))`\n- Input manifest: `$INPUT_MANIFEST`; SHA-256: `$(bytes2hex(open(sha256,INPUT_MANIFEST)))`\n\nResults include the complete matrix, hard direct-reference gates, contracts, hashes, and true residency A/B. Budget rows include end-to-end, contiguous-transfer, clean ordered-stage, complete-UJ+SFS, and RK3 residual timings. The run script hashes results, budget, report, raw log, synced-source manifests, and submission provenance.\n\n041a anchors: 7.40 ms at 1e5 and 92.3 ms at 1e6; CPU production baseline: 170–230 s/step; target: 3.3 s/step.\n\n## Residency checkpoint\n\nNo threshold chooses a default. Present both A/B results to the user and ask which mode should ship.")
end
isempty(failures)||error("fm049 completed all rows with failed gates: "*join(failures," | "))
println("fm049 complete: all gates passed")

"""Summarize actual run.json evidence without inventing missing training audits."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                pass
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("path",type=Path,nargs="?")
    p.add_argument("--latest",help="Find the most recent completed run.json for this stage, e.g. pilot-d")
    p.add_argument("--replay-tol",type=float,default=1e-4)
    a=p.parse_args()
    if a.latest:
        if a.path is not None:raise SystemExit("Use a path or --latest, not both.")
        found=list((Path(__file__).resolve().parent/"outputs").glob(a.latest+"_*/run.json"))
        if not found:raise SystemExit("No run.json found. Inspect console.log first if training exited abnormally.")
        path=max(found,key=lambda f:f.stat().st_mtime)
    elif a.path is not None:path=a.path/"run.json" if a.path.is_dir() else a.path
    else:raise SystemExit("Supply a run.json path or --latest STAGE.")
    d=json.loads(path.read_text(encoding="utf-8"));errors=[];incomplete=[];notes=[]
    def require(ok,why):
        if not ok:errors.append(why)
    def check_finite(value,loc="report"):
        if isinstance(value,float) and not math.isfinite(value):errors.append("Non-finite number at "+loc)
        elif isinstance(value,dict):
            for k,v in value.items():check_finite(v,loc+"."+k)
        elif isinstance(value,list):
            for i,v in enumerate(value):check_finite(v,f"{loc}[{i}]")
    check_finite(d)
    require(d.get("lightning_optimization_steps",0)>0,"No recorded optimizer update.")
    require(d.get("sampling_replay_sample_count",0)>0,"Replay sample count missing or zero.")
    err=d.get("sampling_replay_max_abs_error")
    require(isinstance(err,(int,float)) and math.isfinite(err) and err<a.replay_tol,"Replay error missing/non-finite/over tolerance.")
    require(d.get("lightning_amp_skipped_steps",0)==0,"AMP skipped updates occurred; inspect before acceptance.")
    require(not d.get("infrastructure_interrupted_workers"),"Infrastructure interruption was recorded.")
    worker_cuda=d.get("environment_worker_cuda_initialized")
    require(isinstance(worker_cuda,list) and len(worker_cuda)>0 and not any(worker_cuda),"Worker CUDA-isolation evidence missing or an environment worker initialized CUDA.")
    require(sum(float(h.get("ppo_updates",0)) for h in d.get("history",[]))>0,"No actual PPO updates recorded.")
    for i,h in enumerate(d.get("history",[])):
        if h.get("ppo_updates",0)>0:
            e=h.get("sampling_replay_max_abs_error")
            require(isinstance(e,(int,float)) and math.isfinite(e) and e<a.replay_tol and h.get("sampling_replay_sample_count",0)>0,
                    f"Rollout {i} has missing or invalid pre-update replay evidence.")
    if d.get("run_mode")=="pilot":
        if d.get("lightning_fit_calls",0)<2:incomplete.append("Pilot did not record two rollout updates.")
        if d.get("successful_batch_count",0)<1:incomplete.append("No complete aircraft batch; partial trajectory is not full feasibility PASS.")
        if d.get("actual_disturbance_hit_count",0)<1:incomplete.append("No confirmed disturbance hit.")
        feasibility=d.get("independent_feasibility",[])
        if any(x.get("feasible") is False or x.get("violations") for x in feasibility):errors.append("Independent feasibility violation.")
        if not any(x.get("feasible") is True and x.get("batch_complete") is True for x in feasibility):
            incomplete.append("No complete independently feasible audited batch in the exported audit.")
        aux=bool(d.get("method_profile",{}).get("use_time_auxiliary",d.get("method_variant")=="D"))
        if aux:
            require(d.get("time_supervision_optimizer_updates",0)>0,"D pilot has no actual auxiliary optimizer update.")
            require(d.get("time_head_training_status")=="trained_online","D time head is not recorded as trained_online.")
    two_cycle=[{"worker_id":s.get("worker_id"),"episode_id":s.get("episode_id"),
                "transfers":len(s.get("actual_transfer_times",[]))}
               for s in d.get("scenario_log",[]) if len(s.get("actual_transfer_times",[]))>=2]
    if d.get("run_mode")=="pilot" and not two_cycle:incomplete.append("No episode with two actual transfers was recorded.")
    peak=d.get("gpu_memory_peak_bytes")
    if isinstance(peak,(int,float)) and peak>6*1024**3:notes.append("Allocated peak exceeds the 6 GiB engineering target; check total nvidia-smi usage before scaling.")
    scope=d.get("sampling_replay_scope")
    notes.append("Replay scope: "+str(scope))
    notes.append("Next-only cross-update identity/masks and bound frozen predictor checksums are not certified by these aggregate fields.")
    result={"status":"FAIL" if errors else "INCOMPLETE" if incomplete else "BASIC_RECORDED_CHECKS_PASS","source_file":str(path),
            "method":d.get("method_variant"),"device":d.get("device"),"amp_dtype":d.get("amp_dtype"),
            "steps":d.get("total_decisions"),"fit_calls":d.get("lightning_fit_calls"),
            "optimizer_steps":d.get("lightning_optimization_steps"),"successful_batches":d.get("successful_batch_count"),
            "actual_hits":d.get("actual_disturbance_hit_count"),"aux_updates":d.get("time_supervision_optimizer_updates"),
            "replay_error":err,"replay_samples":d.get("sampling_replay_sample_count"),
            "gpu_allocated_peak_GiB":None if peak is None else peak/1024**3,
            "episodes_with_two_transfers":two_cycle,"errors":errors,"incomplete":incomplete,"notes":notes}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(2 if errors else 3 if incomplete else 0)
if __name__=="__main__":main()


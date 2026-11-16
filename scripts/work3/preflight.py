"""Check the active local environment without installing or updating packages."""
from __future__ import annotations
import argparse,datetime,importlib.metadata,json,platform,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from runtime.artifacts import create_run_context
def command(args,cwd):
    try:
        r=subprocess.run(args,cwd=cwd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30)
        return {"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:return {"error":str(e)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--device",default="cuda:0")
    p.add_argument("--run-id")
    a=p.parse_args();root=ROOT
    report={"python":sys.version,"platform":platform.platform(),"device_requested":a.device}
    context=create_run_context(root,experiment_name="work3_preflight",run_id=a.run_id)
    if context.run_dir.exists():raise SystemExit("运行目录已存在，请使用新的run-id以保护已有检查报告。")
    create_run_context(root,experiment_name=context.experiment_name,run_id=context.run_id,create_dirs=True)
    out=context.reports_dir
    report["git_head"]=command(["git","rev-parse","HEAD"],root)
    report["git_status"]=command(["git","status","--short"],root)
    report["nvidia_smi"]=command(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"],root)
    freeze=command([sys.executable,"-m","pip","freeze"],root)
    (context.configs_dir/"pip-freeze.txt").write_text(freeze.get("stdout",json.dumps(freeze)),encoding="utf-8")
    try:
        import torch,torch_geometric,lightning.pytorch
        report["versions"]={name:importlib.metadata.version(name) for name in ["torch","torch-geometric","lightning","numpy","omegaconf"]}
        report["torch_cuda_build"]=torch.version.cuda
        report["cuda_available"]=torch.cuda.is_available()
        dev=torch.device(a.device)
        if dev.type=="cuda":
            if not torch.cuda.is_available():raise RuntimeError("CUDA is unavailable in this Python environment.")
            prop=torch.cuda.get_device_properties(dev)
            report.update(device_name=prop.name,total_memory_bytes=int(prop.total_memory),
                          bf16_reported_supported=bool(torch.cuda.is_bf16_supported()))
        x=torch.randn(2,8,device=dev)
        layer=torch.nn.Linear(8,4).to(dev);loss=layer(x).square().mean();loss.backward()
        assert layer.weight.grad is not None and bool(torch.isfinite(layer.weight.grad).all().item())
        if dev.type=="cuda":torch.cuda.synchronize(dev)
        report["small_fp32_backward"]="PASS"
        # Validate every delivered config with the actual repository validator.
        sys.path.insert(0,str(root))
        from training.work3.runtime_config import load_work3_runtime_config
        report["configs"]={}
        for path in sorted((root/"conf"/"work3").glob("local_*.yaml")):
            load_work3_runtime_config(path);report["configs"][path.name]="PASS"
        report["status"]="PASS"
    except Exception as e:
        report.update(status="FAIL",exception_type=type(e).__name__,exception=str(e));raise
    finally:
        (out/"preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False,indent=2));print("Artifacts:",out)
if __name__=="__main__":main()


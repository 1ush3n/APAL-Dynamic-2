"""Check the active local environment without installing or updating packages."""
from __future__ import annotations
import argparse,datetime,importlib.metadata,json,platform,subprocess,sys
from pathlib import Path
def command(args,cwd):
    try:
        r=subprocess.run(args,cwd=cwd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30)
        return {"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:return {"error":str(e)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--device",default="cuda:0")
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    report={"python":sys.version,"platform":platform.platform(),"device_requested":a.device}
    out=root/"work3_4060/outputs"/("preflight_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True,exist_ok=False)
    report["git_head"]=command(["git","rev-parse","HEAD"],root)
    report["git_status"]=command(["git","status","--short"],root)
    report["nvidia_smi"]=command(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"],root)
    freeze=command([sys.executable,"-m","pip","freeze"],root)
    (out/"pip-freeze.txt").write_text(freeze.get("stdout",json.dumps(freeze)),encoding="utf-8")
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
        from training.work3_runtime_config import load_work3_runtime_config
        report["configs"]={}
        for path in sorted((root/"work3_4060/configs").glob("*.yaml")):
            load_work3_runtime_config(path);report["configs"][path.name]="PASS"
        report["status"]="PASS"
    except Exception as e:
        report.update(status="FAIL",exception_type=type(e).__name__,exception=str(e));raise
    finally:
        (out/"preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False,indent=2));print("Artifacts:",out)
if __name__=="__main__":main()


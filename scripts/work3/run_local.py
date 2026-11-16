"""通过现有训练入口执行有界的历史C/D本地检查。"""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.artifacts import create_run_context
from training.work3.runtime_config import load_work3_runtime_config

STAGES=("smoke-c","smoke-d","pilot-c","pilot-d","middle-d","stability-d","bridge-c","bridge-d")
def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                pass
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=STAGES)
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--amp",choices=("fp32","bf16"),default="fp32")
    p.add_argument("--ppo-batch",type=int)
    p.add_argument("--aux-batch",type=int)
    p.add_argument("--steps",type=int)
    p.add_argument("--max-decisions",type=int)
    p.add_argument("--wall-seconds",type=float)
    p.add_argument("--r3-e04-audit",action="store_true",help="Enable R3/E04 audit")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--run-id")
    p.add_argument("--runs-root",type=Path,default=Path("runs"))
    args=p.parse_args();repo=args.repo.resolve()
    train=repo/Path("scripts", "work3", "train_ppo_work3.py")
    if not train.is_file():raise SystemExit("Repository training entry not found.")
    conf=repo/"conf"/"work3"/("local_"+args.stage.replace("-","_")+".yaml")
    config=load_work3_runtime_config(conf)
    for name in ("ppo_batch","aux_batch","steps","max_decisions"):
        v=getattr(args,name)
        if v is not None and v<1:raise SystemExit(name+" must be positive.")
    if args.stage.startswith("smoke"):
        n=args.steps or config["ppo"]["steps_per_iter"]
        limit=args.max_decisions or n
        batch=args.ppo_batch or config["ppo"]["batch_size"]
        if not 1<=limit<=n<=64 or batch<limit:
            raise SystemExit("Smoke requires max_decisions <= steps <=64 and PPO batch >= max_decisions.")
    context=create_run_context(repo,experiment_name="work3_local_"+args.stage.replace("-","_"),
                               run_id=args.run_id,runs_root=args.runs_root)
    out=context.run_dir
    cmd=[sys.executable,str(train),"--config",str(conf),"--seed",str(args.seed),"--device",args.device,
         "--set","runtime.amp_dtype="+args.amp,"--output",str(context.checkpoint_dir/"weights.pt"),
         "--report",str(context.reports_dir/"run.json"),
         "--set","artifacts.experiment_name="+context.experiment_name,
         "--set","artifacts.run_id="+context.run_id,"--set","artifacts.runs_root="+str(args.runs_root)]
    if args.ppo_batch is not None:cmd+=["--batch-size",str(args.ppo_batch)]
    if args.aux_batch is not None:cmd+=["--time-auxiliary-batch-size",str(args.aux_batch)]
    if args.steps is not None:
        cmd+=["--steps",str(args.steps)]
        if args.stage.startswith("smoke") and args.max_decisions is None:
            cmd+=["--max-decisions",str(args.steps)]
    if args.max_decisions is not None:cmd+=["--max-decisions",str(args.max_decisions)]
    if args.wall_seconds is not None:cmd+=["--max-wall-seconds",str(args.wall_seconds)]
    if args.r3_e04_audit or os.environ.get("WORK3_R3_E04_AUDIT") in ("1", "true", "TRUE"):
        cmd+=["--r3-e04-audit"]
    if args.dry_run:
        from omegaconf import OmegaConf
        print(json.dumps({"cwd":str(repo),"argv":cmd,"config":OmegaConf.to_container(config,resolve=True)},ensure_ascii=False,indent=2));return
    if out.exists():
        raise SystemExit("运行目录已存在，请使用新的run-id以保护已有输出。")
    env=os.environ.copy();env["PYTHONUNBUFFERED"]="1";env["PYTHONIOENCODING"]="utf-8"
    print("Stage:",args.stage," Output:",out,flush=True)
    # 训练入口独占创建run目录；启动器只在确认子进程清单后归入日志。
    with tempfile.TemporaryFile(mode="w+",encoding="utf-8") as log:
        child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               text=True,encoding="utf-8",errors="replace",bufsize=1)
        try:
            assert child.stdout is not None
            for line in child.stdout:
                try:
                    print(line,end="",flush=True)
                except UnicodeEncodeError:
                    print(line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"),end="",flush=True)
                log.write(line);log.flush()
            code=child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
        manifest_path=context.configs_dir/"run_manifest.json"
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        if (manifest.get("run_id")==context.run_id and manifest.get("pid")==child.pid
                and manifest.get("command")==cmd[1:]):
            context.logs_dir.mkdir(parents=True,exist_ok=True)
            log.seek(0)
            with (context.logs_dir/"console.log").open("x",encoding="utf-8") as saved:
                shutil.copyfileobj(log,saved)
            with (context.configs_dir/"launch.json").open("x",encoding="utf-8") as saved:
                json.dump({"argv":cmd,"cwd":str(repo),"child_pid":child.pid,"returncode":code},saved,ensure_ascii=False,indent=2)
        elif code==0:
            raise RuntimeError("训练返回成功，但缺少属于本次子进程的运行清单，拒绝归档为成功。")
    print("Exit code:",code," Results:",out,flush=True)
    raise SystemExit(code)
if __name__=="__main__":main()


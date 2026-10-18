"""Run isolated bounded training stages through the repository's existing CLI."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys
from pathlib import Path

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
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
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
    args=p.parse_args();repo=args.repo.resolve();bundle=Path(__file__).resolve().parent
    if bundle.parent!=repo:
        raise SystemExit("Extract work3_4060 directly under the repository root; --repo must point to that parent.")
    train=repo/"scripts/work3/train_ppo_work3.py"
    if not train.is_file():raise SystemExit("Repository training entry not found.")
    conf=bundle/"configs"/(args.stage.replace("-","_")+".yaml")
    config=json.loads(conf.read_text(encoding="utf-8"))
    for name in ("ppo_batch","aux_batch","steps","max_decisions"):
        v=getattr(args,name)
        if v is not None and v<1:raise SystemExit(name+" must be positive.")
    if args.stage.startswith("smoke"):
        n=args.steps or config["ppo"]["steps_per_iter"]
        limit=args.max_decisions or n
        batch=args.ppo_batch or config["ppo"]["batch_size"]
        if not 1<=limit<=n<=64 or batch<limit:
            raise SystemExit("Smoke requires max_decisions <= steps <=64 and PPO batch >= max_decisions.")
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out=bundle/"outputs"/f"{args.stage}_seed{args.seed}_{args.amp}_{stamp}"
    cmd=[sys.executable,str(train),"--config",str(conf),"--seed",str(args.seed),"--device",args.device,
         "--set","runtime.amp_dtype="+args.amp,"--output",str(out/"weights.pt"),"--report",str(out/"run.json")]
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
        print(json.dumps({"cwd":str(repo),"argv":cmd,"config":config},ensure_ascii=False,indent=2));return
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env["PYTHONUNBUFFERED"]="1";env["PYTHONIOENCODING"]="utf-8"
    (out/"launch.json").write_text(json.dumps({"argv":cmd,"cwd":str(repo)},ensure_ascii=False,indent=2),encoding="utf-8")
    print("Stage:",args.stage," Output:",out,flush=True)
    with (out/"console.log").open("w",encoding="utf-8") as log:
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
    print("Exit code:",code," Results:",out,flush=True)
    raise SystemExit(code)
if __name__=="__main__":main()


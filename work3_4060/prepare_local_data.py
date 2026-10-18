"""Prepare bounded Work-3 diagnostic fixtures without changing model/environment code."""
from __future__ import annotations
import argparse,copy,hashlib,json
from collections import Counter
from pathlib import Path

EXPECTED_SOURCE_SHA256="b71031f470680c68b49ee3147d56ed2cf0429feb8745303fb77f560747daa992"
SEEDS=(15,16,8,9,12,13,20,21,24,23)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def build_dataset(source,selected,k):
    out=copy.deepcopy(source)
    selected=set(selected)
    out["tasks"]={key:copy.deepcopy(t) for key,t in source["tasks"].items()
                  if int(t["aircraft_id"])<k and int(t["task_id"]) in selected}
    for task in out["tasks"].values():
        # We require predecessor closure, rather than dropping predecessor edges.
        assert set(task["predecessors"])<=selected,(task["task_key"],task["predecessors"])
    out["metadata"].update(num_aircraft=k,num_stations=5,total_cycles=k+4,
                           physical_tasks_per_aircraft=len(selected),total_physical_tasks=k*len(selected))
    out["diagnostic_fixture"]={"role":"engineering_debug_only","source_sha256":EXPECTED_SOURCE_SHA256,
                              "selected_task_ids":sorted(selected),"predecessor_closed":True}
    return out
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();repo=a.repo.resolve();dest=Path(__file__).resolve().parent
    source_path=repo/"data/work3/real_283_k10_baseline.json"
    if digest(source_path)!=EXPECTED_SOURCE_SHA256:
        raise ValueError("Source baseline changed; compare its process DAG before regenerating these pinned diagnostic fixtures.")
    source=json.loads(source_path.read_text(encoding="utf-8"))
    template={int(t["task_id"]):t for t in source["tasks"].values() if int(t["aircraft_id"])==0}
    def closure(seeds):
        kept=set();todo=list(seeds)
        while todo:
            tid=todo.pop()
            if tid in kept:continue
            kept.add(tid);todo.extend(template[tid]["predecessors"])
        return kept
    fixtures={"mini70":(closure(SEEDS),7),"mini210":(closure((*SEEDS,125)),7),"real566":(set(template),2)}
    summary={}
    for name,(ids,k) in fixtures.items():
        baseline=build_dataset(source,ids,k);bp=dest/f"data/{name}.json";dump(bp,baseline)
        event={"scenario_id":f"LOCAL_{name.upper()}_INITIAL_HIT","valid":True,"timing":"EARLY",
               "intensity":"LOW","station_id":0,"aircraft_id":0,"tau":0.0,"delta":20.0,
               "recovery_time":20.0,"candidate_count":1,"affected_count":1,
               "affected_task_keys":["0_16"],"timing_reference":"local_diagnostic_fixed_event"}
        ep=dest/f"data/{name}_events.json";sp=dest/f"data/{name}_split.json"
        dump(ep,[event]);dump(sp,[event["scenario_id"]])
        per_plane=[t for t in baseline["tasks"].values() if int(t["aircraft_id"])==0]
        summary[name]={"aircraft":k,"stations":5,"tasks_per_aircraft":len(ids),"tasks":len(baseline["tasks"]),
                       "skills":sorted({int(t["skill"]) for t in per_plane}),
                       "demands":sorted({int(t["demand"]) for t in per_plane}),
                       "station_task_counts":dict(Counter(int(t["station_id"]) for t in per_plane)),
                       "files":{str(f.relative_to(dest)):digest(f) for f in [bp,ep,sp]}}
    middle={"scenario_id":"LOCAL_MINI210_MIDDLE_S2","valid":True,"timing":"MID","intensity":"DIAGNOSTIC",
            "station_id":2,"aircraft_id":2,"tau":130.20533674370247,"recovery_time":226.7626745432108,
            "affected_task_keys":["2_13"],"timing_reference":"pinned_nominal_heuristic_full_line_cycle5",
            "diagnostic_only":True}
    dump(dest/"data/mini210_middle_events.json",[middle])
    dump(dest/"data/mini210_middle_split.json",[middle["scenario_id"]])
    dump(dest/"data/fixture_manifest.json",{"source_sha256":EXPECTED_SOURCE_SHA256,"fixtures":summary,
        "middle_event_files":{str(f.relative_to(dest)):digest(f) for f in
            [dest/"data/mini210_middle_events.json",dest/"data/mini210_middle_split.json"]}})
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()


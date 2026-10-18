"""Verify all delivered local fixtures with the actual environment and independent checker."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from envs.work3.environment import AirLineEnvWork3
from envs.work3.core_types import TaskStatus,ActionBranch
from models.work3.heuristic_agent import HeuristicAgentWork3
from utils.work3.trajectory_feasibility import check_environment_trajectory_feasibility
def main():
    results={}
    for name,n,k in [("mini70",70,7),("mini210",210,7),("real566",566,2)]:
        variants=["nominal","hit"]+(["middle"] if name=="mini210" else [])
        for variant in variants:
            disturbed=variant!="nominal"
            env=AirLineEnvWork3(baseline_json_path=ROOT/f"work3_4060/data/{name}.json");env.reset()
            by_ac={a:{t.task_id for t in env.state.tasks.values() if t.aircraft_id==a} for a in range(k)}
            for task in env.state.tasks.values():
                # Check closure against the complete raw physical DAG, not a filtered graph.
                missing=set(env.constraint_engine.physical_predecessors[task.task_id])-by_ac[task.aircraft_id]
                assert not missing,(task.task_key,missing)
            event_name=name+("_middle" if variant=="middle" else "")
            scenario=json.loads((ROOT/f"work3_4060/data/{event_name}_events.json").read_text(encoding="utf-8"))[0]
            if disturbed:env.load_scenario(scenario)
            hit=bool(env.disturbance_event_results.get(scenario["scenario_id"],{}).get("actual_hit_task_keys"))
            if variant=="hit":assert hit,"The initial diagnostic event must hit 0_16 before decisions."
            agent=HeuristicAgentWork3();steps=0;began=time.perf_counter()
            while not env._check_terminated() and steps<10*n+100:
                action=agent.select_action(env) or {"branch":ActionBranch.ADVANCE_TO_NEXT_EVENT}
                _,_,term,trunc,_=env.step(action);steps+=1
                if term or trunc:break
            assert env._check_terminated()
            feasible,violations=check_environment_trajectory_feasibility(env)
            assert feasible and not violations
            completed=sum(t.status==TaskStatus.COMPLETED for t in env.state.tasks.values())
            assert completed==n and len(env.state.transfer_history)==k+4
            ledger=abs(env.cumulative_cost-env.cost_takt-env.cost_time-env.cost_team-env.cost_postpone-env.cost_revision)
            reward=abs(sum(env.step_rewards)+env.cumulative_cost)
            assert ledger<1e-10 and reward<1e-10
            hit=bool(env.disturbance_event_results.get(scenario["scenario_id"],{}).get("actual_hit_task_keys"))
            if disturbed:assert hit,"The pinned diagnostic event did not hit in its reference heuristic trajectory."
            results[name+"_"+variant]={
                "completed":completed,"transfers":len(env.state.transfer_history),"feasible":feasible,
                "violations":violations,"predecessor_closed_against_raw_DAG":True,
                "actual_hit":hit,"target_start":env.state.tasks[scenario["affected_task_keys"][0]].actual_start,
                "ledger_gap":ledger,"reward_cost_gap":reward,"seconds":time.perf_counter()-began}
    path=ROOT/"work3_4060/outputs/fixture_validation.json";path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=="__main__":main()


"""只读公开工艺、资格与同步规则的独立精确下界及见证核验。"""
from __future__ import annotations

import csv
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
from typing import Any

TOL = F(1, 10**9)
ANSWER_KEYS = {'optimal_team', 'expert_team', 'optimal_branch', 'target_makespan', 'optimal_makespan', 'dispatch_priority', 'local_tasks', 'hidden_seed', 'teacher_action'}

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def rational(value: Any) -> F:
    return F(str(value))

def forbidden_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [key for key in value if key in ANSWER_KEYS] + [key for v in value.values() for key in forbidden_keys(v)]
    if isinstance(value, list):
        return [key for v in value for key in forbidden_keys(v)]
    return []

def parse_public(instance: Path) -> dict:
    for path in (instance / 'public').glob('*.json'):
        assert not forbidden_keys(read_json(path)), 'explicit hidden-answer fields in public JSON'
    base = read_json(instance / 'public/baseline.json')
    with (instance / 'public/tasks.csv').open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    with (instance / 'public/workers.csv').open(encoding='utf-8-sig', newline='') as f:
        wrows = list(csv.DictReader(f))
    ao = {r['AO号']: int(r['序号']) - 1 for r in raw}
    assert len(ao) == len(raw)
    tasks = {}
    for r in raw:
        task_id = int(r['序号']) - 1
        fixed = int(r['限定站位']) if r['限定站位'].strip() else None
        tasks[task_id] = dict(standard=rational(r['加工时间/h']), demand=int(r['需求人数']), skill=int(r['工种']), fixed=fixed, predecessors=tuple((ao[x.strip()] for x in r['紧前工序AO号'].split(',') if x.strip())))
        assert tasks[task_id]['standard'] > 0 and tasks[task_id]['demand'] > 0
    assert set(tasks) == set(range(len(raw)))
    workers = {int(r['worker_id']): dict(efficiency=rational(r['efficiency']), skills={i: rational(r[f'skill_{i}']) for i in range(5)}) for r in wrows}
    assert len(workers) == len(wrows) and set(workers) == set(range(len(workers)))
    assert all((w['efficiency'] > 0 for w in workers.values()))
    bindings = {int(s): tuple(map(int, ws)) for s, ws in base['station_workers'].items()}
    assert set(bindings) == set(range(1, 6))
    flattened = [w for pool in bindings.values() for w in pool]
    assert len(flattened) == len(set(flattened)) and set(flattened) == set(workers)
    metadata = base['metadata']
    k = int(metadata['num_aircraft'])
    h0 = rational(metadata['h0'])
    assert int(metadata['num_stations']) == 5 and k >= 1 and (h0 > 0)
    copies = base['tasks']
    expected = {f'{a}_{t}' for a in range(k) for t in tasks}
    assert set(copies) == expected
    for key, r in copies.items():
        tid = int(r['task_id'])
        t = tasks[tid]
        assert key == f"{int(r['aircraft_id'])}_{tid}"
        assert t['demand'] == int(r['demand']) and t['skill'] == int(r['skill'])
        assert set(map(int, r.get('predecessors', []))) == set(t['predecessors'])
        assert t['fixed'] is None or t['fixed'] == int(r['station_id'])
    for tid, t in tasks.items():
        t['initial_station'] = int(copies[f'0_{tid}']['station_id'])
        assert all((int(copies[f'{a}_{tid}']['station_id']) == t['initial_station'] for a in range(k)))
        assert 1 <= t['initial_station'] <= 5
        t['modes_by_station'] = {}
        for station, pool in bindings.items():
            eligible = [w for w in pool if workers[w]['skills'][t['skill']] > 0]
            t['modes_by_station'][station] = {team: t['standard'] * t['demand'] / (sum((workers[w]['efficiency'] for w in team), F(0)) * F(19, 20) ** (len(team) - 1)) for team in combinations(sorted(eligible), t['demand'])}
        assert t['modes_by_station'][t['initial_station']]
    order = []
    remaining = set(tasks)
    while remaining:
        ready = sorted((i for i in remaining if set(tasks[i]['predecessors']) <= set(order)))
        assert ready, 'public precedence is cyclic or names an absent task'
        order.extend(ready)
        remaining.difference_update(ready)
    max_station = {i: tasks[i]['fixed'] or 5 for i in tasks}
    for i in reversed(order):
        for p in tasks[i]['predecessors']:
            max_station[p] = min(max_station[p], max_station[i])
    for i, t in tasks.items():
        t['max_station'] = max_station[i]
        assert t['initial_station'] <= max_station[i]
    cfg_path = instance / 'public/problem_config.json'
    cfg = read_json(cfg_path) if cfg_path.exists() else {}
    assert cfg.get('max_slots_per_station', 2) == 2
    assert cfg.get('objective', 'C_exit_only') in {'C_exit_only', 'C_exit'}
    weights = cfg.get('weights', {})
    assert rational(weights.get('w_makespan', 1)) == 1
    assert all((rational(value) == 0 for key, value in weights.items() if key != 'w_makespan')), 'extra nonzero objective weights'
    policy = metadata['transfer_policy']
    assert policy in {'actual_cleared_v0', 'nominal_anchored_v1'}
    assert cfg.get('transfer_policy', policy) == policy
    return dict(tasks=tasks, workers=workers, bindings=bindings, base=base, aircraft=k, h0=h0, policy=policy, topological_order=order, config=cfg)

def public_lower_bound(problem: dict) -> dict:
    tasks = problem['tasks']
    stage = {}
    for station in range(1, 6):
        ends = {}
        for tid in problem['topological_order']:
            task = tasks[tid]
            if task['fixed'] != station:
                continue
            pmin = min(task['modes_by_station'][station].values())
            ends[tid] = pmin + max((ends[p] for p in task['predecessors'] if p in ends), default=F(0))
        assert ends, 'no mandatory fixed work at an occupied station'
        stage[station] = max(ends.values())
    qtotal = problem['aircraft'] + 4
    pulses = []
    value = F(0)
    for q in range(1, qtotal + 1):
        occupied = [q - a for a in range(problem['aircraft']) if 1 <= q - a <= 5]
        duration = max((stage[s] for s in occupied))
        value += duration
        if problem['policy'] == 'nominal_anchored_v1':
            value = max(value, q * problem['h0'])
        pulses.append(value)
    return dict(station_bounds=stage, pulse_lower_bounds=pulses, lower_bound=value, proof='mandatory fixed-task longest paths using the minimum over all qualified teams; pulse bounds recur before applying absolute nominal boundary')

def check_rows(problem: dict, rows: list[dict], pulses: list[F], *, native: bool) -> dict:
    tolerance = TOL if native else F(0)
    expected = set(problem['base']['tasks'])
    by_key = {r['task_key']: r for r in rows}
    assert len(rows) == len(by_key) == len(expected) and set(by_key) == expected
    worker_intervals = {}
    station_intervals = {}
    assert len(pulses) == problem['aircraft'] + 4 and all((b > a for a, b in zip([F(0)] + pulses, pulses)))
    for r in rows:
        tid = r['tid']
        t = problem['tasks'][tid]
        station = r['station']
        assert r['task_key'] == f"{r['aircraft']}_{tid}"
        assert t['initial_station'] <= station <= t['max_station']
        assert t['fixed'] is None or station == t['fixed']
        modes = t['modes_by_station'][station]
        assert r['team'] in modes and abs(r['end'] - r['start'] - modes[r['team']]) <= tolerance
        qi = r['aircraft'] + station - 1
        entry = F(0) if qi == 0 else pulses[qi - 1]
        assert r['start'] >= entry - tolerance and r['end'] <= pulses[qi] + tolerance
        for pred in t['predecessors']:
            pr = by_key[f"{r['aircraft']}_{pred}"]
            assert pr['station'] <= station and pr['end'] <= r['start'] + tolerance
        for w in r['team']:
            worker_intervals.setdefault(w, []).append((r['start'], r['end'], r['task_key']))
        station_intervals.setdefault(station, []).append((r['start'], r['end'], r['task_key']))
    for intervals in worker_intervals.values():
        ordered = sorted(intervals)
        assert all((a[1] <= b[0] + tolerance for a, b in zip(ordered, ordered[1:]))), 'worker overlap'
    for intervals in station_intervals.values():
        points = sorted({x for a, b, _ in intervals for x in (a, b)})
        for a, b in zip(points, points[1:]):
            if b - a > tolerance:
                mid = (a + b) / 2
                assert sum((start <= mid < end for start, end, _ in intervals)) <= 2, 'station capacity'
    return dict(checked_tasks=len(rows), maximum_end=str(max((r['end'] for r in rows))), pulse_count=len(pulses), native_float_tolerance=str(tolerance))

def public_reference_rows(problem: dict) -> list[dict]:
    return [dict(task_key=key, aircraft=int(r['aircraft_id']), tid=int(r['task_id']), station=int(r['station_id']), team=tuple(sorted(map(int, r['team']))), start=rational(r['baseline_start']), end=rational(r['baseline_end'])) for key, r in problem['base']['tasks'].items()]


def conditional_lower_bound(case: Path, problem: dict) -> dict:
    """迁移原包中两项受限条件证明，不把静态关键链下界当受扰最优值。"""
    scenario_path = case / "public" / "scenario.json"
    if scenario_path.exists():
        lower = public_lower_bound(problem)
        assert problem["policy"] == "nominal_anchored_v1"
        stage = lower["station_bounds"]
        assert len(set(stage.values())) == 1
        period = next(iter(stage.values()))
        assert period == problem["h0"]
        movable = [index for index, task in problem["tasks"].items() if task["fixed"] is None]
        assert len(movable) == 1
        f_id = movable[0]
        f_task = problem["tasks"][f_id]
        successors = [index for index, task in problem["tasks"].items() if f_id in task["predecessors"]]
        assert len(successors) == 1 and not f_task["predecessors"]
        g_task = problem["tasks"][successors[0]]
        source, target = f_task["initial_station"], g_task["fixed"]
        assert target == source + 1 == f_task["max_station"]
        source_modes, target_modes = f_task["modes_by_station"][source], f_task["modes_by_station"][target]
        g_modes = g_task["modes_by_station"][target]
        assert len(source_modes) == len(target_modes) == len(g_modes) == 1
        assert next(iter(target_modes)) == next(iter(g_modes))
        source_duration = min(source_modes.values())
        target_duration = min(target_modes.values()) + min(g_modes.values())
        assert source_duration <= period < target_duration
        scenario = read_json(scenario_path)
        last_key = f"{problem['aircraft'] - 1}_{f_id}"
        assert scenario["valid"] and scenario["affected_task_keys"] == [last_key]
        cycle = problem["aircraft"] + source - 1
        assert 0 < rational(scenario["tau"]) <= (cycle - 1) * period
        release = rational(scenario["recovery_time"])
        base = (problem["aircraft"] + 4) * period
        stay = base + max(F(0), release + source_duration - cycle * period)
        postpone = base + max(F(0), max(cycle * period, release) + target_duration - (cycle + 1) * period)
        return {"lower_bound": min(stay, postpone), "station_bounds": stage,
                "pulse_lower_bounds": [], "conditional_values": {"stay": str(stay), "postpone": str(postpone)},
                "proof_scope": "末架F在已揭示R后留站或后移的两种穷尽分支；其余必经anchor周期长度T"}
    scenarios_path = case / "public" / "scenarios.json"
    assert scenarios_path.exists() and problem["policy"] == "actual_cleared_v0"
    roles = {}
    for station in range(1, 6):
        ids = [index for index, task in problem["tasks"].items() if task["fixed"] == station]
        assert len(ids) == 3
        successors = [index for index in ids if problem["tasks"][index]["predecessors"]]
        assert len(successors) == 1
        g_id = successors[0]
        assert len(problem["tasks"][g_id]["predecessors"]) == 1
        h_id = problem["tasks"][g_id]["predecessors"][0]
        l_id = next(index for index in ids if index not in (h_id, g_id))
        assert not problem["tasks"][h_id]["predecessors"] and not problem["tasks"][l_id]["predecessors"]
        modes = {role: problem["tasks"][index]["modes_by_station"][station]
                 for role, index in (("L", l_id), ("H", h_id), ("G", g_id))}
        assert all(len(value) == 1 for value in modes.values())
        assert next(iter(modes["L"])) == next(iter(modes["H"]))
        assert not set(next(iter(modes["G"]))) & set(next(iter(modes["H"])))
        roles[station] = {"H": h_id, "durations": tuple(min(modes[role].values()) for role in ("L", "H", "G"))}
    shapes = {row["durations"] for row in roles.values()}
    assert len(shapes) == 1
    l_duration, h_duration, g_duration = next(iter(shapes))
    releases = {}
    for event in read_json(scenarios_path):
        assert event["valid"] and rational(event["tau"]) == 0 and len(event["affected_task_keys"]) == 1
        key = event["affected_task_keys"][0]
        assert key not in releases
        releases[key] = rational(event["recovery_time"])
    expected = {f"{aircraft}_{row['H']}" for aircraft in range(problem["aircraft"]) for row in roles.values()}
    assert set(releases) == expected
    release_gap = releases[f"0_{roles[1]['H']}"]
    assert 0 <= release_gap < l_duration
    l_first = max(l_duration, release_gap) + h_duration + g_duration
    h_first = release_gap + h_duration + max(l_duration, g_duration)
    period = min(l_first, h_first)
    for aircraft in range(problem["aircraft"]):
        for station, row in roles.items():
            assert releases[f"{aircraft}_{row['H']}"] == (aircraft + station - 1) * period + release_gap
    return {"lower_bound": (problem["aircraft"] + 4) * period,
            "station_bounds": {station: period for station in range(1, 6)},
            "pulse_lower_bounds": [cycle * period for cycle in range(1, problem["aircraft"] + 5)],
            "conditional_values": {"L_before_H": str(l_first), "H_before_L": str(h_first)},
            "proof_scope": "每站唯一共同工人的L/H两种顺序穷尽下界，按公开R逐周期单调归纳；仅条件对齐标签"}

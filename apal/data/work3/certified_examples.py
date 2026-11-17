"""成熟有证书构造的当前命名空间适配；仅私有教师读取构造答案。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from fractions import Fraction as F
from itertools import combinations
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
from typing import Any

SOURCE_REF = "a710e3427ab87f01d27fca13238172894de6fa09"
COLUMNS = ["序号", "AO号", "类型", "专业编码", "工种", "紧前工序AO号", "需求人数", "加工时间/h", "限定站位", "部位容量"]
PROFESSIONS = ("A", "D", "B", "N", "C")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

GENERATOR_SOURCE_SHA256 = {'templates/structured_generator.py': '5859fe73a531d8c9aac1a7b8108a75b2982619941aee43b4ea66397d9340e331', 'task_order/task_order_examples.py': '8aae2081a255e582b045ecbb11657ee8e21dbea5755122d52d6d5a64e81f89ef', 'templates/five_worker_example.py': '5acbf65c80c8a7fbe8bf301ac18de1c62ca3adee10bc28ca77ff687461be40fd', 'postpone/postpone_examples.py': 'b01c6e0ca49d814826391ed5f888cd809fecb290f738169779c5155586a2b8e1'}

def exact_decimal(x: F) -> str:
    denominator = x.denominator
    for p in (2, 5):
        while denominator % p == 0:
            denominator //= p
    if denominator != 1:
        raise ValueError(f'nonterminating physical input rejected: {x}')
    value = format(float(x), '.12f').rstrip('0').rstrip('.')
    if F(value) != x:
        raise ValueError(f'exact decimal representation lost: {x}')
    return value

def _structured_partition(rng: random.Random, units: int, minimum: int=2) -> list[int]:
    cuts = {t for t in range(1, units) if rng.random() < rng.uniform(0.18, 0.65)}
    while len(cuts) < minimum - 1:
        cuts.add(rng.randrange(1, units))
    boundaries = [0, *sorted(cuts), units]
    return [3 * (b - a) for a, b in zip(boundaries, boundaries[1:])]

def _team_duration(task: dict, team: tuple[int, ...], workers: dict[int, dict]) -> F:
    return task['standard'] * task['demand'] / (sum((workers[w]['efficiency'] for w in team), F(0)) * F(19, 20) ** (len(team) - 1))

def _generate_structured(out: Path, *, seed: int=101, augmentation_seed: int=1, aircraft: int=2, horizon: int=24, transfer_policy: str='actual_cleared_v0', cross_station: bool=True) -> dict:
    if horizon % 3 or horizon < 18 or aircraft not in (2, 3):
        raise ValueError('This verified family uses K=2/3 and integer T>=18 divisible by3')
    if transfer_policy not in {'actual_cleared_v0', 'nominal_anchored_v1'}:
        raise ValueError('unknown transfer semantics')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    rng = random.Random(seed)
    workers: dict[int, dict] = {}
    bindings: dict[int, list[int]] = {}
    tasks = []
    chains = []
    side_shapes = []
    next_worker = 0
    for station in range(1, 6):
        pool_size = rng.randrange(3, 6)
        pool = list(range(next_worker, next_worker + pool_size))
        next_worker += pool_size
        bindings[station] = pool
        fast = pool[0]
        fastest = rng.choice((4, 5, 6))
        for w in pool:
            workers[w] = dict(efficiency=F(fastest if w == fast else rng.randint(1, fastest - 1)), skills=[int(w == fast or rng.random() < 0.4) for _ in range(5)])
        critical_parts = _structured_partition(rng, horizon // 3, minimum=2)
        side_parts = _structured_partition(rng, horizon // 3, minimum=4)
        critical = []
        side = []
        offset = 0
        for index, p in enumerate(critical_parts):
            skill = rng.randrange(5)
            t = dict(uid=len(tasks), station=station, role='critical', segment=index, standard=F(p * fastest), demand=1, skill=skill, team=(fast,), offset=offset, duration=p, predecessors=set())
            if critical:
                t['predecessors'].add(critical[-1]['uid'])
            tasks.append(t)
            critical.append(t)
            offset += p
        chains.append([t['uid'] for t in critical])
        offset = 0
        for index, p in enumerate(side_parts):
            demand = rng.randint(1, min(3, pool_size - 1))
            team = tuple(sorted(rng.sample(pool[1:], demand)))
            skill = rng.randrange(5)
            for w in team:
                workers[w]['skills'][skill] = 1
            standard = F(p) * sum((workers[w]['efficiency'] for w in team), F(0)) * F(19, 20) ** (demand - 1) / demand
            exact_decimal(standard)
            t = dict(uid=len(tasks), station=station, role='parallel', segment=index, standard=standard, demand=demand, skill=skill, team=team, offset=offset, duration=p, predecessors=set())
            tasks.append(t)
            side.append(t)
            offset += p
        shape = rng.choice(('independent', 'chain', 'fork_join', 'cross_linked'))
        if shape == 'chain':
            for left, right in zip(side, side[1:]):
                right['predecessors'].add(left['uid'])
        elif shape == 'fork_join':
            side[1]['predecessors'].add(side[0]['uid'])
            side[2]['predecessors'].add(side[0]['uid'])
            side[-1]['predecessors'].update((side[1]['uid'], side[2]['uid']))
        elif shape == 'cross_linked':
            for right in side:
                candidates = [left for left in critical + side if left['offset'] + left['duration'] <= right['offset']]
                for left in candidates:
                    if rng.random() < 0.35:
                        right['predecessors'].add(left['uid'])
            for right in critical[1:]:
                candidates = [left for left in side if left['offset'] + left['duration'] <= right['offset']]
                if candidates and rng.random() < 0.6:
                    right['predecessors'].add(rng.choice(candidates)['uid'])
        side_shapes.append(shape)
        if cross_station and station > 1:
            earlier = [t for t in tasks if t['station'] == station - 1]
            for right in [critical[0], side[0]]:
                for left in rng.sample(earlier, min(2, len(earlier))):
                    right['predecessors'].add(left['uid'])
    group_spec = {'family': 'protected_chain_parallel_dag_v1', 'horizon': horizon, 'workers': [[w, str(x['efficiency']), x['skills']] for w, x in sorted(workers.items())], 'bindings': bindings, 'tasks': [[t['uid'], t['station'], str(t['standard']), t['demand'], t['skill'], sorted(t['predecessors'])] for t in tasks]}
    group_id = hashlib.sha256(json.dumps(group_spec, sort_keys=True).encode()).hexdigest()
    aug = random.Random(augmentation_seed)
    worker_permutation = list(workers)
    aug.shuffle(worker_permutation)
    worker_map = dict(zip(sorted(workers), worker_permutation))
    task_permutation = list(range(len(tasks)))
    aug.shuffle(task_permutation)
    task_map = dict(enumerate(task_permutation))
    public_workers = {worker_map[w]: x for w, x in workers.items()}
    public_bindings = {s: sorted((worker_map[w] for w in pool)) for s, pool in bindings.items()}
    public_tasks = {}
    for t in tasks:
        i = task_map[t['uid']]
        public_tasks[i] = dict(task_id=i, station=t['station'], standard=t['standard'], demand=t['demand'], skill=t['skill'], predecessors=sorted((task_map[p] for p in t['predecessors'])), ao_code=f"A{PROFESSIONS[t['skill']]}{i:06d}")
    public = out / 'public'
    public.mkdir(parents=True)
    with (public / 'workers.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['worker_id', 'efficiency', *(f'skill_{s}' for s in range(5))])
        writer.writeheader()
        row_ids = list(public_workers)
        aug.shuffle(row_ids)
        for w in row_ids:
            x = public_workers[w]
            writer.writerow({'worker_id': w, 'efficiency': exact_decimal(x['efficiency']), **{f'skill_{s}': x['skills'][s] for s in range(5)}})
    with (public / 'tasks.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i, t in sorted(public_tasks.items()):
            preds = ','.join((public_tasks[p]['ao_code'] for p in t['predecessors']))
            writer.writerow(dict(zip(COLUMNS, [i + 1, t['ao_code'], 2, PROFESSIONS[t['skill']], t['skill'], preds, t['demand'], exact_decimal(t['standard']), t['station'], 2])))
    references = {}
    station_lengths = []
    for station in range(1, 6):
        remaining = {i: t for i, t in public_tasks.items() if t['station'] == station}
        done = {i for i, t in public_tasks.items() if t['station'] < station}
        clock = F(0)
        while remaining:
            ready = sorted((i for i, t in remaining.items() if set(t['predecessors']) <= done))
            if not ready:
                raise AssertionError('public DAG cycle or future-station predecessor')
            i = ready[0]
            t = remaining.pop(i)
            eligible = [w for w in public_bindings[station] if public_workers[w]['skills'][t['skill']] > 0]
            modes = [(_team_duration(t, team, public_workers), team) for team in combinations(eligible, t['demand'])]
            p, team = min(modes)
            references[i] = dict(team=list(team), offset=clock, duration=p)
            clock += p
            done.add(i)
        station_lengths.append(clock)
    h0 = math.ceil(max(station_lengths))
    period = horizon if transfer_policy == 'actual_cleared_v0' else max(horizon, h0)
    cycles = aircraft + 4
    baseline_tasks = {}
    schedule = []
    for k in range(aircraft):
        for i, t in public_tasks.items():
            q = k + t['station']
            entry = (q - 1) * h0
            key = f'{k}_{i}'
            r = references[i]
            baseline_tasks[key] = dict(aircraft_id=k, task_id=i, task_key=key, station_id=t['station'], team=r['team'], duration=float(r['duration']), in_station_offset=float(r['offset']), baseline_start=float(entry + r['offset']), baseline_end=float(entry + r['offset'] + r['duration']), cycle_idx=q, nominal_station_entry=entry, nominal_station_exit=entry + h0, demand=t['demand'], skill=t['skill'], ao_code=t['ao_code'], predecessors=t['predecessors'])
        for t in tasks:
            i = task_map[t['uid']]
            q = k + t['station']
            key = f'{k}_{i}'
            start = (q - 1) * period + t['offset']
            schedule.append(dict(task_key=key, aircraft_id=k, task_id=i, station_id=t['station'], team=sorted((worker_map[w] for w in t['team'])), start=start, end=start + t['duration'], cycle_idx=q))
    write_json(public / 'baseline.json', dict(metadata=dict(num_aircraft=aircraft, num_stations=5, h0=h0, total_cycles=cycles, physical_tasks_per_aircraft=len(tasks), total_physical_tasks=len(schedule), transfer_policy=transfer_policy, version='structured_public_reference_v1'), station_workers=public_bindings, tasks=baseline_tasks))
    write_json(public / 'problem_config.json', dict(schema='apal_certified_example_runtime_v1', source_ref=SOURCE_REF, max_slots_per_station=2, objective='C_exit_only', weights=dict(w_makespan=1, w_h=0, w_t=0, w_w=0, w_p=0, w_revision_time=0, w_revision_team=0), initial_commitments=[], revealed_disturbances=[]))
    schedule.sort(key=lambda t: (t['start'], t['station_id'], t['team'], t['task_id']))
    target = cycles * period
    write_json(out / 'private/witness.json', dict(schema='apal_known_optimum_witness_v1', source_ref=SOURCE_REF, station_id_base=1, transfer_policy=transfer_policy, objective='makespan_only', target_makespan=target, cycle_length=period, schedule=schedule, dispatch_priority=[t['task_key'] for t in schedule]))
    mapped_chains = [[task_map[i] for i in chain] for chain in chains]
    write_json(out / 'private/certificate.json', dict(schema='apal_known_optimum_certificate_v1', family='protected_chain_parallel_dag_v1', objective='C_exit', transfer_policy=transfer_policy, num_aircraft=aircraft, num_stations=5, cycles=cycles, protected_chains=mapped_chains, per_cycle_chain_lower_bound=horizon, station_capacity=2, physical_cycle_sum_lower_bound=cycles * horizon, nominal_final_boundary_lower_bound=cycles * h0 if transfer_policy == 'nominal_anchored_v1' else 0, lower_bound=target, witness_value=target, public_inputs_sha256={str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(public.iterdir())}))
    write_json(out / 'private/augmentation_mapping.json', dict(seed=seed, augmentation_seed=augmentation_seed, template_group=group_id, task_id_mapping=task_map, worker_id_mapping=worker_map, construction_role_by_public_task={task_map[t['uid']]: t['role'] for t in tasks}, note='Private provenance only. Grouping uses parent public problem before ID/K/mode augmentation.'))
    result = dict(family='protected_chain_parallel_dag_v1', seed=seed, augmentation_seed=augmentation_seed, template_group=group_id, aircraft=aircraft, transfer_policy=transfer_policy, horizon=horizon, public_h0=h0, raw_tasks=len(tasks), expanded_tasks=len(schedule), workers=len(workers), workers_per_station=[len(bindings[s]) for s in range(1, 6)], demand_histogram={str(d): sum((t['demand'] == d for t in tasks)) for d in (1, 2, 3)}, side_shapes=side_shapes, precedence_edges=sum((len(t['predecessors']) for t in tasks)), cross_station_edges=sum((sum((tasks[p]['station'] < t['station'] for p in t['predecessors'])) for t in tasks)), certified_C_exit=target, source_ref=SOURCE_REF)
    write_json(out / 'generation_summary.json', result)
    return result


def _generate_task_order(out: Path, *, seed: int, horizon: int=8, aircraft: int=2, augmentation: int=0, station_horizons: tuple[int, ...] | None=None) -> dict:
    if horizon < 4 or aircraft < 1:
        raise ValueError('horizon >= 4 and aircraft >= 1 required')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    if station_horizons is not None and (len(station_horizons) != 5 or min(station_horizons) < 4):
        raise ValueError('Exactly five station horizons >=4 required')
    horizons = list(station_horizons) if station_horizons is not None else [horizon] * 5
    structural = random.Random(seed)
    labels = random.Random((seed + 1) * 10007 + augmentation)
    worker_ids = list(range(10))
    labels.shuffle(worker_ids)
    tasks, workers, station_workers, local_facts = ([], {}, {}, [])
    for station in range(1, 6):
        local_horizon = horizons[station - 1]
        a = structural.randint(1, local_horizon - 2)
        d = local_horizon - a
        b = structural.randint(1, d)
        primary_eff, secondary_eff = (structural.randint(1, 3), structural.randint(1, 3))
        first_skill = structural.randrange(5)
        second_skill = (first_skill + structural.randint(1, 4)) % 5
        primary, secondary = worker_ids[2 * station - 2:2 * station]
        station_workers[str(station)] = [primary, secondary]
        for worker, efficiency, skill in ((primary, primary_eff, first_skill), (secondary, secondary_eff, second_skill)):
            workers[worker] = dict(worker_id=worker, efficiency=efficiency, **{f'skill_{i}': int(i == skill) for i in range(5)})
        for role, duration, worker, skill, offset in (('A', a, primary, first_skill, 0), ('B', b, primary, first_skill, a), ('D', d, secondary, second_skill, a)):
            tasks.append(dict(station_id=station, role=role, duration=duration, standard_duration=duration * workers[worker]['efficiency'], demand=1, skill=skill, team=[worker], offset=offset))
        local_facts.append(dict(station=station, a=a, b=b, d=d, primary_efficiency=primary_eff, secondary_efficiency=secondary_eff, distinct_skills=[first_skill, second_skill]))
    labels.shuffle(tasks)
    for i, task in enumerate(tasks):
        task.update(task_id=i, ao_code=f"A{PROFESSIONS[task['skill']]}{i:06d}")
    by_role = {(t['station_id'], t['role']): t for t in tasks}
    for task in tasks:
        task['predecessors'] = [by_role[task['station_id'], 'A']['task_id']] if task['role'] == 'D' else []
    public = out / 'public'
    public.mkdir(parents=True)
    with (public / 'tasks.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for t in tasks:
            predecessors = ','.join((tasks[p]['ao_code'] for p in t['predecessors']))
            writer.writerow(dict(zip(COLUMNS, [t['task_id'] + 1, t['ao_code'], 2, PROFESSIONS[t['skill']], t['skill'], predecessors, 1, t['standard_duration'], t['station_id'], 2])))
    with (public / 'workers.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['worker_id', 'efficiency', *(f'skill_{i}' for i in range(5))])
        writer.writeheader()
        order = list(workers)
        labels.shuffle(order)
        writer.writerows((workers[w] for w in order))
    refs = {}
    for station in range(1, 6):
        todo = {t['task_id']: t for t in tasks if t['station_id'] == station}
        done, calendar = ({}, {w: [] for w in station_workers[str(station)]})
        while todo:
            i = min((i for i, t in todo.items() if set(t['predecessors']) <= done.keys()))
            t = todo.pop(i)
            eligible = [w for w in station_workers[str(station)] if workers[w][f"skill_{t['skill']}"]]
            assert len(eligible) == 1
            w = eligible[0]
            duration = F(t['standard_duration'], workers[w]['efficiency'])
            start = max((done[p] for p in t['predecessors']), default=F(0))
            for left, right in sorted(calendar[w]):
                if start + duration <= left:
                    break
                if start < right:
                    start = right
            end = start + duration
            calendar[w].append((start, end))
            refs[i] = dict(team=[w], offset=int(start), duration=int(duration))
            done[i] = end
    h0 = max((r['offset'] + r['duration'] for r in refs.values()))
    cycles = aircraft + 4
    cycle_lengths = [max((horizons[q - k] for k in range(aircraft) if 0 <= q - k < 5)) for q in range(cycles)]
    boundaries = [0]
    for length in cycle_lengths:
        boundaries.append(boundaries[-1] + length)
    baseline_tasks, schedule = ({}, [])
    for k in range(aircraft):
        for t in tasks:
            i, station = (t['task_id'], t['station_id'])
            q, key = (k + station, f'{k}_{i}')
            ref = refs[i]
            entry = (q - 1) * h0
            baseline_tasks[key] = dict(aircraft_id=k, task_id=i, task_key=key, station_id=station, team=ref['team'], duration=ref['duration'], in_station_offset=ref['offset'], baseline_start=entry + ref['offset'], baseline_end=entry + ref['offset'] + ref['duration'], cycle_idx=q, nominal_station_entry=entry, nominal_station_exit=entry + h0, demand=1, skill=t['skill'], ao_code=t['ao_code'], predecessors=t['predecessors'])
            start = boundaries[q - 1] + t['offset']
            schedule.append(dict(task_key=key, aircraft_id=k, task_id=i, station_id=station, team=t['team'], start=start, end=start + t['duration'], cycle_idx=q))
    baseline = dict(metadata=dict(num_aircraft=aircraft, num_stations=5, h0=h0, total_cycles=cycles, physical_tasks_per_aircraft=len(tasks), total_physical_tasks=len(schedule), transfer_policy='actual_cleared_v0', version='task_order_public_reference_v1'), station_workers=station_workers, tasks=baseline_tasks)
    write_json(public / 'baseline.json', baseline)
    write_json(public / 'problem_config.json', dict(max_slots_per_station=2, objective='C_exit_only', revealed_disturbances=[], initial_commitments=[]))
    schedule.sort(key=lambda r: (r['start'], r['station_id'], r['task_id']))
    witness = dict(schema='apal_known_optimum_witness_v1', source_ref=SOURCE_REF, station_id_base=1, transfer_policy='actual_cleared_v0', objective='makespan_only', target_makespan=boundaries[-1], cycle_length=horizon if station_horizons is None else None, cycle_lengths=cycle_lengths, pulse_boundaries=boundaries, local_tasks=tasks, local_facts=local_facts, schedule=schedule, dispatch_priority=[r['task_key'] for r in schedule])
    write_json(out / 'private/witness.json', witness)
    write_json(out / 'private/certificate.json', dict(objective='C_exit', transfer_policy='actual_cleared_v0', family='competing_roots_critical_successor', lower_bound=boundaries[-1], witness_value=boundaries[-1], per_station_mandatory_chain_lower_bound=horizons, assumptions=['five mandatory fixed stages', 'disjoint station personnel', 'two independent skills', 'A and B have exactly the same single qualified worker', 'D has another unique qualified worker', 'A precedes D', 'a+d=T', '0<b<=d', 'no initial/future disturbance', 'C_exit only']))
    station_parameters = [{k: v for k, v in r.items() if k != 'station'} for r in local_facts]
    normalized = dict(horizon=horizon, stations=station_parameters) if station_horizons is None else dict(station_horizons=horizons, stations=station_parameters)
    group = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    summary = dict(family='competing_roots_critical_successor', seed=seed, augmentation=augmentation, aircraft=aircraft, template_group=group, raw_tasks=len(tasks), expanded_tasks=len(schedule), horizon=horizon, station_horizons=horizons, cycle_lengths=cycle_lengths, certified_C_exit=boundaries[-1], source_ref=SOURCE_REF, reference_rule='topological_public_id_earliest_resource_insertion', local_facts=local_facts, public_inputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in public.iterdir()})
    write_json(out / 'generation_summary.json', summary)
    return summary


def _generate_five_person(out: Path, *, seed: int=401, aircraft: int=2, horizon: int=8) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    if horizon < 1 or aircraft not in (2, 3):
        raise ValueError('positive horizon, K2/3')
    rng = random.Random(seed)
    ids = list(range(30))
    rng.shuffle(ids)
    pools = {s: ids[6 * (s - 1):6 * s] for s in range(1, 6)}
    local = []
    for station, pool in pools.items():
        local.extend([dict(station=station, role='anchor', standard=F(2 * horizon), demand=1, skill=0, team=[pool[0]]), dict(station=station, role='five', standard=F(horizon) * F(19, 20) ** 4, demand=5, skill=1, team=sorted(pool[1:]))])
    rng.shuffle(local)
    for i, t in enumerate(local):
        t.update(task_id=i, ao_code=f"A{('A' if t['skill'] == 0 else 'D')}{i:06d}")
    public = out / 'public'
    public.mkdir(parents=True)
    with (public / 'tasks.csv').open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        for t in local:
            wr.writerow(dict(zip(COLUMNS, [t['task_id'] + 1, t['ao_code'], 2, 'A' if t['skill'] == 0 else 'D', t['skill'], '', t['demand'], exact_decimal(t['standard']), t['station'], 2])))
    with (public / 'workers.csv').open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['worker_id', 'efficiency', *(f'skill_{i}' for i in range(5))])
        wr.writeheader()
        fast = {pool[0] for pool in pools.values()}
        order = ids.copy()
        rng.shuffle(order)
        for w in order:
            wr.writerow(dict(worker_id=w, efficiency=2 if w in fast else 1, **{f'skill_{i}': int(i == 1 or (i == 0 and w in fast)) for i in range(5)}))
    refs = {}
    for station, pool in pools.items():
        clock = F(0)
        for t in sorted((t for t in local if t['station'] == station), key=lambda t: t['task_id']):
            team = [pool[0]] if t['demand'] == 1 else sorted([pool[0], *sorted(pool[1:])[:4]])
            p = F(horizon) if t['demand'] == 1 else F(5 * horizon, 6)
            refs[t['task_id']] = dict(team=team, offset=clock, duration=p)
            clock += p
    H0 = math.ceil(F(11 * horizon, 6))
    Q = aircraft + 4
    copies = {}
    schedule = []
    for k in range(aircraft):
        for t in local:
            i = t['task_id']
            q = k + t['station']
            entry = (q - 1) * H0
            r = refs[i]
            key = f'{k}_{i}'
            copies[key] = dict(aircraft_id=k, task_id=i, task_key=key, station_id=t['station'], team=r['team'], duration=float(r['duration']), in_station_offset=float(r['offset']), baseline_start=float(entry + r['offset']), baseline_end=float(entry + r['offset'] + r['duration']), cycle_idx=q, nominal_station_entry=entry, nominal_station_exit=entry + H0, demand=t['demand'], skill=t['skill'], ao_code=t['ao_code'], predecessors=[])
            schedule.append(dict(task_key=key, aircraft_id=k, task_id=i, station_id=t['station'], team=t['team'], start=(q - 1) * horizon, end=q * horizon, cycle_idx=q))
    write_json(public / 'baseline.json', dict(metadata=dict(num_aircraft=aircraft, num_stations=5, h0=H0, total_cycles=Q, physical_tasks_per_aircraft=len(local), total_physical_tasks=len(copies), transfer_policy='actual_cleared_v0', version='five_person_public_serial_fast_reference_v1'), station_workers={s: sorted(ws) for s, ws in pools.items()}, tasks=copies))
    write_json(public / 'problem_config.json', dict(objective='C_exit_only', max_slots_per_station=2, initial_commitments=[], revealed_disturbances=[]))
    schedule.sort(key=lambda t: (t['start'], t['station_id'], t['team'], t['task_id']))
    write_json(out / 'private/witness.json', dict(schema='apal_known_optimum_witness_v1', source_ref=SOURCE_REF, station_id_base=1, objective='makespan_only', transfer_policy='actual_cleared_v0', cycle_length=horizon, target_makespan=Q * horizon, schedule=schedule, dispatch_priority=[t['task_key'] for t in schedule]))
    write_json(out / 'private/certificate.json', dict(schema='apal_known_optimum_certificate_v1', family='five_person_protected_anchor_v1', objective='C_exit', transfer_policy='actual_cleared_v0', lower_bound=Q * horizon, witness_value=Q * horizon, proof='Unique qualified anchor worker takes T in every occupied station. Other five workers execute five-person task of duration T concurrently; Q=K+4 pulses. All six legitimate five-person teams are considered in pmin, including five teams containing the faster anchor worker.', public_inputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in public.iterdir()}))
    summary = dict(source_ref=SOURCE_REF, seed=seed, aircraft=aircraft, stations=5, raw_tasks=10, expanded_tasks=10 * aircraft, workers=30, workers_per_station=6, five_person_tasks=5 * aircraft, legal_five_person_teams_per_task=6, certified_C_exit=Q * horizon, public_H0=H0, standard_five_person_work=exact_decimal(F(horizon) * F(19, 20) ** 4))
    write_json(out / 'generation_summary.json', summary)
    return summary


def _json_number(value: F | int) -> int | float:
    f = F(value)
    return int(f) if f.denominator == 1 else float(f)

def _anchor_partition(total: int, rng: random.Random) -> list[int]:
    cuts = [0, *(i for i in range(1, total) if rng.random() < 0.38), total]
    return [b - a for a, b in zip(cuts, cuts[1:])]

def write_supervision_plan(out: Path) -> dict:
    w = json.loads((out / 'private/witness.json').read_text())
    c = json.loads((out / 'private/certificate.json').read_text())
    if 'conditional_optima' in w:
        values = {0: F(w['conditional_optima']['stay']), 1: F(w['conditional_optima']['postpone'])}
        best = min(values.values())
        support = [branch for branch, value in values.items() if value == best]
        labels = [dict(task_key=w['release_target_task_key'], source_station=w['source_station'] - 1, optimal_action_support=support, conditional_C_exit={str(b): str(v) for b, v in values.items()}, strict=len(support) == 1)]
        scope = 'Only last released F has a complete proof-backed pair of branch labels; earlier teacher branch choices can become equivalent after downstream waiting.'
    else:
        correct = 0 if w['optimal_branch'] == 'stay' else 1
        values = {correct: F(w['certified_C_exit']), 1 - correct: F(w['certified_C_exit']) + F(c['wrong_first_branch_excess'])}
        labels = [dict(task_key=f"{k}_{w['roles']['F']}", source_station=w['source_station'] - 1, optimal_action_support=[correct], conditional_C_exit={str(b): str(v) for b, v in sorted(values.items())}, strict=True) for k in range(c['aircraft'])]
        scope = 'Every F source-station choice on the optimal prefix is strictly identified. Bounds consider every legal continuation.'
    plan = dict(schema='apal_branch_supervision_plan_v1', source_ref=SOURCE_REF, objective='C_exit_only', required_state_scope='Source station, uncommitted F, prefix drawn from the certified optimal trajectory', component='stay_postpone_branch', labels=labels, excluded_components=['task_order', 'worker_choice', 'alignment'], teacher_order_is_unique=False, scope=scope, equivalent_action_handling='Use probability of the support set or skip the equivalent decision; do not penalize another proved-optimal branch.')
    write_json(out / 'private/supervision_plan.json', plan)
    return plan

def write_template_group(out: Path) -> dict:
    w = json.loads((out / 'private/witness.json').read_text())
    T = F(w['anchor_duration'])
    anchor_rows = {}
    for t in w['local_tasks']:
        if t['role'].startswith('anchor_'):
            _, station, index = t['role'].split('_')
            anchor_rows.setdefault(station, []).append((int(index), str(F(str(t['standard'])) / T)))
    local = {t['role']: t for t in w['local_tasks']}
    payload = dict(family='mandatory_anchors_with_movable_F_fixed_G', source_station=w['source_station'], anchor_chains={s: [v for _, v in sorted(rows)] for s, rows in sorted(anchor_rows.items())}, F_work_over_T=str(F(str(local['F']['standard'])) / T), G_work_over_T=str(F(str(local['G']['standard'])) / T))
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path = out / 'generation_summary.json'
    summary = json.loads(path.read_text())
    summary.update(template_group_id='anchor_FG_' + signature[:20], family_group='mandatory_anchors_with_movable_F_fixed_G', group_ignores=['task_ids', 'worker_ids', 'CSV_order', 'K', 'release_R', 'reveal_tau', 'opposite_efficiency_condition'], grouping_scope='Restricted constructed family only; not generic DAG isomorphism or a claim of independent industrial samples')
    write_json(path, summary)
    return {k: summary[k] for k in ('template_group_id', 'family_group', 'grouping_scope')}

def _generate_postpone(out: Path, *, seed: int=11, aircraft: int=2, source_station: int=1, anchor_duration: int=5, successor_work: int=1, optimal_branch: str='postpone', split_anchors: bool=True) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    if not 1 <= source_station <= 4 or aircraft < 1:
        raise ValueError('five stations; source must be 1..4')
    if anchor_duration < 3 or not 1 <= successor_work <= anchor_duration - 1:
        raise ValueError('insufficient parallel slack')
    if optimal_branch not in {'stay', 'postpone'}:
        raise ValueError(optimal_branch)
    rng = random.Random(seed)
    T = anchor_duration
    s = source_station
    target = s + 1
    worker_ids = list(range(10))
    rng.shuffle(worker_ids)
    bindings = {station: [worker_ids[2 * (station - 1)], worker_ids[2 * (station - 1) + 1]] for station in range(1, 6)}
    efficiency = {w: 1 for w in worker_ids}
    efficiency[bindings[s][1]] = 2 if optimal_branch == 'stay' else 1
    efficiency[bindings[target][1]] = 1 if optimal_branch == 'stay' else 2
    tasks = []
    for station in range(1, 6):
        sizes = _anchor_partition(T, rng) if split_anchors else [T]
        previous = None
        for seq, size in enumerate(sizes):
            role = f'anchor_{station}_{seq}'
            tasks.append(dict(role=role, base_station=station, fixed_station=station, skill=0, standard=size, predecessor_roles=[] if previous is None else [previous]))
            previous = role
    tasks.extend([dict(role='F', base_station=s, fixed_station=None, skill=1, standard=T + 1, predecessor_roles=[]), dict(role='G', base_station=target, fixed_station=target, skill=1, standard=successor_work, predecessor_roles=['F'])])
    rng.shuffle(tasks)
    for i, t in enumerate(tasks):
        t.update(task_id=i, ao_code=f"A{('A' if t['skill'] == 0 else 'D')}{i:06d}")
    role_to_task = {t['role']: t for t in tasks}
    for t in tasks:
        t['predecessors'] = [role_to_task[r]['task_id'] for r in t['predecessor_roles']]
    public = out / 'public'
    public.mkdir(parents=True)
    with (public / 'tasks.csv').open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=COLUMNS)
        wr.writeheader()
        for t in tasks:
            wr.writerow(dict(zip(COLUMNS, [t['task_id'] + 1, t['ao_code'], 2, 'A' if t['skill'] == 0 else 'D', t['skill'], ','.join((role_to_task[r]['ao_code'] for r in t['predecessor_roles'])), 1, t['standard'], '' if t['fixed_station'] is None else t['fixed_station'], 2])))
    with (public / 'workers.csv').open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['worker_id', 'efficiency', *(f'skill_{i}' for i in range(5))])
        wr.writeheader()
        order = worker_ids.copy()
        rng.shuffle(order)
        by_worker = {w: (station, lane) for station, ws in bindings.items() for lane, w in enumerate(ws)}
        for w in order:
            lane = by_worker[w][1]
            wr.writerow(dict(worker_id=w, efficiency=efficiency[w], **{f'skill_{i}': int(i == lane) for i in range(5)}))
    offset = {}
    reference_duration = {}
    for station in range(1, 6):
        todo = {t['task_id']: t for t in tasks if t['base_station'] == station}
        done = set()
        loads = {0: F(0), 1: F(0)}
        while todo:
            ready = [t for t in todo.values() if all((role_to_task[r]['base_station'] < station or role_to_task[r]['task_id'] in done for r in t['predecessor_roles']))]
            t = min(ready, key=lambda t: t['task_id'])
            worker = bindings[station][t['skill']]
            duration = F(t['standard'], efficiency[worker])
            start = loads[t['skill']]
            for pred in t['predecessor_roles']:
                pt = role_to_task[pred]
                if pt['base_station'] == station:
                    start = max(start, offset[pt['task_id']] + reference_duration[pt['task_id']])
            offset[t['task_id']] = start
            reference_duration[t['task_id']] = duration
            loads[t['skill']] = start + duration
            done.add(t['task_id'])
            del todo[t['task_id']]
    H0 = math.ceil(max((offset[t['task_id']] + reference_duration[t['task_id']] for t in tasks)))
    copies = {}
    for k in range(aircraft):
        for t in tasks:
            i = t['task_id']
            station = t['base_station']
            cycle = k + station
            entry = (cycle - 1) * H0
            copies[f'{k}_{i}'] = dict(aircraft_id=k, task_id=i, task_key=f'{k}_{i}', station_id=station, team=[bindings[station][t['skill']]], duration=_json_number(reference_duration[i]), in_station_offset=_json_number(offset[i]), baseline_start=_json_number(entry + offset[i]), baseline_end=_json_number(entry + offset[i] + reference_duration[i]), cycle_idx=cycle, nominal_station_entry=entry, nominal_station_exit=entry + H0, demand=1, skill=t['skill'], ao_code=t['ao_code'], predecessors=t['predecessors'])
    baseline = dict(metadata=dict(num_aircraft=aircraft, num_stations=5, h0=H0, total_cycles=aircraft + 4, physical_tasks_per_aircraft=len(tasks), total_physical_tasks=len(copies), transfer_policy='actual_cleared_v0', version='diversity_independent_stay_reference_v1'), station_workers={str(k): v for k, v in bindings.items()}, tasks=copies)
    write_json(public / 'baseline.json', baseline)
    write_json(public / 'problem_config.json', dict(objective='C_exit_only', transfer_policy='actual_cleared_v0', max_slots_per_station=2, weights=dict(w_makespan=1, w_h=0, w_t=0, w_w=0, w_p=0, w_revision_time=0, w_revision_team=0)))
    source_time = F(T + 1, efficiency[bindings[s][1]])
    target_chain = F(T + 1 + successor_work, efficiency[bindings[target][1]])
    assert source_time <= T < target_chain if optimal_branch == 'stay' else target_chain <= T < source_time
    certificate = dict(schema='apal_stay_postpone_certificate_v1', source_ref=SOURCE_REF, objective='C_exit', transfer_policy='actual_cleared_v0', stations=5, aircraft=aircraft, cycles=aircraft + 4, anchor_duration=T, source_station=s, target_station=target, source_F_duration=str(source_time), target_FG_duration=str(target_chain), lower_bound=(aircraft + 4) * T, optimal_branch=optimal_branch, F_task_id=role_to_task['F']['task_id'], G_task_id=role_to_task['G']['task_id'], anchor_task_ids={str(station): [t['task_id'] for t in tasks if t['role'].startswith(f'anchor_{station}_')] for station in range(1, 6)}, source_F_worker=bindings[s][1], target_FG_worker=bindings[target][1], wrong_first_aircraft=0, wrong_first_task_key=f"0_{role_to_task['F']['task_id']}", wrong_first_branch_excess=str((target_chain if optimal_branch == 'stay' else source_time) - T), proof='Each occupied station has a mandatory fixed skill-0 chain of total T with one qualified worker. Every one of K+4 cycles is nonempty. F may execute only at s or s+1 because of fixed successor G. Source F or target F->G use the sole skill-1 worker, so wrong branch forces one cycle above T. Correct branch fits every cycle within T.', public_inputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in public.iterdir() if p.is_file()})
    witness = dict(schema='apal_stay_postpone_witness_v1', source_ref=SOURCE_REF, objective='C_exit_only', seed=seed, optimal_branch=optimal_branch, certified_C_exit=(aircraft + 4) * T, anchor_duration=T, source_station=s, target_station=target, roles={t['role']: t['task_id'] for t in tasks}, local_tasks=tasks, worker_roles={str(k): v for k, v in bindings.items()}, dispatch_rule='earliest aircraft then anchors before F before G; exact native witness generated on replay', note='Private roles are for teacher and checker only; they must not enter model features.')
    write_json(out / 'private/certificate.json', certificate)
    write_json(out / 'private/witness.json', witness)
    summary = dict(seed=seed, aircraft=aircraft, source_station=s, anchor_duration=T, successor_work=successor_work, raw_tasks=len(tasks), expanded_tasks=len(copies), optimal_branch=optimal_branch, certified_C_exit=(aircraft + 4) * T, wrong_first_branch_certified_C_exit=str(F((aircraft + 4) * T) + F(certificate['wrong_first_branch_excess'])), public_H0=H0, worker_ids_permuted=True, task_ids_permuted=True)
    write_json(out / 'generation_summary.json', summary)
    write_supervision_plan(out)
    summary.update(write_template_group(out))
    return summary



def _alignment_finish(entry: F, release: F, l: F, h: F, g: F) -> tuple[F, F]:
    l_then_h = max(entry + l, release) + h + g
    h_then_l = max(entry, release) + h + max(l, g)
    return (l_then_h, h_then_l)

def _generate_alignment(out: Path, *, g: int=20, l: int=6, h: int=3, release_gap: int=2, scale: int=1, aircraft: int=2, seed: int=11) -> dict:
    if min(g, l, h, release_gap, scale, aircraft) < 1 or release_gap >= l:
        raise ValueError('positive integer parameters and 0 < release_gap < l required')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    rng = random.Random(seed)
    worker_ids = list(range(10))
    rng.shuffle(worker_ids)
    local = [{'station_id': s, 'role': role, 'skill': int(role == 'G'), 'duration': (g if role == 'G' else l if role == 'L' else h) * scale} for s in range(1, 6) for role in ('L', 'H', 'G')]
    rng.shuffle(local)
    for task_id, task in enumerate(local):
        task.update(task_id=task_id, ao_code=f"A{('D' if task['skill'] else 'A')}{task_id:06d}")
    roles = {(t['station_id'], t['role']): t for t in local}
    a_workers = {s: worker_ids[2 * (s - 1)] for s in range(1, 6)}
    b_workers = {s: worker_ids[2 * (s - 1) + 1] for s in range(1, 6)}
    public = out / 'public'
    public.mkdir(parents=True)
    with (public / 'tasks.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=COLUMNS)
        w.writeheader()
        for t in local:
            pred = roles[t['station_id'], 'H']['ao_code'] if t['role'] == 'G' else ''
            w.writerow(dict(zip(COLUMNS, [t['task_id'] + 1, t['ao_code'], 2, 'D' if t['skill'] else 'A', t['skill'], pred, 1, t['duration'], t['station_id'], 2])))
    with (public / 'workers.csv').open('w', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=['worker_id', 'efficiency', *(f'skill_{s}' for s in range(5))])
        w.writeheader()
        profiles = {a_workers[s]: 0 for s in range(1, 6)} | {b_workers[s]: 1 for s in range(1, 6)}
        for worker_id in range(10):
            w.writerow(dict(worker_id=worker_id, efficiency=1, **{f'skill_{s}': int(s == profiles[worker_id]) for s in range(5)}))
    ll, hh, gg, rr = (l * scale, h * scale, g * scale, release_gap * scale)
    two_orders = _alignment_finish(F(0), F(rr), F(ll), F(hh), F(gg))
    period = int(min(two_orders))
    q_total = aircraft + 4
    h0 = (release_gap + h + max(l, g)) * scale
    baseline_tasks = {}
    schedule = []
    scenarios = []
    optimal_align = int(two_orders[1] <= two_orders[0])
    for k in range(aircraft):
        for t in local:
            s = t['station_id']
            role = t['role']
            q = k + s
            key = f"{k}_{t['task_id']}"
            worker = b_workers[s] if role == 'G' else a_workers[s]
            offset = rr if role == 'H' else rr + hh
            entry = (q - 1) * h0
            baseline_tasks[key] = dict(aircraft_id=k, task_id=t['task_id'], task_key=key, station_id=s, team=[worker], duration=t['duration'], in_station_offset=offset, baseline_start=entry + offset, baseline_end=entry + offset + t['duration'], cycle_idx=q, nominal_station_entry=entry, nominal_station_exit=entry + h0, demand=1, skill=t['skill'], ao_code=t['ao_code'], predecessors=[roles[s, 'H']['task_id']] if role == 'G' else [])
            e = (q - 1) * period
            if optimal_align:
                start = e + (rr if role == 'H' else rr + hh)
            else:
                start = e + {'L': 0, 'H': ll, 'G': ll + hh}[role]
            schedule.append(dict(task_key=key, aircraft_id=k, task_id=t['task_id'], station_id=s, team=[worker], start=start, end=start + t['duration'], cycle_idx=q, align=optimal_align if role == 'L' else 0, role=role))
            if role == 'H':
                scenarios.append(dict(scenario_id=f'REVEALED_{key}', tau=0.0, recovery_time=e + rr, affected_task_keys=[key], valid=True))
    baseline = dict(metadata=dict(num_aircraft=aircraft, num_stations=5, h0=h0, total_cycles=q_total, physical_tasks_per_aircraft=len(local), total_physical_tasks=len(baseline_tasks), transfer_policy='actual_cleared_v0', version='alignment_public_h_first_reference_v1'), station_workers={str(s): [a_workers[s], b_workers[s]] for s in range(1, 6)}, tasks=baseline_tasks)
    write_json(public / 'baseline.json', baseline)
    write_json(public / 'scenarios.json', scenarios)
    write_json(public / 'problem_config.json', dict(objective='C_exit_only', max_slots_per_station=2, transfer_policy='actual_cleared_v0', known_release_events=scenarios, public_reference_rule='H-first with its release handoff buffer; then L on A, G on B follows H', reference_is_not_a_hidden_optimal_label=True))
    write_json(out / 'private/witness.json', dict(source_ref=SOURCE_REF, local_tasks=local, schedule=schedule, optimal_L_align=optimal_align, period=period, target_makespan=period * q_total, policy='all occupied L, then all H, then all G; L align chosen by local two-order bound'))
    certificate = dict(source_ref=SOURCE_REF, family='conditional_alignment_two_resource_chain_v1', objective='C_exit', num_stations=5, num_aircraft=aircraft, cycles=q_total, period=period, lower_bound=period * q_total, witness_value=period * q_total, l_before_h=float(two_orders[0]), h_before_l=float(two_orders[1]), recurrence='F(e,R)=min(max(e+l,R)+h+g, max(e,R)+h+max(l,g)); monotone in e', induction='R_q=(q-1)T+r; E_(q-1)>=(q-1)T gives E_q>=F(E_(q-1),R_q)>=qT', scope='Fixed stations and unique qualified worker per skill; known releases at tau0; pure C_exit; alignment1 is not globally indispensable', public_inputs_sha256={str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(public.iterdir())})
    write_json(out / 'private/certificate.json', certificate)
    summary = dict(seed=seed, g=g, l=l, h=h, release_gap=release_gap, scale=scale, aircraft=aircraft, raw_tasks=15, expanded_tasks=15 * aircraft, period=period, optimal_C_exit=period * q_total, optimal_L_align=optimal_align, conditional_comparison=dict(align0=float(two_orders[0]), align1=float(two_orders[1])), workers_permuted=True, task_ids_permuted=True, source_ref=SOURCE_REF)
    write_json(out / 'generation_summary.json', summary)
    return summary

def _generate_revealed(out: Path, *, seed: int=11, aircraft: int=2, source_station: int=1, anchor_duration: int=5, successor_work: int=1, release: int=8, transfer_policy: str='nominal_anchored_v1', reveal_time: int=0) -> dict:
    T = anchor_duration
    g = successor_work
    if not T > 2 * g + 1:
        raise ValueError('This strict reversal family needs T > 2*g+1')
    if release < 0:
        raise ValueError('release must be nonnegative and disclosed at tau0')
    if reveal_time < 0 or reveal_time > (aircraft + source_station - 2) * T:
        raise ValueError('Reveal must occur no later than last F first becoming selectable; no anticipatory target choice')
    if transfer_policy not in {'actual_cleared_v0', 'nominal_anchored_v1'}:
        raise ValueError(transfer_policy)
    _generate_postpone(out, seed=seed, aircraft=aircraft, source_station=source_station, anchor_duration=T, successor_work=g, optimal_branch='stay', split_anchors=True)
    baseline = json.loads((out / 'public/baseline.json').read_text())
    baseline['metadata'].update(transfer_policy=transfer_policy, version='release_branch_independent_stay_reference_v1')
    assert baseline['metadata']['h0'] == T
    write_json(out / 'public/baseline.json', baseline)
    config = json.loads((out / 'public/problem_config.json').read_text())
    config['transfer_policy'] = transfer_policy
    write_json(out / 'public/problem_config.json', config)
    witness = json.loads((out / 'private/witness.json').read_text())
    certificate = json.loads((out / 'private/certificate.json').read_text())
    last = aircraft - 1
    F_id = witness['roles']['F']
    G_id = witness['roles']['G']
    key = f'{last}_{F_id}'
    scenario_id = f'LAST_F_K{aircraft}_S{source_station}_R{release}' if reveal_time == 0 else f'LAST_F_K{aircraft}_S{source_station}_TAU{reveal_time}_R{release}'
    scenario = dict(scenario_id=scenario_id, tau=float(reveal_time), recovery_time=release, affected_task_keys=[key], valid=True)
    write_json(out / 'public/scenario.json', scenario)
    Q = aircraft + 4
    qsource = aircraft + source_station - 1
    source_time = F(T + 1, 2)
    target_chain = F(T + 1 + g)
    stay = F(Q * T) + max(F(0), F(release) + source_time - qsource * T)
    post = F(Q * T) + max(F(0), max(F(qsource * T), F(release)) + target_chain - (qsource + 1) * T)
    optimum = min(stay, post)
    branch = 'stay' if stay <= post else 'postpone'
    witness.update(schema='apal_released_branch_witness_v1', optimal_branch=branch, certified_C_exit=_json_number(optimum), release_target_aircraft=last, release_target_task_key=key, per_aircraft_optimal_branch={str(k): 'stay' if k != last else branch for k in range(aircraft)}, conditional_optima={'stay': str(stay), 'postpone': str(post)}, target_actions_equivalent=stay == post)
    write_json(out / 'private/witness.json', witness)
    certificate.update(schema='apal_released_branch_certificate_v1', family='last_plane_source_release_branch_reversal', transfer_policy=transfer_policy, lower_bound=str(optimum), witness_value=str(optimum), optimal_branch=branch, target_actions_equivalent=stay == post, release=release, reveal_time=reveal_time, release_target_aircraft=last, release_target_task_key=key, target_successor_key=f'{last}_{G_id}', source_cycle=qsource, source_earliest_entry=(qsource - 1) * T, source_earliest_transfer=qsource * T, source_F_duration=str(source_time), target_FG_duration=str(target_chain), conditional_optima={'stay': str(stay), 'postpone': str(post)}, proof='Let q=K+s-1 and Q=K+4. Every cycle takes >=T. Stay forces Pq>=max(q*T,R+p_source), so C_exit>=Q*T+max(0,R+p_source-q*T). Postpone forces P(q+1)>=max(q*T,R)+p_target_FG, so C_exit>=Q*T+max(0,max(q*T,R)+p_target_FG-(q+1)*T). First K-1 planes stay; correct scheduling attains each conditional bound. F is prohibited beyond s+1 by its fixed successor G. Taking min proves the global optimum.', public_inputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / 'public').iterdir() if p.is_file()})
    write_json(out / 'private/certificate.json', certificate)
    summary = dict(family=certificate['family'], seed=seed, aircraft=aircraft, source_station=source_station, anchor_duration=T, successor_work=g, release=release, raw_tasks=len(witness['local_tasks']), expanded_tasks=len(baseline['tasks']), transfer_policy=transfer_policy, H0=T, reveal_time=reveal_time, source_cycle=qsource, certified_C_exit=str(optimum), optimal_branch=branch, target_actions_equivalent=stay == post, conditional_optima=witness['conditional_optima'])
    write_json(out / 'generation_summary.json', summary)
    write_supervision_plan(out)
    summary.update(write_template_group(out))
    return summary

GENERATOR_SOURCE_SHA256.update({'alignment/alignment_examples.py': '869eb6caff25df4f578d9705e42922270ca9ada2f625508bee889e47711b4577', 'postpone/release_branch_examples.py': '044fb80bb43d53602620788cfdba2e2d6925c98464511420596b8891735a5e81'})

def generate_instance(out: Path, *, family: str = "structured", seed: int = 101,
                      aircraft: int = 2, horizon: int | None = None,
                      augmentation_seed: int = 1, optimal_branch: str = "postpone",
                      release: int | None = None, reveal_time: int = 1) -> dict[str, Any]:
    """迁移已核过的生成规则，输出路径必须是尚未使用的实例目录。"""
    out = Path(out)
    if horizon is not None and horizon <= 0:
        raise ValueError("构造周期必须为正数")
    if family == "structured":
        summary = _generate_structured(out, seed=seed, augmentation_seed=augmentation_seed,
                                       aircraft=aircraft, horizon=horizon or 18)
    elif family == "task_order":
        summary = _generate_task_order(out, seed=seed, aircraft=aircraft, horizon=horizon or 8,
                                       augmentation=augmentation_seed)
    elif family == "five_person":
        summary = _generate_five_person(out, seed=seed, aircraft=aircraft, horizon=horizon or 8)
        summary["template_group"] = "five_person_protected_anchor_v1"
    elif family == "postpone":
        summary = _generate_postpone(out, seed=seed, aircraft=aircraft,
                                    anchor_duration=horizon or 5, optimal_branch=optimal_branch,
                                    source_station=1 + seed % 4)
    elif family == "alignment":
        summary = _generate_alignment(out, seed=seed, aircraft=aircraft, g=1 if seed % 2 else 20)
    elif family == "revealed":
        station, period = 1 + seed % 4, horizon or 5
        summary = _generate_revealed(out, seed=seed, aircraft=aircraft, source_station=station,
                                    anchor_duration=period, reveal_time=reveal_time,
                                    release=release if release is not None else (aircraft + station - 1) * period + 2)
    else:
        raise ValueError(f"未迁移的构造族: {family}")
    certificate_path = out / "private" / "certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["public_inputs_sha256"] = {
        path.relative_to(out).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((out / "public").iterdir()) if path.is_file()
    }
    write_json(certificate_path, certificate)
    transfer_policy = json.loads((out / "public" / "baseline.json").read_text(encoding="utf-8"))["metadata"]["transfer_policy"]
    summary.update(generator_family=family, generator_version="work3_certified_examples_v1",
                   generator_source_sha256=GENERATOR_SOURCE_SHA256,
                   transfer_policy=transfer_policy, objective="C_exit_only")
    write_json(out / "generation_summary.json", summary)
    return summary


def verify_certificate(case: Path) -> dict[str, Any]:
    """先从公开输入重推下界，再读私有声明核对；更新文件哈希不能绕过语义核验。"""
    from apal.data.work3.certificates import parse_public, public_lower_bound, conditional_lower_bound, check_rows, public_reference_rows

    case = Path(case)
    public = parse_public(case)
    conditional = (case / "public" / "scenario.json").exists() or (case / "public" / "scenarios.json").exists()
    if not conditional and public["policy"] != "actual_cleared_v0":
        raise ValueError("本轮静态主实验仅核验实际放行时钟")
    lower = conditional_lower_bound(case, public) if conditional else public_lower_bound(public)
    check_rows(public, public_reference_rows(public),
               [q * public["h0"] for q in range(1, public["aircraft"] + 5)], native=True)
    certificate = json.loads((case / "private" / "certificate.json").read_text(encoding="utf-8"))
    if certificate["objective"] != "C_exit":
        raise ValueError("证书目标不匹配")
    required_inputs = {path.relative_to(case).as_posix() for path in (case / "public").iterdir() if path.is_file()}
    if set(certificate["public_inputs_sha256"]) != required_inputs:
        raise ValueError("证书输入身份清单不完整")
    for relative, expected in certificate["public_inputs_sha256"].items():
        path = case / relative
        if not path.resolve().is_relative_to((case / "public").resolve()):
            raise ValueError("证书只能索引本实例公开输入")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("公开输入字节已改变")
    if F(str(certificate["lower_bound"])) != lower["lower_bound"]:
        raise ValueError("证书下界与公开语义重算不一致")
    return {"certificate_valid": True, "certified_C_exit": float(lower["lower_bound"]),
            "station_bounds": {str(key): str(value) for key, value in lower["station_bounds"].items()},
            "pulse_lower_bounds": [str(value) for value in lower["pulse_lower_bounds"]],
            "proof_scope": lower.get("proof_scope", "五个必经站位的固定任务最短工时前序链；同步实际放行的逐周期下界"),
            "conditional_values": lower.get("conditional_values", {})}


def make_instance_env(case: Path, *, weights: Any | None = None) -> Any:
    """只向环境传公开文件；私有证书、最优值和教师不属于环境输入。"""
    from apal.envs.work3.environment import AirLineEnvWork3
    from apal.utils.work3.objective_evaluator import ObjectiveWeights

    case = Path(case)
    env = AirLineEnvWork3(baseline_json_path=case / "public" / "baseline.json",
                          raw_data_path=case / "public" / "tasks.csv",
                          worker_pool_path=case / "public" / "workers.csv", max_slots_per_station=2,
                          weights=weights if weights is not None else ObjectiveWeights(
                              w_makespan=1, w_h=0, w_t=0, w_w=0, w_p=0,
                              w_revision_time=0, w_revision_team=0))
    env.reset()
    scenario_path, scenarios_path = case / "public" / "scenario.json", case / "public" / "scenarios.json"
    if scenario_path.exists():
        env.load_scenario(json.loads(scenario_path.read_text(encoding="utf-8")))
    if scenarios_path.exists():
        for scenario in json.loads(scenarios_path.read_text(encoding="utf-8")):
            env.load_scenario(scenario)
    return env


def replay_witness(case: Path, *, first_action_override: dict[str, Any] | None = None) -> dict[str, Any]:
    """用当前原生引擎重放私有见证并保存实际候选、动作和监督范围。"""
    from apal.data.work3.certificates import parse_public, check_rows
    from apal.utils.work3.objective_evaluator import evaluate_trajectory_objective
    from apal.utils.work3.trajectory_feasibility import TaskConstraintRecord, TrajectoryExecutionRecord, validate_trajectory

    case = Path(case)
    verified = verify_certificate(case)
    witness = json.loads((case / "private" / "witness.json").read_text(encoding="utf-8"))
    summary = json.loads((case / "generation_summary.json").read_text(encoding="utf-8"))
    env = make_instance_env(case)
    family = summary["generator_family"]
    rows = {row["task_key"]: row for row in witness.get("schedule", [])}
    rank = {key: index for index, key in enumerate(witness.get("dispatch_priority", rows))}
    roles = {value: key for key, value in witness.get("roles", {}).items()}
    actions: list[dict[str, Any]] = []
    for index in range(len(env.state.tasks) * 4 + 100):
        candidates = env.get_action_candidates_with_masks()
        before = {"time": env.state.current_time, "cycle": env.state.current_cycle,
                  "candidates": [{"task_key": task.task_key, "mask": list(mask)} for task, mask in candidates]}
        scope: list[str] = []
        if candidates:
            if family in ("postpone", "revealed"):
                task, mask = min(candidates, key=lambda pair: (pair[0].aircraft_id,
                    0 if roles[pair[0].task_id].startswith("anchor_") else 1 if roles[pair[0].task_id] == "F" else 2,
                    pair[0].task_id))
                is_f_source = roles[task.task_id] == "F" and task.current_station == witness["source_station"] - 1
                branch_name = witness.get("per_aircraft_optimal_branch", {}).get(str(task.aircraft_id), witness["optimal_branch"])
                postpone = is_f_source and branch_name == "postpone"
                if is_f_source and (family == "postpone" or task.task_key == witness.get("release_target_task_key")):
                    assert tuple(mask) == (True, True)
                    scope = ["branch"]
                if postpone:
                    action = {"task_key": task.task_key, "branch": 1}
                else:
                    team = env.valid_team_completion_workers(task, [])
                    assert len(team) == 1 and task.demand == 1
                    action = {"task_key": task.task_key, "branch": 0, "team": team, "align": 0}
            else:
                if family == "alignment":
                    role_by_task = {row["task_id"]: row["role"] for row in witness["local_tasks"]}
                    task, mask = min(candidates, key=lambda pair: (
                        ("L", "H", "G").index(role_by_task[pair[0].task_id]), pair[0].current_station, pair[0].aircraft_id))
                else:
                    task, mask = min(candidates, key=lambda pair: rank[pair[0].task_key])
                assert mask[0]
                row = rows[task.task_key]
                action = {"task_key": task.task_key, "branch": 0, "team": row["team"], "align": row.get("align", 0)}
                scope = (["align"] if row.get("role") == "L" else []) if family == "alignment" else ["advance", "task", "branch", "team", "align"]
        else:
            action = {"branch": 2}
        if index == 0 and first_action_override is not None:
            action = dict(first_action_override)
        _, reward, terminated, truncated, info = env.step(action)
        actions.append({"action": action, "label_scope": scope, "before": before, "reward": reward,
                        "after_time": env.state.current_time, "terminated": terminated,
                        "truncated": truncated})
        if terminated or truncated:
            if not terminated or not info.get("success"):
                raise ValueError(f"见证未自然完成: {info}")
            break
    else:
        raise ValueError("见证重放超过有界步数")
    records = [TrajectoryExecutionRecord(aircraft_id=task.aircraft_id, task_id=task.task_id,
        station_id=task.actual_start_station, team=tuple(task.assigned_team), start=task.actual_start,
        end=task.actual_end, material_ready_time=task.material_ready_time,
        station_entry_time=env.state.aircraft[task.aircraft_id].entry_times[task.actual_start_station],
        aircraft_station_at_start=task.actual_start_station,
        station_exit_time=env.state.aircraft[task.aircraft_id].exit_times[task.actual_start_station])
        for task in env.state.tasks.values()]
    constraints = {task.task_id: TaskConstraintRecord(demand=task.demand, required_skill=task.skill,
        predecessors=task.predecessors, fixed_station=task.fixed_station,
        max_allowed_station=task.max_allowed_station, standard_duration=task.standard_duration)
        for task in env.state.tasks.values()}
    physical = validate_trajectory(records, task_constraints=constraints, worker_skills=env.worker_skills,
        worker_station_bindings={worker: station for station, pool in env.state.station_worker_bindings.items() for worker in pool},
        station_capacities={station: 2 for station in range(5)}, worker_efficiencies=env.worker_efficiencies)
    if not physical.is_feasible:
        raise ValueError(f"原生见证违反物理规则: {asdict(physical)}")
    actual_rows = [{"task_key": task.task_key, "aircraft": task.aircraft_id, "tid": task.task_id,
                   "station": task.actual_start_station + 1, "team": tuple(sorted(task.assigned_team)),
                   "start": F(str(task.actual_start)), "end": F(str(task.actual_end))}
                  for task in env.state.tasks.values()]
    check_rows(parse_public(case), actual_rows, [F(str(value)) for value in env.state.transfer_history], native=True)
    objective = evaluate_trajectory_objective(env, env.weights)
    cost_ok = math.isclose(objective.j_total, env.cumulative_cost, abs_tol=1e-9, rel_tol=0)
    cost_ok = cost_ok and math.isclose(-sum(env.step_rewards), env.cumulative_cost, abs_tol=1e-9, rel_tol=0)
    attained = math.isclose(env.state.current_time, verified["certified_C_exit"], abs_tol=1e-9, rel_tol=0)
    if not attained or not cost_ok:
        raise ValueError("原生见证未达到独立下界或费用账本不一致")
    result = {**verified, "native_verified": True, "independent_feasible": True,
              "cost_ledger_matches": cost_ok, "C_exit": env.state.current_time,
              "C_task": max(task.actual_end for task in env.state.tasks.values()),
              "environment_steps": env.step_count, "completed_tasks": len(env.state.tasks),
              "cumulative_cost": env.cumulative_cost, "rewards_sum": sum(env.step_rewards),
              "transfer_history": list(env.state.transfer_history), "transfer_policy": env.transfer_policy,
              "actual_disturbance_hits": sum(len(row["actual_hit_task_keys"]) for row in env.disturbance_event_results.values()),
              "revealed_release_increase": sum(task.material_ready_time for task in env.state.tasks.values()),
              "environment_source_sha256": hashlib.sha256(Path(inspect.getfile(type(env))).read_bytes()).hexdigest()}
    write_json(case / "private" / "native_witness_actions.json", actions)
    write_json(case / "private" / "native_verification.json", result)
    return result


def verify_instance(case: Path) -> dict[str, Any]:
    return replay_witness(case)


def load_witness_actions(case: Path) -> list[dict[str, Any]]:
    path = Path(case) / "private" / "native_witness_actions.json"
    if not path.is_file():
        replay_witness(Path(case))
    return json.loads(path.read_text(encoding="utf-8"))


def generate_dataset(output: Path, *, train_count: int = 32, validation_count: int = 8,
                     test_count: int = 12, seed: int = 20261117) -> dict[str, Path]:
    """按母实例分组冻结划分；五人同构族只进入开发训练，不冒充留出新母实例。"""
    output = Path(output)
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("训练、验证与测试至少各需一个母实例")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    all_groups: set[str] = set()
    manifests: dict[str, Path] = {}
    registry = []
    for split_index, (split, count) in enumerate((("train", train_count), ("val", validation_count), ("test", test_count))):
        entries = []
        for index in range(count):
            family = ("structured", "structured", "task_order", "postpone")[index % 4]
            if split == "train" and index == 1:
                family = "five_person"
            case_seed = seed + split_index * 100000 + index * 97
            case = output / "instances" / f"{split}_{index:03d}"
            summary = generate_instance(case, family=family, seed=case_seed,
                                        optimal_branch="stay" if index % 8 == 7 else "postpone")
            group = str(summary.get("template_group", summary.get("template_group_id")))
            if not group or group == "None" or group in all_groups:
                raise ValueError(f"来源分组重复或缺失，停止数据生成: {group}")
            all_groups.add(group)
            public_paths = {"baseline": case / "public" / "baseline.json", "raw_data": case / "public" / "tasks.csv",
                            "worker_pool": case / "public" / "workers.csv"}
            entry = {"instance_id": case.name, "parent_group": group, "family": family, "split": split,
                     **{key: path.relative_to(output).as_posix() for key, path in public_paths.items()},
                     "public_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in public_paths.items()},
                     "scenario": None, "env_config": {"max_slots_per_station": 2, "transfer_policy": "actual_cleared_v0"},
                     "generator_version": "work3_certified_examples_v1", "generation_seed": case_seed,
                     "source_ref": SOURCE_REF}
            verification = verify_instance(case)
            registry.append({"instance_id": case.name, "split": split, "family": family,
                             "parent_group": group, "generation_seed": case_seed,
                             "certificate_valid": True, "native_verified": True,
                             "certified_C_exit": verification["certified_C_exit"],
                             "native_verification": (case / "private" / "native_verification.json").relative_to(output).as_posix(),
                             "witness_actions": (case / "private" / "native_witness_actions.json").relative_to(output).as_posix()})
            entries.append(entry)
        path = output / f"{split}_instances.json"
        write_json(path, {"version": "work3_episode_instances_v1", "instances": entries,
                          "episode_sequence": [entry["instance_id"] for entry in entries]})
        manifests[split] = path
    write_json(output / "private" / "dataset_registry.json", {
        "version": "work3_certified_dataset_v1", "source_ref": SOURCE_REF,
        "generator_source_sha256": GENERATOR_SOURCE_SHA256, "instances": registry,
        "grouping_scope": "受限构造族的参数母实例；编号和规模变体不作为新来源；不声称工业或未见拓扑泛化"})
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="生成并独立核验工作三有证书示例")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    result = generate_dataset(args.output, train_count=args.train_count, validation_count=args.val_count,
                              test_count=args.test_count, seed=args.seed)
    print(json.dumps({key: str(path) for key, path in result.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

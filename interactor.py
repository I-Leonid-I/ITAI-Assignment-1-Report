from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SIZE = 13
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
SENTINEL = object()

ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"
DEFAULT_MAPS_PER_RUN = 1000

COMMAND_TIMEOUT_SEC = 5.0
PROCESS_TIMEOUT_SEC = 15.0
DEFAULT_SEED: Optional[int] = None


def inside(x: int, y: int) -> bool:
    return 0 <= x < SIZE and 0 <= y < SIZE


def moore(radius: int) -> List[Tuple[int, int]]:
    return [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if max(abs(dx), abs(dy)) <= radius
    ]


def von_neumann(radius: int) -> List[Tuple[int, int]]:
    return [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if abs(dx) + abs(dy) <= radius
    ]


def translate(pos: Tuple[int, int], offsets: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    x, y = pos
    return [(x + dx, y + dy) for dx, dy in offsets if inside(x + dx, y + dy)]


@dataclass(frozen=True)
class Enemy:
    kind: str
    x: int
    y: int

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)


def nazgul_zone(pos: Tuple[int, int], ring_on: bool, has_coat: bool) -> set[Tuple[int, int]]:
    if ring_on:
        zone = set(translate(pos, moore(2)))
        ears = [(3, 0), (-3, 0), (0, 3), (0, -3)]
        zone.update([p for p in translate(pos, ears)])
        return zone
    if has_coat:
        return set(translate(pos, moore(1)))
    zone = set(translate(pos, moore(1)))
    ears = [(2, 0), (-2, 0), (0, 2), (0, -2)]
    zone.update([p for p in translate(pos, ears)])
    return zone


def watchtower_zone(pos: Tuple[int, int], ring_on: bool) -> set[Tuple[int, int]]:
    zone = set(translate(pos, moore(2)))
    if ring_on:
        ears = [(3, 0), (-3, 0), (0, 3), (0, -3)]
        zone.update([p for p in translate(pos, ears)])
    return zone


def enemy_zone(enemy: Enemy, ring_on: bool, has_coat: bool) -> set[Tuple[int, int]]:
    pos = enemy.pos
    if enemy.kind == "O":
        if ring_on or has_coat:
            return {pos}
        return set(translate(pos, von_neumann(1)))
    if enemy.kind == "U":
        r = 1 if (ring_on or has_coat) else 2
        return set(translate(pos, von_neumann(r)))
    if enemy.kind == "N":
        return nazgul_zone(pos, ring_on, has_coat)
    if enemy.kind == "W":
        return watchtower_zone(pos, ring_on)
    raise ValueError(f"Unknown enemy type {enemy.kind}")


@dataclass(frozen=True)
class MapDefinition:
    g_pos: Tuple[int, int]
    m_pos: Tuple[int, int]
    c_pos: Tuple[int, int]
    enemies: Tuple[Enemy, ...]
    variant: int = 1

    def enemy_positions(self) -> set[Tuple[int, int]]:
        return {enemy.pos for enemy in self.enemies}


def compute_hazard_cache(map_def: MapDefinition) -> Dict[Tuple[bool, bool], set[Tuple[int, int]]]:
    cache: Dict[Tuple[bool, bool], set[Tuple[int, int]]] = {}
    for ring_on in (False, True):
        for has_coat in (False, True):
            zone: set[Tuple[int, int]] = set()
            for enemy in map_def.enemies:
                zone |= enemy_zone(enemy, ring_on, has_coat)
            cache[(ring_on, has_coat)] = zone
    return cache


def ensure_binaries_built() -> Dict[str, Optional[str]]:
    sources = {
        "astar": ROOT / "a_star.cpp",
        "backtracking": ROOT / "backtracking.cpp",
    }
    outputs = {
        "astar": ROOT / "a_star",
        "backtracking": ROOT / "backtracking",
    }
    to_build: list[tuple[str, Path, Path]] = []
    errors: Dict[str, Optional[str]] = {"astar": None, "backtracking": None}
    for key, src in sources.items():
        out = outputs[key]
        if not src.exists():
            errors[key] = f"Missing source file: {src}"
            continue
        if (not out.exists()) or (out.stat().st_mtime < src.stat().st_mtime):
            to_build.append((key, src, out))
    if not to_build:
        return errors
    cxx = os.environ.get("CXX", "g++")
    common_flags = ["-std=c++17", "-O2", "-pipe"]
    for key, src, out in to_build:
        cmd = [cxx, *common_flags, str(src), "-o", str(out)]
        try:
            res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        except Exception as e:
            errors[key] = f"Failed to invoke compiler for {key}: {e}"
            continue
        if res.returncode != 0:
            msg = (
                f"Compilation failed for {src.name} (exit {res.returncode}).\n\n"
                f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}\n"
            )
            errors[key] = msg
        else:
            errors[key] = None
    return errors


def stage_transition(stage: int, position: Tuple[int, int], g_pos: Tuple[int, int]) -> int:
    if stage == 0 and position == g_pos:
        return 1
    return stage


def compute_shortest_paths(map_def: MapDefinition) -> Dict[str, Optional[int]]:
    hazards = compute_hazard_cache(map_def)
    enemy_cells = map_def.enemy_positions()
    c_pos = map_def.c_pos
    g_pos = map_def.g_pos
    m_pos = map_def.m_pos
    start_state = (0, 0, 0, 0, 0)
    dist: Dict[Tuple[int, int, int, int, int], int] = {start_state: 0}
    dq: deque[Tuple[int, int, int, int, int]] = deque([start_state])
    while dq:
        x, y, ring, coat, stage = dq.popleft()
        moves = dist[(x, y, ring, coat, stage)]
        current_hazard = hazards[(bool(ring), bool(coat))]
        new_ring = 1 - ring
        new_hazard = hazards[(bool(new_ring), bool(coat))]
        if (x, y) not in new_hazard and (x, y) not in enemy_cells:
            nxt = (x, y, new_ring, coat, stage)
            if nxt not in dist or moves < dist[nxt]:
                dist[nxt] = moves
                dq.appendleft(nxt)
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not inside(nx, ny):
                continue
            if (nx, ny) in enemy_cells:
                continue
            if (nx, ny) in current_hazard:
                continue
            next_coat = 1 if (coat or (nx, ny) == c_pos) else 0
            next_stage = stage_transition(stage, (nx, ny), g_pos)
            nxt = (nx, ny, ring, next_coat, next_stage)
            if nxt not in dist or moves + 1 < dist[nxt]:
                dist[nxt] = moves + 1
                dq.append(nxt)
    g_candidates = [
        dist[state]
        for state in dist
        if state[0] == g_pos[0] and state[1] == g_pos[1] and state[4] == 1
    ]
    m_candidates = [
        dist[state]
        for state in dist
        if state[0] == m_pos[0] and state[1] == m_pos[1] and state[4] == 1
    ]
    return {
        "dist_to_g": min(g_candidates) if g_candidates else None,
        "dist_to_m": min(m_candidates) if m_candidates else None,
    }


def random_cell(rng: random.Random, excluded: set[Tuple[int, int]]) -> Tuple[int, int]:
    candidates = [(x, y) for x in range(SIZE) for y in range(SIZE) if (x, y) not in excluded]
    if not candidates:
        raise RuntimeError("No free cells available for placement")
    return rng.choice(candidates)


def generate_map(rng: random.Random, variant: int | None = None) -> MapDefinition:
    while True:
        occupied = {(0, 0)}
        g_pos = random_cell(rng, occupied)
        occupied.add(g_pos)
        m_pos = random_cell(rng, occupied)
        occupied.add(m_pos)
        c_pos = random_cell(rng, occupied)
        occupied.add(c_pos)
        enemies: List[Enemy] = []
        w_pos = random_cell(rng, occupied)
        occupied.add(w_pos)
        enemies.append(Enemy("W", *w_pos))
        u_pos = random_cell(rng, occupied)
        occupied.add(u_pos)
        enemies.append(Enemy("U", *u_pos))
        if rng.random() < 0.6:
            n_pos = random_cell(rng, occupied)
            occupied.add(n_pos)
            enemies.append(Enemy("N", *n_pos))
        orc_count = rng.choice((1, 2))
        for _ in range(orc_count):
            o_pos = random_cell(rng, occupied)
            occupied.add(o_pos)
            enemies.append(Enemy("O", *o_pos))
        map_def = MapDefinition(
            g_pos=g_pos,
            m_pos=m_pos,
            c_pos=c_pos,
            enemies=tuple(enemies),
            variant=(variant if variant in (1, 2) else rng.choice((1, 2))),
        )
        hazards = compute_hazard_cache(map_def)
        base_hazard = hazards[(False, False)]
        if (0, 0) not in base_hazard and g_pos not in base_hazard and c_pos not in base_hazard and m_pos not in base_hazard:
            return map_def


def get_visible_entries(
    map_def: MapDefinition,
    hazards: Dict[Tuple[bool, bool], set[Tuple[int, int]]],
    position: Tuple[int, int],
    ring_on: bool,
    has_coat: bool,
    g_active: Optional[Tuple[int, int]],
    c_active: Optional[Tuple[int, int]],
    m_active: Optional[Tuple[int, int]],
) -> List[Tuple[int, int, str]]:
    radius = 1 if map_def.variant == 1 else 2
    hazard_set = hazards[(ring_on, has_coat)]
    entries: List[Tuple[int, int, str]] = []
    px, py = position
    items: Dict[Tuple[int, int], str] = {}
    if g_active is not None:
        items[g_active] = "G"
    if c_active is not None:
        items[c_active] = "C"
    if m_active is not None:
        items[m_active] = "M"
    enemy_at = {enemy.pos: enemy.kind for enemy in map_def.enemies}
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if max(abs(dx), abs(dy)) > radius:
                continue
            nx, ny = px + dx, py + dy
            if not inside(nx, ny) or (nx, ny) == (px, py):
                continue
            if (nx, ny) in enemy_at:
                entries.append((nx, ny, enemy_at[(nx, ny)]))
            elif (nx, ny) in items:
                entries.append((nx, ny, items[(nx, ny)]))
            elif (nx, ny) in hazard_set:
                pass
    entries.sort()
    return entries


def spawn_algorithm(algo: str) -> subprocess.Popen:
    if algo == "astar":
        exe = ROOT / "a_star"
    elif algo == "backtracking":
        exe = ROOT / "backtracking"
    else:
        raise ValueError(f"Unknown algorithm {algo}")
    if not exe.exists():
        ensure_binaries_built()
    if not exe.exists():
        raise FileNotFoundError(f"Executable not found: {exe}")
    command = [str(exe)]
    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def reader_thread(stream, queue: Queue):
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            queue.put(line.rstrip("\n"))
    finally:
        queue.put(SENTINEL)


def send_line(proc: subprocess.Popen, data: str) -> None:
    assert proc.stdin is not None
    try:
        proc.stdin.write(data + "\n")
        proc.stdin.flush()
    except BrokenPipeError:
        pass


def receive_line(queue: Queue, timeout: float) -> Tuple[Optional[str], bool]:
    try:
        item = queue.get(timeout=timeout)
    except Empty:
        return None, False
    if item is SENTINEL:
        return None, True
    return item, False


def cleanup_process(proc: subprocess.Popen, stdout_thread: threading.Thread, stderr_thread: threading.Thread) -> None:
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=0.5)
    except Exception:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    try:
        stdout_thread.join(timeout=0.5)
    except Exception:
        pass
    try:
        stderr_thread.join(timeout=0.5)
    except Exception:
        pass
    for stream in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
        try:
            if stream and not stream.closed:
                stream.close()
        except Exception:
            pass


@dataclass
class RunResult:
    success: bool
    reason: str
    moves: int
    toggles: int
    reported_length: Optional[int]
    runtime_sec: float
    claimed_unsolvable: bool
    was_solvable: bool
    stderr: str = ""
    log: List[str] = field(default_factory=list)


def run_single_simulation(map_def: MapDefinition, algo: str, hazards: Optional[Dict[Tuple[bool, bool], set[Tuple[int, int]]]] = None) -> RunResult:
    hazards = hazards or compute_hazard_cache(map_def)
    try:
        proc = spawn_algorithm(algo)
    except Exception as e:
        return RunResult(False, "spawn_error", 0, 0, None, 0.0, False, False, str(e), [])

    stdout_queue: Queue = Queue()
    stderr_queue: Queue = Queue()
    stdout_thread = threading.Thread(target=reader_thread, args=(proc.stdout, stdout_queue), daemon=True)
    stderr_thread = threading.Thread(target=reader_thread, args=(proc.stderr, stderr_queue), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    start_time = time.perf_counter()

    ring_on = False
    has_coat = False
    position = (0, 0)
    g_reached = False
    moves = 0
    toggles = 0
    log: List[str] = []

    send_line(proc, str(map_def.variant))
    send_line(proc, f"{map_def.g_pos[0]} {map_def.g_pos[1]}")
    initial_entries = get_visible_entries(map_def, hazards, position, ring_on, has_coat, map_def.g_pos, map_def.c_pos, None)
    send_line(proc, str(len(initial_entries)))
    for x, y, kind in initial_entries:
        send_line(proc, f"{x} {y} {kind}")

    def consume_stderr() -> str:
        collected: List[str] = []
        while True:
            try:
                line = stderr_queue.get_nowait()
            except Empty:
                break
            if line is SENTINEL:
                break
            collected.append(line)
        return "\n".join(collected)

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed > PROCESS_TIMEOUT_SEC:
                return RunResult(False, "timeout", moves, toggles, None, elapsed, False, False, consume_stderr(), log)

            line, eof = receive_line(stdout_queue, COMMAND_TIMEOUT_SEC)
            if eof:
                return RunResult(False, "no_output", moves, toggles, None, time.perf_counter() - start_time, False, False, consume_stderr(), log)
            if line is None:
                return RunResult(False, "no_output", moves, toggles, None, time.perf_counter() - start_time, False, False, consume_stderr(), log)

            s = line.strip()
            if not s:
                continue
            log.append(s)

            if s.startswith("e "):
                parts = s.split()
                reported = None
                if len(parts) >= 2:
                    try:
                        reported = int(parts[1])
                    except ValueError:
                        reported = None
                success = (reported is not None) and (reported != -1)
                return RunResult(success, "ok" if success else "unsolvable", moves, toggles, reported, time.perf_counter() - start_time, reported == -1, False, consume_stderr(), log)

            if s == "r":
                ring_on = True
                toggles += 1
            elif s == "rr":
                ring_on = False
                toggles += 1
            elif s.startswith("m "):
                try:
                    _, xs, ys = s.split()
                    position = (int(xs), int(ys))
                except Exception:
                    pass
                moves += 1
                if (not has_coat) and position == map_def.c_pos:
                    has_coat = True
                if position == map_def.g_pos and not g_reached:
                    g_reached = True

            entries = get_visible_entries(map_def, hazards, position, ring_on, has_coat, None if g_reached else map_def.g_pos, map_def.c_pos, None)
            send_line(proc, str(len(entries)))
            for x, y, kind in entries:
                send_line(proc, f"{x} {y} {kind}")

            if g_reached:
                send_line(proc, f"{map_def.m_pos[0]} {map_def.m_pos[1]}")
                g_reached = False

    finally:
        cleanup_process(proc, stdout_thread, stderr_thread)


def aggregate_statistics(results: Sequence[RunResult]) -> Dict[str, object]:
    runtimes_all = [res.runtime_sec for res in results]
    success_count = sum(1 for res in results if res.success)
    failure_count = len(results) - success_count
    def safe_mode(values: Sequence[float | int]) -> Optional[float]:
        if not values:
            return None
        try:
            return statistics.mode(values)
        except statistics.StatisticsError:
            bins: Dict[int, int] = {}
            for v in values:
                b = int(round(float(v) * 1000.0))
                bins[b] = bins.get(b, 0) + 1
            if not bins:
                return None
            max_count = max(bins.values())
            candidates = [b for b, c in bins.items() if c == max_count]
            return (statistics.median(candidates) / 1000.0) if candidates else None
    return {
        "total_runs": len(results),
        "successes": success_count,
        "failures": failure_count,
        "runtime_mean_all": statistics.fmean(runtimes_all) if runtimes_all else None,
        "runtime_median_all": statistics.median(runtimes_all) if runtimes_all else None,
        "runtime_mode_all": safe_mode(runtimes_all),
        "runtime_std_all": statistics.pstdev(runtimes_all) if len(runtimes_all) > 1 else 0.0,
    }


def save_tests(tests: Sequence[MapDefinition], out_path: Path) -> None:
    payload = {
        "tests": [
            {
                "variant": t.variant,
                "g_pos": [t.g_pos[0], t.g_pos[1]],
                "m_pos": [t.m_pos[0], t.m_pos[1]],
                "c_pos": [t.c_pos[0], t.c_pos[1]],
                "enemies": [{"kind": e.kind, "x": e.x, "y": e.y} for e in t.enemies],
            }
            for t in tests
        ]
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_tests(in_path: Path) -> List[MapDefinition]:
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    tests_json = payload.get("tests", [])
    tests: List[MapDefinition] = []
    for item in tests_json:
        enemies = tuple(Enemy(str(e["kind"]), int(e["x"]), int(e["y"])) for e in item.get("enemies", []))
        t = MapDefinition(
            g_pos=(int(item["g_pos"][0]), int(item["g_pos"][1])),
            m_pos=(int(item["m_pos"][0]), int(item["m_pos"][1])),
            c_pos=(int(item["c_pos"][0]), int(item["c_pos"][1])),
            enemies=enemies,
            variant=int(item.get("variant", 1)),
        )
        tests.append(t)
    return tests


def generate_guaranteed_tests(count: int, rng: random.Random, prefer_algo: str = "astar") -> List[MapDefinition]:
    tests: List[MapDefinition] = []
    while len(tests) < count:
        md = generate_map(rng, variant=rng.choice((1, 2)))
        stats = compute_shortest_paths(md)
        if stats["dist_to_m"] is None:
            continue
        hazards = compute_hazard_cache(md)
        res = run_single_simulation(md, prefer_algo, hazards)
        if res.success:
            tests.append(md)
    return tests


def generate_random_tests(count: int, rng: random.Random) -> List[MapDefinition]:
    return [generate_map(rng) for _ in range(count)]


def run_suite(tests: Sequence[MapDefinition], algo: str) -> List[RunResult]:
    results: List[RunResult] = []
    for md in tests:
        hazards = compute_hazard_cache(md)
        res = run_single_simulation(md, algo, hazards)
        results.append(res)
    return results


def write_stats_summary(out_path: Path, seed: int, tests_count: int, summaries: Dict[str, Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"seed: {seed}\n")
        f.write(f"tests: {tests_count}\n\n")
        for key, data in summaries.items():
            f.write(f"[{key}]\n")
            for dk, dv in data.items():
                f.write(f"{dk}: {dv}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactor + generator + runner")
    parser.add_argument("--maps", type=int, default=DEFAULT_MAPS_PER_RUN, help="Total tests (default: 1000)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument("--tests-file", type=str, default=str(ROOT / "tests.json"), help="Tests file path")
    parser.add_argument("--generate", action="store_true", help="Generate tests and save to file")
    parser.add_argument("--run", action="store_true", help="Run tests from file and write stats")
    args = parser.parse_args()

    chosen_seed = int.from_bytes(os.urandom(8), "little") if args.seed is None else args.seed
    rng = random.Random(chosen_seed)

    build_errors = ensure_binaries_built()
    if any(v for v in build_errors.values()):
        for algo_key, err in build_errors.items():
            if err:
                print(f"[build] {algo_key}: {err}")

    tests_path = Path(args.tests_file)

    if args.generate:
        total = int(args.maps)
        if total != 1000:
            print("Adjusting --maps to 1000 as required.")
            total = 1000
        guaranteed = 800
        print(f"Generating {guaranteed} guaranteed-success tests...")
        tests = generate_guaranteed_tests(guaranteed, rng)
        extra = total - guaranteed
        print(f"Generating {extra} random tests...")
        tests += generate_random_tests(extra, rng)
        save_tests(tests, tests_path)
        print(f"Saved {len(tests)} tests to {tests_path}")

    if args.run:
        tests = load_tests(tests_path)
        if len(tests) != 1000:
            print(f"Warning: loaded {len(tests)} tests, expected 1000")
        print("Running a_star...")
        astar_results = run_suite(tests, "astar")
        print("Running backtracking...")
        backtracking_results = run_suite(tests, "backtracking")
        summaries = {
            "astar": aggregate_statistics(astar_results),
            "backtracking": aggregate_statistics(backtracking_results),
        }
        out_path = ANALYSIS_DIR / "stats_summary.txt"
        write_stats_summary(out_path, chosen_seed, len(tests), summaries)
        print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

# Reuse interactor utilities without test generation
from interactor import (
    MapDefinition,
    load_tests,
    run_suite,
    aggregate_statistics,
    write_stats_summary,
    ensure_binaries_built,
)

ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"


def draw_map_ascii(size: int, enemies: List[tuple[str, int, int]], c_pos: tuple[int, int], m_pos: tuple[int, int], g_pos: tuple[int, int]) -> str:
    # grid[y][x] with y=row, x=col
    grid = [["." for _ in range(size)] for _ in range(size)]
    # Place items first
    cx, cy = c_pos
    mx, my = m_pos
    gx, gy = g_pos
    grid[cy][cx] = "C"
    grid[my][mx] = "M"
    grid[gy][gx] = "G"
    # Place enemies (override items if overlaps happen)
    for kind, x, y in enemies:
        if 0 <= x < size and 0 <= y < size:
            grid[y][x] = kind
    # Build string
    return "\n".join("".join(row) for row in grid)


def force_variant_tests(tests: List[MapDefinition], variant: int) -> List[MapDefinition]:
    # Rebuild MapDefinition objects with the forced variant while keeping positions/enemies intact
    return [
        MapDefinition(
            g_pos=t.g_pos,
            m_pos=t.m_pos,
            c_pos=t.c_pos,
            enemies=t.enemies,
            variant=variant,
        )
        for t in tests
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-generated JSON tests through both algorithms, write stats, and dump unsolvable maps")
    parser.add_argument("--tests-file", type=str, default=str(ROOT / "tests.json"), help="Path to JSON tests file")
    parser.add_argument("--out", type=str, default=str(ANALYSIS_DIR / "stats_summary.txt"), help="Output stats file path")
    parser.add_argument("--astar-unsolvable-out", type=str, default=str(ROOT / "a_star_unsolvable_maps.txt"), help="File to write unsolvable maps for a_star")
    parser.add_argument("--backtracking-unsolvable-out", type=str, default=str(ROOT / "backtracking_unsolvable_maps.txt"), help="File to write unsolvable maps for backtracking")
    args = parser.parse_args()

    # Ensure C++ binaries are available
    build_errors = ensure_binaries_built()
    if any(v for v in build_errors.values()):
        for algo_key, err in build_errors.items():
            if err:
                print(f"[build] {algo_key}: {err}")

    tests_path = Path(args.tests_file)
    tests = load_tests(tests_path)
    if not tests:
        raise SystemExit(f"No tests loaded from {tests_path}")

    # Build two forced-variant suites
    tests_v1 = force_variant_tests(tests, 1)
    tests_v2 = force_variant_tests(tests, 2)

    # Run both algorithms on both variants
    print("Running a_star on variant=1 test suite...")
    astar_v1 = run_suite(tests_v1, "astar")
    print("Running a_star on variant=2 test suite...")
    astar_v2 = run_suite(tests_v2, "astar")

    print("Running backtracking on variant=1 test suite...")
    back_v1 = run_suite(tests_v1, "backtracking")
    print("Running backtracking on variant=2 test suite...")
    back_v2 = run_suite(tests_v2, "backtracking")

    # Write unsolvable maps per variant (claimed e -1)
    astar_out_v1 = Path(args.astar_unsolvable_out).with_name("a_star_unsolvable_maps_v1.txt")
    astar_out_v2 = Path(args.astar_unsolvable_out).with_name("a_star_unsolvable_maps_v2.txt")
    back_out_v1 = Path(args.backtracking_unsolvable_out).with_name("backtracking_unsolvable_maps_v1.txt")
    back_out_v2 = Path(args.backtracking_unsolvable_out).with_name("backtracking_unsolvable_maps_v2.txt")

    def write_unsolvable(file_path: Path, base_tests: List[MapDefinition], results) -> int:
        cnt = 0
        with file_path.open("w", encoding="utf-8") as f:
            for idx, md in enumerate(base_tests):
                if idx < len(results) and results[idx].claimed_unsolvable:
                    cnt += 1
                    enemies = [(e.kind, e.x, e.y) for e in md.enemies]
                    ascii_map = draw_map_ascii(13, enemies, md.c_pos, md.m_pos, md.g_pos)
                    f.write(f"# Variant={md.variant}\n")
                    f.write(ascii_map + "\n\n")
        return cnt

    count_a_v1 = write_unsolvable(astar_out_v1, tests_v1, astar_v1)
    count_a_v2 = write_unsolvable(astar_out_v2, tests_v2, astar_v2)
    count_b_v1 = write_unsolvable(back_out_v1, tests_v1, back_v1)
    count_b_v2 = write_unsolvable(back_out_v2, tests_v2, back_v2)
    print(
        "Unsolvable (claimed) maps written: "
        f"a_star v1={count_a_v1}, a_star v2={count_a_v2}, "
        f"backtracking v1={count_b_v1}, backtracking v2={count_b_v2}"
    )

    summaries: Dict[str, object] = {
        "astar_v1": aggregate_statistics(astar_v1),
        "astar_v2": aggregate_statistics(astar_v2),
        "backtracking_v1": aggregate_statistics(back_v1),
        "backtracking_v2": aggregate_statistics(back_v2),
    }

    out_path = Path(args.out)
    # We don't track a seed here; pass -1 for visibility and the exact count
    write_stats_summary(out_path, seed=-1, tests_count=len(tests), summaries=summaries)  # type: ignore[arg-type]
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()

"""
LALR(1) lookahead propagation by INIT/PASS rounds.

This script implements the table-construction algorithm shown in image.png:
build the LR(0) item sets first, attach lookaheads to kernel items, then
propagate those lookaheads until a fixed point is reached.

It intentionally reuses the existing project code for grammar parsing,
FIRST sets, LR(0) automaton construction, and LR(1)-style table building.

Usage:
  python lalr1_pass.py
  python lalr1_pass.py grammar.txt
  python lalr1_pass.py grammar.txt "id = id"
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import DefaultDict, Dict, FrozenSet, Iterable, List, Set, Tuple

from grammar_utils import Grammar, compute_first, fmt_set, print_section_header
from lr0_automaton import Item, build_lr0_automaton, print_table
from lr1_parser import LR1Item, build_lr1_table, lr1_parse_demo


EPSILON = "\u851a"
ENDMARKER = "$"
PROPAGATE = "#"

KernelRef = Tuple[int, Item]
PassSnapshot = Dict[KernelRef, Set[str]]
PassAdditions = Dict[KernelRef, Set[str]]


IMAGE_GRAMMAR = """
S -> L = R | R
L -> * R | id
R -> L
"""


def item_to_str(item: Item) -> str:
    lhs, rhs, dot = item
    parts = list(rhs)
    parts.insert(dot, ".")
    return f"{lhs} -> {' '.join(parts)}"


def lr1_item_to_str(item: LR1Item) -> str:
    lhs, rhs, dot, lookahead = item
    parts = list(rhs)
    parts.insert(dot, ".")
    return f"[{lhs} -> {' '.join(parts)}, {lookahead}]"


def ref_to_str(ref: KernelRef) -> str:
    state_id, item = ref
    return f"I{state_id}: {item_to_str(item)}"


def sorted_symbols(symbols: Iterable[str]) -> List[str]:
    return sorted(symbols, key=lambda x: (x != ENDMARKER, x != PROPAGATE, x))


def fmt_symbols(symbols: Iterable[str]) -> str:
    values = list(sorted_symbols(symbols))
    return ", ".join(values) if values else "-"


def first_sequence(first: Dict[str, Set[str]], symbols: List[str]) -> Set[str]:
    result: Set[str] = set()
    if not symbols:
        return {EPSILON}

    for sym in symbols:
        if sym in {ENDMARKER, PROPAGATE}:
            sym_first = {sym}
        elif sym == EPSILON:
            sym_first = {EPSILON}
        else:
            sym_first = set(first.get(sym, {sym}))

        result |= (sym_first - {EPSILON})
        if EPSILON not in sym_first:
            return result

    result.add(EPSILON)
    return result


def lr1_closure_with_placeholder(
    items: FrozenSet[LR1Item],
    grammar: Grammar,
    first: Dict[str, Set[str]],
) -> FrozenSet[LR1Item]:
    closure_set: Set[LR1Item] = set(items)
    worklist: List[LR1Item] = list(items)

    while worklist:
        lhs, rhs, dot, lookahead = worklist.pop()
        if dot >= len(rhs):
            continue

        next_symbol = rhs[dot]
        if next_symbol not in grammar.non_terminals:
            continue

        beta_a = list(rhs[dot + 1 :]) + [lookahead]
        lookaheads = first_sequence(first, beta_a)
        for prod_rhs in grammar.productions[next_symbol]:
            for la in lookaheads:
                if la == EPSILON:
                    continue
                new_item: LR1Item = (next_symbol, tuple(prod_rhs), 0, la)
                if new_item not in closure_set:
                    closure_set.add(new_item)
                    worklist.append(new_item)

    return frozenset(closure_set)


def kernel_items(state_id: int, state: FrozenSet[Item], aug_start: str) -> List[Item]:
    result: List[Item] = []
    for item in state:
        lhs, _rhs, dot = item
        if dot > 0 or (state_id == 0 and lhs == aug_start):
            result.append(item)
    return sorted(result)


def collect_kernel_refs(
    states: List[FrozenSet[Item]],
    aug_start: str,
) -> List[KernelRef]:
    refs: List[KernelRef] = []
    for state_id, state in enumerate(states):
        for item in kernel_items(state_id, state, aug_start):
            refs.append((state_id, item))
    return refs


def initial_ref(states: List[FrozenSet[Item]], aug_start: str) -> KernelRef:
    start_matches = [
        (0, item)
        for item in states[0]
        if item[0] == aug_start and item[2] == 0
    ]
    if len(start_matches) != 1:
        raise ValueError("Could not find the unique augmented start item in I0.")
    return start_matches[0]


def build_propagation_data(
    grammar: Grammar,
    states: List[FrozenSet[Item]],
    gotos: Dict[Tuple[int, str], int],
    first: Dict[str, Set[str]],
) -> Tuple[
    List[KernelRef],
    Dict[KernelRef, Set[str]],
    Dict[KernelRef, Set[KernelRef]],
    Dict[KernelRef, List[Tuple[KernelRef, str, str]]],
]:
    aug = grammar.augmented()
    aug_start = aug.start
    kernel_refs = collect_kernel_refs(states, aug_start)
    kernel_ref_set = set(kernel_refs)

    spontaneous: Dict[KernelRef, Set[str]] = defaultdict(set)
    propagation: Dict[KernelRef, Set[KernelRef]] = defaultdict(set)
    generated_by: Dict[KernelRef, List[Tuple[KernelRef, str, str]]] = defaultdict(list)

    start_ref = initial_ref(states, aug_start)
    spontaneous[start_ref].add(ENDMARKER)
    generated_by[start_ref].append((start_ref, "initial item", ENDMARKER))

    outgoing_by_state: DefaultDict[int, List[Tuple[str, int]]] = defaultdict(list)
    for (src, symbol), dst in gotos.items():
        outgoing_by_state[src].append((symbol, dst))

    for source_ref in kernel_refs:
        state_id, source_item = source_ref
        lhs, rhs, dot = source_item
        seed: LR1Item = (lhs, rhs, dot, PROPAGATE)
        closure = lr1_closure_with_placeholder(frozenset({seed}), aug, first)

        for symbol, dst_state in sorted(outgoing_by_state[state_id]):
            for c_lhs, c_rhs, c_dot, lookahead in closure:
                if c_dot >= len(c_rhs) or c_rhs[c_dot] != symbol:
                    continue

                moved_item: Item = (c_lhs, c_rhs, c_dot + 1)
                target_ref: KernelRef = (dst_state, moved_item)
                if target_ref not in kernel_ref_set:
                    continue

                if lookahead == PROPAGATE:
                    propagation[source_ref].add(target_ref)
                else:
                    spontaneous[target_ref].add(lookahead)
                    generated_by[target_ref].append((source_ref, symbol, lookahead))

    return kernel_refs, spontaneous, propagation, generated_by


def run_passes(
    kernel_refs: List[KernelRef],
    spontaneous: Dict[KernelRef, Set[str]],
    propagation: Dict[KernelRef, Set[KernelRef]],
) -> Tuple[PassSnapshot, List[PassAdditions], List[PassSnapshot]]:
    current: PassSnapshot = {ref: set(spontaneous.get(ref, set())) for ref in kernel_refs}
    pass_additions: List[PassAdditions] = []
    pass_snapshots: List[PassSnapshot] = []

    while True:
        additions: PassAdditions = {ref: set() for ref in kernel_refs}
        for source_ref, targets in propagation.items():
            source_lookaheads = current.get(source_ref, set())
            for target_ref in targets:
                additions[target_ref] |= (source_lookaheads - current.get(target_ref, set()))

        additions = {ref: symbols for ref, symbols in additions.items() if symbols}
        if not additions:
            break

        for ref, symbols in additions.items():
            current[ref] |= symbols

        pass_additions.append(additions)
        pass_snapshots.append({ref: set(current.get(ref, set())) for ref in kernel_refs})

    return current, pass_additions, pass_snapshots


def lalr_states_from_passes(
    grammar: Grammar,
    lr0_states: List[FrozenSet[Item]],
    lookaheads: PassSnapshot,
) -> List[FrozenSet[LR1Item]]:
    aug = grammar.augmented()
    first = compute_first(aug)
    result: List[FrozenSet[LR1Item]] = []

    for state_id, state in enumerate(lr0_states):
        seeds: Set[LR1Item] = set()
        for item in kernel_items(state_id, state, aug.start):
            lhs, rhs, dot = item
            for la in lookaheads.get((state_id, item), set()):
                seeds.add((lhs, rhs, dot, la))
        result.append(lr1_closure_with_placeholder(frozenset(seeds), aug, first))

    return result


def print_lr0_states(states: List[FrozenSet[Item]], gotos: Dict[Tuple[int, str], int]) -> None:
    for state_id, state in enumerate(states):
        print(f"\n  I{state_id}")
        for item in sorted(state):
            print(f"    {item_to_str(item)}")

        edges = [(symbol, dst) for (src, symbol), dst in gotos.items() if src == state_id]
        if edges:
            print("    transitions:")
            for symbol, dst in sorted(edges):
                print(f"      on {symbol!r} -> I{dst}")


def print_kernel_table(kernel_refs: List[KernelRef], aug_start: str) -> None:
    print("  Kernel items that receive propagated lookaheads:")
    for ref in kernel_refs:
        state_id, item = ref
        prefix = "start" if state_id == 0 and item[0] == aug_start else "kernel"
        print(f"    {prefix:6} {ref_to_str(ref)}")


def print_spontaneous(
    spontaneous: Dict[KernelRef, Set[str]],
    generated_by: Dict[KernelRef, List[Tuple[KernelRef, str, str]]],
) -> None:
    for target_ref in sorted(spontaneous, key=lambda r: (r[0], item_to_str(r[1]))):
        print(f"  {ref_to_str(target_ref)} gets INIT {{{fmt_symbols(spontaneous[target_ref])}}}")
        for source_ref, symbol, lookahead in generated_by.get(target_ref, []):
            if source_ref == target_ref and symbol == "initial item":
                print(f"      from augmented start item: {lookahead}")
            else:
                print(
                    f"      generated from {ref_to_str(source_ref)} "
                    f"while taking GOTO on {symbol!r}: {lookahead}"
                )


def print_propagation_edges(propagation: Dict[KernelRef, Set[KernelRef]]) -> None:
    if not propagation:
        print("  (no propagation edges)")
        return

    for source_ref in sorted(propagation, key=lambda r: (r[0], item_to_str(r[1]))):
        targets = sorted(propagation[source_ref], key=lambda r: (r[0], item_to_str(r[1])))
        print(f"  FROM {ref_to_str(source_ref)}")
        for target_ref in targets:
            print(f"       TO {ref_to_str(target_ref)}")


def print_pass_details(pass_additions: List[PassAdditions]) -> None:
    if not pass_additions:
        print("  No PASS rounds were needed; INIT was already a fixed point.")
        return

    for index, additions in enumerate(pass_additions, start=1):
        print(f"  PASS {index}")
        for ref in sorted(additions, key=lambda r: (r[0], item_to_str(r[1]))):
            print(f"    {ref_to_str(ref)} adds {{{fmt_symbols(additions[ref])}}}")


def print_snapshot_table(
    kernel_refs: List[KernelRef],
    spontaneous: Dict[KernelRef, Set[str]],
    pass_snapshots: List[PassSnapshot],
    final_lookaheads: PassSnapshot,
) -> None:
    headers = ["State", "Item", "INIT"]
    headers += [f"PASS {i}" for i in range(1, len(pass_snapshots) + 1)]
    headers.append("FINAL")

    rows: List[List[str]] = []
    for ref in kernel_refs:
        state_id, item = ref
        row = [f"I{state_id}", item_to_str(item), fmt_symbols(spontaneous.get(ref, set()))]
        for snapshot in pass_snapshots:
            row.append(fmt_symbols(snapshot.get(ref, set())))
        row.append(fmt_symbols(final_lookaheads.get(ref, set())))
        rows.append(row)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(values: List[str]) -> str:
        return "  " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print(line(headers))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def print_lalr_item_sets(lalr_states: List[FrozenSet[LR1Item]]) -> None:
    for state_id, state in enumerate(lalr_states):
        print(f"\n  I{state_id}")
        by_core: DefaultDict[Item, Set[str]] = defaultdict(set)
        for lhs, rhs, dot, lookahead in state:
            by_core[(lhs, rhs, dot)].add(lookahead)
        for core, lookaheads in sorted(by_core.items(), key=lambda x: item_to_str(x[0])):
            print(f"    {item_to_str(core)}, {{{fmt_symbols(lookaheads)}}}")


def analyze_lalr_pass(grammar_text: str, input_string: str | None = None) -> Tuple[List[FrozenSet[LR1Item]], Dict[Tuple[int, str], int]]:
    grammar = Grammar(grammar_text)
    aug = grammar.augmented()
    first = compute_first(aug)

    print_section_header("Step 1. Grammar and FIRST sets")
    print(f"  Start symbol: {grammar.start}")
    print("  Productions:")
    for lhs, rhs in grammar.all_productions:
        print(f"    {lhs} -> {' '.join(rhs)}")
    print("  FIRST:")
    for symbol in sorted(grammar.non_terminals):
        print(f"    FIRST({symbol}) = {{{fmt_set(first[symbol])}}}")

    print_section_header("Step 2. LR(0) item sets")
    lr0_states, lr0_gotos, _trace = build_lr0_automaton(grammar)
    print_lr0_states(lr0_states, lr0_gotos)

    print_section_header("Step 3. Kernel items")
    kernel_refs, spontaneous, propagation, generated_by = build_propagation_data(
        grammar,
        lr0_states,
        lr0_gotos,
        first,
    )
    print_kernel_table(kernel_refs, aug.start)

    print_section_header("Step 4. INIT lookaheads")
    print_spontaneous(spontaneous, generated_by)

    print_section_header("Step 5. Propagation graph")
    print_propagation_edges(propagation)

    print_section_header("Step 6. PASS rounds")
    final_lookaheads, pass_additions, pass_snapshots = run_passes(
        kernel_refs,
        spontaneous,
        propagation,
    )
    print_pass_details(pass_additions)

    print_section_header("Step 7. Lookahead table")
    print_snapshot_table(kernel_refs, spontaneous, pass_snapshots, final_lookaheads)

    print_section_header("Step 8. Final LALR(1) item sets")
    lalr_states = lalr_states_from_passes(grammar, lr0_states, final_lookaheads)
    print_lalr_item_sets(lalr_states)

    print_section_header("Step 9. LALR(1) parsing table")
    table = build_lr1_table(grammar, aug, lalr_states, lr0_gotos)
    print_table(table, grammar, len(lalr_states))
    conflicts = [(key, value) for key, value in table.items() if value[0] == "conflict"]
    if conflicts:
        print(f"\n  Conflicts: {len(conflicts)}")
        for (state_id, symbol), value in conflicts:
            print(f"    I{state_id}, {symbol}: {value}")
    else:
        print("\n  No LALR(1) conflicts.")

    if input_string:
        print_section_header("Step 10. Parse input with the PASS-built table")
        lr1_parse_demo(grammar, lalr_states, lr0_gotos, input_string, label="LALR(1)-PASS")

    return lalr_states, lr0_gotos


def main() -> None:
    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            grammar_text = handle.read()
        input_string = sys.argv[2] if len(sys.argv) >= 3 else None
    else:
        grammar_text = IMAGE_GRAMMAR
        input_string = None

    analyze_lalr_pass(grammar_text, input_string)


if __name__ == "__main__":
    main()

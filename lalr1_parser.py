"""
LALR(1) 语法分析器 (Look-Ahead LR)
==================================
先构造完整的 LR(1) 项集族，然后将具有相同「核」（core = 忽略前瞻符后的 LR(0) 项）
的状态合并，合并后前瞻符取并集。

核心优势: LR(1) 的判别力 + LR(0) 的状态数。

依赖:
  grammar_utils   — Grammar, compute_first, fmt_set, print_section_header
  lr0_automaton   — print_table, _format_action, _print_parse_table
  lr1_parser      — build_lr1_automaton (构造 LR(1) 项集族)

用法:
  python lalr1_parser.py                              # 交互模式
  python lalr1_parser.py grammar.txt                  # 从文件读取文法
  python lalr1_parser.py grammar.txt "id * id + id"   # 指定输入串分析
"""

import sys
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

from grammar_utils import (
    Grammar, compute_first, fmt_set, print_section_header,
)
from lr0_automaton import (
    print_table, _format_action, _print_parse_table,
)
from lr1_parser import (
    LR1Item, lr1_item_str, lr1_items_str,
    lr1_goto,  # LALR 核的 GOTO 计算需要 LR(1) 闭包
    build_lr1_automaton, build_lr1_table,
    lr1_parse_demo,  # 复用 LR(1) 分析过程（LALR 只是状态集不同）
)


# ═══════════════════════════════════════════════════════════════
# 1. LALR(1) 核心: 按核合并 LR(1) 状态
# ═══════════════════════════════════════════════════════════════

def _core(state: FrozenSet[LR1Item]) -> FrozenSet[Tuple[str, Tuple[str, ...], int]]:
    """提取状态的核（core）：忽略前瞻符。"""
    return frozenset((lhs, rhs, dot) for (lhs, rhs, dot, _) in state)


def lalr1_merge_states(
    lr1_states: List[FrozenSet[LR1Item]],
) -> Tuple[
    List[FrozenSet[LR1Item]],           # 合并后的状态列表
    List[Set[int]],                      # 每个新状态由哪些旧状态合并而来
    Dict[int, int],                      # 旧状态 ID → 新状态 ID
]:
    """合并具有相同核的 LR(1) 状态。

    返回: (merged_states, merge_sources, old_to_new)
    """
    # 按核分组
    core_to_items: Dict[FrozenSet, Set[LR1Item]] = {}
    core_to_old_ids: Dict[FrozenSet, List[int]] = defaultdict(list)

    for old_id, state in enumerate(lr1_states):
        c = _core(state)
        core_to_old_ids[c].append(old_id)
        if c in core_to_items:
            # 合并: 同一核 + 不同前瞻符 → 都保留在集合中
            core_to_items[c] |= set(state)
        else:
            core_to_items[c] = set(state)

    # 生成合并后的状态列表（保持稳定顺序）
    # 按首次出现的核排序
    merged_states: List[FrozenSet[LR1Item]] = []
    merge_sources: List[Set[int]] = []
    old_to_new: Dict[int, int] = {}

    seen_cores = set()
    for old_id, state in enumerate(lr1_states):
        c = _core(state)
        if c in seen_cores:
            continue
        seen_cores.add(c)
        new_id = len(merged_states)
        merged_states.append(frozenset(core_to_items[c]))
        merge_sources.append(set(core_to_old_ids[c]))
        for oid in core_to_old_ids[c]:
            old_to_new[oid] = new_id

    return merged_states, merge_sources, old_to_new


# ═══════════════════════════════════════════════════════════════
# 2. LALR(1) GOTO 转移重映射
# ═══════════════════════════════════════════════════════════════

def lalr1_remap_gotos(
    lr1_gotos: Dict[Tuple[int, str], int],
    old_to_new: Dict[int, int],
    num_lalr_states: int,
) -> Dict[Tuple[int, str], int]:
    """将 LR(1) 的 GOTO 重映射为 LALR(1) 的 GOTO。

    规则: LALR GOTO(new_s, X) = new_t
         其中 new_s = old_to_new[s], new_t = old_to_new[t],
         (s, X, t) 是 LR(1) 的 GOTO 边。

    如果合并后产生冲突（同一 (new_s, X) 映射到不同 new_t），
    说明该文法不是 LALR(1)。
    """
    lalr_gotos: Dict[Tuple[int, str], int] = {}
    lalr_goto_conflicts = []

    for (s, X), t in lr1_gotos.items():
        ns = old_to_new.get(s)
        nt = old_to_new.get(t)
        if ns is None or nt is None:
            continue
        existing = lalr_gotos.get((ns, X))
        if existing is not None and existing != nt:
            lalr_goto_conflicts.append((ns, X, existing, nt))
        lalr_gotos[(ns, X)] = nt

    if lalr_goto_conflicts:
        print(f"  ⚠ LALR(1) GOTO 冲突（文法不是 LALR(1)）:")
        for ns, X, t1, t2 in lalr_goto_conflicts[:5]:
            print(f"    状态 I{ns}, '{X}': → I{t1} vs → I{t2}")

    return lalr_gotos


# ═══════════════════════════════════════════════════════════════
# 3. LALR(1) 分析表构造
# ═══════════════════════════════════════════════════════════════

def build_lalr1_table(
    grammar: Grammar,
    aug: Grammar,
    lalr_states: List[FrozenSet[LR1Item]],
    lalr_gotos: Dict[Tuple[int, str], int],
) -> Dict[Tuple[int, str], tuple]:
    """从 LALR(1) 合并后的状态构造分析表。

    逻辑与 build_lr1_table 完全相同——LALR 只是状态集不同。
    """
    return build_lr1_table(grammar, aug, lalr_states, lalr_gotos)


# ═══════════════════════════════════════════════════════════════
# 4. 顶层流水线
# ═══════════════════════════════════════════════════════════════

def build_lalr1_automaton(
    grammar: Grammar,
    FIRST: Dict[str, Set[str]],
) -> Tuple[
    List[FrozenSet[LR1Item]],
    Dict[Tuple[int, str], int],
    List[str],
    List[FrozenSet[LR1Item]],   # LR(1) states (for comparison)
]:
    """构造 LALR(1) 自动机: LR(1) 构造 → 状态合并 → GOTO 重映射。

    返回: (lalr_states, lalr_gotos, trace, lr1_states)
    """
    trace: List[str] = []

    # Step 1: 完整的 LR(1) 项集族
    trace.append("===== 第一步: 构造完整 LR(1) 项集族 =====")
    trace.append("")
    lr1_states, lr1_gotos, lr1_trace = build_lr1_automaton(grammar, FIRST)
    # 保留 LR(1) 的构造过程但简化输出（太长了）
    trace.append(f"  LR(1) 项集族构造完成: {len(lr1_states)} 个状态")
    trace.append("")

    # Step 2: 按核合并
    trace.append("===== 第二步: 按核合并 LR(1) 状态 =====")
    trace.append("")

    merged, sources, old_to_new = lalr1_merge_states(lr1_states)
    trace.append(f"  合并前 (LR(1)): {len(lr1_states)} 个状态")
    trace.append(f"  合并后 (LALR):  {len(merged)} 个状态")

    # 展示合并情况
    for new_id, src_ids in enumerate(sources):
        if len(src_ids) > 1:
            # 展示哪些状态被合并了
            trace.append(f"    I{new_id} ← 合并了 LR(1) 状态: {sorted(src_ids)}")
            # 展示合并后新增的前瞻符
            core_items = {}
            for oid in sorted(src_ids):
                for item in lr1_states[oid]:
                    key = item[:3]
                    if key not in core_items:
                        core_items[key] = set()
                    core_items[key].add(item[3])
            for key in sorted(core_items):
                if len(core_items[key]) > 1:
                    lhs, rhs, dot = key
                    rhs_list = list(rhs)
                    rhs_list.insert(dot, '·')
                    trace.append(f"      [{lhs} → {' '.join(rhs_list)}, {{{fmt_set(core_items[key])}}}]")

    trace.append("")

    # Step 3: 重映射 GOTO
    trace.append("===== 第三步: 重映射 GOTO 转移 =====")
    trace.append("")
    lalr_gotos = lalr1_remap_gotos(lr1_gotos, old_to_new, len(merged))
    trace.append(f"  LALR(1) GOTO 重映射完成: {len(lalr_gotos)} 条边")
    trace.append("")

    # 汇总
    aug = grammar.augmented()
    aug_start = aug.start

    trace.append("=" * 55)
    trace.append("LALR(1) 自动机汇总")
    trace.append("=" * 55)
    for i, I in enumerate(merged):
        trace.append(f"\n  状态 I{i}{' (合并自 LR(1): ' + str(sorted(sources[i])) if len(sources[i]) > 1 else ''}:")
        for it in sorted(I):
            lhs, rhs, dot, la = it
            marker = ""
            if dot == len(rhs):
                if lhs == aug_start:
                    marker = "  ← 接受项 (ACC)"
                else:
                    marker = f"  ← 归约项 (在 '{la}' 上归约)"
            trace.append(f"    {lr1_item_str(it)}{marker}")
        out_edges = [(sym, dst) for (src, sym), dst in lalr_gotos.items() if src == i]
        if out_edges:
            trace.append(f"    出边:")
            for sym, dst in sorted(out_edges, key=lambda x: x[0]):
                trace.append(f"      --{sym}--> I{dst}")

    return merged, lalr_gotos, trace, lr1_states


def analyze_lalr1(grammar_text: str, input_string: str = None):
    """LALR(1) 完整流程: LR(1) → 合并 → 分析表 → 分析。"""

    grammar = Grammar(grammar_text)
    aug = grammar.augmented()
    print_section_header("第一步: 解析原文法")
    print(f"  开始符号: {grammar.start}")
    print(f"  非终结符: {sorted(grammar.non_terminals)}")
    print(f"  终结符:   {sorted(grammar.terminals)}")
    print(f"  产生式:")
    print(grammar)

    # FIRST
    print_section_header("第二步: 计算 FIRST 集")
    FIRST = compute_first(grammar)
    for nt in sorted(grammar.non_terminals):
        print(f"    FIRST({nt}) = {{{fmt_set(FIRST[nt])}}}")

    # LALR(1) 自动机
    print_section_header("第三步: 构造 LALR(1) 自动机")
    lalr_states, lalr_gotos, trace, lr1_states = build_lalr1_automaton(grammar, FIRST)
    for line in trace:
        print(line)

    # 状态数对比
    print(f"\n  状态数对比: LR(1)={len(lr1_states)}, LALR(1)={len(lalr_states)}, LR(0)={len(lr1_states) if len(lr1_states) < 90 else '?'}")

    # 分析
    if input_string:
        print_section_header("第四步: LALR(1) 分析")
        print(f"  LALR(1) = LR(1) 的判别力 + LR(0) 的状态数")
        lr1_parse_demo(grammar, lalr_states, lalr_gotos, input_string, label="LALR(1)")
    else:
        print_section_header("第四步: LALR(1) 分析表")
        table = build_lalr1_table(grammar, aug, lalr_states, lalr_gotos)
        print(f"\n  LALR(1) 分析表 (ACTION / GOTO):")
        print_table(table, grammar, len(lalr_states))
        conflicts = [(k, v) for k, v in table.items() if v[0] == 'conflict']
        if conflicts:
            print(f"\n  ⚠ 有 {len(conflicts)} 个冲突")
        else:
            print(f"\n  ✓ LALR(1) 分析表无冲突")

    return lalr_states, lalr_gotos


# ═══════════════════════════════════════════════════════════════
# 5. 测试用例
# ═══════════════════════════════════════════════════════════════

ARITHMETIC_GRAMMAR = """
S → S S + | S S * | a
"""

# 经典 LALR(1) vs SLR(1) vs LR(1) 对比文法
LR1_DEMO_GRAMMAR = """
S → L = R | R
L → * R | id
R → L
"""


def run_builtin_tests():
    print("=" * 60)
    print("  测试 1: 算术表达式文法 — LALR(1) 自动机")
    print("=" * 60)
    analyze_lalr1(ARITHMETIC_GRAMMAR)

    print("\n\n")

    # print("=" * 60)
    # print("  测试 2: LALR(1) 分析 'id * id + id'")
    # print("=" * 60)
    # analyze_lalr1(ARITHMETIC_GRAMMAR, "id * id + id")

    print("\n\n")

    # print("=" * 60)
    # print("  测试 3: S→L=R|R, L→*R|id, R→L (经典对比文法)")
    # print("=" * 60)
    # analyze_lalr1(LR1_DEMO_GRAMMAR)


# ═══════════════════════════════════════════════════════════════
# 6. main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        grammar_path = sys.argv[1]
        with open(grammar_path, 'r', encoding='utf-8') as f:
            grammar_text = f.read()
        input_str = sys.argv[2] if len(sys.argv) >= 3 else None
        analyze_lalr1(grammar_text, input_str)
    else:
        run_builtin_tests()

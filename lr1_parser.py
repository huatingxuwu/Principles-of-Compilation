"""
LR(1) 语法分析器
================
LR(1) 项含前瞻符: [A → α·β, a], 归约仅在此前瞻符上执行。

比 SLR(1) 更强: 前瞻符按上下文传播（FIRST(βa)），而非笼统的 FOLLOW(A)。

依赖:
  grammar_utils  — Grammar, compute_first, first_of_symbols, fmt_set, print_section_header
  lr0_automaton  — print_table, _format_action, _print_parse_table

用法:
  python lr1_parser.py                              # 交互模式
  python lr1_parser.py grammar.txt                  # 从文件读取文法
  python lr1_parser.py grammar.txt "id * id + id"   # 指定输入串分析
"""

import sys
from collections import defaultdict, deque
from typing import Dict, FrozenSet, List, Set, Tuple

from grammar_utils import (
    Grammar, compute_first, first_of_symbols, fmt_set, print_section_header,
)
from lr0_automaton import (
    print_table, _format_action, _print_parse_table,
)


# ═══════════════════════════════════════════════════════════════
# 1. LR(1) 项定义
# ═══════════════════════════════════════════════════════════════

# LR(1) 项: (lhs, rhs_tuple, dot_pos, lookahead)
LR1Item = Tuple[str, Tuple[str, ...], int, str]


def lr1_item_str(item: LR1Item) -> str:
    """格式化 LR(1) 项: [A → α·β, a]。"""
    lhs, rhs, dot, la = item
    rhs_list = list(rhs)
    rhs_list.insert(dot, '·')
    return f"[{lhs} → {' '.join(rhs_list)}, {la}]"


def lr1_items_str(items) -> str:
    """格式化 LR(1) 项集。"""
    return '{' + ',  '.join(lr1_item_str(it) for it in sorted(items)) + '}'


# ═══════════════════════════════════════════════════════════════
# 2. LR(1) CLOSURE — 含前瞻符传播
# ═══════════════════════════════════════════════════════════════

def lr1_closure(
    items: FrozenSet[LR1Item],
    grammar: Grammar,
    FIRST: Dict[str, Set[str]],
    trace: List[str] = None,
) -> FrozenSet[LR1Item]:
    """LR(1) CLOSURE 计算。

    规则: 若 [A → α·Bβ, a] ∈ closure(I)
         则对每条 B → γ 和每个 b ∈ FIRST(βa)，加入 [B → ·γ, b]。
    """
    closure_set = set(items)
    worklist = list(items)

    while worklist:
        item = worklist.pop()
        lhs, rhs, dot, la = item
        if dot >= len(rhs):
            continue
        next_sym = rhs[dot]
        if next_sym not in grammar.non_terminals:
            continue

        # β = 点后面的剩余符号
        beta = list(rhs[dot + 1:]) + [la]  # βa
        # 计算 FIRST(βa)
        first_beta_a = first_of_symbols(FIRST, beta)

        for prod_rhs in grammar.productions[next_sym]:
            for b in first_beta_a:
                if b == 'ε':
                    continue  # ε 不能做前瞻符
                new_item: LR1Item = (next_sym, tuple(prod_rhs), 0, b)
                if new_item not in closure_set:
                    closure_set.add(new_item)
                    worklist.append(new_item)
                    if trace is not None:
                        trace.append(
                            f"      由 {lr1_item_str(item)} 预测: "
                            f"加入 {lr1_item_str(new_item)} "
                            f"(FIRST({' '.join(beta)}) = {{{fmt_set(first_beta_a)}}})"
                        )

    return frozenset(closure_set)


# ═══════════════════════════════════════════════════════════════
# 3. LR(1) GOTO
# ═══════════════════════════════════════════════════════════════

def lr1_goto(
    items: FrozenSet[LR1Item],
    symbol: str,
    grammar: Grammar,
    FIRST: Dict[str, Set[str]],
    trace: List[str] = None,
) -> FrozenSet[LR1Item]:
    """GOTO(I, X): 把 I 中所有 [A → α·Xβ, a] 的点过 X 右移，再求闭包。"""
    kernel: Set[LR1Item] = set()
    for item in items:
        lhs, rhs, dot, la = item
        if dot < len(rhs) and rhs[dot] == symbol:
            kernel.add((lhs, rhs, dot + 1, la))

    if not kernel:
        return frozenset()

    if trace is not None:
        trace.append(f"      kernel 项 (点右移过 '{symbol}'):")
        for it in sorted(kernel):
            trace.append(f"        {lr1_item_str(it)}")

    return lr1_closure(frozenset(kernel), grammar, FIRST, trace)


# ═══════════════════════════════════════════════════════════════
# 4. LR(1) 规范项集族构造
# ═══════════════════════════════════════════════════════════════

def build_lr1_automaton(
    grammar: Grammar,
    FIRST: Dict[str, Set[str]],
) -> Tuple[
    List[FrozenSet[LR1Item]],
    Dict[Tuple[int, str], int],
    List[str],
]:
    """构造 LR(1) 规范项集族。

    返回: (states, gotos, trace)
    """
    trace: List[str] = []

    aug = grammar.augmented()
    aug_start = aug.start
    trace.append(f"增广文法: 添加 {aug_start} → {grammar.start}")
    trace.append("")

    # 初始项: [S' → ·S, $]
    start_item: LR1Item = (aug_start, (grammar.start,), 0, '$')
    trace.append(f"初始项: {lr1_item_str(start_item)}")

    I0_kernel = frozenset({start_item})
    trace.append(f"\n计算 LR(1) CLOSURE({{{lr1_item_str(start_item)}}}):")
    I0 = lr1_closure(I0_kernel, aug, FIRST, trace)
    trace.append(f"  → I₀ = {lr1_items_str(I0)}")
    trace.append("")

    states: List[FrozenSet[LR1Item]] = [I0]
    gotos: Dict[Tuple[int, str], int] = {}
    all_symbols = sorted(aug.terminals | aug.non_terminals)

    worklist = deque([0])
    round_num = 0

    while worklist:
        state_id = worklist.popleft()
        I = states[state_id]
        round_num += 1

        trace.append(f"── 处理状态 I{state_id} ──")
        trace.append(f"  项集: {lr1_items_str(I)}")
        trace.append("")

        for X in all_symbols:
            has_item = any(
                dot < len(rhs) and rhs[dot] == X
                for (_, rhs, dot, _) in I
            )
            if not has_item:
                continue

            trace.append(f"  GOTO(I{state_id}, '{X}'):")
            J = lr1_goto(I, X, aug, FIRST, trace)

            if not J:
                trace.append(f"    → 空集，无转移")
                continue

            try:
                existing_id = states.index(J)
                trace.append(f"    → 闭包后项集 = I{existing_id}（已存在）")
                gotos[(state_id, X)] = existing_id
            except ValueError:
                new_id = len(states)
                states.append(J)
                gotos[(state_id, X)] = new_id
                worklist.append(new_id)
                trace.append(f"    → 新状态 I{new_id} = {lr1_items_str(J)}")
            trace.append("")

    # 汇总
    trace.append("=" * 55)
    trace.append("LR(1) 自动机汇总")
    trace.append("=" * 55)
    for i, I in enumerate(states):
        trace.append(f"\n  状态 I{i}:")
        for it in sorted(I):
            lhs, rhs, dot, la = it
            marker = ""
            if dot == len(rhs):
                if lhs == aug_start:
                    marker = "  ← 接受项 (ACC)"
                else:
                    marker = f"  ← 归约项 (在 '{la}' 上归约)"
            trace.append(f"    {lr1_item_str(it)}{marker}")
        out_edges = [(sym, dst) for (src, sym), dst in gotos.items() if src == i]
        if out_edges:
            trace.append(f"    出边:")
            for sym, dst in sorted(out_edges, key=lambda x: x[0]):
                trace.append(f"      --{sym}--> I{dst}")

    return states, gotos, trace


# ═══════════════════════════════════════════════════════════════
# 5. LR(1) 分析表构造
# ═══════════════════════════════════════════════════════════════

def build_lr1_table(
    grammar: Grammar,
    aug: Grammar,
    states: List[FrozenSet[LR1Item]],
    gotos: Dict[Tuple[int, str], int],
) -> Dict[Tuple[int, str], tuple]:
    """构造 LR(1) 分析表。

    LR(1) 归约仅在项的前瞻符上执行，比 SLR(1) 更精确。
    """
    table: Dict[Tuple[int, str], tuple] = {}
    all_prods = aug.all_productions

    for i, I in enumerate(states):
        for item in I:
            lhs, rhs, dot, la = item

            if dot == len(rhs):
                if lhs == aug.start:
                    table[(i, '$')] = ('acc',)
                else:
                    # LR(1): 只在前瞻符 la 上归约
                    prod_idx = all_prods.index((lhs, rhs))
                    existing = table.get((i, la))
                    if existing and existing[0] == 'shift':
                        table[(i, la)] = ('conflict', existing, ('reduce', prod_idx))
                    elif existing and existing[0] == 'reduce' and existing[1] != prod_idx:
                        table[(i, la)] = ('conflict', existing, ('reduce', prod_idx))
                    else:
                        table[(i, la)] = ('reduce', prod_idx)

        for (src, sym), dst in gotos.items():
            if src == i:
                if sym in grammar.non_terminals:
                    table[(i, sym)] = ('goto', dst)
                else:
                    existing = table.get((i, sym))
                    if existing and existing[0] == 'reduce':
                        table[(i, sym)] = ('conflict', ('shift', dst), existing)
                    else:
                        table[(i, sym)] = ('shift', dst)

    return table


# ═══════════════════════════════════════════════════════════════
# 6. LR(1) 分析过程演示
# ═══════════════════════════════════════════════════════════════

def lr1_parse_demo(
    grammar: Grammar,
    states: List[FrozenSet[LR1Item]],
    gotos: Dict[Tuple[int, str], int],
    input_string: str,
    label: str = "LR(1)",
) -> bool:
    """LR(1) / LALR(1) 表驱动语法分析（表格输出）。"""
    aug = grammar.augmented()
    tokens = input_string.strip().split()
    table = build_lr1_table(grammar, aug, states, gotos)
    all_prods = aug.all_productions

    print(f"\n  输入串: {' '.join(tokens)}")
    print(f"\n  {label} 分析表 (ACTION / GOTO):")
    print_table(table, grammar, len(states))

    conflicts = [(k, v) for k, v in table.items() if v[0] == 'conflict']
    if conflicts:
        print(f"\n  ⚠ 有 {len(conflicts)} 个冲突（文法不是 LR(1)）:")
        for (s, a), v in conflicts:
            print(f"    状态 I{s}, 符号 '{a}': {_format_action(v[1])} vs {_format_action(v[2])}")
        return False
    else:
        print(f"\n  ✓ LR(1) 分析表无冲突！")

    print(f"\n  ── 分析过程 ──")

    rows = []
    state_stack: List[int] = [0]
    sym_stack: List[str] = ['$']
    input_tokens = tokens + ['$']
    pos = 0
    step = 0

    while True:
        step += 1
        s = state_stack[-1]
        a = input_tokens[pos]

        state_str = ' '.join(str(x) for x in state_stack)
        sym_str = ' '.join(sym_stack)
        remaining = ' '.join(input_tokens[pos:])

        action_entry = table.get((s, a))

        if action_entry is None:
            rows.append((f"({step})", state_str, sym_str, remaining,
                         f"✗ 错误: M[{s}, {a}] 为空"))
            _print_parse_table(rows)
            print(f"\n  ✗ 分析失败")
            return False

        action = action_entry[0]

        if action == 'shift':
            _, next_state = action_entry
            rows.append((f"({step})", state_str, sym_str, remaining, f"移入{next_state}"))
            state_stack.append(next_state)
            sym_stack.append(a)
            pos += 1

        elif action == 'reduce':
            _, prod_idx = action_entry
            lhs, rhs = all_prods[prod_idx]
            rhs_len = 0 if rhs == ('ε',) else len(rhs)
            rhs_str = ' '.join(rhs)
            rows.append((f"({step})", state_str, sym_str, remaining,
                         f"按 {lhs} → {rhs_str} 归约"))

            for _ in range(rhs_len):
                state_stack.pop()
                sym_stack.pop()

            new_s = state_stack[-1]
            goto_entry = table.get((new_s, lhs))
            if goto_entry is None or goto_entry[0] != 'goto':
                rows.append(("", "", "", "", f"✗ GOTO 错误"))
                _print_parse_table(rows)
                return False

            _, next_state = goto_entry
            state_stack.append(next_state)
            sym_stack.append(lhs)

        elif action == 'acc':
            rows.append((f"({step})", state_str, sym_str, remaining, "接受"))
            _print_parse_table(rows)
            print(f"\n  ✓ {label} 分析成功！")
            return True

    return True


# ═══════════════════════════════════════════════════════════════
# 7. 顶层流水线
# ═══════════════════════════════════════════════════════════════

def analyze_lr1(grammar_text: str, input_string: str = None):
    """LR(1) 完整流程: FIRST → LR(1) 自动机 → 分析表 → 分析。"""

    grammar = Grammar(grammar_text)
    print_section_header("第一步: 解析原文法")
    print(f"  开始符号: {grammar.start}")
    print(f"  非终结符: {sorted(grammar.non_terminals)}")
    print(f"  终结符:   {sorted(grammar.terminals)}")
    print(f"  产生式:")
    print(grammar)

    # FIRST（LR(1) CLOSURE 需要）
    print_section_header("第二步: 计算 FIRST 集")
    FIRST = compute_first(grammar)
    for nt in sorted(grammar.non_terminals):
        print(f"    FIRST({nt}) = {{{fmt_set(FIRST[nt])}}}")

    # LR(1) 自动机
    print_section_header("第三步: 构造 LR(1) 自动机")
    states, gotos, trace = build_lr1_automaton(grammar, FIRST)
    for line in trace:
        print(line)

    # 分析
    if input_string:
        print_section_header("第四步: LR(1) 分析")
        print(f"  LR(1) 前瞻符按上下文精确传播，比 SLR(1) 的 FOLLOW 更精细")
        lr1_parse_demo(grammar, states, gotos, input_string)
    else:
        print_section_header("第四步: LR(1) 分析表")
        aug = grammar.augmented()
        table = build_lr1_table(grammar, aug, states, gotos)
        print(f"\n  LR(1) 分析表 (ACTION / GOTO):")
        print_table(table, grammar, len(states))
        conflicts = [(k, v) for k, v in table.items() if v[0] == 'conflict']
        if conflicts:
            print(f"\n  ⚠ 有 {len(conflicts)} 个冲突")
        else:
            print(f"\n  ✓ LR(1) 分析表无冲突")

    return states, gotos


# ═══════════════════════════════════════════════════════════════
# 8. 测试用例
# ═══════════════════════════════════════════════════════════════

ARITHMETIC_GRAMMAR = """
E → E + T | T
T → T * F | F
F → ( E ) | id
"""

# 经典 LR(1) vs SLR(1) 差异示例
# 文法: S → C C, C → c C | d
# SLR(1) 有冲突, LR(1) 无冲突
LR1_DEMO_GRAMMAR = """
S → C C
C → c C | d
"""


def run_builtin_tests():
    print("=" * 60)
    print("  测试 1: 算术表达式文法 — LR(1) 自动机")
    print("=" * 60)
    analyze_lr1(ARITHMETIC_GRAMMAR)

    print("\n\n")

    print("=" * 60)
    print("  测试 2: LR(1) 分析 'id * id + id'")
    print("=" * 60)
    analyze_lr1(ARITHMETIC_GRAMMAR, "id * id + id")

    print("\n\n")

    print("=" * 60)
    print("  测试 3: S → CC, C → cC | d (SLR(1) 冲突, LR(1) 无冲突)")
    print("=" * 60)
    analyze_lr1(LR1_DEMO_GRAMMAR)


# ═══════════════════════════════════════════════════════════════
# 9. main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        grammar_path = sys.argv[1]
        with open(grammar_path, 'r', encoding='utf-8') as f:
            grammar_text = f.read()
        input_str = sys.argv[2] if len(sys.argv) >= 3 else None
        analyze_lr1(grammar_text, input_str)
    else:
        run_builtin_tests()

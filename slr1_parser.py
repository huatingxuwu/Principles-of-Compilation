"""
SLR(1) 语法分析器
=================
在 LR(0) 自动机基础上，用 FOLLOW 集限制归约动作，消除移进-归约冲突。

依赖:
  grammar_utils  — Grammar, compute_follow, fmt_set
  lr0_automaton   — build_lr0_automaton, build_lr0_table, print_table,
                     _format_action, print_section_header,
                     Item, item_str, items_str, FrozenSet, ...

核心区别（SLR vs LR0）:
  LR(0): 归约项 A→α· 在所有输入符号上归约
  SLR(1): 归约项 A→α· 只在 FOLLOW(A) 中的符号上归约

用法:
  python slr1_parser.py                           # 交互模式
  python slr1_parser.py grammar.txt               # 从文件读取文法
  python slr1_parser.py grammar.txt "id * id"     # 指定输入串分析
"""

import sys
from typing import Dict, FrozenSet, List, Set, Tuple

from grammar_utils import Grammar, compute_follow, fmt_set
from lr0_automaton import (
    # 自动化构造
    build_lr0_automaton,
    # LR(0) Item 相关
    Item, item_str, items_str,
    # 输出工具
    print_table, _format_action, print_section_header, _print_parse_table,
)


# ═══════════════════════════════════════════════════════════════
# 1. SLR(1) 分析表构造
# ═══════════════════════════════════════════════════════════════

def build_slr1_table(
    grammar: Grammar,
    aug: Grammar,
    states: List[FrozenSet[Item]],
    gotos: Dict[Tuple[int, str], int],
    FOLLOW: Dict[str, Set[str]],
) -> Dict[Tuple[int, str], tuple]:
    """构造 SLR(1) 分析表。

    LR(0) 算法认为归约项应对所有输入符号执行归约；
    SLR(1) 限制为只在 FOLLOW(A) 上执行归约。

    返回: {(state, symbol): ('shift', dst) | ('reduce', idx) | ('acc',) | ('goto', dst) | ('conflict', ...)}
    """
    table: Dict[Tuple[int, str], tuple] = {}
    all_prods = []
    for lhs in aug.productions:
        for rhs in aug.productions[lhs]:
            all_prods.append((lhs, tuple(rhs)))

    for i, I in enumerate(states):
        for item in I:
            lhs, rhs, dot = item

            # 归约项: 点在末尾
            if dot == len(rhs):
                if lhs == aug.start:
                    table[(i, '$')] = ('acc',)
                else:
                    # SLR(1): 只在 FOLLOW(lhs) 上归约
                    prod_idx = all_prods.index((lhs, rhs))
                    for t in FOLLOW.get(lhs, set()):
                        existing = table.get((i, t))
                        if existing and existing[0] == 'shift':
                            table[(i, t)] = ('conflict', existing, ('reduce', prod_idx))
                        else:
                            table[(i, t)] = ('reduce', prod_idx)

        # GOTO 转移: 终结符→SHIFT, 非终结符→GOTO
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
# 2. SLR(1) 分析过程
# ═══════════════════════════════════════════════════════════════

def slr1_parse(
    grammar: Grammar,
    table: Dict[Tuple[int, str], tuple],
    aug: Grammar,
    all_prods: List[Tuple[str, Tuple[str, ...]]],
    input_string: str,
) -> Tuple[bool, List[str]]:
    """用 SLR(1) 分析表做表驱动语法分析。返回 (success, trace_lines)。

    与 lr0_automaton.lr0_parse_demo 结构相同，复用了 _print_parse_table。
    区别仅在使用 SLR(1) 表（而非 LR(0) 表）。
    """
    trace: List[str] = []
    tokens = input_string.strip().split()

    # 冲突预检
    conflicts = [(k, v) for k, v in table.items() if v[0] == 'conflict']
    if conflicts:
        trace.append(f"\n  ⚠ 仍有 {len(conflicts)} 个冲突（文法不是 SLR(1)）:")
        for (s, a), v in conflicts:
            trace.append(f"    状态 I{s}, 符号 '{a}': {_format_action(v[1])} vs {_format_action(v[2])}")
        trace.append(f"  → 需要 LR(1) 或 LALR(1)")
        return False, trace

    trace.append(f"  输入串: {' '.join(tokens)}")
    trace.append("")

    rows: List[Tuple[str, str, str, str, str]] = []
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
                         f"✗ 错误: ACTION[{s}, {a}] 为空"))
            _print_parse_table(rows)
            trace.append(f"\n  ✗ 分析失败: ACTION[{s}, {a}] 为空")
            return False, trace

        action = action_entry[0]

        if action == 'shift':
            _, next_state = action_entry
            rows.append((f"({step})", state_str, sym_str, remaining,
                         f"移进 s{next_state}"))
            state_stack.append(next_state)
            sym_stack.append(a)
            pos += 1

        elif action == 'reduce':
            _, prod_idx = action_entry
            lhs, rhs = all_prods[prod_idx]
            rhs_len = 0 if rhs == ('ε',) else len(rhs)
            rhs_str = ' '.join(rhs)
            rows.append((f"({step})", state_str, sym_str, remaining,
                         f"按 {lhs} → {rhs_str} 归约 r{prod_idx}"))

            for _ in range(rhs_len):
                state_stack.pop()
                sym_stack.pop()

            new_s = state_stack[-1]
            goto_entry = table.get((new_s, lhs))
            if goto_entry is None or goto_entry[0] != 'goto':
                rows.append(("", "", "", "",
                             f"✗ GOTO[{new_s}, {lhs}] 不存在"))
                _print_parse_table(rows)
                trace.append(f"\n  ✗ 分析失败: GOTO[{new_s}, {lhs}] 不存在")
                return False, trace

            _, next_state = goto_entry
            state_stack.append(next_state)
            sym_stack.append(lhs)

        elif action == 'acc':
            rows.append((f"({step})", state_str, sym_str, remaining, "接受"))
            _print_parse_table(rows)
            trace.append(f"\n  ✓ SLR(1) 分析成功！输入串是文法的句子。")
            return True, trace

    return True, trace


# ═══════════════════════════════════════════════════════════════
# 3. 顶层流水线
# ═══════════════════════════════════════════════════════════════

def analyze_slr1(grammar_text: str, input_string: str = None):
    """SLR(1) 完整流程: LR(0) 自动机 → FOLLOW → SLR(1) 表 → 分析过程。

    input_string 为 None 时只构造 SLR(1) 表并检查冲突。
    """

    # Step 1: 解析文法
    grammar = Grammar(grammar_text)
    print_section_header("第一步: 解析原文法")
    print(f"  开始符号: {grammar.start}")
    print(f"  非终结符: {sorted(grammar.non_terminals)}")
    print(f"  终结符:   {sorted(grammar.terminals)}")
    print(f"  产生式:")
    print(grammar)

    # Step 2: LR(0) 自动机（复用 lr0_automaton）
    print_section_header("第二步: 构造 LR(0) 自动机")
    states, gotos, trace = build_lr0_automaton(grammar)
    for line in trace:
        print(line)

    # Step 3: FOLLOW 集
    print_section_header("第三步: 计算 FOLLOW 集")
    FOLLOW = compute_follow(grammar)
    for nt in sorted(grammar.non_terminals):
        print(f"    FOLLOW({nt}) = {{{fmt_set(FOLLOW[nt])}}}")

    # Step 4: SLR(1) 分析表（建表一次，复用）
    print_section_header("第四步: 构造 SLR(1) 分析表")
    print(f"  SLR(1) vs LR(0): 归约动作只在 FOLLOW(A) 中的符号上执行")
    print(f"  这消除了 LR(0) 的移进-归约冲突")
    print()

    aug = grammar.augmented()
    table = build_slr1_table(grammar, aug, states, gotos, FOLLOW)

    # 准备产生式编号列表（供分析阶段用）
    all_prods = []
    for lhs in aug.productions:
        for rhs in aug.productions[lhs]:
            all_prods.append((lhs, tuple(rhs)))

    # 打印分析表
    print(f"  SLR(1) 分析表 (ACTION / GOTO):")
    print_table(table, grammar, len(states))

    # 冲突检查
    conflicts = [(k, v) for k, v in table.items() if v[0] == 'conflict']
    if conflicts:
        print(f"\n  ⚠ 有 {len(conflicts)} 个冲突（文法不是 SLR(1)）:")
        for (s, a), v in conflicts:
            print(f"    状态 I{s}, 符号 '{a}': {_format_action(v[1])} vs {_format_action(v[2])}")
    else:
        print(f"\n  ✓ SLR(1) 分析表无冲突！")

    # Step 5: 分析过程（若有输入串）
    if input_string:
        print_section_header("第五步: SLR(1) 表驱动分析过程")
        success, parse_trace = slr1_parse(grammar, table, aug, all_prods, input_string)
        for line in parse_trace:
            print(line)
    else:
        print_section_header("第五步: SLR(1) 语法分析")
        print("  （未提供输入串，跳过分析。用法: python slr1_parser.py grammar.txt \"id * id\"）")

    return states, gotos


# ═══════════════════════════════════════════════════════════════
# 4. 测试用例
# ═══════════════════════════════════════════════════════════════

ARITHMETIC_GRAMMAR = """
S → S S + | S S * | a
"""

SIMPLE_GRAMMAR = """
S → A B
A → a | ε
B → b | ε
"""


def run_builtin_tests():
    """运行内置测试。"""
    # 测试 1: 只建表 + 检查冲突（无输入串）
    print("=" * 60)
    print("  测试 1: 算术表达式文法 — SLR(1) 表 (无冲突)")
    print("=" * 60)
    analyze_slr1(ARITHMETIC_GRAMMAR)

    print("\n\n")

    # 测试 2: 含输入串的 SLR(1) 完整分析过程
    print("=" * 60)
    print("  测试 2: SLR(1) 分析过程 — 输入 'a a +'")
    print("=" * 60)
    analyze_slr1(ARITHMETIC_GRAMMAR, "a a * a +")


# ═══════════════════════════════════════════════════════════════
# 5. main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        grammar_path = sys.argv[1]
        with open(grammar_path, 'r', encoding='utf-8') as f:
            grammar_text = f.read()
        input_str = sys.argv[2] if len(sys.argv) >= 3 else None
        analyze_slr1(grammar_text, input_str)
    else:
        run_builtin_tests()

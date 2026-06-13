"""
练习 4.4.1 (P147) — 预测分析器设计
===================================
（1）S → 0S1 | 01           → 提取左公因子 → LL(1)
（2）S → +SS | *SS | a      → 直接 LL(1)
（3）S → S+S | SS | (S) | S* | a  → 消除左递归 → LL(1)
（4）针对（1），输入 000111 的预测分析过程

输出写入 exercise_441_output.txt
"""

import sys
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# 确保能导入同目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grammar_utils import (
    Grammar, compute_first, compute_follow,
    first_of_symbols, is_nullable, fmt_set,
)

# ═══════════════════════════════════════════════════════════════
# 复用 ll1_parser 的核心函数（直接 import 会执行 main，故手动复制）
# ═══════════════════════════════════════════════════════════════

def fmt_rhs(symbols: List[str]) -> str:
    return ' '.join(symbols) if symbols else 'ε'


def build_ll1_table(
    grammar: Grammar,
    FIRST: Dict[str, Set[str]],
    FOLLOW: Dict[str, Set[str]],
) -> Tuple[Dict[Tuple[str, str], List[str]], List[str]]:
    """构造 LL(1) 预测分析表。返回 (table, trace)。"""
    trace: List[str] = []
    table: Dict[Tuple[str, str], List[str]] = {}
    prods = grammar.productions

    trace.append("构造 LL(1) 预测分析表:")
    trace.append("")

    for nt in sorted(grammar.non_terminals):
        for ri, rhs in enumerate(prods[nt]):
            rhs_str = ' '.join(rhs)

            first_rhs = first_of_symbols(FIRST, rhs)

            for a in sorted(first_rhs - {'ε'}):
                key = (nt, a)
                if key in table:
                    trace.append(
                        f"  ⚠ 冲突! M[{nt}, {a}] 已有 "
                        f"{nt}→{fmt_rhs(table[key])}, "
                        f"试图填入 {nt}→{fmt_rhs(rhs)}"
                    )
                else:
                    table[key] = rhs
                    trace.append(f"  M[{nt}, {a}] = {nt} → {fmt_rhs(rhs)}")

            if 'ε' in first_rhs:
                for b in sorted(FOLLOW[nt]):
                    key = (nt, b)
                    if key in table:
                        trace.append(
                            f"  ⚠ 冲突! M[{nt}, {b}] 已有 "
                            f"{nt}→{fmt_rhs(table[key])}, "
                            f"试图填入 {nt}→{fmt_rhs(rhs)} （ε ∈ FIRST）"
                        )
                    else:
                        table[key] = rhs
                        trace.append(
                            f"  M[{nt}, {b}] = {nt} → {fmt_rhs(rhs)} "
                            f"（ε ∈ FIRST, {b} ∈ FOLLOW({nt})）"
                        )

    # 格式化输出表格
    trace.append(f"\n  预测分析表:")
    all_terminals = sorted(grammar.terminals | {'$'})
    header = "  " + "".join(f"{t:>10}" for t in all_terminals)
    trace.append(header)
    trace.append("  " + "-" * (10 * len(all_terminals)))
    for nt in sorted(grammar.non_terminals):
        row = f"  {nt:<8}"
        for a in all_terminals:
            key = (nt, a)
            if key in table:
                cell = f"{nt}→{fmt_rhs(table[key])}"
                row += f"{cell:>10}"
            else:
                row += f"{'':>10}"
        trace.append(row)

    return table, trace


def predictive_parse(
    grammar: Grammar,
    table: Dict[Tuple[str, str], List[str]],
    input_tokens: List[str],
) -> Tuple[bool, List[str]]:
    """表驱动的预测语法分析。返回 (success, trace)。"""
    trace: List[str] = []
    trace.append(f"输入串: {' '.join(input_tokens)}")
    trace.append("")

    stack: List[str] = ['$', grammar.start]
    tokens = input_tokens + ['$']
    pos = 0

    # 表头
    trace.append(
        f"{'步骤':<5} {'已匹配':<20} {'栈':<30} {'输入':<20} {'动作'}"
    )
    trace.append("-" * 100)

    step = 0
    matched: List[str] = []

    while True:
        step += 1
        X = stack[-1]
        a = tokens[pos]

        matched_str = ''.join(matched)
        stack_str = ' '.join(stack)
        input_str = ' '.join(tokens[pos:])

        # 情况1: 栈顶是终结符（或 $）
        if X in grammar.terminals or X == '$':
            if X == a:
                action = f"匹配 '{X}'"
                trace.append(
                    f"{step:<5} {matched_str:<20} {stack_str:<30} {input_str:<20} {action}"
                )
                stack.pop()
                if X != '$':
                    matched.append(X)
                pos += 1
                if X == '$':
                    trace.append(f"\n✓ 分析成功！输入串是文法的句子。")
                    return True, trace
            else:
                action = f"错误: 期待 '{X}'，遇到 '{a}'"
                trace.append(
                    f"{step:<5} {matched_str:<20} {stack_str:<30} {input_str:<20} {action}"
                )
                return False, trace

        # 情况2: 栈顶是非终结符
        elif X in grammar.non_terminals:
            key = (X, a)
            if key in table:
                rhs = table[key]
                rhs_str = ' '.join(rhs)
                action = f"输出 {X} → {rhs_str}"
                trace.append(
                    f"{step:<5} {matched_str:<20} {stack_str:<30} {input_str:<20} {action}"
                )
                stack.pop()
                if rhs != ['ε']:
                    for sym in reversed(rhs):
                        stack.append(sym)
            else:
                action = f"错误: M[{X}, {a}] 为空"
                trace.append(
                    f"{step:<5} {matched_str:<20} {stack_str:<30} {input_str:<20} {action}"
                )
                return False, trace
        else:
            action = f"错误: 未知符号 '{X}'"
            trace.append(
                f"{step:<5} {matched_str:<20} {stack_str:<30} {input_str:<20} {action}"
            )
            return False, trace


def compute_first_sets_trace(grammar: Grammar) -> Tuple[Dict[str, Set[str]], List[str]]:
    """带 trace 的 FIRST 计算。"""
    trace: List[str] = []

    trace.append("计算 FIRST 集:")
    trace.append(f"  终结符 FIRST:")
    for t in sorted(grammar.terminals):
        trace.append(f"    FIRST({t}) = {{{t}}}")
    trace.append("")

    FIRST = compute_first(grammar)

    trace.append("  非终结符 FIRST:")
    for nt in sorted(grammar.non_terminals):
        trace.append(f"    FIRST({nt}) = {{{fmt_set(FIRST[nt])}}}")

    return FIRST, trace


def compute_follow_sets_trace(grammar: Grammar, FIRST: Dict[str, Set[str]]) -> Tuple[Dict[str, Set[str]], List[str]]:
    """带 trace 的 FOLLOW 计算。"""
    trace: List[str] = []
    FOLLOW = compute_follow(grammar, FIRST)

    trace.append("计算 FOLLOW 集:")
    for nt in sorted(grammar.non_terminals):
        trace.append(f"    FOLLOW({nt}) = {{{fmt_set(FOLLOW[nt])}}}")

    return FOLLOW, trace


# ═══════════════════════════════════════════════════════════════
# 输出文件
# ═══════════════════════════════════════════════════════════════

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "exercise_441_output.txt")
out_lines: List[str] = []


def out(s: str = ""):
    out_lines.append(s)


# ═══════════════════════════════════════════════════════════════
# 第（1）题: S → 0S1 | 01
# ═══════════════════════════════════════════════════════════════

def problem_1():
    out("=" * 70)
    out("  （1）S → 0S1 | 01")
    out("=" * 70)
    out()

    out("【步骤 1】检查是否需要提取左公因子 / 消除左递归")
    out()
    out("  产生式: S → 0S1 | 01")
    out("  FIRST(0S1) = {0}, FIRST(01) = {0}")
    out("  两个候选式都以 0 开头 → 需要提取左公因子！")
    out()
    out("  提取左公因子 '0':")
    out("    原: S → 0S1 | 01")
    out("    新: S  → 0 X")
    out("        X  → S1 | 1")
    out()

    grammar_text = """S → 0 X
X → S 1 | 1
"""
    grammar = Grammar(grammar_text)
    out("  变换后文法:")
    out(f"    开始符号: {grammar.start}")
    out(f"    非终结符: {sorted(grammar.non_terminals)}")
    out(f"    终结符:   {sorted(grammar.terminals)}")
    out(str(grammar))
    out()

    # FIRST
    out("【步骤 2】计算 FIRST 集")
    FIRST, first_trace = compute_first_sets_trace(grammar)
    for line in first_trace:
        out(line)
    out()

    # FOLLOW
    out("【步骤 3】计算 FOLLOW 集")
    out("  FOLLOW(S) ← 开始符号，加入 {$}")
    out("  在 X → S 1 中，S 后面跟着 '1'，所以 1 ∈ FOLLOW(S)")
    out("  在 S → 0 X 中，X 在末尾，FOLLOW(S) ⊆ FOLLOW(X)")
    out("  在 X → S 1 中，X 在末尾（该产生式最后），FOLLOW(X) = FOLLOW(X)（无新信息）")
    out("  ... 但更准确地说 FOLLOW(X) = FOLLOW(S) = {1, $}")
    out()
    FOLLOW, follow_trace = compute_follow_sets_trace(grammar, FIRST)
    for line in follow_trace:
        out(line)
    out()

    # LL(1) table
    out("【步骤 4】构造 LL(1) 预测分析表")
    table, table_trace = build_ll1_table(grammar, FIRST, FOLLOW)
    for line in table_trace:
        out(line)
    out()

    return grammar, table


# ═══════════════════════════════════════════════════════════════
# 第（2）题: S → +SS | *SS | a
# ═══════════════════════════════════════════════════════════════

def problem_2():
    out()
    out("=" * 70)
    out("  （2）S → +SS | *SS | a")
    out("=" * 70)
    out()

    out("【步骤 1】检查是否需要提取左公因子 / 消除左递归")
    out()
    out("  产生式: S → +SS | *SS | a")
    out("  FIRST(+SS) = {+}, FIRST(*SS) = {*}, FIRST(a) = {a}")
    out("  三个候选式的 FIRST 互不相交 → 无需任何变换，直接 LL(1)！")
    out()

    grammar_text = """S → + S S | * S S | a
"""
    grammar = Grammar(grammar_text)
    out("  原文法:")
    out(str(grammar))
    out()

    # FIRST
    out("【步骤 2】计算 FIRST 集")
    FIRST, first_trace = compute_first_sets_trace(grammar)
    for line in first_trace:
        out(line)
    out()

    # FOLLOW
    out("【步骤 3】计算 FOLLOW 集")
    out("  FOLLOW(S) ← 开始符号，加入 {$}")
    out("  在 S → + S S 中，第一个 S 后面跟着 S，所以 FIRST(S) ⊆ FOLLOW(S)")
    out("  类似地，其他产生式中 S 出现在非末尾位置时也如此。")
    out()
    FOLLOW, follow_trace = compute_follow_sets_trace(grammar, FIRST)
    for line in follow_trace:
        out(line)
    out()

    # LL(1) table
    out("【步骤 4】构造 LL(1) 预测分析表")
    table, table_trace = build_ll1_table(grammar, FIRST, FOLLOW)
    for line in table_trace:
        out(line)
    out()

    return grammar, table


# ═══════════════════════════════════════════════════════════════
# 第（3）题: S → S+S | SS | (S) | S* | a
# ═══════════════════════════════════════════════════════════════

def problem_3():
    out()
    out("=" * 70)
    out("  （3）S → S+S | SS | (S) | S* | a")
    out("=" * 70)
    out()

    out("【步骤 1】检查是否需要提取左公因子 / 消除左递归")
    out()
    out("  存在直接左递归的产生式: S → S+S, S → SS, S → S*")
    out("  不含左递归的产生式:    S → (S), S → a")
    out()
    out("  消除左递归 (标准技法):")
    out("    将所有产生式分为两类:")
    out("      A → Aα₁ | Aα₂ | ... | Aαₘ      (左递归)")
    out("      A → β₁ | β₂ | ... | βₙ          (非左递归)")
    out("    变换为:")
    out("      A  → β₁A' | β₂A' | ... | βₙA'")
    out("      A' → α₁A' | α₂A' | ... | αₘA' | ε")
    out()
    out("    此处:")
    out("      α₁ = +S,  α₂ = S,  α₃ = *")
    out("      β₁ = (S),  β₂ = a")
    out()
    out("    变换后:")
    out("      S  → (S) S' | a S'")
    out("      S' → +S S' | S S' | * S' | ε")
    out()

    grammar_text = """S → ( S ) S' | a S'
S' → + S S' | S S' | * S' | ε
"""
    grammar = Grammar(grammar_text)
    out("  变换后文法:")
    out(f"    开始符号: {grammar.start}")
    out(f"    非终结符: {sorted(grammar.non_terminals)}")
    out(f"    终结符:   {sorted(grammar.terminals)}")
    out(str(grammar))
    out()

    # FIRST
    out("【步骤 2】计算 FIRST 集")
    FIRST, first_trace = compute_first_sets_trace(grammar)
    for line in first_trace:
        out(line)
    out()

    # FOLLOW
    out("【步骤 3】计算 FOLLOW 集")
    FOLLOW, follow_trace = compute_follow_sets_trace(grammar, FIRST)
    for line in follow_trace:
        out(line)
    out()

    # LL(1) table
    out("【步骤 4】构造 LL(1) 预测分析表")
    table, table_trace = build_ll1_table(grammar, FIRST, FOLLOW)
    for line in table_trace:
        out(line)
    out()

    return grammar, table


# ═══════════════════════════════════════════════════════════════
# 第（4）题: 对（1），输入 000111 的预测分析过程
# ═══════════════════════════════════════════════════════════════

def problem_4(grammar, table):
    out()
    out("=" * 70)
    out("  （4）针对（1）的文法，输入 000111 的预测分析过程")
    out("=" * 70)
    out()

    out("  文法 (左公因子提取后):")
    out("    S → 0 X")
    out("    X → S 1 | 1")
    out()
    out("  输入串: 0 0 0 1 1 1")
    out()

    tokens = list("000111")
    success, parse_trace = predictive_parse(grammar, table, tokens)
    for line in parse_trace:
        out(line)
    out()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    out("练习 4.4.1 (P147) 预测分析器设计")
    out("=" * 70)
    out()

    grammar1, table1 = problem_1()
    grammar2, table2 = problem_2()
    grammar3, table3 = problem_3()
    problem_4(grammar1, table1)

    # 写入文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

    print(f"结果已写入: {OUTPUT_FILE}")
    # 也打印到屏幕
    print('\n'.join(out_lines))


if __name__ == '__main__':
    main()

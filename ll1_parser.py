"""
FIRST / FOLLOW 集计算 → LL(1) 预测分析表 → 表驱动语法分析
==========================================================
完整流水线，含详细中间过程输出:
  1. 解析上下文无关文法
  2. 计算 FIRST 集（迭代不动点）
  3. 计算 FOLLOW 集（迭代不动点）
  4. 构造 LL(1) 预测分析表
  5. 表驱动的预测语法分析器

文法输入格式:
  每行一条产生式（或 '|' 分隔的多个右部）:
    非终结符 → 右部1 | 右部2 | ...
  符号之间用空格分隔。
  用 ε 表示空串。
  第一条产生式的左部为开始符号。

用法:
  python ll1_parser.py                              # 交互模式
  python ll1_parser.py grammar.txt                  # 从文件读取文法
  python ll1_parser.py grammar.txt "id + id * id"   # 指定输入串分析
"""

import sys
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from grammar_utils import (
    Grammar, compute_first, compute_follow,
    first_of_symbols, is_nullable, fmt_set,
    print_section_header,
)


# ═══════════════════════════════════════════════════════════════
# 1. 文法解析
# ═══════════════════════════════════════════════════════════════

# Grammar 类已提取到 grammar_utils.Grammar


# ═══════════════════════════════════════════════════════════════
# 2. FIRST 集计算
# ═══════════════════════════════════════════════════════════════

def compute_first_sets(grammar: Grammar) -> Tuple[Dict[str, Set[str]], List[str]]:
    """计算所有符号的 FIRST 集。返回 (FIRST, trace)。"""
    trace: List[str] = []

    trace.append("===== 计算 FIRST 集 =====")
    trace.append("")
    for t in grammar.terminals:
        trace.append(f"  FIRST({t}) = {{{t}}}  （终结符，直接得到）")
    trace.append("")

    # 复现迭代过程用于展示 trace
    FIRST_trace: Dict[str, Set[str]] = defaultdict(set)
    for t in grammar.terminals:
        FIRST_trace[t] = {t}
    for nt in grammar.non_terminals:
        FIRST_trace[nt] = set()

    round_num = 0
    while True:
        changed = False
        round_num += 1
        trace.append(f"  --- 第 {round_num} 轮迭代 ---")

        for nt in grammar.non_terminals:
            for rhs in grammar.productions[nt]:
                before = len(FIRST_trace[nt])
                if rhs == ['ε']:
                    FIRST_trace[nt].add('ε')
                else:
                    all_nullable = True
                    for sym in rhs:
                        if sym == 'ε':
                            break
                        FIRST_trace[nt] |= (FIRST_trace.get(sym, set()) - {'ε'})
                        if 'ε' not in FIRST_trace.get(sym, set()):
                            all_nullable = False
                            break
                    if all_nullable:
                        FIRST_trace[nt].add('ε')

                after = len(FIRST_trace[nt])
                if after > before:
                    changed = True
                    rhs_str = ' '.join(rhs)
                    trace.append(
                        f"    {nt} → {rhs_str}: FIRST({nt}) 增加 → "
                        f"{{{fmt_set(FIRST_trace[nt])}}}"
                    )
        if not changed:
            break

    # 使用共享版得到最终结果（确保正确性）
    FIRST = compute_first(grammar)

    trace.append(f"\n  ◆ 最终 FIRST 集:")
    for nt in sorted(grammar.non_terminals):
        trace.append(f"    FIRST({nt}) = {{{fmt_set(FIRST[nt])}}}")

    return FIRST, trace


# ═══════════════════════════════════════════════════════════════
# 3. FOLLOW 集计算
# ═══════════════════════════════════════════════════════════════

def compute_follow_sets(
    grammar: Grammar, FIRST: Dict[str, Set[str]]
) -> Tuple[Dict[str, Set[str]], List[str]]:
    """计算 FOLLOW 集。返回 (FOLLOW, trace)。"""
    trace: List[str] = []
    prods = grammar.productions

    trace.append("\n===== 计算 FOLLOW 集 =====")
    trace.append("")
    trace.append(f"  FOLLOW({grammar.start}) 加入 '$'  （开始符号规则）")

    # 扫描出现位置（复用共享版的扫描逻辑，trace 复现迭代）
    occurrences: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    for nt in grammar.non_terminals:
        for lhs, rhs_list in prods.items():
            for ri, rhs in enumerate(rhs_list):
                for pos, sym in enumerate(rhs):
                    if sym == nt:
                        occurrences[nt].append((lhs, ri, pos))

    # 复现迭代过程用于展示 trace
    FOLLOW_trace: Dict[str, Set[str]] = defaultdict(set)
    FOLLOW_trace[grammar.start].add('$')

    round_num = 0
    while True:
        changed = False
        round_num += 1
        trace.append(f"\n  --- 第 {round_num} 轮迭代 ---")

        for nt in sorted(grammar.non_terminals):
            for lhs, ri, pos in occurrences[nt]:
                rhs = prods[lhs][ri]
                before = len(FOLLOW_trace[nt])
                beta = rhs[pos + 1:]

                if beta:
                    first_beta = first_of_symbols(FIRST, beta)
                    FOLLOW_trace[nt] |= (first_beta - {'ε'})
                if not beta or is_nullable(FIRST, beta):
                    FOLLOW_trace[nt] |= FOLLOW_trace[lhs]

                after = len(FOLLOW_trace[nt])
                if after > before:
                    changed = True
                    rhs_str = ' '.join(rhs)
                    trace.append(
                        f"    {lhs} → {rhs_str}: 位置 {pos} 的 {nt} 后跟 "
                        f"{' '.join(beta) if beta else 'ε（末尾）'}, "
                        f"FOLLOW({nt}) 增加 → {{{fmt_set(FOLLOW_trace[nt])}}}"
                    )
        if not changed:
            break

    # 使用共享版得到最终结果（确保正确性）
    FOLLOW = compute_follow(grammar, FIRST)

    trace.append(f"\n  ◆ 最终 FOLLOW 集:")
    for nt in sorted(grammar.non_terminals):
        trace.append(f"    FOLLOW({nt}) = {{{fmt_set(FOLLOW[nt])}}}")

    return FOLLOW, trace


# first_of_symbols / is_nullable 已提取到 grammar_utils


# ═══════════════════════════════════════════════════════════════
# 4. LL(1) 预测分析表
# ═══════════════════════════════════════════════════════════════

def build_ll1_table(
    grammar: Grammar,
    FIRST: Dict[str, Set[str]],
    FOLLOW: Dict[str, Set[str]],
) -> Tuple[Dict[Tuple[str, str], List[str]], List[str]]:
    """构造 LL(1) 预测分析表。返回 (table, trace)。"""
    trace: List[str] = []
    table: Dict[Tuple[str, str], List[str]] = {}
    prods = grammar.productions

    trace.append("\n===== 构造 LL(1) 预测分析表 =====")
    trace.append("")

    for nt in sorted(grammar.non_terminals):
        for ri, rhs in enumerate(prods[nt]):
            rhs_str = ' '.join(rhs)
            trace.append(f"  处理产生式: {nt} → {rhs_str}")

            # 计算 rhs 的 FIRST
            first_rhs = first_of_symbols(FIRST, rhs)
            trace.append(f"    FIRST({rhs_str}) = {{{fmt_set(first_rhs)}}}")

            # 规则1: 对 FIRST(rhs) 中的每个非 ε 终结符
            for a in sorted(first_rhs - {'ε'}):
                key = (nt, a)
                if key in table:
                    trace.append(
                        f"    ⚠ 冲突! M[{nt}, {a}] 已有 {fmt_rhs(table[key])}, "
                        f"试图填入 {fmt_rhs(rhs)}"
                    )
                else:
                    table[key] = rhs
                    trace.append(f"    M[{nt}, {a}] = {nt} → {fmt_rhs(rhs)}")

            # 规则2: 如果 ε ∈ FIRST(rhs)
            if 'ε' in first_rhs:
                for b in sorted(FOLLOW[nt]):
                    key = (nt, b)
                    if key in table:
                        trace.append(
                            f"    ⚠ 冲突! M[{nt}, {b}] 已有 {fmt_rhs(table[key])}, "
                            f"试图填入 {fmt_rhs(rhs)} （因 ε ∈ FIRST）"
                        )
                    else:
                        table[key] = rhs
                        trace.append(
                            f"    M[{nt}, {b}] = {nt} → {fmt_rhs(rhs)} "
                            f"（ε ∈ FIRST, {b} ∈ FOLLOW({nt})）"
                        )

    # 格式化输出表格
    trace.append(f"\n  ◆ 预测分析表:")
    all_terminals = sorted(grammar.terminals | {'$'})
    # 表头
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


# ═══════════════════════════════════════════════════════════════
# 5. 表驱动的预测语法分析器
# ═══════════════════════════════════════════════════════════════

def predictive_parse(
    grammar: Grammar,
    table: Dict[Tuple[str, str], List[str]],
    input_tokens: List[str],
) -> Tuple[bool, List[str]]:
    """表驱动的预测语法分析。返回 (success, trace)。"""
    trace: List[str] = []
    trace.append("\n===== 表驱动预测语法分析 =====")
    trace.append(f"  输入串: {' '.join(input_tokens)}")
    trace.append("")

    # 初始化
    stack: List[str] = ['$', grammar.start]
    tokens = input_tokens + ['$']
    pos = 0

    trace.append(f"  初始栈: [{', '.join(stack)}]")
    trace.append(f"  输入: {' '.join(tokens[pos:])}")
    trace.append("")

    step = 0
    while True:
        step += 1
        X = stack[-1]
        a = tokens[pos]

        trace.append(
            f"  步骤 {step}: 栈顶={X}, 当前输入={a}, "
            f"栈=[{', '.join(stack)}]"
        )

        # 情况1: 栈顶是终结符（或 $）
        if X in grammar.terminals or X == '$':
            if X == a:
                stack.pop()
                pos += 1
                trace.append(f"    匹配终结符 '{X}'，弹出栈，输入指针前进")
                if X == '$':
                    trace.append(f"\n  ✓ 分析成功！输入串是文法的句子。")
                    return True, trace
            else:
                trace.append(
                    f"    ✗ 错误: 期待 '{X}'，但遇到 '{a}'"
                )
                return False, trace

        # 情况2: 栈顶是非终结符
        elif X in grammar.non_terminals:
            key = (X, a)
            if key in table:
                rhs = table[key]
                rhs_str = ' '.join(rhs)
                stack.pop()
                # 将产生式右部逆序压栈（跳过 ε）
                if rhs != ['ε']:
                    for sym in reversed(rhs):
                        stack.append(sym)
                trace.append(
                    f"    应用产生式: {X} → {rhs_str}"
                )
                trace.append(f"    栈更新为: [{', '.join(stack)}]")
            else:
                # 尝试输出更有用的错误信息
                expected = []
                for (nt, t), prod in table.items():
                    if nt == X:
                        expected.append(t)
                trace.append(
                    f"    ✗ 错误: M[{X}, {a}] 为空，"
                    f"期待 {expected if expected else '无可用产生式'}"
                )
                return False, trace

        else:
            trace.append(f"    ✗ 错误: 未知符号 '{X}'")
            return False, trace


# ═══════════════════════════════════════════════════════════════
# 6. 辅助函数
# ═══════════════════════════════════════════════════════════════

# fmt_set 已提取到 grammar_utils


# print_section_header 已提取到 grammar_utils.print_section_header


def fmt_rhs(symbols: List[str]) -> str:
    return ' '.join(symbols) if symbols else 'ε'


# print_section_header 已提取到 grammar_utils.print_section_header


# ═══════════════════════════════════════════════════════════════
# 7. 顶层流水线
# ═══════════════════════════════════════════════════════════════

def analyze_grammar(grammar_text: str, input_string: str = None):
    """完整流水线: 文法解析 → FIRST → FOLLOW → 分析表 → 语法分析。"""

    # Step 1: 解析文法
    grammar = Grammar(grammar_text)
    print_section_header("第一步: 解析文法")
    print(f"  开始符号: {grammar.start}")
    print(f"  非终结符: {sorted(grammar.non_terminals)}")
    print(f"  终结符:   {sorted(grammar.terminals)}")
    print(f"  产生式:")
    print(grammar)

    # Step 2: FIRST 集
    print_section_header("第二步: 计算 FIRST 集（不动点迭代）")
    FIRST, first_trace = compute_first_sets(grammar)
    for line in first_trace:
        print(line)

    # Step 3: FOLLOW 集
    print_section_header("第三步: 计算 FOLLOW 集（不动点迭代）")
    FOLLOW, follow_trace = compute_follow_sets(grammar, FIRST)
    for line in follow_trace:
        print(line)

    # Step 4: LL(1) 分析表
    print_section_header("第四步: 构造 LL(1) 预测分析表")
    table, table_trace = build_ll1_table(grammar, FIRST, FOLLOW)
    for line in table_trace:
        print(line)

    # Step 5: 表驱动语法分析
    if input_string:
        tokens = input_string.strip().split()
        print_section_header("第五步: 表驱动预测语法分析")
        success, parse_trace = predictive_parse(grammar, table, tokens)
        for line in parse_trace:
            print(line)
    else:
        print_section_header("第五步: 表驱动预测语法分析")
        print("  （未提供输入串，跳过分析。用法: python ll1_parser.py grammar.txt \"id + id\"）")

    return FIRST, FOLLOW, table


# ═══════════════════════════════════════════════════════════════
# 8. 内置测试文法
# ═══════════════════════════════════════════════════════════════

ARITHMETIC_GRAMMAR = """
E  → T E'
E' → + T E' | ε
T  → F T'
T' → * F T' | ε
F  → ( E ) | id
"""

SIMPLER_GRAMMAR = """
S → A B
A → a | ε
B → b | ε
"""


def run_builtin_tests():
    print("=" * 60)
    print("  测试 1: 经典算术表达式文法")
    print("=" * 60)
    analyze_grammar(ARITHMETIC_GRAMMAR, "id + id * id")

    print("\n\n")
    print("=" * 60)
    print("  测试 2: 含 ε 的简单文法")
    print("=" * 60)
    analyze_grammar(SIMPLER_GRAMMAR, "a b")


# ═══════════════════════════════════════════════════════════════
# 9. main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        grammar_path = sys.argv[1]
        with open(grammar_path, 'r', encoding='utf-8') as f:
            grammar_text = f.read()
        input_str = sys.argv[2] if len(sys.argv) >= 3 else None
        analyze_grammar(grammar_text, input_str)
    else:
        run_builtin_tests()

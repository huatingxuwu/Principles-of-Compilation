"""
LR(0) 自动机构造
================
从上下文无关文法出发，构造完整的 LR(0) 自动机:
  1. 增广文法 (S' → S)
  2. LR(0) 项        — A → α·β
  3. CLOSURE 闭包    — 预测可能用到的产生式
  4. GOTO 转移       — 读入一个符号后到达的新状态
  5. 规范项集族       — 自动机的所有状态

输出: 每个状态的 LR(0) 项集 + 状态间的 GOTO 转移。

文法输入格式（同 ll1_parser.py）:
  非终结符 → 右部1 | 右部2 | ...
  符号之间用空格分隔，ε 表示空串。
  第一条产生式的左部为开始符号。

用法:
  python lr0_automaton.py                           # 交互模式
  python lr0_automaton.py grammar.txt               # 从文件读取文法
"""

import sys
from collections import defaultdict, deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from grammar_utils import Grammar, print_section_header


# ═══════════════════════════════════════════════════════════════
# 1. 文法解析 → 已提取到 grammar_utils.Grammar
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# 2. LR(0) 项 与 自动机核心算法
# ═══════════════════════════════════════════════════════════════

# LR(0) 项: (lhs, rhs_tuple, dot_pos)
# rhs_tuple 是 tuple 以支持哈希（放入 set / frozenset）
Item = Tuple[str, Tuple[str, ...], int]


def item_str(item: Item) -> str:
    """格式化一个 LR(0) 项: A → α·β。"""
    lhs, rhs, dot = item
    rhs_list = list(rhs)
    rhs_list.insert(dot, '·')
    return f"{lhs} → {' '.join(rhs_list)}"


def items_str(items: FrozenSet[Item] | Set[Item]) -> str:
    """格式化一个项集。"""
    return '{' + ',  '.join(item_str(it) for it in sorted(items)) + '}'


# ═══════════════════════════════════════════════════════════════

def closure(
    items: FrozenSet[Item],
    grammar: Grammar,
    trace: Optional[List[str]] = None,
) -> FrozenSet[Item]:
    """计算 LR(0) 项集的 CLOSURE 闭包。

    规则: 若 A → α·Bβ ∈ closure 且 B → γ 是产生式,
         则 B → ·γ 也加入 closure。
    """
    closure_set = set(items)
    worklist = list(items)

    while worklist:
        item = worklist.pop()
        _lhs, rhs, dot = item
        if dot >= len(rhs):
            continue  # 点在末尾，无后继符号
        next_sym = rhs[dot]
        if next_sym in grammar.non_terminals:
            for prod_rhs in grammar.productions[next_sym]:
                new_item = (next_sym, tuple(prod_rhs), 0)
                if new_item not in closure_set:
                    closure_set.add(new_item)
                    worklist.append(new_item)
                    if trace is not None:
                        trace.append(
                            f"      由 {item_str(item)} 预测: 加入 {item_str(new_item)}"
                        )

    return frozenset(closure_set)


def goto(
    items: FrozenSet[Item],
    symbol: str,
    grammar: Grammar,
    trace: Optional[List[str]] = None,
) -> FrozenSet[Item]:
    """计算 GOTO(I, X): 将 I 中所有 A → α·Xβ 的点右移一位，再求闭包。

    返回空集表示该符号下无合法转移。
    """
    kernel = set()
    for item in items:
        lhs, rhs, dot = item
        if dot < len(rhs) and rhs[dot] == symbol:
            kernel.add((lhs, rhs, dot + 1))

    if not kernel:
        return frozenset()

    if trace is not None:
        trace.append(f"      kernel 项 (点右移过 '{symbol}'):")
        for it in sorted(kernel):
            trace.append(f"        {item_str(it)}")

    return closure(frozenset(kernel), grammar, trace)


# ═══════════════════════════════════════════════════════════════

def build_lr0_automaton(
    grammar: Grammar,
) -> Tuple[
    List[FrozenSet[Item]],           # 状态列表 (每个状态是一个项集)
    Dict[Tuple[int, str], int],      # GOTO 转移: (state_id, symbol) → next_state
    List[str],                       # trace 输出
]:
    """构造 LR(0) 规范项集族 (canonical collection)。

    返回: (states, gotos, trace)
    """
    trace: List[str] = []

    # ── 增广文法 ──
    aug = grammar.augmented()
    aug_start = aug.start
    trace.append(f"增广文法: 添加 {aug_start} → {grammar.start}")
    trace.append("")

    # ── 初始项集 I₀ ──
    start_item: Item = (aug_start, (grammar.start,), 0)
    trace.append(f"初始项: {item_str(start_item)}")

    I0_kernel = frozenset({start_item})
    trace.append(f"\n计算 CLOSURE({{{item_str(start_item)}}}):")
    I0 = closure(I0_kernel, aug, trace)
    trace.append(f"  → I₀ = {items_str(I0)}")
    trace.append("")

    # ── 规范项集族 ──
    states: List[FrozenSet[Item]] = [I0]
    gotos: Dict[Tuple[int, str], int] = {}

    # 收集所有文法符号 (终结符 + 非终结符)
    all_symbols = sorted(aug.terminals | aug.non_terminals)

    worklist = deque([0])  # 状态 id 的工作列表
    round_num = 0

    while worklist:
        state_id = worklist.popleft()
        I = states[state_id]
        round_num += 1

        trace.append(f"── 处理状态 I{state_id} ──")
        trace.append(f"  项集: {items_str(I)}")
        trace.append("")

        for X in all_symbols:
            # 先检查 I 中是否有 A → α·Xβ 形式的项
            has_item_with_X = any(
                dot < len(rhs) and rhs[dot] == X
                for (_, rhs, dot) in I
            )
            if not has_item_with_X:
                continue

            trace.append(f"  GOTO(I{state_id}, '{X}'):")

            J = goto(I, X, aug, trace)

            if not J:
                trace.append(f"    → 空集，无转移")
                continue

            # 检查是否已存在
            try:
                existing_id = states.index(J)
                trace.append(f"    → 闭包后项集 = I{existing_id}（已存在）")
                gotos[(state_id, X)] = existing_id
            except ValueError:
                new_id = len(states)
                states.append(J)
                gotos[(state_id, X)] = new_id
                worklist.append(new_id)
                trace.append(f"    → 新状态 I{new_id} = {items_str(J)}")

            trace.append("")

    # ── 汇总 ──
    trace.append("=" * 55)
    trace.append("LR(0) 自动机汇总")
    trace.append("=" * 55)
    for i, I in enumerate(states):
        trace.append(f"\n  状态 I{i}:")
        for it in sorted(I):
            marker = ""
            # 归约项: 点在末尾
            if it[2] == len(it[1]):
                marker = "  ← 归约项 (可归约)"
            # 接受项: S' → S·
            if it[0] == aug_start and it[2] == len(it[1]):
                marker = "  ← 接受项 (ACC)"
            trace.append(f"    {item_str(it)}{marker}")

        # 该状态的出边
        out_edges = [(sym, dst) for (src, sym), dst in gotos.items() if src == i]
        if out_edges:
            trace.append(f"    出边:")
            for sym, dst in sorted(out_edges, key=lambda x: x[0]):
                trace.append(f"      --{sym}--> I{dst}")

    return states, gotos, trace


# ═══════════════════════════════════════════════════════════════
# 3. LR(0) 分析表构造 与 分析过程演示
# ═══════════════════════════════════════════════════════════════

def build_lr0_table(
    grammar: Grammar,
    aug: Grammar,
    states: List[FrozenSet[Item]],
    gotos: Dict[Tuple[int, str], int],
) -> Dict[Tuple[int, str], Tuple[str, int]]:
    """从 LR(0) 自动机构造 ACTION / GOTO 表。

    ACTION[i, a] = ('shift', j)  |  ('reduce', prod_index)
    GOTO[i, A] = j

    为方便演示，合并成一张表:
      table[(i, sym)] = ('shift', j)      移进到状态 j
      table[(i, sym)] = ('reduce', p_idx)  用第 p_idx 条产生式归约
      table[(i, sym)] = ('acc',)           接受
      table[(i, sym)] = ('goto', j)        非终结符的 GOTO
    """
    table: Dict[Tuple[int, str], tuple] = {}
    # 给所有产生式编号
    all_prods = []
    for lhs in aug.productions:
        for rhs in aug.productions[lhs]:
            all_prods.append((lhs, tuple(rhs)))

    for i, I in enumerate(states):
        for item in I:
            lhs, rhs, dot = item

            # 归约项: 点在末尾
            if dot == len(rhs):
                # 接受项: S' → S·
                if lhs == aug.start:
                    table[(i, '$')] = ('acc',)
                else:
                    # LR(0): 对所有输入符号归约 (无前瞻)
                    prod_idx = all_prods.index((lhs, rhs))
                    for t in grammar.terminals | {'$'}:
                        existing = table.get((i, t))
                        if existing and existing != ('reduce', prod_idx):
                            # 冲突!
                            table[(i, t)] = ('conflict', existing, ('reduce', prod_idx))
                        else:
                            table[(i, t)] = ('reduce', prod_idx)

        # GOTO 转移 (分两种: 终结符→SHIFT, 非终结符→GOTO)
        for (src, sym), dst in gotos.items():
            if src == i:
                if sym in grammar.non_terminals:
                    table[(i, sym)] = ('goto', dst)
                else:
                    # 终结符的 GOTO 就是 SHIFT 动作
                    existing = table.get((i, sym))
                    if existing and existing[0] == 'reduce':
                        table[(i, sym)] = ('conflict', ('shift', dst), existing)
                    else:
                        table[(i, sym)] = ('shift', dst)

    return table


def lr0_parse(
    grammar: Grammar,
    table: Dict[Tuple[int, str], tuple],
    aug: Grammar,
    all_prods: List[Tuple[str, Tuple[str, ...]]],
    input_string: str,
) -> Tuple[bool, List[str]]:
    """用 LR(0) 分析表做表驱动语法分析。返回 (success, trace_lines)。"""
    trace: List[str] = []
    tokens = input_string.strip().split()

    trace.append(f"  输入串: {' '.join(tokens)}")
    trace.append("")

    rows: List[Tuple[str, str, str, str, str]] = []
    state_stack: List[int] = [0]
    sym_stack: List[str] = ['$']
    input_tokens = tokens + ['$']
    pos = 0
    step = 0
    error_msg = ""

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
            error_msg = f"ACTION[{s}, {a}] 为空，分析失败"
            break

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
                error_msg = f"GOTO[{new_s}, {lhs}] 不存在"
                break

            _, next_state = goto_entry
            state_stack.append(next_state)
            sym_stack.append(lhs)

        elif action == 'acc':
            rows.append((f"({step})", state_str, sym_str, remaining, "接受"))
            break

        elif action == 'conflict':
            existing = action_entry[1]
            new = action_entry[2]
            rows.append((f"({step})", state_str, sym_str, remaining,
                         f"⚠ 冲突: {_format_action(existing)} vs {_format_action(new)}"))
            error_msg = (f"LR(0) 冲突! 状态 I{s} 含移进-归约冲突，"
                         f"需要 SLR(1) 或 LR(1)")
            break

    _print_parse_table(rows)

    if error_msg:
        trace.append(f"\n  ✗ {error_msg}")
        return False, trace
    else:
        trace.append(f"\n  ✓ LR(0) 分析成功！输入串是文法的句子。")
        return True, trace


def _format_action(entry) -> str:
    """格式化 ACTION 表项: ('shift', n) | ('reduce', n) | ('acc',) | ('conflict', ...)"""
    if entry[0] == 'shift':
        return f"移进 s{entry[1]}"
    elif entry[0] == 'reduce':
        return f"归约 r{entry[1]}"
    elif entry[0] == 'acc':
        return "acc"
    else:
        return str(entry)


def print_table(
    table: Dict[Tuple[int, str], tuple],
    grammar: Grammar,
    num_states: int,
):
    """格式化打印 LR(0) 分析表。"""
    all_terminals = sorted(grammar.terminals | {'$'})
    all_nonterms = sorted(grammar.non_terminals)

    # ACTION 部分
    header = "  状态 | " + " | ".join(f"{t:^8}" for t in all_terminals)
    header += " || " + " | ".join(f"{nt:^6}" for nt in all_nonterms)
    print(f"  {header}")
    sep = "  " + "-" * (len(header) - 2)
    print(f"  {sep}")

    for i in range(num_states):
        row = f"  I{i:<4}"
        for t in all_terminals:
            entry = table.get((i, t))
            if entry:
                cell = _format_action(entry)
                row += f" | {cell:<8}"
            else:
                row += f" | {'':8}"
        row += " ||"
        for nt in all_nonterms:
            entry = table.get((i, nt))
            if entry and entry[0] == 'goto':
                row += f" | {entry[1]:<6}"
            else:
                row += f" | {'':6}"
        print(f"  {row}")




# ═══════════════════════════════════════════════════════════════
# 4. 格式化输出
# ═══════════════════════════════════════════════════════════════

def _print_parse_table(rows: List[Tuple[str, str, str, str, str]]):
    """打印语法分析过程表格。"""
    if not rows:
        return

    # 计算列宽
    col_widths = [6, 10, 12, 16, 20]  # 默认最小宽度
    for row in rows:
        for i, cell in enumerate(row):
            # 粗略估算中文字符宽度（中文占2，ASCII占1）
            w = sum(2 if ord(c) > 127 else 1 for c in str(cell))
            col_widths[i] = max(col_widths[i], w + 2)

    # 表头
    headers = ["行号", "栈", "符号", "输入", "动作"]
    header_line = "|"
    for i, h in enumerate(headers):
        header_line += f" {h:<{col_widths[i]}} |"
    sep_line = "|"
    for i in range(len(headers)):
        sep_line += " " + "-" * col_widths[i] + " |"

    print(f"\n  {header_line}")
    print(f"  {sep_line}")

    for row in rows:
        line = "|"
        for i, cell in enumerate(row):
            line += f" {str(cell):<{col_widths[i]}} |"
        print(f"  {line}")


# print_section_header 已提取到 grammar_utils.print_section_header
# fmt_set 已提取到 grammar_utils.fmt_set


# ═══════════════════════════════════════════════════════════════
# 4. 顶层函数
# ═══════════════════════════════════════════════════════════════

def analyze_lr0(grammar_text: str, input_string: Optional[str] = None):
    """构造 LR(0) 自动机 → 构造 LR(0) 分析表 → 可选分析过程。

    LR(0) 无前瞻，遇到冲突时会显示冲突原因。
    需要消除冲突请使用 slr1_parser.analyze_slr1()。
    """

    # Step 1: 解析文法
    grammar = Grammar(grammar_text)
    print_section_header("第一步: 解析原文法")
    print(f"  开始符号: {grammar.start}")
    print(f"  非终结符: {sorted(grammar.non_terminals)}")
    print(f"  终结符:   {sorted(grammar.terminals)}")
    print(f"  产生式:")
    print(grammar)

    # Step 2: LR(0) 自动机
    print_section_header("第二步: 构造 LR(0) 自动机")
    states, gotos, trace = build_lr0_automaton(grammar)
    for line in trace:
        print(line)

    # Step 3: LR(0) 分析表（始终输出）
    print_section_header("第三步: 构造 LR(0) 分析表")
    aug = grammar.augmented()
    table = build_lr0_table(grammar, aug, states, gotos)

    all_prods = []
    for lhs in aug.productions:
        for rhs in aug.productions[lhs]:
            all_prods.append((lhs, tuple(rhs)))

    print(f"\n  LR(0) 分析表 (ACTION / GOTO):")
    print_table(table, grammar, len(states))

    # 冲突汇总
    conflicts = [(k, v) for k, v in table.items() if v[0] == 'conflict']
    if conflicts:
        print(f"\n  ⚠ 有 {len(conflicts)} 个冲突（LR(0) 分析表非无冲突）:")
        for (s, a), v in conflicts:
            print(f"    状态 I{s}, 符号 '{a}': {_format_action(v[1])} vs {_format_action(v[2])}")
        print(f"  → 需要 SLR(1) / LR(1) / LALR(1) 消除冲突")
    else:
        print(f"\n  ✓ LR(0) 分析表无冲突！")

    # Step 4: 分析过程（若有输入串）
    if input_string:
        print_section_header("第四步: LR(0) 表驱动分析过程")
        success, parse_trace = lr0_parse(grammar, table, aug, all_prods, input_string)
        for line in parse_trace:
            print(line)
    else:
        print_section_header("第四步: LR(0) 语法分析")
        print("  （未提供输入串，跳过分析。用法: python lr0_automaton.py grammar.txt \"a a +\"）")

    return states, gotos


# ═══════════════════════════════════════════════════════════════
# 5. 内置测试
# ═══════════════════════════════════════════════════════════════

ARITHMETIC_GRAMMAR = """
S → S S + | S S * | a
"""



def run_builtin_tests():
    tests = [
        ("算术表达式文法 — LR(0) 表 (含冲突)", ARITHMETIC_GRAMMAR),
    ]
    for desc, grammar_text in tests:
        print("=" * 60)
        print(f"  测试: {desc}")
        print("=" * 60)
        analyze_lr0(grammar_text)
        print("\n\n")

    # 额外: 带输入串的 LR(0) 完整分析过程
    print("=" * 60)
    print("  测试: LR(0) 分析过程 — 输入 'a a +'")
    print("=" * 60)
    analyze_lr0(ARITHMETIC_GRAMMAR, "a a a * +")


# ═══════════════════════════════════════════════════════════════
# 6. main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        grammar_path = sys.argv[1]
        with open(grammar_path, 'r', encoding='utf-8') as f:
            grammar_text = f.read()
        input_str = sys.argv[2] if len(sys.argv) >= 3 else None
        analyze_lr0(grammar_text, input_str)
    else:
        run_builtin_tests()

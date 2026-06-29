"""
NFA → DFA → 最小化 DFA
======================
完整流水线，含详细中间过程输出:
  1. 子集构造法                  →  DFA（展示每轮工作列表、ε-闭包、move）
  2. Hopcroft 算法               →  最小化 DFA（展示每轮划分细化）

用法:
  python regex_to_dfa.py                     # 运行内置测试
  python regex_to_dfa.py "a(b|c)*"           # 命令行传入正则
  python regex_to_dfa.py --nfa "1+a=2,2+b=3" # 直接输入 NFA 图
  python regex_to_dfa.py --nfa               # 交互式粘贴 NFA 图
"""

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple

from grammar_utils import print_section_header
from regex_to_nfa import (
    CharNode,
    ConcatNode,
    EpsilonNode,
    NFA,
    RegexNode,
    RegexParser,
    StarNode,
    StateCounter,
    UnionNode,
    describe_nfa_connections,
    parse_nfa_graph,
)


# ═══════════════════════════════════════════════════════════════
# 1. DFA 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class DFA:
    transitions: Dict[Tuple[int, str], int] = field(default_factory=dict)
    start: int = 0
    accepts: Set[int] = field(default_factory=set)
    num_states: int = 0

    def add_state(self) -> int:
        s = self.num_states
        self.num_states += 1
        return s

    @property
    def alphabet(self) -> Set[str]:
        return {sym for (_, sym) in self.transitions}

    @property
    def states(self) -> Set[int]:
        return set(range(self.num_states))


# ═══════════════════════════════════════════════════════════════
# 2. NFA → DFA (子集构造法) — 含详细过程
# ═══════════════════════════════════════════════════════════════

def nfa_to_dfa(nfa: NFA, trace: List[str]) -> DFA:
    """子集构造法，全程记录 trace。"""
    dfa = DFA()
    alphabet = nfa.alphabet
    subset_to_id: Dict[FrozenSet[int], int] = {}
    subset_labels: Dict[int, FrozenSet[int]] = {}  # id → NFA 子集

    trace.append(f"  字母表: {sorted(alphabet)}")
    trace.append("")

    # 起始子集
    start_subset = nfa.epsilon_closure({nfa.start})
    start_id = dfa.add_state()
    subset_to_id[start_subset] = start_id
    subset_labels[start_id] = start_subset
    dfa.start = start_id
    trace.append(
        f"  初始: ε-闭包({{{nfa.start}}}) = {sorted(start_subset)} → DFA 状态 {start_id}"
    )

    worklist = deque([start_subset])
    round_num = 0

    while worklist:
        round_num += 1
        current_subset = worklist.popleft()
        current_id = subset_to_id[current_subset]
        is_accept = bool(current_subset & nfa.accepts)
        if is_accept:
            dfa.accepts.add(current_id)

        label = "接受态" if is_accept else "非接受态"
        trace.append(
            f"\n  —— 第 {round_num} 轮: 处理 DFA 状态 {current_id} "
            f"(NFA 子集 {sorted(current_subset)}) [{label}] ——"
        )

        for symbol in sorted(alphabet):
            # move
            move_set: Set[int] = set()
            for s in current_subset:
                for nxt in nfa.transitions.get(s, {}).get(symbol, set()):
                    move_set.add(nxt)

            if not move_set:
                trace.append(
                    f"    move({sorted(current_subset)}, '{symbol}') = {{}} — 无迁移，跳过"
                )
                continue

            move_sorted = sorted(move_set)

            # ε-闭包
            next_subset = nfa.epsilon_closure(move_set)
            next_sorted = sorted(next_subset)

            trace.append(
                f"    move({sorted(current_subset)}, '{symbol}') = {move_sorted}"
            )
            if next_subset != frozenset(move_set):
                trace.append(
                    f"    ε-闭包({move_sorted}) = {next_sorted}"
                )

            # 是否新子集
            if next_subset not in subset_to_id:
                new_id = dfa.add_state()
                subset_to_id[next_subset] = new_id
                subset_labels[new_id] = next_subset
                worklist.append(next_subset)
                trace.append(
                    f"    → 新子集, 分配 DFA 状态 {new_id}"
                )
            else:
                existing_id = subset_to_id[next_subset]
                trace.append(
                    f"    → 已有子集, 对应 DFA 状态 {existing_id}"
                )

            target_id = subset_to_id[next_subset]
            dfa.transitions[(current_id, symbol)] = target_id
            trace.append(
                f"    DFA 迁移: 状态 {current_id} --{symbol}--> 状态 {target_id}"
            )

    trace.append(f"\n  子集构造完成: 共 {dfa.num_states} 个 DFA 状态")
    trace.append(f"  DFA 状态 → NFA 子集 对应关系:")
    for dfa_id in sorted(subset_labels.keys()):
        nfa_set = sorted(subset_labels[dfa_id])
        marker = " *接受" if dfa_id in dfa.accepts else ""
        trace.append(f"    状态 {dfa_id} ↔ NFA 子集 {nfa_set}{marker}")

    return dfa


# ═══════════════════════════════════════════════════════════════
# 3. DFA 最小化 (Hopcroft) — 含详细过程
# ═══════════════════════════════════════════════════════════════

def dfa_minimize(dfa: DFA, trace: List[str]) -> DFA:
    """Hopcroft 算法，全程记录 trace。"""
    alphabet = dfa.alphabet
    states = dfa.states

    F = states & dfa.accepts
    notF = states - F
    P: List[Set[int]] = []
    if notF:
        P.append(notF)
    if F:
        P.append(F)

    trace.append(f"\n  初始划分:")
    if notF:
        trace.append(f"    非接受态块: {sorted(notF)}")
    if F:
        trace.append(f"    接受态块:   {sorted(F)}")

    W = deque(P.copy())
    round_num = 0

    while W:
        round_num += 1
        A = W.popleft()
        trace.append(f"\n  —— 第 {round_num} 轮细化: 取块 A = {sorted(A)} ——")

        any_split = False
        for c in sorted(alphabet):
            # X = { s | δ(s, c) ∈ A }
            X: Set[int] = set()
            for s in states:
                target = dfa.transitions.get((s, c))
                if target is not None and target in A:
                    X.add(s)

            if not X:
                continue

            new_P = []
            for Y in P:
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    any_split = True
                    trace.append(
                        f"    符号 '{c}': 块 {sorted(Y)} 被分裂 → "
                        f"移入A的={sorted(inter)}, 不移动的={sorted(diff)}"
                    )
                    new_P.append(inter)
                    new_P.append(diff)
                    if Y in W:
                        W.remove(Y)
                        W.append(inter)
                        W.append(diff)
                    else:
                        if len(inter) <= len(diff):
                            W.append(inter)
                        else:
                            W.append(diff)
                else:
                    new_P.append(Y)
            P = new_P

        if not any_split:
            trace.append(f"    本轮无分裂，划分稳定。")

    trace.append(f"\n  最终等价类划分:")
    for i, block in enumerate(P):
        trace.append(f"    块 {i}: {sorted(block)}")

    # 构建最小化 DFA
    block_to_id: Dict[FrozenSet[int], int] = {}
    state_to_block: Dict[int, FrozenSet[int]] = {}

    for block in P:
        b = frozenset(block)
        block_to_id[b] = len(block_to_id)
        for s in block:
            state_to_block[s] = b

    min_dfa = DFA()
    min_dfa.num_states = len(block_to_id)
    min_dfa.start = block_to_id[state_to_block[dfa.start]]

    for acc in dfa.accepts:
        min_dfa.accepts.add(block_to_id[state_to_block[acc]])

    for (s, sym), t in dfa.transitions.items():
        src_block = block_to_id[state_to_block[s]]
        dst_block = block_to_id[state_to_block[t]]
        min_dfa.transitions[(src_block, sym)] = dst_block

    trace.append(f"\n  重新编号后的映射: 旧 DFA 状态 → 最小化 DFA 状态")
    seen = set()
    for old_state in sorted(dfa.states, key=lambda x: block_to_id[state_to_block[x]]):
        new_id = block_to_id[state_to_block[old_state]]
        if new_id not in seen:
            block_states = sorted([s for s in dfa.states
                                   if block_to_id[state_to_block[s]] == new_id])
            acc_mark = " *接受" if new_id in min_dfa.accepts else ""
            trace.append(f"    DFA 状态 {block_states} → 新状态 {new_id}{acc_mark}")
            seen.add(new_id)

    return min_dfa


# ═══════════════════════════════════════════════════════════════
# 4. DFA 输出函数
# ═══════════════════════════════════════════════════════════════

def describe_dfa_connections(dfa: DFA, title: str = "DFA"):
    """用文字描述 DFA 的所有连接。"""
    print(f"    [{title}] 状态总数: {dfa.num_states}")
    print(f"    起始态: {dfa.start}")
    print(f"    接受态: {sorted(dfa.accepts)}")
    print(f"    字母表: {sorted(dfa.alphabet)}")
    print(f"    状态间连接:")
    for (src, sym), dst in sorted(dfa.transitions.items()):
        start_mark = " (起始)" if src == dfa.start else ""
        accept_mark = " (接受)" if dst in dfa.accepts else ""
        print(f"      状态{src}{start_mark} 经过 '{sym}' 到达 状态{dst}{accept_mark}")

    # 额外: 标注哪些状态无出边
    all_srcs = {s for (s, _) in dfa.transitions}
    for s in range(dfa.num_states):
        outgoing = [(sym, dst) for (src, sym), dst in dfa.transitions.items() if src == s]
        if not outgoing:
            acc_label = " (接受)" if s in dfa.accepts else ""
            print(f"      状态{s}{acc_label} 没有任何出边（死状态/陷阱）")


# ═══════════════════════════════════════════════════════════════
# 5. 顶层流水线
# ═══════════════════════════════════════════════════════════════

def regex_to_min_dfa(pattern: str):
    """正则 → NFA → DFA → 最小化 DFA 完整流水线。"""
    print(f"\n{'#'*60}")
    print(f"#  正则表达式:  \"{pattern}\"")
    print(f"{'#'*60}")

    # ── Step 1: 解析正则 → AST ──
    parser = RegexParser(pattern)
    ast = parser.parse()
    print_section_header("第一步: 解析正则表达式 → 抽象语法树 (AST)")
    print(f"  原始模式: \"{pattern}\"")
    print(f"  AST 结构: {_describe_ast(ast, 0)}")

    # ── Step 2: AST → NFA (Thompson 构造法) ──
    print_section_header("第二步: Thompson 构造法 —— AST → NFA")
    trace_thompson: List[str] = []
    sc = StateCounter()
    nfa = ast.to_nfa(sc, trace_thompson)
    for line in trace_thompson:
        print(line)
    print(f"\n  ◆ 构造结果 — NFA 连接关系:")
    describe_nfa_connections(nfa)

    # ── Step 3: NFA → DFA (子集构造法) ──
    print_section_header("第三步: 子集构造法 —— NFA → DFA")
    trace_subset: List[str] = []
    dfa = nfa_to_dfa(nfa, trace_subset)
    for line in trace_subset:
        print(line)
    print(f"\n  ◆ 构造结果 — DFA 连接关系:")
    describe_dfa_connections(dfa, "子集构造 DFA")

    # ── Step 4: DFA 最小化 (Hopcroft) ──
    print_section_header("第四步: Hopcroft 算法 —— DFA 最小化")
    trace_hopcroft: List[str] = []
    min_dfa = dfa_minimize(dfa, trace_hopcroft)
    for line in trace_hopcroft:
        print(line)
    print(f"\n  ◆ 最终结果 — 最小化 DFA 连接关系:")
    describe_dfa_connections(min_dfa, "最小化 DFA")

    return nfa, dfa, min_dfa


def nfa_to_min_dfa(nfa: NFA):
    """从已有的 NFA 出发，执行子集构造 + 最小化（跳过正则解析和 Thompson 构造）。"""
    print(f"\n{'#'*60}")
    print(f"#  直接输入 NFA 图")
    print(f"{'#'*60}")

    print_section_header("NFA 连接关系")
    describe_nfa_connections(nfa)

    # ── Step 1: NFA → DFA (子集构造法) ──
    print_section_header("第一步: 子集构造法 —— NFA → DFA")
    trace_subset: List[str] = []
    dfa = nfa_to_dfa(nfa, trace_subset)
    for line in trace_subset:
        print(line)
    print(f"\n  ◆ 构造结果 — DFA 连接关系:")
    describe_dfa_connections(dfa, "子集构造 DFA")

    # ── Step 2: DFA 最小化 (Hopcroft) ──
    print_section_header("第二步: Hopcroft 算法 —— DFA 最小化")
    trace_hopcroft: List[str] = []
    min_dfa = dfa_minimize(dfa, trace_hopcroft)
    for line in trace_hopcroft:
        print(line)
    print(f"\n  ◆ 最终结果 — 最小化 DFA 连接关系:")
    describe_dfa_connections(min_dfa, "最小化 DFA")

    return nfa, dfa, min_dfa


def _describe_ast(node: RegexNode, depth: int) -> str:
    prefix = "  " * depth
    if isinstance(node, EpsilonNode):
        return "ε"
    elif isinstance(node, CharNode):
        return f"字符 '{node.char}'"
    elif isinstance(node, ConcatNode):
        parts = [_describe_ast(c, depth + 1) for c in node.children]
        return "连接(\n" + ",\n".join(
            f"{prefix}  {p}" for p in parts
        ) + f"\n{prefix})"
    elif isinstance(node, UnionNode):
        left = _describe_ast(node.left, depth + 1)
        right = _describe_ast(node.right, depth + 1)
        return f"并(\n{prefix}  {left},\n{prefix}  {right}\n{prefix})"
    elif isinstance(node, StarNode):
        inner = _describe_ast(node.child, depth + 1)
        return f"星号(\n{prefix}  {inner}\n{prefix})"


# ═══════════════════════════════════════════════════════════════
# 6. 测试与命令行入口
# ═══════════════════════════════════════════════════════════════

def run_all_tests():
    tests = [
        ("(a|ab)*ab",       "嵌套星 (ab)*"),
    ]
    for pattern, desc in tests:
        print(f"\n{'='*60}")
        print(f"  测试用例: {desc}")
        print(f"{'='*60}")
        regex_to_min_dfa(pattern)

    # NFA 图直接输入测试
    print(f"\n{'='*60}")
    print(f"  测试用例: 直接输入 NFA 图")
    print(f"{'='*60}")
    nfa = parse_nfa_graph("1+a=2, 2+b=3")
    nfa_to_min_dfa(nfa)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--nfa':
        if len(sys.argv) > 2:
            nfa_text = sys.argv[2]
        else:
            print("请输入 NFA 图（每行一条规则，格式: 状态+字符=状态，空行结束）:")
            lines = []
            try:
                while True:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
            except EOFError:
                pass
            nfa_text = '\n'.join(lines)
        nfa = parse_nfa_graph(nfa_text)
        nfa_to_min_dfa(nfa)
    elif len(sys.argv) > 1:
        pattern = sys.argv[1]
        regex_to_min_dfa(pattern)
    else:
        run_all_tests()

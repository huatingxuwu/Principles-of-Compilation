"""
正则表达式 → NFA → DFA → 最小化 DFA
=====================================
完整流水线，含详细中间过程输出:
  1. 正则解析 + Thompson 构造法  →  NFA（逐步骤展示）
  2. 子集构造法                  →  DFA（展示每轮工作列表、ε-闭包、move）
  3. Hopcroft 算法               →  最小化 DFA（展示每轮划分细化）

支持运算符（优先级从高到低）:  *  Kleene星  >  连接（省略） >  |  并

用法:
  python regex_to_dfa.py                  # 运行内置测试
  python regex_to_dfa.py "a(b|c)*"        # 命令行传入正则
"""

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from grammar_utils import print_section_header


# ═══════════════════════════════════════════════════════════════
# 1. 正则表达式解析器
# ═══════════════════════════════════════════════════════════════

class RegexParser:
    """递归下降解析器，生成 AST。

    grammar:
      regex    → term ('|' term)*
      term     → factor+
      factor   → atom '*'
      atom     → char | '(' regex ')'
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos < len(self.pattern):
            return self.pattern[self.pos]
        return None

    def consume(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def parse(self) -> 'RegexNode':
        node = self._regex()
        if self.pos != len(self.pattern):
            raise ValueError(f"多余字符: '{self.pattern[self.pos:]}'")
        return node

    def _regex(self) -> 'RegexNode':
        left = self._term()
        while self.peek() == '|':
            self.consume()
            right = self._term()
            left = UnionNode(left, right)
        return left

    def _term(self) -> 'RegexNode':
        children = []
        while self.peek() is not None and self.peek() not in (')', '|'):
            children.append(self._factor())
        if not children:
            raise ValueError("期待一个表达式，但遇到了空串或非法字符")
        if len(children) == 1:
            return children[0]
        return ConcatNode(children)

    def _factor(self) -> 'RegexNode':
        node = self._atom()
        if self.peek() == '*':
            self.consume()
            node = StarNode(node)
        return node

    def _atom(self) -> 'RegexNode':
        ch = self.peek()
        if ch is None:
            raise ValueError("意外的表达式结尾")
        if ch == '(':
            self.consume()
            node = self._regex()
            if self.peek() != ')':
                raise ValueError("期待 ')'")
            self.consume()
            return node
        elif ch == 'ε':
            self.consume()
            return EpsilonNode()
        elif ch in ('*', '|', ')'):
            raise ValueError(f"意外的运算符 '{ch}'")
        else:
            self.consume()
            return CharNode(ch)


# ── AST 节点 ────────────────────────────────────────────────

class RegexNode:
    def to_nfa(self, state_counter: 'StateCounter', trace: List[str]) -> 'NFA':
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


@dataclass
class EpsilonNode(RegexNode):
    """空串 ε — 匹配空字符串。"""

    def describe(self) -> str:
        return 'ε'

    def to_nfa(self, sc: 'StateCounter', trace: List[str]) -> 'NFA':
        s = sc.fresh()
        nfa = NFA()
        nfa.start = s
        nfa.accepts = {s}
        trace.append(
            f"  空串 ε: 新建状态 {s}, 起始=接受={{{s}}}（无迁移边）"
        )
        return nfa


@dataclass
class CharNode(RegexNode):
    char: str

    def describe(self) -> str:
        return self.char

    def to_nfa(self, sc: 'StateCounter', trace: List[str]) -> 'NFA':
        s0 = sc.fresh()
        s1 = sc.fresh()
        nfa = NFA()
        nfa.start = s0
        nfa.accepts = {s1}
        nfa.add_transition(s0, self.char, s1)
        trace.append(
            f"  字符 '{self.char}': 新建状态 {s0} --{self.char}--> {s1}, "
            f"起始={s0}, 接受={{{s1}}}"
        )
        return nfa


@dataclass
class ConcatNode(RegexNode):
    children: List[RegexNode]

    def describe(self) -> str:
        return ''.join(c.describe() for c in self.children)

    def to_nfa(self, sc: 'StateCounter', trace: List[str]) -> 'NFA':
        nfa = self.children[0].to_nfa(sc, trace)
        trace.append(f"  连接: 先构造左部 NFA（起始={nfa.start}, 接受={nfa.accepts}）")
        for i, child in enumerate(self.children[1:], 1):
            nfa2 = child.to_nfa(sc, trace)
            trace.append(
                f"  连接第{i+1}部分: 将左部接受态 {nfa.accepts} "
                f"通过 ε 连到右部起始态 {nfa2.start}"
            )
            nfa = nfa.concat(nfa2)
        trace.append(
            f"  连接完成: 起始={nfa.start}, 接受={nfa.accepts}, "
            f"共 {len(nfa.states)} 个状态"
        )
        return nfa


@dataclass
class UnionNode(RegexNode):
    left: RegexNode
    right: RegexNode

    def describe(self) -> str:
        return f"({self.left.describe()}|{self.right.describe()})"

    def to_nfa(self, sc: 'StateCounter', trace: List[str]) -> 'NFA':
        trace.append("  Union: 分别构造左右分支 NFA...")
        nfa1 = self.left.to_nfa(sc, trace)
        nfa2 = self.right.to_nfa(sc, trace)
        trace.append(
            f"  Union 合并: 左分支（起始={nfa1.start}, 接受={nfa1.accepts}）, "
            f"右分支（起始={nfa2.start}, 接受={nfa2.accepts}）"
        )
        result = nfa1.union(nfa2, sc)
        trace.append(
            f"  Union 完成: 新建起始 {result.start}, 新建接受 {result.accepts}, "
            f"通过 ε 分叉到左右分支"
        )
        return result


@dataclass
class StarNode(RegexNode):
    child: RegexNode

    def describe(self) -> str:
        return f"({self.child.describe()})*"

    def to_nfa(self, sc: 'StateCounter', trace: List[str]) -> 'NFA':
        nfa = self.child.to_nfa(sc, trace)
        trace.append(
            f"  Kleene 星: 对子 NFA（起始={nfa.start}, 接受={nfa.accepts}）加环"
        )
        result = nfa.kleene_star(sc)
        trace.append(
            f"  Kleene 星完成: 新建起始 {result.start}, 新建接受 {result.accepts}, "
            f"ε-回边从接受态回到子 NFA 起始 {nfa.start}"
        )
        return result


# ═══════════════════════════════════════════════════════════════
# 2. NFA 数据结构及操作
# ═══════════════════════════════════════════════════════════════

class StateCounter:
    def __init__(self):
        self._id = 0

    def fresh(self) -> int:
        self._id += 1
        return self._id


@dataclass
class NFA:
    """epsilon 用 None 表示。"""
    transitions: Dict[int, Dict[Optional[str], Set[int]]] = field(default_factory=dict)
    start: int = 0
    accepts: Set[int] = field(default_factory=set)

    @property
    def states(self) -> Set[int]:
        result = set(self.transitions.keys())
        result.add(self.start)
        result.update(self.accepts)
        return result

    def add_transition(self, src: int, symbol: Optional[str], dst: int):
        if src not in self.transitions:
            self.transitions[src] = {}
        edge_map = self.transitions[src]
        if symbol not in edge_map:
            edge_map[symbol] = set()
        edge_map[symbol].add(dst)

    def epsilon_closure(self, states: Set[int]) -> FrozenSet[int]:
        stack = list(states)
        closure = set(states)
        while stack:
            s = stack.pop()
            for nxt in self.transitions.get(s, {}).get(None, set()):
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)
        return frozenset(closure)

    def concat(self, other: 'NFA') -> 'NFA':
        nfa = NFA()
        nfa.transitions = _merge_dicts(self.transitions, other.transitions)
        nfa.start = self.start
        nfa.accepts = other.accepts
        for acc in self.accepts:
            nfa.add_transition(acc, None, other.start)
        return nfa

    def union(self, other: 'NFA', sc: StateCounter) -> 'NFA':
        nfa = NFA()
        nfa.transitions = _merge_dicts(self.transitions, other.transitions)
        new_start = sc.fresh()
        new_accept = sc.fresh()
        nfa.start = new_start
        nfa.accepts = {new_accept}
        nfa.add_transition(new_start, None, self.start)
        nfa.add_transition(new_start, None, other.start)
        for acc in self.accepts:
            nfa.add_transition(acc, None, new_accept)
        for acc in other.accepts:
            nfa.add_transition(acc, None, new_accept)
        return nfa

    def kleene_star(self, sc: StateCounter) -> 'NFA':
        nfa = NFA()
        nfa.transitions = self.transitions
        new_start = sc.fresh()
        new_accept = sc.fresh()
        nfa.start = new_start
        nfa.accepts = {new_accept}
        nfa.add_transition(new_start, None, self.start)
        nfa.add_transition(new_start, None, new_accept)
        for acc in self.accepts:
            nfa.add_transition(acc, None, self.start)
            nfa.add_transition(acc, None, new_accept)
        return nfa

    @property
    def alphabet(self) -> Set[str]:
        symbols: Set[str] = set()
        for edge_map in self.transitions.values():
            for sym in edge_map:
                if sym is not None:
                    symbols.add(sym)
        return symbols


def _merge_dicts(a, b):
    result = {}
    for d in (a, b):
        for k, v in d.items():
            if k in result:
                for sym, dsts in v.items():
                    if sym in result[k]:
                        result[k][sym] |= dsts
                    else:
                        result[k][sym] = set(dsts)
            else:
                result[k] = {sym: set(dsts) for sym, dsts in v.items()}
    return result


# ═══════════════════════════════════════════════════════════════
# 3. NFA → DFA (子集构造法) — 含详细过程
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
# 4. DFA 最小化 (Hopcroft) — 含详细过程
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
# 5. 输出函数 —— 纯文字描述，不用 ASCII 图
# ═══════════════════════════════════════════════════════════════

def describe_nfa_connections(nfa: NFA):
    """用文字描述 NFA 的所有连接。"""
    print(f"    状态总数: {len(nfa.states)}")
    print(f"    起始态: {nfa.start}")
    print(f"    接受态: {sorted(nfa.accepts)}")
    print(f"    字母表: {sorted(nfa.alphabet)}")
    print(f"    状态间连接:")
    for src in sorted(nfa.states):
        edge_map = nfa.transitions.get(src, {})
        if not edge_map:
            continue
        for sym in sorted(edge_map.keys(), key=lambda x: (x is None, x or '')):
            label = "ε" if sym is None else f"'{sym}'"
            dsts = sorted(edge_map[sym])
            for dst in dsts:
                start_mark = " (起始)" if src == nfa.start else ""
                accept_mark = " (接受)" if dst in nfa.accepts else ""
                print(f"      状态{src}{start_mark} 经过 {label} 到达 状态{dst}{accept_mark}")


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


# print_section_header 已提取到 grammar_utils.print_section_header


# ═══════════════════════════════════════════════════════════════
# 6. 顶层流水线
# ═══════════════════════════════════════════════════════════════

def regex_to_min_dfa(pattern: str, verbose: bool = True):
    """一键流水线，verbose=True 时输出全部中间过程。"""
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
# 7. 测试用例
# ═══════════════════════════════════════════════════════════════

def run_all_tests():
    tests = [
        ("a",           "单个字符"),
        ("a|ε",         "并 a|ε"),
        ("aε",          "连接 a·ε"),
        ("((ε|a)b*)*",       "嵌套星 (ab)*"),
    ]
    for pattern, desc in tests:
        print(f"\n{'='*60}")
        print(f"  测试用例: {desc}")
        print(f"{'='*60}")
        regex_to_min_dfa(pattern)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        regex_to_min_dfa(pattern)
    else:
        run_all_tests()

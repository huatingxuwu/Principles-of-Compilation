"""
正则表达式 → NFA
================
支持两种输入方式:
  1. 正则解析 + Thompson 构造法 → NFA
  2. 直接输入 NFA 图 (x+y=z 格式)

用法:
  from regex_to_nfa import RegexParser, parse_nfa_graph, NFA, StateCounter
"""

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


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
# 3. 直接输入 NFA 图 → NFA
# ═══════════════════════════════════════════════════════════════

_EPSILON_NAMES = {'ε', 'epsilon', 'eps', 'εpsilon', ''}


def parse_nfa_graph(text: str) -> NFA:
    """从 "x+y=z" 格式的文本解析 NFA 图。

    格式:
      1+a=2     状态1 经过字符'a' 到达状态2
      2+ε=3     状态2 经过ε 到达状态3
      start=1   显式指定起始状态（可选）
      accept=3  显式指定接受状态，多个用逗号分隔（可选）

    自动推断（未显式指定时）:
      - 起始状态: 无入边的状态中取最小ID
      - 接受状态: 无出边的状态；若无，取最大状态ID
    """
    nfa = NFA()
    explicit_start = None
    explicit_accepts = None

    # 支持换行、分号、逗号分隔
    rules: List[str] = []
    for line in text.strip().splitlines():
        for part in re.split(r'[;,，]', line.strip()):
            part = part.strip()
            if part:
                rules.append(part)

    for rule in rules:
        # start=N
        if rule.startswith('start='):
            explicit_start = int(rule.split('=')[1])
            continue

        # accept=N 或 accept=N,M,...
        if rule.startswith('accept='):
            explicit_accepts = {
                int(x.strip()) for x in rule.split('=')[1].split(',') if x.strip()
            }
            continue

        # x+y=z
        m = re.match(r'(\d+)\+(.+?)=(\d+)$', rule)
        if not m:
            raise ValueError(f"无法解析规则: '{rule}'，期望格式: 状态+字符=状态")

        src = int(m.group(1))
        symbol = m.group(2).strip()
        dst = int(m.group(3))

        if symbol.lower() in _EPSILON_NAMES:
            symbol = None

        nfa.add_transition(src, symbol, dst)

    # ── 自动推断 start / accept ──
    if explicit_start is not None:
        nfa.start = explicit_start
    else:
        nfa.start = _auto_detect_start(nfa)

    if explicit_accepts is not None:
        nfa.accepts = explicit_accepts
    else:
        nfa.accepts = _auto_detect_accepts(nfa)

    return nfa


def _collect_edge_sets(nfa: NFA):
    """返回 (sources_set, destinations_set, all_states_set)，仅基于 transitions。"""
    sources: Set[int] = set(nfa.transitions.keys())
    destinations: Set[int] = set()
    for edge_map in nfa.transitions.values():
        for dsts in edge_map.values():
            destinations.update(dsts)
    return sources, destinations, sources | destinations


def _auto_detect_start(nfa: NFA) -> int:
    """起始状态: 无入边的状态中取最小ID；若全有入边，取最小状态ID。"""
    _sources, destinations, all_s = _collect_edge_sets(nfa)
    no_incoming = all_s - destinations
    if no_incoming:
        return min(no_incoming)
    return min(all_s) if all_s else 0


def _auto_detect_accepts(nfa: NFA) -> Set[int]:
    """接受状态: 无出边的状态；若无，取最大状态ID。"""
    sources, _destinations, all_s = _collect_edge_sets(nfa)
    no_outgoing = all_s - sources
    if no_outgoing:
        return no_outgoing
    return {max(all_s)} if all_s else set()


# ═══════════════════════════════════════════════════════════════
# 4. NFA 输出函数
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

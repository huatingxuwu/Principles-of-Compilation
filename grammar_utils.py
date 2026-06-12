"""
文法工具模块 — 供 ll1_parser / lr0_automaton 共用
==================================================
  - Grammar 类:       解析上下文无关文法，支持增广
  - compute_first():  FIRST 集（不动点迭代）
  - compute_follow():  FOLLOW 集（不动点迭代）
  - first_of_symbols() / fmt_set():  辅助函数

用法:
  from grammar_utils import Grammar, compute_first, compute_follow, fmt_set
"""

from collections import defaultdict
from typing import Dict, List, Set


# ═══════════════════════════════════════════════════════════════
# 1. Grammar 类
# ═══════════════════════════════════════════════════════════════

class Grammar:
    """上下文无关文法。

    输入格式（同 ll1_parser / lr0_automaton）:
      非终结符 → 右部1 | 右部2 | ...
      符号之间空格分隔，ε 表示空串。
      第一条产生式的左部为开始符号。
    """

    def __init__(self, text: str):
        self.productions: Dict[str, List[List[str]]] = defaultdict(list)
        self.non_terminals: Set[str] = set()
        self.terminals: Set[str] = set()
        self.start: str = ""
        self._parse(text)
        self._collect_symbols()

    def _parse(self, text: str):
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '→' in line:
                lhs, rhs = line.split('→', 1)
            elif '->' in line:
                lhs, rhs = line.split('->', 1)
            else:
                continue
            lhs = lhs.strip()
            if not self.start:
                self.start = lhs
            for alt in rhs.split('|'):
                symbols = alt.strip().split()
                if not symbols:
                    symbols = ['ε']
                self.productions[lhs].append(symbols)

    def _collect_symbols(self):
        self.non_terminals = set(self.productions.keys())
        for _lhs, rhs_list in self.productions.items():
            for rhs in rhs_list:
                for sym in rhs:
                    if sym != 'ε' and sym not in self.non_terminals:
                        self.terminals.add(sym)

    def augmented(self) -> 'Grammar':
        """返回增广文法: 添加 S' → S（LR 自动机构造用）。"""
        text_lines = []
        new_start = self.start + "'"
        while new_start in self.non_terminals:
            new_start += "'"
        text_lines.append(f"{new_start} → {self.start}")
        for lhs, rhs_list in self.productions.items():
            for rhs in rhs_list:
                rhs_str = ' '.join(rhs)
                text_lines.append(f"{lhs} → {rhs_str}")
        return Grammar('\n'.join(text_lines))

    @property
    def all_productions(self) -> List[tuple]:
        """所有产生式编号: [(lhs, rhs_tuple), ...]。"""
        result = []
        for lhs in self.productions:
            for rhs in self.productions[lhs]:
                result.append((lhs, tuple(rhs)))
        return result

    def __str__(self):
        lines = []
        for lhs in self.productions:
            for rhs in self.productions[lhs]:
                rhs_str = ' '.join(rhs)
                lines.append(f"  {lhs} → {rhs_str}")
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# 2. FIRST 集
# ═══════════════════════════════════════════════════════════════

def compute_first(grammar: Grammar) -> Dict[str, Set[str]]:
    """计算所有文法符号的 FIRST 集（不动点迭代）。

    规则:
      1. 终结符 x:  FIRST(x) = {x}
      2. X → ε:      ε ∈ FIRST(X)
      3. X → Y₁Y₂...Yₖ:
           FIRST(Y₁)-{ε} ⊆ FIRST(X)
           若 ε∈FIRST(Y₁) 则继续 Y₂...
           若全部可空则 ε∈FIRST(X)
    """
    FIRST: Dict[str, Set[str]] = defaultdict(set)

    # 终结符
    for t in grammar.terminals:
        FIRST[t] = {t}

    # 非终结符初始化为空
    for nt in grammar.non_terminals:
        FIRST[nt] = set()

    changed = True
    while changed:
        changed = False
        for nt in grammar.non_terminals:
            for rhs in grammar.productions[nt]:
                before = len(FIRST[nt])

                if rhs == ['ε']:
                    FIRST[nt].add('ε')
                else:
                    all_nullable = True
                    for sym in rhs:
                        if sym == 'ε':
                            break
                        FIRST[nt] |= (FIRST.get(sym, set()) - {'ε'})
                        if 'ε' not in FIRST.get(sym, set()):
                            all_nullable = False
                            break
                    if all_nullable:
                        FIRST[nt].add('ε')

                if len(FIRST[nt]) > before:
                    changed = True

    return FIRST


# ═══════════════════════════════════════════════════════════════
# 3. FOLLOW 集
# ═══════════════════════════════════════════════════════════════

def compute_follow(grammar: Grammar, FIRST: Dict[str, Set[str]] = None) -> Dict[str, Set[str]]:
    """计算所有非终结符的 FOLLOW 集（不动点迭代）。

    规则:
      1. $ ∈ FOLLOW(S)
      2. A → αBβ  ⇒ FIRST(β)-{ε} ⊆ FOLLOW(B)
      3. A → αB   ⇒ FOLLOW(A) ⊆ FOLLOW(B)
      4. A → αBβ 且 ε∈FIRST(β) ⇒ FOLLOW(A) ⊆ FOLLOW(B)
    """
    if FIRST is None:
        FIRST = compute_first(grammar)

    FOLLOW: Dict[str, Set[str]] = defaultdict(set)
    FOLLOW[grammar.start].add('$')

    # 扫描每个非终结符 B 在所有产生式中的出现位置
    # occurrences[B] = [(lhs, rhs_index, pos)]
    occurrences: Dict[str, List[tuple]] = defaultdict(list)
    for lhs, rhs_list in grammar.productions.items():
        for ri, rhs in enumerate(rhs_list):
            for pos, sym in enumerate(rhs):
                if sym in grammar.non_terminals:
                    occurrences[sym].append((lhs, ri, pos))

    changed = True
    while changed:
        changed = False
        for B, occs in occurrences.items():
            for lhs, ri, pos in occs:
                rhs = grammar.productions[lhs][ri]
                beta = rhs[pos + 1:]
                before = len(FOLLOW[B])

                if beta:
                    # 计算 FIRST(β)
                    first_beta: Set[str] = set()
                    all_nullable = True
                    for s in beta:
                        if s == 'ε':
                            first_beta.add('ε')
                            break
                        first_beta |= (FIRST.get(s, set()) - {'ε'})
                        if 'ε' not in FIRST.get(s, set()):
                            all_nullable = False
                            break
                    if all_nullable:
                        first_beta.add('ε')

                    FOLLOW[B] |= (first_beta - {'ε'})
                    if 'ε' in first_beta:
                        FOLLOW[B] |= FOLLOW[lhs]
                else:
                    # beta 为空
                    FOLLOW[B] |= FOLLOW[lhs]

                if len(FOLLOW[B]) > before:
                    changed = True

    return FOLLOW


# ═══════════════════════════════════════════════════════════════
# 4. 辅助函数
# ═══════════════════════════════════════════════════════════════

def first_of_symbols(FIRST: Dict[str, Set[str]], symbols: List[str]) -> Set[str]:
    """计算符号串 X₁X₂...Xₖ 的 FIRST 集。

    $ 视为终结符: FIRST($) = {$}。
    """
    result: Set[str] = set()
    for sym in symbols:
        if sym == 'ε':
            result.add('ε')
            return result
        if sym == '$':
            result.add('$')
            return result
        result |= (FIRST.get(sym, set()) - {'ε'})
        if 'ε' not in FIRST.get(sym, set()):
            return result
    result.add('ε')
    return result


def is_nullable(FIRST: Dict[str, Set[str]], symbols: List[str]) -> bool:
    """符号串是否可为空。$ 不可空。"""
    for sym in symbols:
        if sym == 'ε':
            return True
        if sym == '$':
            return False
        if 'ε' not in FIRST.get(sym, set()):
            return False
    return True


def fmt_set(s: Set[str]) -> str:
    """格式化集合: {$, id, +}。"""
    items = sorted(s, key=lambda x: (x != '$', x != 'ε', x))
    return ', '.join(items)


def print_section_header(title: str):
    """打印分隔标题。"""
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

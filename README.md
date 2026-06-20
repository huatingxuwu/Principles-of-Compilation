# Compiler Principles — Lexical & Syntax Analysis Toolkit

编译原理课程实验代码集：从正则表达式到词法分析，再到 LL/LR/SLR/LALR 语法分析，包含 SDD 三地址码翻译。

## Quick Start

```bash
# 词法分析 — 正则 → NFA（Thompson 构造法）
python regex_to_nfa.py
python regex_to_nfa.py "a(b|c)*"

# 词法分析 — NFA → DFA → 最小化 DFA（子集构造 + Hopcroft）
python regex_to_dfa.py
python regex_to_dfa.py "a(b|c)*"

# 直接从 NFA 图输入（跳过正则解析）
python regex_to_dfa.py --nfa "0+a=1; 1+b=2"

# LL(1) 语法分析 — FIRST/FOLLOW → 预测分析表 → 表驱动
python ll1_parser.py
python ll1_parser.py grammar.txt "id + id * id"

# LR(0) 自动机构造 + 分析过程表格
python lr0_automaton.py

# SLR(1) — FOLLOW 消冲突 + 分析过程表格
python slr1_parser.py
python slr1_parser.py grammar.txt "id * id"

# LR(1) — 前瞻符精确传播
python lr1_parser.py

# LALR(1) — LR(1) 判别力 + LR(0) 状态数 (yacc/bison 同款)
python lalr1_parser.py

# 布尔表达式 SDD — 三地址码 + 回填
python sdd_boolean.py
python sdd_boolean.py "(a < b || c > d) && e == f"

# 算术表达式 SDD — 三地址码
python sdd_arithmetic.py
python sdd_arithmetic.py "x * y + z / (a - b)"
```

所有输出已保存在 `test/` 目录下。

## File Structure

```
.
├── grammar_utils.py        # 公共模块: Grammar 解析, FIRST/FOLLOW 计算 (支持 ε)
├── regex_to_nfa.py         # 词法分析: 正则解析 + Thompson 构造 → ε-NFA
├── regex_to_dfa.py         # 词法分析: NFA→DFA→最小化 DFA（支持 --nfa 直接输入 NFA 图）
├── ll1_parser.py           # LL(1): FIRST/FOLLOW → 预测分析表 → 表驱动
├── lr0_automaton.py        # LR(0): 项集族, ACTION/GOTO 表, 逐步分析过程
├── slr1_parser.py          # SLR(1): FOLLOW 消冲突, 逐步分析过程表格
├── lr1_parser.py           # LR(1): 前瞻符精确传播
├── lalr1_parser.py         # LALR(1): 状态合并 (yacc/bison)
├── sdd_boolean.py          # 布尔表达式 SDD: 回填技术 + 三地址码
├── sdd_arithmetic.py       # 算术表达式 SDD: S-属性 + 三地址码
├── README.md
└── test/                   # 所有程序的运行输出
```

## Dependency Graph

```
grammar_utils.py  ←── ll1_parser.py
    ↑                    ↑
    ├── lr0_automaton.py  ←── slr1_parser.py
    │        ↑                    ↑
    │        └── lr1_parser.py  ←── lalr1_parser.py
    │
    └── regex_to_nfa.py ←── regex_to_dfa.py (共用 NFA 数据结构)
```

## Features

### Lexical Analysis

**regex_to_nfa.py** — 正则 → ε-NFA（Thompson 构造法）
- Supports `*` (Kleene star), `|` (union), concatenation, `()` grouping
- Regex parser → AST → Thompson construction → ε-NFA
- ε-NFA data structure with auto-detection of start/accept states

**regex_to_dfa.py** — NFA → DFA → 最小化 DFA
- **输入 `--nfa`** 直接传入 NFA 图（格式: `状态+字符=状态`，用 `;` 或换行分隔）
- **无参数** 运行内置正则测试用例
- **传入正则** 则链式调用 regex_to_nfa 完成全流程
- Subset construction: NFA → DFA
- Hopcroft algorithm: DFA minimization
- Output: transition tables, minimized state diagrams

### Syntax Analysis

- Grammar supports **ε (epsilon)** for empty productions
- LR(0) and SLR(1) output detailed **step-by-step analysis tables** (state stack, symbol stack, remaining input, action per step)

| Parser | Lookahead | State Count | Conflict Resolution |
|--------|-----------|-------------|---------------------|
| LL(1) | FIRST/FOLLOW | — | Predictive table |
| LR(0) | None | Baseline | None (many conflicts) |
| SLR(1) | FOLLOW(A) | = LR(0) | Coarse |
| LR(1) | FIRST(βa) per-item | Larger | Precise |
| LALR(1) | Merged per-item | = LR(0) | Near-LR(1) |

### SDD Translation
- **Boolean expressions**: short-circuit evaluation via backpatching (Dragon Book §6.6)
- **Arithmetic expressions**: S-attributed SDD with temp variables (Dragon Book §6.4)
- Both generate readable three-address code

## Grammar Format

```
# 每行一条产生式, | 分隔多个右部, ε 表示空串
E → T E'
E' → + T E' | ε
T → F T'
T' → * F T' | ε
F → ( E ) | id
```

保存为 `.txt` 文件，通过命令行传入：

```bash
python ll1_parser.py grammar.txt "id + id * id"
python slr1_parser.py grammar.txt "id * id + id"
```

## Classic Verification Cases

| Grammar | LR(0) | SLR(1) | LALR(1) | LR(1) |
|---------|-------|--------|---------|-------|
| E→E+T\|T, T→T\*F\|F, F→(E)\|id | 2 conflicts | 0 | 0 | 0 |
| S→L=R\|R, L→\*R\|id, R→L | — | 1 conflict | 0 | 0 |

Both grammars verified — LALR(1) resolves the conflict that SLR(1) cannot, matching LR(1) with fewer states.

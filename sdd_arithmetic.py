"""
算术表达式 SDD 翻译 — 三地址码生成
====================================
支持算术表达式:
  + - * /  (加减乘除)，括号分组，标识符，数字字面量

SDD 规则 (Dragon Book §6.4):
  产生式                    语义规则
  ─────────────────────────────────────────────────
  E → E₁ + T               E.place = newtemp()
                           emit(E.place = E₁.place + T.place)
  E → E₁ - T               E.place = newtemp()
                           emit(E.place = E₁.place - T.place)
  E → T                    E.place = T.place

  T → T₁ * F               T.place = newtemp()
                           emit(T.place = T₁.place * F.place)
  T → T₁ / F               T.place = newtemp()
                           emit(T.place = T₁.place / F.place)
  T → F                    T.place = F.place

  F → ( E )                F.place = E.place
  F → id                   F.place = id
  F → num                  F.place = num

复用: sdd_boolean.Instr (三地址指令数据结构)

用法:
  python sdd_arithmetic.py                              # 内置测试
  python sdd_arithmetic.py "a + b * c"                  # 单条表达式
  python sdd_arithmetic.py "x * y + z / (a - b)"        # 复合表达式
"""

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# 1. 三地址指令（与 sdd_boolean.Instr 结构兼容，独立定义以保持模块自足）
# ═══════════════════════════════════════════════════════════════

@dataclass
class Instr:
    op: str
    arg1: str = ""
    arg2: str = ""
    result: str = ""


# ═══════════════════════════════════════════════════════════════
# 2. 词法分析器
# ═══════════════════════════════════════════════════════════════

TOKEN_EOF = 'EOF'
TOKEN_ID = 'ID'
TOKEN_NUM = 'NUM'
TOKEN_PLUS = '+'
TOKEN_MINUS = '-'
TOKEN_STAR = '*'
TOKEN_SLASH = '/'
TOKEN_LPAREN = '('
TOKEN_RPAREN = ')'


class Lexer:
    def __init__(self, source: str):
        self.tokens: List[Tuple[str, str]] = []
        self._tokenize(source)

    def _tokenize(self, s: str):
        p = 0
        while p < len(s):
            ch = s[p]
            if ch.isspace():
                p += 1
                continue
            if ch == '(':
                self.tokens.append((TOKEN_LPAREN, '(')); p += 1
            elif ch == ')':
                self.tokens.append((TOKEN_RPAREN, ')')); p += 1
            elif ch == '+':
                self.tokens.append((TOKEN_PLUS, '+')); p += 1
            elif ch == '-':
                self.tokens.append((TOKEN_MINUS, '-')); p += 1
            elif ch == '*':
                self.tokens.append((TOKEN_STAR, '*')); p += 1
            elif ch == '/':
                self.tokens.append((TOKEN_SLASH, '/')); p += 1
            elif ch.isalpha() or ch == '_':
                start = p
                while p < len(s) and (s[p].isalnum() or s[p] == '_'):
                    p += 1
                self.tokens.append((TOKEN_ID, s[start:p]))
            elif ch.isdigit():
                start = p
                while p < len(s) and (s[p].isdigit() or s[p] == '.'):
                    p += 1
                self.tokens.append((TOKEN_NUM, s[start:p]))
            else:
                raise ValueError(f"非法字符: '{ch}' 位置 {p}")
        self.tokens.append((TOKEN_EOF, ''))


# ═══════════════════════════════════════════════════════════════
# 3. 代码生成器（带临时变量管理）
# ═══════════════════════════════════════════════════════════════

class CodeGen:
    def __init__(self):
        self.instrs: List[Instr] = []
        self._temp_counter = 0

    def newtemp(self) -> str:
        self._temp_counter += 1
        return f"t{self._temp_counter}"

    def emit(self, op: str, arg1: str = "", arg2: str = "", result: str = ""):
        self.instrs.append(Instr(op, arg1, arg2, result))

    def get_code(self) -> List[str]:
        lines = []
        for instr in self.instrs:
            if instr.op == '=':
                # result = arg1 op arg2
                if instr.arg2:
                    lines.append(f"  {instr.result} = {instr.arg1} {instr.op} {instr.arg2}")
                else:
                    lines.append(f"  {instr.result} = {instr.arg1}")
            elif instr.op in ('+', '-', '*', '/'):
                lines.append(f"  {instr.result} = {instr.arg1} {instr.op} {instr.arg2}")
            else:
                parts = [instr.op, instr.arg1, instr.arg2, instr.result]
                text = ' '.join(p for p in parts if p)
                lines.append(f"  {text}")
        return lines


# ═══════════════════════════════════════════════════════════════
# 4. SDD 翻译方案 (递归下降 + S-属性文法)
# ═══════════════════════════════════════════════════════════════

class SDDTranslator:
    """算术表达式的 SDD 翻译器。

    每个非终结符有一个综合属性 .place（存放值的变量名/临时变量名）。
    消除左递归: E → T { (+|-) T }*,  T → F { (*|/) F }*
    """

    def __init__(self, tokens: List[Tuple[str, str]], gen: CodeGen,
                 trace: List[str] = None):
        self.tokens = tokens
        self.pos = 0
        self.gen = gen
        self.trace = trace

    # ── 辅助 ──

    def peek(self):       return self.tokens[self.pos]
    def consume(self):
        tok = self.tokens[self.pos]; self.pos += 1; return tok

    def _log(self, msg: str, indent: int = 0):
        if self.trace is not None:
            self.trace.append(f"  {'  ' * indent}{msg}")

    # ── E → T { (+|-) T }* ──

    def _E(self) -> str:
        """返回 E.place。"""
        place = self._T()
        self._log(f"E.place ← T.place = {place}")

        while self.peek()[0] in (TOKEN_PLUS, TOKEN_MINUS):
            op_tok = self.consume()
            op = op_tok[1]
            self._log(f"  看到 '{op}'")
            T2_place = self._T()

            temp = self.gen.newtemp()
            self.gen.emit(op, place, T2_place, temp)
            self._log(f"  emit: {temp} = {place} {op} {T2_place}  (E.place → {temp})")
            place = temp

        return place

    # ── T → F { (*|/) F }* ──

    def _T(self) -> str:
        """返回 T.place。"""
        place = self._F()
        self._log(f"T.place ← F.place = {place}")

        while self.peek()[0] in (TOKEN_STAR, TOKEN_SLASH):
            op_tok = self.consume()
            op = op_tok[1]
            self._log(f"  看到 '{op}'")
            F2_place = self._F()

            temp = self.gen.newtemp()
            self.gen.emit(op, place, F2_place, temp)
            self._log(f"  emit: {temp} = {place} {op} {F2_place}  (T.place → {temp})")
            place = temp

        return place

    # ── F → ( E ) | id | num ──

    def _F(self) -> str:
        """返回 F.place。"""
        tok = self.peek()

        if tok[0] == TOKEN_LPAREN:
            self.consume()
            self._log(f"F → ( E )")
            place = self._E()
            self.match(TOKEN_RPAREN)
            self._log(f"F.place = E.place = {place}")
            return place

        elif tok[0] == TOKEN_ID:
            id_val = self.consume()[1]
            self._log(f"F → id ({id_val}),  F.place = '{id_val}'")
            return id_val

        elif tok[0] == TOKEN_NUM:
            num_val = self.consume()[1]
            self._log(f"F → num ({num_val}),  F.place = '{num_val}'")
            return num_val

        else:
            raise ValueError(f"意外的 token: {tok}")

    def match(self, expected: str):
        t = self.consume()
        if t[0] != expected:
            raise ValueError(f"期待 {expected}，但遇到 {t}")


# ═══════════════════════════════════════════════════════════════
# 5. 顶层流水线
# ═══════════════════════════════════════════════════════════════

def translate_arithmetic(source: str):
    print(f"\n{'='*60}")
    print(f"  算术表达式:  {source}")
    print(f"{'='*60}")

    # 词法
    lexer = Lexer(source)
    tokens = lexer.tokens
    token_str = ' '.join(tok for tok, _ in tokens[:-1])
    print(f"\n  Tokens: {token_str}")

    # SDD 翻译
    trace: List[str] = []
    gen = CodeGen()
    translator = SDDTranslator(tokens, gen, trace=trace)

    root_place = translator._E()

    # 输出 SDD 过程
    print(f"\n  ── SDD 翻译过程 ──")
    for line in trace:
        print(line)

    # 最终结果
    code = gen.get_code()
    print(f"\n  生成的三地址码:")
    print(f"  {'─'*35}")
    for line in code:
        print(line)
    print(f"  {'─'*35}")
    print(f"  根节点: {root_place}")
    if gen._temp_counter > 0:
        print(f"  临时变量: {', '.join(f't{i}' for i in range(1, gen._temp_counter + 1))}")

    return gen


# ═══════════════════════════════════════════════════════════════
# 6. 测试用例
# ═══════════════════════════════════════════════════════════════

def run_builtin_tests():
    tests = [
        "1 * 2 * 3 * (4 + 5)",
    ]
    for expr in tests:
        translate_arithmetic(expr)
        print()


# ═══════════════════════════════════════════════════════════════
# 7. main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) > 1:
        translate_arithmetic(sys.argv[1])
    else:
        run_builtin_tests()

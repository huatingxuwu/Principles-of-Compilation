"""
布尔表达式 SDD 翻译 — 三地址码生成与回填
=========================================
支持布尔表达式:
  && (与), || (或), ! (非), == != < > <= >= (比较)
  true / false 字面量, 括号分组, 标识符

翻译方案:
  使用回填 (backpatching) 技术实现短路求值:
    - B.truelist:  当 B 为真时需要回填的指令列表
    - B.falselist: 当 B 为假时需要回填的指令列表
    - M.quad:      标记当前位置（下一条指令的编号）

文法 (消除左递归):
  B  → B || M B  |  B && M B  |  ! B  |  ( B )  |  E relop E  |  true  |  false  |  id
  M  → ε

用法:
  python sdd_boolean.py                                     # 内置测试
  python sdd_boolean.py "a < b && c > d"                   # 单条表达式
  python sdd_boolean.py "x > 0 && x < 10 || y == 0"        # 复合表达式
"""

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# 1. 三地址指令
# ═══════════════════════════════════════════════════════════════

@dataclass
class Instr:
    """一条三地址指令。"""
    op: str      # 操作: 'jump', 'jtrue', 'jfalse', '=', '<', '>', '<=', '>=', '==', '!=', '!', 'label'
    arg1: str = ""
    arg2: str = ""
    result: str = ""   # 目标标签或结果


# ═══════════════════════════════════════════════════════════════
# 2. 词法分析器
# ═══════════════════════════════════════════════════════════════

TOKEN_EOF = 'EOF'
TOKEN_ID = 'ID'
TOKEN_TRUE = 'TRUE'
TOKEN_FALSE = 'FALSE'
TOKEN_AND = 'AND'
TOKEN_OR = 'OR'
TOKEN_NOT = 'NOT'
TOKEN_LPAREN = '('
TOKEN_RPAREN = ')'

RELOP_TOKENS = {'==', '!=', '<', '>', '<=', '>='}


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.tokens: List[Tuple[str, str]] = []  # [(type, value), ...]
        self._tokenize()

    def _tokenize(self):
        s = self.source
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
            elif ch == '!':
                if p + 1 < len(s) and s[p + 1] == '=':
                    self.tokens.append(('RELOP', '!=')); p += 2
                else:
                    self.tokens.append((TOKEN_NOT, '!')); p += 1
            elif ch == '&' and p + 1 < len(s) and s[p + 1] == '&':
                self.tokens.append((TOKEN_AND, '&&')); p += 2
            elif ch == '|' and p + 1 < len(s) and s[p + 1] == '|':
                self.tokens.append((TOKEN_OR, '||')); p += 2
            elif ch in '<>=':
                if p + 1 < len(s) and s[p + 1] == '=':
                    self.tokens.append(('RELOP', ch + '=')); p += 2
                else:
                    self.tokens.append(('RELOP', ch)); p += 1
            elif ch.isalpha() or ch == '_':
                start = p
                while p < len(s) and (s[p].isalnum() or s[p] == '_'):
                    p += 1
                word = s[start:p]
                if word == 'true':
                    self.tokens.append((TOKEN_TRUE, 'true'))
                elif word == 'false':
                    self.tokens.append((TOKEN_FALSE, 'false'))
                else:
                    self.tokens.append((TOKEN_ID, word))
            elif ch.isdigit():
                start = p
                while p < len(s) and (s[p].isdigit() or s[p] == '.'):
                    p += 1
                self.tokens.append((TOKEN_ID, s[start:p]))
            else:
                raise ValueError(f"非法字符: '{ch}' 位置 {p}")
        self.tokens.append((TOKEN_EOF, ''))

    def token_list(self) -> List[Tuple[str, str]]:
        return self.tokens


# ═══════════════════════════════════════════════════════════════
# 3. SDD 代码生成器 (回填技术)
# ═══════════════════════════════════════════════════════════════

class CodeGen:
    """三地址码生成器，支持回填。

    用指令位置（nextinstr）作为标签，不需要独立的标签计数器。
    """

    def __init__(self, trace: List[str] = None):
        self.instrs: List[Instr] = []
        self.trace = trace  # 外部 trace list

    def _log(self, msg: str):
        if self.trace is not None:
            self.trace.append(msg)

    def emit(self, op: str, arg1: str = "", arg2: str = "", result: str = ""):
        idx = len(self.instrs)
        self.instrs.append(Instr(op, arg1, arg2, result))
        parts = [op, arg1, arg2]
        text = ' '.join(p for p in parts if p)
        self._log(f"    [{idx}] emit: {text}  (目标待回填)" if not result else f"    [{idx}] emit: {text}")
        return idx

    @property
    def nextinstr(self) -> int:
        """下一条指令的编号（也是它可以充当的标签号）。"""
        return len(self.instrs)

    def makelist(self, i: int) -> List[int]:
        return [i]

    def merge(self, p1: List[int], p2: List[int]) -> List[int]:
        return p1 + p2

    def backpatch(self, instr_indices: List[int], target: int):
        """回填: 将跳转指令的目标填充为 target 标签。"""
        if not instr_indices:
            return
        for idx in instr_indices:
            instr = self.instrs[idx]
            instr.result = str(target)
        self._log(f"    backpatch({instr_indices}, L{target})  ← 指令 {instr_indices} 的目标填为 L{target}")

    def get_code(self) -> List[str]:
        """返回格式化的代码列表。

        只在实际跳转目标处显示标签，无跳转的指令不标号。
        """
        # 1) 收集所有跳转目标
        targets: set[int] = set()
        for instr in self.instrs:
            if instr.result and instr.result.isdigit():
                targets.add(int(instr.result))

        # 2) 格式化输出
        lines: list[str] = []
        for i, instr in enumerate(self.instrs):
            if instr.op == 'label':
                continue  # 不再需要手动 label 指令

            text = ""
            # 仅在被跳转到的位置前加标签
            if i in targets:
                text += f"L{i}: "

            if instr.op == 'jump':
                target = f"L{instr.result}" if instr.result else "_"
                text += f"goto {target}"
            elif instr.op == 'jtrue':
                target = f"L{instr.result}" if instr.result else "_"
                text += f"if {instr.arg1} goto {target}"
            elif instr.op == 'jfalse':
                target = f"L{instr.result}" if instr.result else "_"
                text += f"ifFalse {instr.arg1} goto {target}"
            elif instr.op in ('<', '>', '<=', '>=', '==', '!='):
                target = f"L{instr.result}" if instr.result else "_"
                text += f"if {instr.arg1} {instr.op} {instr.arg2} goto {target}"
            elif instr.op == '!':
                text += f"{instr.result} = ! {instr.arg1}"
            else:
                parts = [instr.op, instr.arg1, instr.arg2, instr.result]
                text += ' '.join(p for p in parts if p)

            lines.append(text)
        return lines


# ═══════════════════════════════════════════════════════════════
# 4. SDD 翻译方案 (递归下降 + 属性文法)
# ═══════════════════════════════════════════════════════════════

class SDDTranslator:
    """布尔表达式的 SDD 翻译器。

    SDD 规则 (Dragon Book §6.6):
      产生式                    语义规则
      ─────────────────────────────────────────────────
      B → B1 || M B2           backpatch(B1.falselist, M.quad)
                               B.truelist  = merge(B1.truelist, B2.truelist)
                               B.falselist = B2.falselist

      B → B1 && M B2           backpatch(B1.truelist, M.quad)
                               B.truelist  = B2.truelist
                               B.falselist = merge(B1.falselist, B2.falselist)

      B → ! B1                 B.truelist  = B1.falselist
                               B.falselist = B1.truelist

      B → ( B1 )               B.truelist  = B1.truelist
                               B.falselist = B1.falselist

      B → E1 relop E2          B.truelist  = makelist(nextinstr)
                               B.falselist = makelist(nextinstr+1)
                               emit('if' E1 relop E2 'goto _')
                               emit('goto _')

      B → true                 B.truelist  = makelist(nextinstr)
                               emit('goto _')

      B → false                B.falselist = makelist(nextinstr)
                               emit('goto _')

      M → ε                    M.quad = nextinstr
    """

    def __init__(self, tokens: List[Tuple[str, str]], codegen: CodeGen,
                 trace: List[str] = None):
        self.tokens = tokens
        self.pos = 0
        self.gen = codegen
        self.trace = trace
        self._indent = 0

    def _log(self, msg: str):
        if self.trace is not None:
            prefix = "  " * self._indent
            self.trace.append(f"{prefix}{msg}")

    def _enter(self, rule: str):
        self._log(f"┌─ {rule}")
        self._indent += 1

    def _exit(self, rule: str, tlist: List[int], flist: List[int]):
        self._indent -= 1
        self._log(f"└─ {rule} → truelist={tlist}, falselist={flist}")
        return tlist, flist

    # ── 辅助 ──

    def peek(self) -> Tuple[str, str]:
        return self.tokens[self.pos]

    def consume(self) -> Tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def match(self, expected_type: str):
        t = self.consume()
        if t[0] != expected_type:
            raise ValueError(f"期待 {expected_type}，但遇到 {t}")

    # ── 翻译入口 ──

    def translate(self) -> Tuple[List[int], List[int]]:
        """B → ..."""
        result = self._B()
        if self.peek()[0] != TOKEN_EOF:
            raise ValueError(f"多余 token: {self.peek()}")
        return result

    # ── B → B || M B  ──

    def _B(self) -> Tuple[List[int], List[int]]:
        """B → T { || M T }*"""
        self._enter("B → T")

        tlist, flist = self._T()

        while self.peek()[0] == TOKEN_OR:
            self.consume()
            self._log(f"│ 看到 '||'")
            M_quad = self._M()
            self._log(f"│ B₁.falselist = {flist}  →  回填到 M.quad = L{M_quad}")
            self.gen.backpatch(flist, M_quad)
            tlist2, flist2 = self._T()
            self._log(f"│ 合并: truelist = merge({tlist}, {tlist2}) = {self.gen.merge(tlist, tlist2)}")
            tlist = self.gen.merge(tlist, tlist2)
            self._log(f"│       falselist = {flist2}  (B₂ 的假出口)")
            flist = flist2

        return self._exit("B", tlist, flist)

    # ── T → F { && M F }*  ──

    def _T(self) -> Tuple[List[int], List[int]]:
        """T → F { && M F }*"""
        self._enter("T → F")

        tlist, flist = self._F()

        while self.peek()[0] == TOKEN_AND:
            self.consume()
            self._log(f"│ 看到 '&&'")
            M_quad = self._M()
            self._log(f"│ T₁.truelist = {tlist}  →  回填到 M.quad = L{M_quad}")
            self.gen.backpatch(tlist, M_quad)
            tlist2, flist2 = self._F()
            self._log(f"│ 合并: falselist = merge({flist}, {flist2}) = {self.gen.merge(flist, flist2)}")
            flist = self.gen.merge(flist, flist2)
            self._log(f"│       truelist = {tlist2}  (F₂ 的真出口)")
            tlist = tlist2

        return self._exit("T", tlist, flist)

    # ── F → ! F | ( B ) | E relop E | true | false | id ──

    def _F(self) -> Tuple[List[int], List[int]]:
        tok = self.peek()

        if tok[0] == TOKEN_NOT:
            self.consume()
            self._enter("F → ! F")
            tlist, flist = self._F()
            self._log(f"│ ! 取反: truelist ↔ falselist")
            self._log(f"│   原: truelist={tlist}, falselist={flist}")
            self._log(f"│   新: truelist={flist}, falselist={tlist}")
            return self._exit("F (not)", flist, tlist)

        elif tok[0] == TOKEN_LPAREN:
            self.consume()
            self._enter("F → ( B )")
            tlist, flist = self._B()
            self.match(TOKEN_RPAREN)
            return self._exit("F (paren)", tlist, flist)

        elif tok[0] == TOKEN_TRUE:
            self.consume()
            self._enter("F → true")
            i = self.gen.nextinstr
            self.gen.emit('jump')
            self._log(f"│ 生成 goto _  [{i}], truelist=[{i}]")
            return self._exit("F (true)", [i], [])

        elif tok[0] == TOKEN_FALSE:
            self.consume()
            self._enter("F → false")
            i = self.gen.nextinstr
            self.gen.emit('jump')
            self._log(f"│ 生成 goto _  [{i}], falselist=[{i}]")
            return self._exit("F (false)", [], [i])

        elif tok[0] == TOKEN_ID:
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == 'RELOP':
                # E1 relop E2
                id1 = self.consume()[1]
                relop = self.consume()[1]
                id2 = self.consume()[1]
                self._enter(f"F → id relop id  ({id1} {relop} {id2})")
                i1 = self.gen.nextinstr
                self.gen.emit(relop, id1, id2)
                i2 = self.gen.nextinstr
                self.gen.emit('jump')
                self._log(f"│ 生成 [{i1}] if {id1} {relop} {id2} goto _, [{i2}] goto _")
                self._log(f"│ truelist=[{i1}], falselist=[{i2}]")
                return self._exit("F (relop)", [i1], [i2])
            else:
                id_val = self.consume()[1]
                self._enter(f"F → id  ({id_val})")
                i1 = self.gen.nextinstr
                self.gen.emit('jtrue', id_val)
                i2 = self.gen.nextinstr
                self.gen.emit('jump')
                self._log(f"│ 生成 [{i1}] if {id_val} goto _, [{i2}] goto _")
                self._log(f"│ truelist=[{i1}], falselist=[{i2}]")
                return self._exit("F (id)", [i1], [i2])

        else:
            raise ValueError(f"意外的 token: {tok}")

    # ── M → ε ──

    def _M(self) -> int:
        """标记非终结符: 返回下一条指令的编号。"""
        m = self.gen.nextinstr
        self._log(f"│ M → ε,  M.quad = {m}  (L{m})")
        return m


# ═══════════════════════════════════════════════════════════════
# 5. 顶层流水线
# ═══════════════════════════════════════════════════════════════

def translate_boolean(source: str):
    """翻译布尔表达式为三地址码。"""

    print(f"\n{'='*60}")
    print(f"  布尔表达式:  {source}")
    print(f"{'='*60}")

    # 词法分析
    lexer = Lexer(source)
    tokens = lexer.token_list()
    token_str = ' '.join(f"{t}({v})" if v else t for t, v in tokens[:-1])
    print(f"\n  Tokens: {token_str}")

    # SDD 翻译（含中间调试过程）
    sdd_trace: List[str] = []
    gen = CodeGen(trace=sdd_trace)
    translator = SDDTranslator(tokens, gen, trace=sdd_trace)

    tlist, flist = translator.translate()

    # 最终出口回填
    L_true = gen.nextinstr
    L_false = L_true + 1
    sdd_trace.append(f"\n  最终回填:")
    sdd_trace.append(f"    B.truelist  = {tlist}  →  L{L_true}  (true 出口)")
    sdd_trace.append(f"    B.falselist = {flist}  →  L{L_false}  (false 出口)")
    gen.backpatch(tlist, L_true)
    gen.backpatch(flist, L_false)

    # ── 输出 SDD 翻译过程 ──
    print(f"\n  ── SDD 翻译过程 (属性求值 & 回填) ──")
    for line in sdd_trace:
        print(f"  {line}")

    # ── 输出最终代码 ──
    code_lines = gen.get_code()
    print(f"\n  生成的三地址码:")
    print(f"  {'─'*45}")
    for line in code_lines:
        print(f"  {line}")
    print(f"  L{L_true}: ...  // true  出口")
    print(f"  L{L_false}: ...  // false 出口")
    print(f"  {'─'*45}")

    return gen


# ═══════════════════════════════════════════════════════════════
# 6. 测试用例
# ═══════════════════════════════════════════════════════════════

def run_builtin_tests():
    tests = [
        # 混合 && || !
        "x > 5 && x < 10 || x ==y",
    ]
    for expr in tests:
        translate_boolean(expr)
        print()


# ═══════════════════════════════════════════════════════════════
# 7. main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) > 1:
        translate_boolean(sys.argv[1])
    else:
        run_builtin_tests()

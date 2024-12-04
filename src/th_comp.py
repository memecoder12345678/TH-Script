#-------------------------------------------------------#
#                       TH-Script                       #
#       Author: MemeCoder (memecoder17@gmail.com)       #
#-------------------------------------------------------#
from typing import List, Optional, Union
import subprocess
import os
import sys
import platform

have_return = False
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

class Node:
    def __str__(self):
        return f"{CYAN}{self.__class__.__name__}{RESET}"

class NodeTerm(Node):
    pass

class NodeStmt(Node):
    pass

class NodeExpr(Node):
    pass

class NodeTermIntLit(NodeTerm):
    def __init__(self, value: int):
        self.value = value
    
    def __str__(self):
        return f"{YELLOW}IntLit({self.value}){RESET}"

class NodeTermIdent(NodeTerm):
    def __init__(self, value: str):
        self.value = value
    
    def __str__(self):
        return f"{MAGENTA}Ident({self.value}){RESET}"

class NodeTermParen(NodeTerm):
    def __init__(self, expr: NodeExpr):
        self.expr = expr
    
    def __str__(self):
        return f"{BLUE}Paren({self.expr}){RESET}"

class NodeBinExpr(NodeExpr):
    def __init__(self, lhs: NodeExpr, rhs: NodeExpr):
        self.lhs = lhs
        self.rhs = rhs

class NodeBinExprAdd(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Add({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprSub(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Sub({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprMulti(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Multi({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprDiv(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Div({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprAnd(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}And({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprOr(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Or({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprEq(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Eq({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprNeq(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Neq({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprLt(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Lt({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprGt(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Gt({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprLte(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Lte({self.lhs}, {self.rhs}){RESET}"

class NodeBinExprGte(NodeBinExpr):
    def __str__(self):
        return f"{GREEN}Gte({self.lhs}, {self.rhs}){RESET}"

class NodeStmtReturn(NodeStmt):
    def __init__(self, expr: NodeExpr):
        self.expr = expr
    
    def __str__(self):
        return f"{RED}Return({self.expr}){RESET}"

class NodeStmtAssign(NodeStmt):
    def __init__(self, ident: str, expr: NodeExpr):
        self.ident = ident
        self.expr = expr
    
    def __str__(self):
        return f"{BLUE}Assign({self.ident}, {self.expr}){RESET}"

class NodeStmtComment(NodeStmt):
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return f"{CYAN}Comment({self.content}){RESET}"

class NodeStmtIf(NodeStmt):
    def __init__(self, condition: NodeExpr, true_block: List[NodeStmt], false_block: Optional[List[NodeStmt]] = None):
        self.condition = condition
        self.true_block = true_block
        self.false_block = false_block
    
    def __str__(self):
        return f"{BLUE}If({self.condition}, {self.true_block}, {self.false_block}){RESET}"

class NodeStmtWhile(NodeStmt):
    def __init__(self, condition: NodeExpr, block: List[NodeStmt]):
        self.condition = condition
        self.block = block
    
    def __str__(self):
        return f"{BLUE}While({self.condition}, {self.block}){RESET}"

class NodeStmtFor(NodeStmt):
    def __init__(self, init: NodeStmt, condition: NodeExpr, update: NodeStmt, block: List[NodeStmt]):
        self.init = init
        self.condition = condition
        self.update = update
        self.block = block
    
    def __str__(self):
        return f"{BLUE}For({self.init}, {self.condition}, {self.update}, {self.block}){RESET}"

class NodeScope:
    def __init__(self, stmts: List[NodeStmt]):
        self.stmts = stmts
    
    def __str__(self):
        return f"{MAGENTA}Scope({', '.join(str(stmt) for stmt in self.stmts)}){RESET}"

class NodeProg:
    def __init__(self, stmts: List[NodeStmt]):
        self.stmts = stmts
    
    def __str__(self):
        return f"{RED}Program({', '.join(str(stmt) for stmt in self.stmts)}){RESET}"

# --- Token Definitions ---
class TokenType:
    RETURN = "return"
    INT_LIT = "int_lit"
    SEMI = ";"
    IDENT = "ident"
    ASSIGN = "assign"
    EQ = "="
    PLUS = "+"
    STAR = "*"
    MINUS = "-"
    FSLASH = "/"
    OPEN_PAREN = "("
    CLOSE_PAREN = ")"
    COMMENT = "comment"
    IF = "if"
    ELSE = "else"
    WHILE = "while"
    FOR = "for"
    OPEN_BRACE = "{"
    CLOSE_BRACE = "}"
    COMMA = ","
    AND = "&&"
    OR = "||"
    EQEQ = "=="
    NEQ = "!="
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="

class Token:
    def __init__(self, type: str, line: int, value: Optional[Union[int, str]] = None):
        self.type = type
        self.line = line
        self.value = value

# --- Tokenizer ---
class Tokenizer:
    def __init__(self, src: str):
        self.src = src
        self.index = 0

    def peek(self, offset: int = 0) -> Optional[str]:
        if self.index + offset >= len(self.src):
            return None
        return self.src[self.index + offset]

    def consume(self) -> str:
        char = self.src[self.index]
        self.index += 1
        return char

    def tokenize(self) -> List[Token]:
        global have_return
        tokens = []
        line_count = 1

        while self.index < len(self.src):
            char = self.peek()

            if char == "@":
                self.consume()
                char = self.peek()
                while char and char in " \t":
                    self.consume()
                    char = self.peek()
                comment = ""
                while char and char != "\n":
                    comment += self.consume()
                    char = self.peek()
                tokens.append(Token(TokenType.COMMENT, line_count, comment))

            elif char.isalpha():
                buffer = ""
                while char and char.isalnum():
                    buffer += self.consume()
                    char = self.peek()
                if buffer == "return":
                    have_return = True
                    tokens.append(Token(TokenType.RETURN, line_count))
                elif buffer == "assign":
                    tokens.append(Token(TokenType.ASSIGN, line_count))
                elif buffer == "if":
                    tokens.append(Token(TokenType.IF, line_count))
                elif buffer == "else":
                    tokens.append(Token(TokenType.ELSE, line_count))
                elif buffer == "while":
                    tokens.append(Token(TokenType.WHILE, line_count))
                elif buffer == "for":
                    tokens.append(Token(TokenType.FOR, line_count))
                else:
                    tokens.append(Token(TokenType.IDENT, line_count, buffer))

            elif char.isdigit():
                buffer = ""
                while char and char.isdigit():
                    buffer += self.consume()
                    char = self.peek()
                tokens.append(Token(TokenType.INT_LIT, line_count, int(buffer)))

            elif char in ";=+-*/(){},":
                tokens.append(Token(getattr(TokenType, {
                    ";": "SEMI",
                    "=": "EQ",
                    "+": "PLUS",
                    "-": "MINUS",
                    "*": "STAR",
                    "/": "FSLASH",
                    "(": "OPEN_PAREN",
                    ")": "CLOSE_PAREN",
                    "{": "OPEN_BRACE",
                    "}": "CLOSE_BRACE",
                    ",": "COMMA"
                }[char]), line_count))
                self.consume()

            elif char == "&" and self.peek(1) == "&":
                tokens.append(Token(TokenType.AND, line_count))
                self.consume()
                self.consume()

            elif char == "|" and self.peek(1) == "|":
                tokens.append(Token(TokenType.OR, line_count))
                self.consume()
                self.consume()

            elif char == "=" and self.peek(1) == "=":
                tokens.append(Token(TokenType.EQEQ, line_count))
                self.consume()
                self.consume()

            elif char == "!" and self.peek(1) == "=":
                tokens.append(Token(TokenType.NEQ, line_count))
                self.consume()
                self.consume()

            elif char == "<":
                if self.peek(1) == "=":
                    tokens.append(Token(TokenType.LTE, line_count))
                    self.consume()
                    self.consume()
                else:
                    tokens.append(Token(TokenType.LT, line_count))
                    self.consume()

            elif char == ">":
                if self.peek(1) == "=":
                    tokens.append(Token(TokenType.GTE, line_count))
                    self.consume()
                    self.consume()
                else:
                    tokens.append(Token(TokenType.GT, line_count))
                    self.consume()

            elif char.isspace():
                if char == "\n":
                    line_count += 1
                self.consume()
            else:
                print(f"{RED}ERROR: Unexpected character: {char} at line {line_count}{RESET}")
                sys.exit(1)

        return tokens

# --- Parser ---
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.index = 0

    def peek(self, offset: int = 0) -> Optional[Token]:
        if self.index + offset >= len(self.tokens):
            return None
        return self.tokens[self.index + offset]

    def consume(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def parse_term(self) -> NodeTerm:
        token = self.consume()
        if token.type == TokenType.INT_LIT:
            return NodeTermIntLit(token.value)
        elif token.type == TokenType.IDENT:
            return NodeTermIdent(token.value)
        elif token.type == TokenType.OPEN_PAREN:
            expr = self.parse_expr()
            if self.consume().type != TokenType.CLOSE_PAREN:
                print(f"{RED}ERROR: Expected closing parenthesis at line {token.line}{RESET}")
                sys.exit(1)
            return NodeTermParen(expr)
        else:
            print(f"{RED}ERROR: Unexpected token in term at line {token.line}{RESET}")
            sys.exit(1)

    def parse_expr(self) -> NodeExpr:
        lhs = self.parse_term()
        while True:
            token = self.peek()
            if token and token.type in {TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.FSLASH, 
                                        TokenType.AND, TokenType.OR, TokenType.EQEQ, TokenType.NEQ, 
                                        TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE}:
                op = self.consume().type
                rhs = self.parse_term()
                if op == TokenType.PLUS:
                    lhs = NodeBinExprAdd(lhs, rhs)
                elif op == TokenType.MINUS:
                    lhs = NodeBinExprSub(lhs, rhs)
                elif op == TokenType.STAR:
                    lhs = NodeBinExprMulti(lhs, rhs)
                elif op == TokenType.FSLASH:
                    lhs = NodeBinExprDiv(lhs, rhs)
                elif op == TokenType.AND:
                    lhs = NodeBinExprAnd(lhs, rhs)
                elif op == TokenType.OR:
                    lhs = NodeBinExprOr(lhs, rhs)
                elif op == TokenType.EQEQ:
                    lhs = NodeBinExprEq(lhs, rhs)
                elif op == TokenType.NEQ:
                    lhs = NodeBinExprNeq(lhs, rhs)
                elif op == TokenType.LT:
                    lhs = NodeBinExprLt(lhs, rhs)
                elif op == TokenType.GT:
                    lhs = NodeBinExprGt(lhs, rhs)
                elif op == TokenType.LTE:
                    lhs = NodeBinExprLte(lhs, rhs)
                elif op == TokenType.GTE:
                    lhs = NodeBinExprGte(lhs, rhs)
            else:
                break
        return lhs

    def parse_block(self) -> List[NodeStmt]:
        stmts = []
        if self.peek() is None or self.peek().type != TokenType.OPEN_BRACE:
            print(f"{RED}ERROR: Expected opening brace at line {self.peek().line if self.peek() else 'EOF'}{RESET}")
            sys.exit(1)
        self.consume()
        while self.peek() is not None and self.peek().type != TokenType.CLOSE_BRACE:
            stmts.append(self.parse_stmt())
        if self.peek() is None:
            print(f"{RED}ERROR: Unexpected end of file. Expected closing brace.{RESET}")
            sys.exit(1)
        self.consume()
        return stmts

    def parse_stmt(self) -> NodeStmt:
        token = self.peek()
        if token is None:
            print(f"{RED}ERROR: Unexpected end of file while parsing statement.{RESET}")
            sys.exit(1)
        if token.type == TokenType.COMMENT:
            return NodeStmtComment(self.consume().value)
        elif token.type == TokenType.ASSIGN:
            self.consume()
            ident = self.consume()
            if ident.type != TokenType.IDENT:
                print(f"{RED}ERROR: Expected identifier after 'assign' at line {ident.line}{RESET}")
                sys.exit(1)
            if self.consume().type != TokenType.EQ:
                print(f"{RED}ERROR: Expected '=' after identifier at line {ident.line}{RESET}")
                sys.exit(1)
            expr = self.parse_expr()
            if self.consume().type != TokenType.SEMI:
                print(f"{RED}ERROR: Expected ';' at the end of statement at line {ident.line}{RESET}")
                sys.exit(1)
            return NodeStmtAssign(ident.value, expr)
        elif token.type == TokenType.RETURN:
            self.consume()
            expr = self.parse_expr()
            if self.consume().type != TokenType.SEMI:
                print(f"{RED}ERROR: Expected ';' at the end of statement at line {token.line}{RESET}")
                sys.exit(1)
            return NodeStmtReturn(expr)
        elif token.type == TokenType.IDENT:
            ident = self.consume().value
            if self.consume().type != TokenType.EQ:
                print(f"{RED}ERROR: Expected '=' after identifier at line {token.line}{RESET}")
                sys.exit(1)
            expr = self.parse_expr()
            if self.consume().type != TokenType.SEMI:
                print(f"{RED}ERROR: Expected ';' at the end of statement at line {token.line}{RESET}")
                sys.exit(1)
            return NodeStmtAssign(ident, expr)
        elif token.type == TokenType.IF:
            self.consume()
            condition = self.parse_expr()
            true_block = self.parse_block()
            false_block = None
            if self.peek() and self.peek().type == TokenType.ELSE:
                self.consume()
                false_block = self.parse_block()
            return NodeStmtIf(condition, true_block, false_block)
        elif token.type == TokenType.WHILE:
            self.consume()
            condition = self.parse_expr()
            block = self.parse_block()
            return NodeStmtWhile(condition, block)
        elif token.type == TokenType.FOR:
            self.consume()
            init = self.parse_stmt()
            condition = self.parse_expr()
            if self.consume().type != TokenType.SEMI:
                print(f"{RED}ERROR: Expected ';' after for loop condition at line {token.line}{RESET}")
                sys.exit(1)
            update = self.parse_stmt()
            block = self.parse_block()
            return NodeStmtFor(init, condition, update, block)
        else:
            print(f"{RED}ERROR: Unexpected statement at line {token.line}{RESET}")
            sys.exit(1)

    def parse_prog(self) -> NodeProg:
        stmts = []
        while self.index < len(self.tokens):
            stmts.append(self.parse_stmt())
        return NodeProg(stmts)

# --- Generator ---
class Generator:
    def __init__(self, prog: NodeProg):
        self.m_prog = prog
        self.m_output = []
        self.m_stack_size = 0
        self.m_vars = []
        self.m_scopes = []
        self.m_label_count = 0

    def push(self, reg: str):
        self.m_output.append(f"\tpush {reg}")
        self.m_stack_size += 1

    def pop(self, reg: str):
        self.m_output.append(f"\tpop {reg}")
        self.m_stack_size -= 1

    def begin_scope(self):
        self.m_scopes.append(len(self.m_vars))

    def end_scope(self):
        pop_count = len(self.m_vars) - self.m_scopes.pop()
        if pop_count != 0:
            self.m_output.append(f"\tadd rsp, {pop_count * 8}")
        self.m_stack_size -= pop_count
        self.m_vars = self.m_vars[:len(self.m_vars) - pop_count]

    def gen_term(self, term: NodeTerm):
        if isinstance(term, NodeTermIntLit):
            self.m_output.append(f"\tmov rax, {term.value}")
            self.push("rax")
        elif isinstance(term, NodeTermIdent):
            for var in self.m_vars:
                if var["name"] == term.value:
                    offset = f"[rsp + {(self.m_stack_size - var['stack_loc'] - 1) * 8}]"
                    self.m_output.append(f"\tmov rax, {offset}")
                    self.push("rax")
                    return
            print(f"{RED}ERROR: Undeclared identifier: {term.value}{RESET}")
            sys.exit(1)
        elif isinstance(term, NodeTermParen):
            self.gen_expr(term.expr)

    def gen_bin_expr(self, bin_expr: NodeBinExpr):
        self.gen_expr(bin_expr.rhs)
        self.gen_expr(bin_expr.lhs)
        self.pop("rax")
        self.pop("rbx")
        if isinstance(bin_expr, NodeBinExprAdd):
            self.m_output.append("\tadd rax, rbx")
        elif isinstance(bin_expr, NodeBinExprSub):
            self.m_output.append("\tsub rax, rbx")
        elif isinstance(bin_expr, NodeBinExprMulti):
            self.m_output.append("\timul rbx")
        elif isinstance(bin_expr, NodeBinExprDiv):
            self.m_output.append("\txor rdx, rdx")
            self.m_output.append("\tidiv rbx")
        elif isinstance(bin_expr, NodeBinExprAnd):
            self.m_output.append("\tand rax, rbx")
        elif isinstance(bin_expr, NodeBinExprOr):
            self.m_output.append("\tor rax, rbx")
        elif isinstance(bin_expr, NodeBinExprEq):
            self.m_output.append("\tcmp rax, rbx")
            self.m_output.append("\tsete al")
            self.m_output.append("\tmovzx rax, al")
        elif isinstance(bin_expr, NodeBinExprNeq):
            self.m_output.append("\tcmp rax, rbx")
            self.m_output.append("\tsetne al")
            self.m_output.append("\tmovzx rax, al")
        elif isinstance(bin_expr, NodeBinExprLt):
            self.m_output.append("\tcmp rax, rbx")
            self.m_output.append("\tsetl al")
            self.m_output.append("\tmovzx rax, al")
        elif isinstance(bin_expr, NodeBinExprGt):
            self.m_output.append("\tcmp rax, rbx")
            self.m_output.append("\tsetg al")
            self.m_output.append("\tmovzx rax, al")
        elif isinstance(bin_expr, NodeBinExprLte):
            self.m_output.append("\tcmp rax, rbx")
            self.m_output.append("\tsetle al")
            self.m_output.append("\tmovzx rax, al")
        elif isinstance(bin_expr, NodeBinExprGte):
            self.m_output.append("\tcmp rax, rbx")
            self.m_output.append("\tsetge al")
            self.m_output.append("\tmovzx rax, al")
        self.push("rax")

    def gen_expr(self, expr: NodeExpr):
        if isinstance(expr, NodeTerm):
            self.gen_term(expr)
        elif isinstance(expr, NodeBinExpr):
            self.gen_bin_expr(expr)

    def gen_stmt(self, stmt: NodeStmt):
        if isinstance(stmt, NodeStmtComment):
            self.m_output.append(f"\t; {stmt.content}")
        elif isinstance(stmt, NodeStmtReturn):
            self.gen_expr(stmt.expr)
            self.m_output.append("\tmov rcx, rax")
            self.m_output.append("\tcall ExitProcess")
        elif isinstance(stmt, NodeStmtAssign):
            self.gen_expr(stmt.expr)
            for var in self.m_vars:
                if var["name"] == stmt.ident:
                    self.pop("rax")
                    offset = f"[rsp + {(self.m_stack_size - var['stack_loc'] - 1) * 8}]"
                    self.m_output.append(f"\tmov {offset}, rax")
                    return
            self.m_vars.append({"name": stmt.ident, "stack_loc": self.m_stack_size - 1})
        elif isinstance(stmt, NodeStmtIf):
            self.gen_expr(stmt.condition)
            self.pop("rax")
            self.m_output.append("\tcmp rax, 0")
            label_else = f".L_else_{self.m_label_count}"
            label_end = f".L_end_{self.m_label_count}"
            self.m_label_count += 1
            self.m_output.append(f"\tje {label_else}")
            self.begin_scope()
            for s in stmt.true_block:
                self.gen_stmt(s)
            self.end_scope()
            self.m_output.append(f"\tjmp {label_end}")
            self.m_output.append(f"{label_else}:")
            if stmt.false_block:
                self.begin_scope()
                for s in stmt.false_block:
                    self.gen_stmt(s)
                self.end_scope()
            self.m_output.append(f"{label_end}:")
        elif isinstance(stmt, NodeStmtWhile):
            label_start = f".L_while_start_{self.m_label_count}"
            label_end = f".L_while_end_{self.m_label_count}"
            self.m_label_count += 1
            self.m_output.append(f"{label_start}:")
            self.gen_expr(stmt.condition)
            self.pop("rax")
            self.m_output.append("\tcmp rax, 0")
            self.m_output.append(f"\tje {label_end}")
            self.begin_scope()
            for s in stmt.block:
                self.gen_stmt(s)
            self.end_scope()
            self.m_output.append(f"\tjmp {label_start}")
            self.m_output.append(f"{label_end}:")
        elif isinstance(stmt, NodeStmtFor):
            self.gen_stmt(stmt.init)
            label_start = f".L_for_start_{self.m_label_count}"
            label_end = f".L_for_end_{self.m_label_count}"
            self.m_label_count += 1
            self.m_output.append(f"{label_start}:")
            self.gen_expr(stmt.condition)
            self.pop("rax")
            self.m_output.append("\tcmp rax, 0")
            self.m_output.append(f"\tje {label_end}")
            self.begin_scope()
            for s in stmt.block:
                self.gen_stmt(s)
            self.end_scope()
            self.gen_stmt(stmt.update)
            self.m_output.append(f"\tjmp {label_start}")
            self.m_output.append(f"{label_end}:")

    def gen_prog(self) -> str:
        self.m_output.append("extern ExitProcess")
        self.m_output.append("section .text")
        self.m_output.append("global main")
        self.m_output.append("main:")
        for stmt in self.m_prog.stmts:
            self.gen_stmt(stmt)
        if not have_return:
            self.m_output.append("\txor rcx, rcx")
            self.m_output.append("\tcall ExitProcess")
        return "\n".join(self.m_output)

# --- Main Program ---
if platform.system() != "Windows":
    print(f"{RED}ERROR: This program is intended to run on Windows.{RESET}")
    sys.exit(1)
if platform.machine().lower() != "amd64":
    print(f"{RED}ERROR: This program is intended to run on 64-bit Windows.{RESET}")
    sys.exit(1)
if len(sys.argv) == 2:
    path = os.path.dirname(sys.argv[1])
    filename = os.path.basename(sys.argv[1]).rsplit(".", 1)[0]
    try:
        source_code = open(sys.argv[1], "r").read()
    except FileNotFoundError:
        print(f"{RED}ERROR: File '{sys.argv[1]}' not found.{RESET}")
        sys.exit(1)
    except IOError as e:
        print(f"{RED}ERROR: Unable to read file '{sys.argv[1]}': {e}{RESET}")
elif len(sys.argv) == 3:
    path = os.path.dirname(sys.argv[1])
    filename = os.path.basename(sys.argv[1]).rsplit(".", 1)[0]
    try:
        source_code = open(sys.argv[1], "r").read()
    except FileNotFoundError:
        print(f"{RED}ERROR: File '{sys.argv[1]}' not found.{RESET}")
        sys.exit(1)
    except IOError as e:
        print(f"{RED}ERROR: Unable to read file '{sys.argv[1]}': {e}{RESET}")
else:
    print(f"{RED}ERROR: Please provide a source file as an argument.{RESET}")
    sys.exit(2)

output_dir = os.path.join(path, "output")
os.makedirs(output_dir, exist_ok=True)

# Tokenize and Parse
try:
    tokenizer = Tokenizer(source_code)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    prog = parser.parse_prog()
except Exception as e:
    print(f"{RED}COMPILER ERROR: Failed to parse source code: {e}{RESET}")
    sys.exit(3)
if len(sys.argv) == 3:
    if sys.argv[2] == "--asm":
        generator = Generator(prog)
        assembly_code = generator.gen_prog()
        with open(os.path.join(output_dir, filename + ".asm"), "w", encoding="utf-8") as f:
            f.write(assembly_code)
        sys.exit(0)
    elif sys.argv[2] == "--help":
        print(f"{GREEN}Help Menu:{RESET}")
        print(f"  {GREEN}--asm{RESET}          Generates assembly code from the source")
        print(f"  {GREEN}--help{RESET}         Displays this help menu and exits")
    else:
        print(f"{RED}ERROR: Unknown argument: \"{sys.argv[2]}\"{RESET}\nUse --help for more information")

# Generate Assembly
generator = Generator(prog)
assembly_code = generator.gen_prog()

# Output
with open(os.path.join(output_dir, filename + ".asm"), "w", encoding="utf-8") as f:
    f.write(assembly_code)

subprocess.run(["nasm", "-f", "win64", os.path.join(output_dir, filename + ".asm")])
subprocess.run(["gcc", "-o", os.path.join(output_dir, filename + ".exe"), os.path.join(output_dir, filename + ".obj")])
result = subprocess.run(os.path.join(output_dir, filename + ".exe"), capture_output=True, text=True)
print(result.returncode)

## grammar
```plain text
<Prog> ::= <Stmt>*

<Stmt> ::= 
    "return" <Expr> ";"
    | "assign" <Ident> "=" <Expr> ";"
    | <Ident> "=" <Expr> ";"
    | "if" "(" <Expr> ")" <Scope> <IfPred>

<Scope> ::= <Stmt>*

<IfPred> ::= 
    | "else" <Scope> 
    | ε

<Expr> ::= <Term> 
    | <BinExpr>

<BinExpr> ::= <Expr> "*" <Expr>        {prec = 1}
    | <Expr> "/" <Expr>        {prec = 1}
    | <Expr> "+" <Expr>        {prec = 0}
    | <Expr> "-" <Expr>        {prec = 0}

<Term> ::= <IntLit> 
    | <Ident> 
    | "(" <Expr> ")"

<IntLit> ::= [0-9]+   // A sequence of digits representing an integer literal

<Ident> ::= [a-zA-Z_][a-zA-Z0-9_]*   // A valid identifier starting with a letter or underscore, followed by letters, digits, or underscores
```

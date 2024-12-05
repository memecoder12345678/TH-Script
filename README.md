# **TH-Script (THS)**

## **Key Features**:

- **Variable Assignment**: Uses assign to declare and set variable values (e.g., assign x = 5;).

- **For Loops**: Iterates through a numerical range, specifying start, condition, and increment (e.g., for assign i = 0; i < 10; assign i = i + 1;).

- **Return Statement**: Ends program execution and returns a numerical exit code (e.g., return y;).

- **Math Operations**: Supports addition, subtraction, multiplication, and division.

- **Conditional Statements**: Uses if statements for comparisons (e.g., if x < 10).

- The syntax of this language is very simple, using keywords like `assign`, `for`, `return`, `while`, and `if` to clarify the programming structures. The `return` statement returns the value as an exit code. These exit codes can be accessed through system commands such as `echo $LASTEXITCODE` or `echo %errorlevel%`. Basic arithmetic and comparisons are supported within the core functionality of the language.

- The code snippet calculates a value based on a loop and returns it.

## **Run THS Program**

```bash
python .\src\th_comp.py .\test\test.ths
```
The provided syntax describes variable assignment, for loops, conditional statements, and return values within the THS programming language.

The instructions detail how to create, write, and run a THS program.

To see how the syntax looks, checks the `test` and `docs` folder.

**Dependencies**: `gcc`, `nasm`

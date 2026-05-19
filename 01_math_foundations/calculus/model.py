# ==========================================
# IMPLEMENT
# ==========================================

import numpy as np
from typing import Callable
import regex
import math


# ==========================================
# Exceptions
# ==========================================

class GradientDescentError(Exception):
    """Base error cho gradient descent."""
    pass


class DivergenceError(GradientDescentError):
    """Khi loss tăng thay vì giảm."""
    pass


class InvalidInputError(GradientDescentError):
    """Khi input không hợp lệ."""
    pass


# ==========================================
# Numerical Derivative
# ==========================================

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h


def central_dif_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


# ==========================================
# Regex Detection
# ==========================================

def has_ln(function: str) -> bool:
    pattern = r"(?<![a-zA-Z])ln(\((?:[^()]++|(?1))*\))"
    return bool(regex.search(pattern, function.replace(" ", "")))


def has_log(function: str) -> bool:
    patterns = [
        r'(?<![a-zA-Z])log(?:_[a-zA-Z0-9]+)?\(',
        r'(?<![a-zA-Z])log\('
    ]

    function = function.replace(" ", "")

    return any(
        regex.search(p, function)
        for p in patterns
    )


def has_exp(function: str) -> bool:
    pattern = r"(?<![a-zA-Z])e\^(\((?:[^()]++|(?1))*\))"
    return bool(regex.search(pattern, function.replace(" ", "")))


# ==========================================
# Formula Transform
# ==========================================

def transform_expression(expr: str) -> str:

    expr = expr.replace("^", "**")

    # ln(x) -> math.log(x)
    expr = regex.sub(
        r'(?<![a-zA-Z0-9_.])ln\(',
        'math.log(',
        expr
    )

    # log(x,2) -> math.log(x,2)
    expr = regex.sub(
        r'(?<![a-zA-Z0-9_.])log\(',
        'math.log(',
        expr
    )

    # log_2(x) -> math.log(x, 2)
    expr = regex.sub(
        r'log_([a-zA-Z0-9]+)\((.*?)\)',
        r'math.log(\2, \1)',
        expr
    )

    # e^(x) -> math.exp(x)
    expr = regex.sub(
        r'e\^\((.*?)\)',
        r'math.exp(\1)',
        expr
    )

    return expr


# ==========================================
# String -> Callable Function
# ==========================================

def str_to_formula(function_str: str) -> Callable[[float], float]:

    expr = transform_expression(function_str)

    allowed_globals = {
        "math": math,
        "__builtins__": {}
    }

    return lambda x: eval(
        expr,
        allowed_globals,
        {"x": x}
    )


# ==========================================
# Derivative Wrapper
# ==========================================

def trans_to_derivative(function_str: str) -> Callable[[float], float]:

    f = str_to_formula(function_str)

    if has_ln(function_str) or has_log(function_str):
        print("Warning: log/ln detected — avoid invalid domain")

    if has_exp(function_str):
        print("Warning: exp detected — avoid very large x")

    return lambda x: central_dif_derivative(f, x)


# ==========================================
# Gradient Descent
# ==========================================

def gradient_descent(
    w: np.ndarray,
    loss_fn: callable,
    grad_fn: callable,
    learning_rate: float,
    x: float,
    n_steps: int
) -> tuple[np.ndarray, list[float]]:

    losses = []

    for step in range(n_steps):

        loss = loss_fn(x)
        grad = grad_fn(x)

        # gradient descent update
        x = x - learning_rate * grad

        # weight update
        w = w - learning_rate * (2 * w)

        losses.append(loss)

        # divergence detection
        if step > 0 and losses[-1] > losses[-2]:
            raise DivergenceError(
                f"Loss diverged at step {step}"
            )

    return w, losses


# ==========================================
# CLI
# ==========================================

def cli():

    print("=" * 50)
    print(" SIMPLE FORMULA DERIVATIVE TESTER ")
    print("=" * 50)

    while True:

        try:

            formula = input("\nFormula f(x) = ")

            if formula.lower() in ["exit", "quit"]:
                print("Bye.")
                break

            x = float(input("x = "))

            print("\n[INFO] Parsing formula...")

            # build function
            f = str_to_formula(formula)

            # build derivative
            df = trans_to_derivative(formula)

            # evaluate
            fx = f(x)
            dfx = df(x)

            print("\n========== RESULT ==========")
            print(f"Parsed expression : {transform_expression(formula)}")
            print(f"f({x})  = {fx}")
            print(f"f'({x}) = {dfx}")
            print("============================")

        except ZeroDivisionError:
            print("[ERROR] Division by zero")

        except ValueError:
            print("[ERROR] Invalid numeric input or math domain")

        except SyntaxError:
            print("[ERROR] Invalid formula syntax")

        except OverflowError:
            print("[ERROR] Numerical overflow")

        except DivergenceError as e:
            print(f"[DIVERGENCE] {e}")

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    cli()
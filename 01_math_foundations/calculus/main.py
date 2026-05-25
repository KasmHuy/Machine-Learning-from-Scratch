# ==========================================
# IMPLEMENT
# ==========================================

import numpy as np
from typing import Callable
import regex
import math
import matplotlib.pyplot as plt
import numpy as np

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
    print("PARSED:", expr)
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



# String -> Callable Function

def str_to_formula(function_str: str) -> Callable[[float], float]:

    expr = transform_expression(function_str)

    allowed_globals = {
        "math": math,
        "e": math.e,
        "abs": abs,
        "__builtins__": {}
    }

    return lambda x: eval(
        expr,
        allowed_globals,
        {"x": x}
    )

# Derivative Wrapper

def trans_to_derivative(function_str: str) -> Callable[[float], float]:

    f = str_to_formula(function_str)

    if has_ln(function_str) or has_log(function_str):
        print("Warning: log/ln detected — avoid invalid domain")

    if has_exp(function_str):
        print("Warning: exp detected — avoid very large x")

    return lambda x: central_dif_derivative(f, x)


# Gradient Descent

def gradient_descent(
    loss_fn: callable,
    grad_fn: callable,
    learning_rate: float,
    x: float,
    n_steps: int
) -> tuple[list[float], list[tuple[float, float]]]:

    losses = []
    points = []

    for step in range(n_steps):
        
        loss = loss_fn(x)
        points.append((x, loss))
        grad = grad_fn(x)

        # gradient descent update
        x = x - learning_rate * grad

        losses.append(loss)
        

        # divergence detection
        if not np.isfinite(loss):
            raise DivergenceError(
                f"Loss diverged at step {step}"
            )

    return losses, points


# CLI

def fomula_test_cli():
    print(" SIMPLE FORMULA DERIVATIVE TESTER ")

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

            print(f"Parsed expression : {transform_expression(formula)}")
            print(f"f({x})  = {fx}")
            print(f"f'({x}) = {dfx}")

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
        graph(func = formula, learning_rate=0.05, a=-x-1, x0 =x+1)

# Gradient descent showcase:
def graph(func: str, learning_rate: float, x0: float, a: float):
    f = str_to_formula(func)
    x_vals = np.linspace(a, x0, 500)
    y_vals = [f(i) for i in x_vals]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # function graph
    ax[0].plot(
        x_vals,
        y_vals,
        label=f'y = {func}'
    )

    losses, points = gradient_descent(
        loss_fn=f,
        grad_fn=trans_to_derivative(func),
        learning_rate=learning_rate,
        x=x0,
        n_steps=1000
    )

    px = [p[0] for p in points]
    py = [p[1] for p in points]

    ax[0].plot(
        px,
        py,
        marker='o',
        linestyle='--',
        label='gradient descent'
    )

    ax[0].scatter(px, py)

    ax[0].set_xlim(a, x0)

    ax[0].set_title("Function Graph")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("f(x)")
    ax[0].legend()
    ax[0].grid(True)

    # loss curve
    ax[1].plot(
        range(len(losses)),
        losses,
        marker='o',
        label='loss curve'
    )

    ax[1].set_title("Loss Curve")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fomula_test_cli()
import model
import matplotlib.pyplot as plt
import numpy as np
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
            f = model.str_to_formula(formula)

            # build derivative
            df = model.trans_to_derivative(formula)

            # evaluate
            fx = f(x)
            dfx = df(x)

            print(f"Parsed expression : {model.transform_expression(formula)}")
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

        except model.DivergenceError as e:
            print(f"[DIVERGENCE] {e}")

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")

# Gradient descent showcase:
def graph(learning_rate: float, x0: float, a: float):
    func = 'x**2'
    f = model.str_to_formula(func)
    x_vals = np.linspace(a, x0, 500)
    y_vals = [f(i) for i in x_vals]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # function graph
    ax[0].plot(
        x_vals,
        y_vals,
        label=f'y = {func}'
    )

    losses, points = model.gradient_descent(
        loss_fn=f,
        grad_fn=model.trans_to_derivative(func),
        learning_rate=learning_rate,
        x=x0,
        n_steps=100
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
    graph(learning_rate=0.02, a=-30, x0=30.0)
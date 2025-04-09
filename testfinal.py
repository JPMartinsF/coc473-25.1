import numpy as np
import matplotlib.pyplot as plt

def gauss_elimination(A, b, pivot=False):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    for i in range(n):
        if pivot:
            max_row = np.argmax(abs(A[i:, i])) + i
            if A[max_row, i] == 0:
                raise ValueError("Pivoting failed: zero column.")
            if max_row != i:
                A[[i, max_row]] = A[[max_row, i]]
                b[[i, max_row]] = b[[max_row, i]]
        elif A[i, i] == 0:
            raise ValueError("Zero pivot detected without partial pivoting.")

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

def lu_factorization(A, pivot=False):
    n = A.shape[0]
    A = A.copy().astype(float)
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for i in range(n):
        if pivot:
            max_row = np.argmax(abs(U[i:, i])) + i
            if U[max_row, i] == 0:
                raise ValueError("Zero pivot with partial pivoting")
            if max_row != i:
                U[[i, max_row]] = U[[max_row, i]]
                P[[i, max_row]] = P[[max_row, i]]
                if i > 0:
                    L[[i, max_row], :i] = L[[max_row, i], :i]
        elif U[i, i] == 0:
            raise ZeroDivisionError("Zero pivot encountered without pivoting")

        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]

    return L, U, P

def solve_lu(L, U, P, b):
    b = np.dot(P, b)
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def jacobi_method(A, b, tol=1e-8, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    res_norms = []

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            if A[i, i] == 0:
                raise ZeroDivisionError("Division by zero detected in Jacobi.")
            x_new[i] = (b[i] - s) / A[i, i]

        res = np.linalg.norm(np.dot(A, x_new) - b)
        res_norms.append(res)

        if res < tol:
            break
        x = x_new

    return x, res_norms

def verify_solution(A, x, b):
    b_calc = np.dot(A, x)
    error = np.linalg.norm(b - b_calc)
    print(f"Solution error: {error:.6e}")

def plot_jacobi_errors(cases, max_iter=100, tol=1e-8):
    plt.figure(figsize=(12, 7))
    
    for i, (A, b) in enumerate(cases, 1):
        try:
            if np.any(np.diag(A) == 0):
                print(f"Case {i} skipped: zero on diagonal.")
                continue

            _, res_norms = jacobi_method(A.copy(), b.copy(), tol=tol, max_iter=max_iter)

            A_str = np.array2string(A, precision=1, suppress_small=True, max_line_width=60)
            A_str_short = A_str.replace('\n', ' ')
            label = f"Case {i}: A={A_str_short}"

            plt.semilogy(range(1, len(res_norms) + 1), res_norms, marker='o', label=label)
        except Exception as e:
            print(f"Error in case {i}: {e}")

    plt.title("Jacobi Convergence for Different Systems (matrices in legend)")
    plt.xlabel("Iterations")
    plt.ylabel("Residual norm (log scale)")
    plt.legend(fontsize="small", loc="upper right", bbox_to_anchor=(1.0, 1.0))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Case 1
A1 = np.array([[10, -1, 2], [-1, 11, -1], [2, -1, 10]], dtype=float)
b1 = np.array([6, 25, -11], dtype=float)

# Case 2
A2 = np.array([[4, 1], [2, 3]], dtype=float)
b2 = np.array([1, 2], dtype=float)

# Case 3 – Larger size with guaranteed stability
A3 = np.eye(5) * 5 + np.random.rand(5, 5)
np.fill_diagonal(A3, 10)
b3 = np.ones(5)

# Case 4
A4 = np.array([[5, 2, 1], [1, 6, 1], [1, 1, 7]], dtype=float)
b4 = np.array([1, 2, 3], dtype=float)

# Case 5 – Jacobi fails due to division by zero
A5 = np.array([[0, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
b5 = np.array([1, 2, 3], dtype=float)

# List of test cases
cases = [(A1, b1), (A2, b2), (A3, b3), (A4, b4), (A5, b5)]

# === Run all methods ===
for idx, (A, b) in enumerate(cases):
    print(f"\n=== Case {idx + 1} ===")
    
    print("\nGauss without pivoting:")
    try:
        x_gauss = gauss_elimination(A.copy(), b.copy(), pivot=False)
        verify_solution(A, x_gauss, b)
    except Exception as e:
        print(f"Error: {e}")

    print("\nGauss with pivoting:")
    try:
        x_gauss_pivot = gauss_elimination(A.copy(), b.copy(), pivot=True)
        verify_solution(A, x_gauss_pivot, b)
    except Exception as e:
        print(f"Error: {e}")

    print("\nLU without pivoting:")
    try:
        L, U, P = lu_factorization(A.copy(), pivot=False)
        x_lu = solve_lu(L, U, P, b.copy())
        verify_solution(A, x_lu, b)
    except Exception as e:
        print(f"Error: {e}")

    print("\nLU with pivoting:")
    try:
        Lp, Up, Pp = lu_factorization(A.copy(), pivot=True)
        x_lup = solve_lu(Lp, Up, Pp, b.copy())
        verify_solution(A, x_lup, b)
    except Exception as e:
        print(f"Error: {e}")

    print("\nJacobi:")
    try:
        x_jacobi, res_norms = jacobi_method(A.copy(), b.copy(), tol=1e-8, max_iter=100)
        verify_solution(A, x_jacobi, b)
    except Exception as e:
        print(f"Error: {e}")

# === Plot errors ===
plot_jacobi_errors(cases, max_iter=100, tol=1e-8)

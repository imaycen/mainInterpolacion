#   Codigo que implementa el esquema numerico 
#   de interpolacion para determinar la raiz de
#   una ecuacion
# 
#           Autor:
#   Dr. Ivan de Jesus May-Cen
#   imaycen@hotmail.com
#   Version 1.0 : 11/02/2025
#
import numpy as np
import matplotlib.pyplot as plt

# Función original
def f(x):
    return x**3 - 4*x + 1

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2  # Retorna la mejor estimación de la raíz

# Selección de tres puntos de interpolación
x0 = 1.0
x1 = 1.5
x2 = 3.0
x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio interpolante
# mediante interpolacion de Lagrange
x_vals = np.linspace(x0, x2, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
# en el intervalo inducido por los puntos donde se hace la interpolacion
root = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = x^3 - 4x + 1", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices.png")  # Guarda la imagen
plt.show()

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.4f}")


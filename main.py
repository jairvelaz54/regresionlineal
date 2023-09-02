import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        calculate_regression(file_path)

def calculate_regression(file_path):
    # Carga los datos desde un archivo CSV
    data = pd.read_csv(file_path)

    # Extrae las columnas de x1, x2 y y
    x1 = data['x1'].values
    x2 = data['x2'].values
    y = data['y'].values

    # Paso 1: Organiza los datos en una matriz de características X
    X = np.column_stack((x1, x2))
    result_text.insert(tk.END, "Matriz de características X:\n")
    result_text.insert(tk.END, f"{X}\n\n")

    # Paso 2: Agrega una columna de unos a X para el término de sesgo (intercept)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    result_text.insert(tk.END, "Matriz X con término de sesgo:\n")
    result_text.insert(tk.END, f"{X_b}\n\n")

    # Paso 3: Calcula X^T (transpuesta de X)
    X_transpose = X_b.T
    result_text.insert(tk.END, "Transpuesta de X:\n")
    result_text.insert(tk.END, f"{X_transpose}\n\n")

    # Paso 4: Calcula X^T * X
    X_transpose_X = np.dot(X_transpose, X_b)
    result_text.insert(tk.END, "Cálculo de X^T * X:\n")
    result_text.insert(tk.END, f"{X_transpose_X}\n\n")

    # Calcula (X^T * X)^(-1) (inversa de X^T * X)
    X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
    result_text.insert(tk.END, "Inversa de (X^T * X):\n")
    result_text.insert(tk.END, f"{X_transpose_X_inverse}\n\n")

    # Calcula X^T * y
    X_transpose_y = np.dot(X_transpose, y)
    result_text.insert(tk.END, "X^T * y:\n")
    result_text.insert(tk.END, f"{X_transpose_y}\n\n")

    # Paso 5: Calcula los coeficientes de la regresión matricial
    beta = np.dot(X_transpose_X_inverse, X_transpose_y)
    result_text.insert(tk.END, "Coeficientes de la regresión:\n")
    for i in range(beta.shape[0]):
        result_text.insert(tk.END, f"beta[{i}] = {beta[i]}\n")

root = tk.Tk()
root.title("Regresión Lineal Matricial")

select_button = tk.Button(root, text="Seleccionar archivo CSV", command=select_file)
select_button.pack()

result_text = tk.Text(root)
result_text.pack()

root.mainloop()

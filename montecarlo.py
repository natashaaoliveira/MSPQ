# Estimação de parâmetro pelo método de Monte Carlo
# Natasha Oliveira dos Santos 

import numpy as np

k = 0
k_max = 100

shape_MM_inter = (k_max, 1)
MM_inter = np.zeros(shape_MM_inter)
obj_MC_inter = np.zeros(k_max)


# Função modelo
def temperatura(t):
    f_calc = -0.058 * (t**5) + 2.127 * (t**4) - 28.565 * (t**3) + 171.46 * (t**2)-443.64 * t + 459.52
    return f_calc


# intervalo de busca
xmin_MM = 1# h
xmax_MM = 13# h

# Método de Monte Carlo

while k < k_max:
    random_MM = np.random.random()  # gera n° aleatórios
    MM_inter[k, 0] = xmin_MM + random_MM * (xmax_MM- xmin_MM)  # ajusta o parâmetro
    obj_MC_inter[k] = temperatura(MM_inter[k, 0])  # Realiza o cálculo utilizando a função objetivo
    k += 1

# Melhor resultado para o método
best_func_ML = np.min(obj_MC_inter)
best_min_loc = np.argmin(obj_MC_inter)
best_MM_MC = MM_inter[best_min_loc, 0]

print("best_func_ML:", best_func_ML)
print("best_min_loc", best_min_loc)
print("best_MM_MC:", best_MM_MC)

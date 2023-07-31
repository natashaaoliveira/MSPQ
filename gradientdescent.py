#Método de busca multivariável pelo método do gradient descent 
#criado 26/06/2023
import numpy as np
from numpy import ndarray

x = [4, 2, -1]  # Valor inicial para x
k = 0 # iteração inicial
k_max = 100 # máximo de iteração
tol = 1.0E-15
k_NR = 0
k_NRmax =  100
tol_NR = 1.0E-20
converge_NR = False
alfa = 0.0
k_1 = 0


# Função objetivo: S(x1, x2, x3) = (x1 - 4)^4 + (x2 - 3)^2 + 4(x3 + 5)^4
def objective_func ( x ):
    return (x[0] - 4)**4 + (x[1] - 3)**2 + 4*(x[2] + 5)**4

# Gradiente da função objetivo
def gradient_func(x):
    gradiente = np.zeros(3)         # o tamanho do vetor
    gradiente[0] = 4*(x[0] - 4)**3
    gradiente[1] = 2*(x[1] - 3)
    gradiente[2] = 16*(x[2] + 5)**3
    return gradiente


#ortogonalidade
def orto(x, alfa):
    tamanho = (len(x), 1)
    eq_i = np.zeros(tamanho)
    eq_i[0, 0] = 4*(x[0] - 4)**3
    eq_i[1, 0] = 2*(x[1] - 3)
    eq_i[2, 0] = 16*(x[2] + 5)**3

    x_k1 = x - alfa* gradient_func(x)

# Cálculo para a transposta
    eq_iT = np.zeros(tamanho)
    eq_iT[0, 0] = 4*(x_k1[0] - 4)**3
    eq_iT[1, 0] = 2*(x_k1[1] - 3)
    eq_iT[2, 0] = 16*(x_k1[2] + 5)**3
    gradient_ft = np.transpose(eq_iT)

# multiplicação dos dois gradientes
    grad_ft_grad_f = -np.matmul(gradient_ft, eq_i)
    return grad_ft_grad_f.item()

def d_orto(x, alfa):
    delta = 1.0E-2
    x10_dfNr = alfa + 2.0 * delta
    x11_dfNr = alfa + delta
    x12_dfNr = alfa - delta
    x13_dfNr = alfa - 2.0 * delta
    dfNR_jcr = (-orto(x, x10_dfNr) + 8.0 * orto(x, x11_dfNr) -8.0 * orto(x, x12_dfNr) + orto(x, x13_dfNr)) / (12.0 * delta)
    return dfNR_jcr

variavel = np.abs(objective_func(x))
while abs(variavel) >= tol and k < k_max:

    k_NR = 0
    converge_NR = False
    while converge_NR == False and k_NR < k_NRmax:
        alfa = alfa - orto(x, alfa)/ d_orto(x, alfa)
        k_NR = k_NR+1
        converge_NR = abs(orto(x, alfa)) <= tol_NR


    x -= alfa*gradient_func(x)
    variavel = np.abs(objective_func(x))
    k += 1

print (k, alfa, x, np.abs(objective_func(x)))

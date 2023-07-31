# Estimação de parametro pelo método de gauss-newton
# Natasha Oliveira dos Santos 


import numpy as np
import scipy as sp
from scipy import stats

#Dados do processo de adsorção
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])  # pressão parcial do gás p(bar)
y = np.array([0.000, 0.338, 0.571, 0.716, 0.771, 0.861, 0.952, 0.946, 1.035, 1.032, 1.046, 1.116, 1.116, 1.103, 1.168, 1.132, 1.130, 1.152, 1.203, 1.146, 1.195])  #volume adsorvido na superficie v(cm^3/g)
k = 0 # iteração inicial
k_max = 100 # máximo de iteração
tol = 1.0E-15 # erro máximo do módulo do gradiente
nparam = 2

#inicialização dos parâmetros
parametro = np.array([1.0, 0.5])   # valores de Vmax e keq

#Função modelo
def func_modelo(x, parametro):
    f_calc = parametro[0]*parametro[1] * x / (1 +parametro[1] * x)
    return f_calc

#calcular o resíduo
def residuo(x, parametro):
    R = (y - func_modelo(x,parametro))
    return R

# definindo a função objetivo
def func_objetivo(x, parametro, y):
    f = np.sum((func_modelo(x, parametro) - y)**2)
    return f

#Jacobiano
def jacobian(x, parametro):
    J = np.zeros([len(x), len(parametro)])
    J[:,0] = (parametro[0] * x)/(1+parametro[1] * x)
    J[:,1] = (parametro[1] * x)/((1 + parametro[1] * x)**2)
    return J

variavel = np.abs(func_objetivo(x,parametro, y))

while abs(variavel) >= tol and k < k_max:
    jacobianoT_jacobiano = np.matmul(np.transpose(jacobian(x,parametro)), jacobian(x,parametro)) #derivadaT*derivada
    jacobianoT_modelo = np.matmul(np.transpose(jacobian(x, parametro)), residuo(x,parametro)) #derivadaT*modelo

    sol_delta = np.linalg.solve(jacobianoT_jacobiano, jacobianoT_modelo)
    variavel = np.abs(func_objetivo(x, parametro, y))
    parametro = parametro + sol_delta

    k +=1
print("Valor do parametro:", parametro)

# Calculo dos erros
RSS = np.sum((y - func_modelo(x, parametro))**2)   # soma do quadrado dos resíduos
dfe = len(y) - len(parametro)                    #graus de liberdade
se2 = RSS / dfe                                  # variancia amostral
raiz_se2 = np.sqrt(se2)
cov = np.linalg.inv(jacobianoT_jacobiano)

std_erro = np.zeros((2,2))#matriz 2x2 de zeros
for i in range(2):
    std_erro[i] = raiz_se2*np.sqrt(cov[i,i])

t_value = parametro / std_erro
#print("t_value", t_value)

p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_value), dfe))
#print("p_value", p_value)

nivel_confianca = 0.95

clts_mp = 1.0 - (1.0 - nivel_confianca) / 2.0
#print("clts_mp", clts_mp)

limite_inferior = parametro - np.dot(std_erro, stats.t.ppf(clts_mp, dfe))
limite_superior = parametro + np.dot(std_erro, stats.t.ppf(clts_mp, dfe))

print("Valor de k:", k, "Valor de delta:", sol_delta,"Valor da função objetivo:", func_objetivo(x, parametro, y))
print ("std.erro:",std_erro[0,0],"t_value:", t_value[0,0],"p_value:", p_value[0,0],"limite inferior:", limite_inferior[0,0],"limite superior:", limite_superior[0,0])
print ("std.erro:",std_erro[1,1],"t_value:", abs(t_value[1,1]),"p_value:", p_value[1,1],"limite inferior:", limite_inferior[1,1],"limite superior::", limite_superior[1,1])

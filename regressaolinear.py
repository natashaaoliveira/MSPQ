#estimação dos parametros pelo método de regressão linear
#Natasha Oliveira dos Santos 


import numpy as np
import scipy as sp
from scipy import stats

#dados do processo

ft = np.array([0.98, 1.02, 1.05, 0.96, 0.99, 0.98, 1.01]) #fator de correção
re = np.array([40, 30, 18, 11, 8, 7, 2]) # número de Reynolds
NP = ([1.02, 1.80, 4.00, 4.99, 7.00, 8.03, 11.0])  #número de potência

# sendo a equação  ln(NP) = Y
# regressão linear usando a formula teta=(XT.X)**-1*XT*Y
y  = np.log(NP)
x_exp = np.column_stack((ft, re))
teta = np.linalg.inv(x_exp.T@x_exp)@x_exp.T@y
alfa_1 = teta[0]
alfa_2 = teta [1]

print("Alfa_1:", alfa_1, "Alfa_2:", alfa_2)

y_modelo= alfa_1*ft + alfa_2*re

#print(y, y_modelo)

# Estatística dos parâmentros:
RSS = np.sum((y - y_modelo)**2)    # soma do erro dos resíduos
grau = len(y) - len(teta)          #graus de liberdade
#print(grau)
Var = RSS/grau                     #variância
raiz_Var  = np.sqrt(Var)
m_cov = np.linalg.inv(x_exp.T@x_exp)
std_erro = np.zeros((2, 2))

for i in range(2):#range = 0 e 1
    std_erro[i] = raiz_Var*np.sqrt(m_cov[i,i])

t_value = teta/std_erro

nivel_confianca = 0.95             #confiança de 95%

clts_mp = 1.0-(1.0-nivel_confianca)/2.0

p_value = 2*(1-stats.t.cdf(abs(t_value),grau))

limite_inferior = teta - np.dot(std_erro, stats.t.ppf(clts_mp, grau))
limite_superior = teta + np.dot(std_erro, stats.t.ppf(clts_mp, grau))

print ("std.erro:",std_erro[0,0],"t_value:", t_value[0,0],"p_value:", p_value[0,0],"limite inferior:", limite_inferior[0,0],"limite superior:", limite_superior[0,0])
print ("std.erro:",std_erro[1,1],"t_value:", abs(t_value[1,1]),"p_value:", p_value[1,1],"limite inferior:", limite_inferior[1,1],"limite superior::", limite_superior[1,1])


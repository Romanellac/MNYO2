import numpy as np


#CHEQUEAR SI EULER ESTA BIEN (lo del arange)***.
def Euler(a, b, N, alpha, f):
    h = (b-a)/N
    t = np.arange(a, b, h)
    w = np.zeros((len(t), len(alpha)))
    w[0] = alpha
    
    for i in range(1,N+1):
        w[i] = w + h*f(t[i-1],w[i-1])
        t[i] = a + i*h
        
    return (t, w)

#CHEQUEAR SI RK4 ESTA BIEN (lo del arange y lo del t dentro del loop)***.
def RK4(a, b, N, alpha, f):
    h = (b-a)/N
    t = np.arange(a, b, h)
    w = np.zeros((len(t), len(alpha)))
    w[0] = alpha
    
    for i in range(1, N+1):
        k1 = h * f(t[i-1],w[i-1])
        k2 = h * f(t[i-1]+(h/2), 2 + k1/2)
        k3 = h * f(t[i-1]+(h/2), 2 + k2/2)
        k4 = h * f(t[i]+h, w[i-1] + k3)
        w[i] = w[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return (t,w)

'''
EJERCICIO 1:
------------

CALCULO:
· Buscar solución exacta de las ecuaciones diferenciales.

NUMERICAMENTE:
- Euler
- RK2 y RK4.
- P. eq. (dN/dt = 0):
    - EXP: N(t) = 0! Ver según el método en qué t es.
    - LOG: N(t) = 0 o N(t) = k. 

GRAFICOS:
- N(t) {graficamos según la aproximación}.
- dN/dt (N) {graficamos utilizando el N de la aproximación}.
Cond. iniciales a variar: N0, r, k (qué significa k para la dinámica
del problema, ver bien cómo cambia).

{Comparar y concluir las características de cada gráfico con diferentes
condiciones iniciales}.{Decir qué método aproxima mejor}

'''

#VARIACIÓN DE LAS CONDICIONES INICIALES:
N0_arr = [10, 25, 50, 75, 100, 200, 500]
r_arr = [0, 0.1, 0.25, 0.5, 0.75, 1]
k_arr = [ 25, 50, 75, 100, 200, 500]

'''
EJERCICIO 2:
------------

CALCULO:
- HAY QUE DESEMPAQUETAR EL SISTEMA DE ODES***

NUMERICAMENTE:
- Método numérico que mejor dio en la anterior, aproximar N1 y N2.
- P. eq. (dN/dt = 0) -> iscolinas cero, cuando se cruzan son los p.eq. del sist.
    - luego de desempaquetar el sistema de odes***
    - ¿Cómo se obtienen las isoclinas cero a partir de los puntos de equilibrio de las odes de un sistema de odes? CHATGPT
- Formas de las isoclinas y determinar p.eq del sistema.
    - luego de desempaquetar el sistema de odes***


GRAFICOS:
- dN1/dt, dN2/dt.
Cond. iniciales a variar: N10, N20, r1, r2, k1, k2, alhpa12, alpha21.
- 4 tipos de gráficos dependiendo de k1, k2, alpha12, alpha21.
    - Variarles los N10 y N20, graficar N1(t) y N2(t).

{Comparar y concluir las características de cada gráfico con diferentes
condiciones iniciales}.

'''





'''
EJERCICIO 3:
------------

CALCULO:
- isoclinas.

NUMERICAMENTE:
- IDEM.

GRAFICOS:
- N(t), P(t).
- P vs N: Dinámica para distintas cond iniciales.
Cond. iniciales a variar: r, 1, alpha, beta, k.

{}
'''
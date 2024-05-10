import numpy as np
import matplotlib.pyplot as plt

# ------------------- Metodos para obtener solucion numerica ----------------------
def Euler(a, b, N, alpha, f):
    h = (b - a) / N
    t = np.linspace(a, b, N+1)
    w = np.zeros((len(t), len(alpha))) 
    w[0] = alpha
    
    for i in range(1,N+1):
        w[i] = w[i-1] + h * f(t[i-1], w[i-1])
    return t, w

def RK4(a, b, N, alpha, f):
    h = (b - a) / N
    t = np.linspace(a, b, N+1)
    w = np.zeros((len(t), len(alpha)))
    w[0] = alpha
    
    for i in range(1, N+1):
        k1 = h * f(t[i-1],w[i-1])
        k2 = h * f(t[i-1]+(h/2), w[i-1] + k1/2)
        k3 = h * f(t[i-1]+(h/2), w[i-1] + k2/2) 
        k4 = h * f(t[i], w[i-1] + k3)
        w[i] = w[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
    
    return t, w

# ----------------- Definicion de EDOs que quiero encontrar su solucion numericamente ------------------
def N1_func(t, N):
    return r1*N[0]*((k1-N[0]-alpha12*N[1])/k1)

def N2_func(t, N):
    return r2*N[1]*((k2-N[1]-alpha21*N[0])/k2)

# ----------------- Definicion de isoclinas a evaluar ------------------
def N1_isoclina2_func(k1, alpha12, N2):
    return k1 - alpha12*N2

def N2_isoclina2_func(k2, alpha21, N1):
    return k2 - alpha21*N1


# ----------------- eleccion de condiciones iniciales (Semillas) ------------------------

# defino condiciones para un caso especifico
a = 0
b = 100
N0 = [1000, 1000] # [N10, N20]
N = 100
r1 = 0.5
r2 = 0.5
k1 = 50000
k2 = 50000
alpha12 = 0.5
alpha21 = 0.5

def main():
    # Euler o RK4: Hay uno que lo vamos a eliminar en base si en el primer ejercicio nos quedó menos error con uno o con otro!
    # De momento asumo que usamos RK4 y solo graficamos eso.

    # Resolver la ecuación diferencial N1 con Euler
    t_euler_N1, w_euler_N1 = Euler(a, b, N, N0, N1_func)

    # Resolver la ecuación diferencial N1 con RK4
    t_rk4_N1, w_rk4_N1 = RK4(a, b, N, N0, N1_func)

    # Resolver la ecuación diferencial N2 con Euler
    t_euler_N2, w_euler_N2 = Euler(a, b, N, N0, N2_func)

    # Resolver la ecuación diferencial N2 con RK4
    t_rk4_N2, w_rk4_N2 = RK4(a, b, N, N0, N2_func)

    # ------------------------------------ Grafico N1 vs t y N2 vs t ---------------------------------------
    # Configurar subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Graficar N1 vs t (RK4)
    axs[0].plot(t_rk4_N1, w_rk4_N1[0:], label='Solución Numérica (RK4)')
    axs[0].set_xlabel('Tiempo (t)')
    axs[0].set_ylabel('Tamaño Poblacional (N1)')
    axs[0].set_title('N1(t): Solución Numérica - Método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar N2 vs t (RK4)
    axs[1].plot(t_rk4_N2, w_rk4_N2[0:], label='Solución Numérica (RK4)', color='green')
    axs[1].set_xlabel('Tiempo (t)')
    axs[1].set_ylabel('Tamaño Poblacional (N2)')
    axs[1].set_title('N2(t): Solución Numérica - Método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

    # ------------------------------------ Grafico N1' vs N1 y N2' vs N2 ---------------------------------------
    N1_prime = [N1_func(t_rk4_N1[i], w_rk4_N1[i]) for i in range(len(t_rk4_N1))]
    N2_prime = [N2_func(t_rk4_N2[i], w_rk4_N2[i]) for i in range(len(t_rk4_N2))]

    # Configurar subplots con tamaño ajustado
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Graficar dN1/dt vs N1
    axs[0].plot(w_rk4_N1[:, 0], N1_prime, label='RK4')
    axs[0].set_xlabel('Tamaño Poblacional (N1)')
    axs[0].set_ylabel('Variación Poblacional (dN1/dt)')
    axs[0].set_title('Variación Poblacional - Método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar dN2/dt vs N2
    axs[1].plot(w_rk4_N2[:, 0], N2_prime, label='RK4')
    axs[1].set_xlabel('Tamaño Poblacional (N2)')
    axs[1].set_ylabel('Variación Poblacional (dN2/dt)')
    axs[1].set_title('Variación Poblacional - Método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

    # ------------------------------------ Cálculo de isoclinas y puntos de equilibrio (N1, N2 y el sistema) ---------------------------------------

    N1_isoclina1 = np.zeros(len(t_rk4_N1))
    N1_isoclina2 = np.array([N1_isoclina2_func(k1, alpha12, w_rk4_N2[i][1]) for i in range(len(t_rk4_N1))])
    N2_isoclina1 = np.zeros(len(t_rk4_N2))
    N2_isoclina2 = np.array([N2_isoclina2_func(k2, alpha21, w_rk4_N1[i][0]) for i in range(len(t_rk4_N1))])

    # Ver si hay alguna forma de decirle al gráfico que ponga dónde es la intersección. Lo tengo en número, pero para no tener que escribirlo o para ver como decirle que 
    # ponga esos puntos en el gráfico. *** Sobretodo porque yo lo tengo en términos de N1 y N2, no en términos de t.

    intersecciones = []

    # ------------------------------------ Gráfico de isoclinas y puntos de equilibrio (N1, N2 y el sistema) ---------------------------------------

    plt.plot(t_rk4_N1, N1_isoclina1, label='N1, isoclina1')
    plt.plot(t_rk4_N1, N1_isoclina2, label='N1, isoclina2')
    plt.plot(t_rk4_N1, N2_isoclina1, label='N2, isoclina1')
    plt.plot(t_rk4_N1, N2_isoclina2, label='N2, isoclina2')

    # Agregar los puntos de intersección a la gráfica
    #plt.scatter(intersecciones[:, 0], intersecciones[:, 1], color='red', label='Intersecciones (puntos de equilibrio del sistema)')

    plt.xlabel('Tiempo (t)')
    plt.ylabel('Tamaño poblacional (N1 o N2)')
    plt.legend()
    plt.grid(True)

    plt.show()
    
if __name__ == "__main__":
    main()
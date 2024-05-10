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
def N_func(t, N):
    return r*N[0] - alpha*N[0]*N[1]

def P_func(t, N):
    return beta*N[0]*N[1] - q*N[1]

# ----------------- eleccion de condiciones iniciales (Semillas) ------------------------

# defino condiciones para un caso especifico
a = 0
b = 100
N0 = [1000, 1000] # [N, P]
N = 100
r = 0.5
k = 50000
alpha = 0.5
beta = 0.5
q = 0.5

def main():
    # Euler o RK4: Hay uno que lo vamos a eliminar en base si en el primer ejercicio nos quedó menos error con uno o con otro!
    # De momento asumo que usamos RK4 y solo graficamos eso.

    # Resolver la ecuación diferencial N1 con Euler
    t_euler_N, w_euler_N = Euler(a, b, N, N0, N_func)

    # Resolver la ecuación diferencial N1 con RK4
    t_rk4_N, w_rk4_N = RK4(a, b, N, N0, N_func)

    # Resolver la ecuación diferencial N2 con Euler
    t_euler_P, w_euler_P = Euler(a, b, N, N0, P_func)

    # Resolver la ecuación diferencial N2 con RK4
    t_rk4_P, w_rk4_P = RK4(a, b, N, N0, P_func)

    # ------------------------------------ Grafico N1 vs t y N2 vs t ---------------------------------------
    # Configurar subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Graficar N1 vs t (RK4)
    axs[0].plot(t_rk4_N, w_rk4_N[0:], label='Solución Numérica (RK4)')
    axs[0].set_xlabel('Tiempo (t)')
    axs[0].set_ylabel('Tamaño Poblacional (N)')
    axs[0].set_title('N(t): Solución Numérica - Método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar N2 vs t (RK4)
    axs[1].plot(t_rk4_P, w_rk4_P[0:], label='Solución Numérica (RK4)', color='green')
    axs[1].set_xlabel('Tiempo (t)')
    axs[1].set_ylabel('Tamaño Poblacional (P)')
    axs[1].set_title('P(t): Solución Numérica - Método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

    # ------------------------------------ Grafico N1' vs N1 y N2' vs N2 ---------------------------------------
    N1_prime = [N_func(t_rk4_N[i], w_rk4_N[i]) for i in range(len(t_rk4_N))]
    N2_prime = [P_func(t_rk4_P[i], w_rk4_P[i]) for i in range(len(t_rk4_P))]

    # Configurar subplots con tamaño ajustado
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Graficar dN1/dt vs N1
    axs[0].plot(w_rk4_N[:, 0], N1_prime, label='RK4')
    axs[0].set_xlabel('Tamaño Poblacional (N)')
    axs[0].set_ylabel('Variación Poblacional (dN/dt)')
    axs[0].set_title('Variación Poblacional - Método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar dN2/dt vs N2
    axs[1].plot(w_rk4_P[:, 0], N2_prime, label='RK4')
    axs[1].set_xlabel('Tamaño Poblacional (P)')
    axs[1].set_ylabel('Variación Poblacional (dP/dt)')
    axs[1].set_title('Variación Poblacional - Método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
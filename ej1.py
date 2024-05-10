import numpy as np
import matplotlib.pyplot as plt

def error_absoluto(real, aprox):
    return np.abs(real - aprox)

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
def odeExponencial(t, N):
    # r es una variable global.
    return r * N

def odeLogistica(t, N):
    # K y r son variables globales.
    return r*N * ((K-N) / K)

# ----------------- Solcuion analitica de EDOs ------------------
def solAnaliExponencial(r, N0, t):
    w = []
    for i in range(len(t)):
       w.append(N0 * np.exp(r * t[i]))
    return w

def solAnaliLogistica(r, K, t):
    w = []
    for i in range(len(t)):
        wi = (K*np.exp(r * t[i]))/(1+np.exp(r * t[i]))
        w.append(wi)
        #(C1*K*np.exp(r * t[i]))/(1+C1*np.exp(r * t[i])) El C1 viene de una CTE que aparece cuando, en una parte de la cuenta, se integra r 
        # respecto de t, pero como nuestro t0 = 0, dicha CTE = 0, y el término C1 = e^CTE = 1.
    return w

# ----------------- eleccion de condiciones iniciales (Semillas) ------------------------
N0_arr = [10, 25, 50, 75, 100, 200, 500] # poblacion inicial
r_arr = [-1, -0.75, -0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 1] 
k_arr = [ 25, 50, 75, 100, 200, 500] #maxima cantidad de poblacion

# defino condiciones para un caso especifico
a = 0
b = 100
N0 = 10000
N = 100
r = 0.1
K = 50000

#*** modularizar código***
def main():

    # Ecuación diferencial exponencial:
    # ---------------------------------

    # Resolver la ecuación diferencial con Euler
    t_euler_expo, w_euler_expo = Euler(a, b, N, [N0], odeExponencial)

    # Resolver la ecuación diferencial con RK4
    t_rk4_expo, w_rk4_expo = RK4(a, b, N, [N0], odeExponencial)

    # Resolver la ecuación diferencial Analiticamente
    t_anali_expo = np.linspace(a, b, N+1)
    w_anali_expo = solAnaliExponencial(r, N0, t_anali_expo)
    
    # Error absoluto de Euler y RK4
    
    err_abs_euler_expo = error_absoluto(w_anali_expo, w_euler_expo[:, 0])

    err_abs_RK4_expo = error_absoluto(w_anali_expo, w_rk4_expo[:, 0])


    # ------------------------------------ Grafico N vs t ---------------------------------------
    # Configurar subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))

    # Graficar solución numérica Euler
    axs[0].plot(t_euler_expo, w_euler_expo[0:], label='Solución Numérica(Euler)')
    axs[0].set_xlabel('Tiempo (t)')
    axs[0].set_ylabel('Tamaño Poblacional (N)')
    axs[0].set_title('(Exponencial) Solución Numérica - Método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar solución numérica RK4
    axs[1].plot(t_rk4_expo, w_rk4_expo[0:], label='Solución Numérica (RK4)', color='green')
    axs[1].set_xlabel('Tiempo (t)')
    axs[1].set_ylabel('Tamaño Poblacional (N)')
    axs[1].set_title('(Exponencial) Solución Numérica - Método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Graficar solución analítica
    axs[2].plot(t_anali_expo , w_anali_expo, label='Solución Analítica', color='red')
    axs[2].set_xlabel('Tiempo (t)')
    axs[2].set_ylabel('Tamaño Poblacional (N)')
    axs[2].set_title('(Exponencial) Solución Analítica - Ecuación Diferencial')
    axs[2].legend()
    axs[2].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

    # ------------------------------------ Grafico N' vs N ---------------------------------------
    # Calcular n' para cada t_anali
    N_prime_anali_expo = [odeExponencial(t_anali_expo[i], w_anali_expo[i]) for i in range(len(t_anali_expo))]
    # Configurar subplots con tamaño ajustado
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Graficar dN/dt vs N para Euler
    axs[0].plot(w_euler_expo[:, 0], N_prime_anali_expo, label='Euler')
    axs[0].set_xlabel('Tamaño Poblacional (N)')
    axs[0].set_ylabel('Variación Poblacional (dN/dt)')
    axs[0].set_title('(Exponencial) Variación Poblacional - Método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar dN/dt vs N para RK4
    axs[1].plot(w_rk4_expo[:, 0], N_prime_anali_expo, label='RK4')
    axs[1].set_xlabel('Tamaño Poblacional (N)')
    axs[1].set_ylabel('Variación Poblacional (dN/dt)')
    axs[1].set_title('(Exponencial) Variación Poblacional - Método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Graficar solución analítica
    axs[2].plot(w_anali_expo, N_prime_anali_expo, label='Analitica')
    axs[2].set_xlabel('Tamaño Poblacional (N)')
    axs[2].set_ylabel('Variación Poblacional (dN/dt)')
    axs[2].set_title('(Exponencial) Variación Poblacional - Solucion Analitica')
    axs[2].legend()
    axs[2].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()
    
    # ------------------------------------ Gráfico errores absolutos -----------------------------
    
    # Configurar los subplots de los errores
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Graficar el error absoluto (solución analítica vs Euler).
    axs[0].plot(t_euler_expo, err_abs_euler_expo, label='Error absoluto (Euler)')
    axs[0].set_xlabel('Tiempo (t)')
    axs[0].set_ylabel('Error absoluto')
    axs[0].set_title('Exponencial. Error absoluto, método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar el error absoluto (solución analítica vs RK4).
    axs[1].plot(t_rk4_expo, err_abs_RK4_expo, label='Error absoluto (RK4)')
    axs[1].set_xlabel('Tiempo (t)')
    axs[1].set_ylabel('Error absoluto')
    axs[1].set_title('Exponencial. Error absoluto, método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    
    
    



    # ----------------------------------------------------------------------------------------------

    # Ecuación diferencial logística:
    # ---------------------------------

    # Resolver la ecuación diferencial con Euler
    t_euler_log, w_euler_log = Euler(a, b, N, [N0], odeLogistica)

    # Resolver la ecuación diferencial con RK4
    t_rk4_log, w_rk4_log = RK4(a, b, N, [N0], odeLogistica)

    # Resolver la ecuación diferencial Analiticamente
    t_anali_log = np.linspace(a, b, N+1)
    w_anali_log = solAnaliLogistica(r, N0, t_anali_log)
    
    # Error absoluto de Euler y RK4
    err_abs_euler_log = error_absoluto(w_anali_log, w_euler_log[:, 0])
    err_abs_RK4_log = error_absoluto(w_anali_log, w_rk4_log[:, 0])

    # ------------------------------------ Grafico N vs t ---------------------------------------
    # Configurar subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))

    # Graficar solución numérica Euler
    axs[0].plot(t_euler_log, w_euler_log[0:], label='Solución Numérica (Euler)')
    axs[0].set_xlabel('Tiempo (t)')
    axs[0].set_ylabel('Tamaño Poblacional (N)')
    axs[0].set_title('(Logística) Solución Numérica - Método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar solución numérica RK4
    axs[1].plot(t_rk4_log, w_rk4_log[0:], label='Solución Numérica (RK4)', color='green')
    axs[1].set_xlabel('Tiempo (t)')
    axs[1].set_ylabel('Tamaño Poblacional (N)')
    axs[1].set_title('(Logística) Solución Numérica - Método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Graficar solución analítica
    axs[2].plot(t_anali_log , w_anali_log, label='Solución Analítica', color='red')
    axs[2].set_xlabel('Tiempo (t)')
    axs[2].set_ylabel('Tamaño Poblacional (N)')
    axs[2].set_title('(Logística) Solución Analítica - Ecuación Diferencial')
    axs[2].legend()
    axs[2].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

    # ------------------------------------ Grafico N' vs N ---------------------------------------
    # Calcular n' para cada t_anali
    N_prime_anali_log = [odeLogistica(t_anali_expo[i], w_anali_log[i]) for i in range(len(t_anali_expo))]
    # Configurar subplots con tamaño ajustado
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Graficar dN/dt vs N para Euler
    axs[0].plot(w_euler_log[:, 0], N_prime_anali_log, label='Euler')
    axs[0].set_xlabel('Tamaño Poblacional (N)')
    axs[0].set_ylabel('Variación Poblacional (dN/dt)')
    axs[0].set_title('(Logística) Variación Poblacional - Método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar dN/dt vs N para RK4
    axs[1].plot(w_rk4_log[:, 0], N_prime_anali_log, label='RK4')
    axs[1].set_xlabel('Tamaño Poblacional (N)')
    axs[1].set_ylabel('Variación Poblacional (dN/dt)')
    axs[1].set_title('(Logística) Variación Poblacional - Método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Graficar solución analítica
    axs[2].plot(w_anali_log, N_prime_anali_log, label='Analitica')
    axs[2].set_xlabel('Tamaño Poblacional (N)')
    axs[2].set_ylabel('Variación Poblacional (dN/dt)')
    axs[2].set_title('(Logística) Variación Poblacional - Solucion Analitica')
    axs[2].legend()
    axs[2].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()
    
    # ------------------------------------ Gráfico errores absolutos -----------------------------
    
    # Configurar los subplots de los errores
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Graficar el error absoluto (solución analítica vs Euler).
    axs[0].plot(t_euler_expo, err_abs_euler_log[0:], label='Error absoluto (Euler)')
    axs[0].set_xlabel('Tiempo (t)')
    axs[0].set_ylabel('Error absoluto')
    axs[0].set_title('Logística. Error absoluto, método de Euler')
    axs[0].legend()
    axs[0].grid(True)

    # Graficar el error absoluto (solución analítica vs RK4).
    axs[1].plot(t_rk4_expo, err_abs_RK4_log[0:], label='Error absoluto (RK4)')
    axs[1].set_xlabel('Tiempo (t)')
    axs[1].set_ylabel('Error absoluto')
    axs[1].set_title('Logística. Error absoluto, método de RK4')
    axs[1].legend()
    axs[1].grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

    #––––––––––––––––––––––––– PUNTOS DE EQUILIBIRIO, ECUACIÓN LOGÍSTICA ––––––––––––––––––––––––
    
    # (cálculo dN/dt = 0, analíticamente. Con dN/dt = rN(1 - N/K))
    N_eq_1 = 0
    N_eq_2 = K
    
    
if __name__ == "__main__":
    main() 
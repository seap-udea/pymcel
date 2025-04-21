
import numpy as np
from scipy.optimize import newton
import spiceypy as spy

def C(z):
    '''Función de Stumpff
    '''

    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        return 1 / 2

def S(z):
    '''Función de Stumpff
    '''

    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
    elif z < 0:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3
    else:
        return 1 / 6

def solucion_lambert(P1, P2, tf, mu=1, direccion='pro', tol=1e-6, maxiter=10000):
    '''Adaptado de: https://github.com/iscoooooo/Porkchop-Plot-Generator

    Resuelve el problema de Lambert, es decir, encuentra la órbita que conecta los puntos P1 y P2
    en un tiempo de vuelo tf. 

    Las fórmulas en este algoritmo están basadas en la teoría del libro "Orbital Mechanics for Engineering Students,"
    Fourth Edition by Howard D. Curtis.
    
    Parameters:
    P1, P2 : np.array
        Posiciones iniciales
    tf : float
        Tiempo de vuelo
    mu = 1 : float
        Parámetro gravitacional.
    tol = 1e-6 : int, optional
        Tolerancia para el solucionador de Newton
    maxiter = 10000: int, optional
        Máximo número de iteraciones para el solucionador de Newton
    direccion = 'pro' : str, optional
        'pro' para la órbita prograda y 'retro' para la órbita retrograda
    
    Returns:
    V1, V2 : np.array
        Velocidades inicial y final

    orbita: dict
        z: Variable universal (z<0 para órbita hiperbólica)
        elts: Elementos orbitales de SPICE (q, e, I, Omega, omega, M, t, mu)
    '''

    # Distancias
    r1 = np.linalg.norm(P1)
    r2 = np.linalg.norm(P2)

    # Ángulo del triángulo espacial
    theta = np.arccos(np.dot(P1, P2) / (r1 * r2))
    cross12 = np.cross(P1, P2)
    if direccion == 'pro':  
        if cross12[2] < 0:
            theta = 2 * np.pi - theta
    elif direccion == 'retro': 
        if cross12[2] >= 0:
            theta = 2 * np.pi - theta
    else:
        raise ValueError("Debe indicar la dirección de movimiento ('pro' de P1 a P2 o 'retro' de P1 a P2 por el lado opuesto).")

    # Función auxiliar
    A = np.sin(theta) * np.sqrt(r1 * r2 / (1 - np.cos(theta)))

    # Funciones auxiliares
    def y(z):
        return r1 + r2 + A * (z * S(z) - 1) / np.sqrt(C(z))

    def F(z):
        return (y(z) / C(z)) ** 1.5 * S(z) + A *  np.sqrt(y(z)) - np.sqrt(mu) * tf

    def dFdz(z):
        if z == 0:
            return np.sqrt(2) / 40 * y(0) ** 1.5 + A / 8 * (np.sqrt(y(0)) + A * np.sqrt(1 / 2 / y(0)))
        else:
            return (y(z) / C(z)) ** 1.5 * (1 / 2 / z * (C(z) - 3 * S(z) / 2 / C(z)) + 3 * S(z) ** 2 / 4 / C(z)) + A / 8 * (3 * S(z) / C(z) * np.sqrt(y(z)) + A * np.sqrt(C(z) / y(z)))
        
    # Busca el valor inicial de z para resolver por Newton
    z = 0.1
    while F(z) < 0:
        z += 0.1
        if z > 1e6: 
            raise ValueError("No pude encontrar un valor inicial apropiado para z.")

    # Resuelve la ecuación de variable universal
    z = newton(F, z, tol=tol, maxiter=maxiter)

    # Determina si encontro una solución
    if np.isnan(z) or np.isinf(z):
        solved = False
    else:
        solved = True 
    
    # Encuentra las velocidades usando las funciones de Lagrange f, g
    default = np.array([0, 0, 0]), np.array([0, 0, 0]), dict(z=0, elts=[0]*8)
    if solved:
        f = 1 - y(z) / r1
        fdot = (np.sqrt(mu) / (r1 * r2)) * np.sqrt(y(z) / C(z)) * (z * S(z) - 1)
        g = A * np.sqrt(y(z) / mu)
        gdot = 1 - y(z) / r2

        # Compute the velocities V1 & V2
        V1 = 1 / g * (P2 - f * P1)
        V2 = 1 / g * (gdot * P2 - P1)
            
        elts = spy.oscelt(list(P1)+list(V1), 0, mu)
        orbit_info = dict(z=z, elts=elts)
        return V1, V2, orbit_info
    else: 
        print('El método de Lambert no convergió') 

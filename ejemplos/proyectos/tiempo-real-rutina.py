"""
Ejemplo de uso de ncuerpos_rebound_visual:
Visualización en tiempo real de un cúmulo de Plummer.

Uso:
    python tiempo-real-rutina.py
"""
import numpy as np
import pymcel as pc
np.random.seed(1)

# Generar condiciones iniciales de un cúmulo de Plummer
N_particulas = 10
masas, pos, vel = pc.condiciones_iniciales_plummer(
    N=N_particulas, masa_total=1.0, radio_escala=1.0
)

# Visualizar en tiempo real (2D)
# pc.ncuerpos_rebound_visual(masas, pos, vel, limite=5.0, titulo="Cúmulo de Plummer")
# pc.ncuerpos_rebound_visual3d(masas, pos, vel, limite=5.0, titulo="Cúmulo de Plummer")
pc.ncuerpos_rebound_visual_gif(masas, pos, vel, t_final=None, limite=5.0, titulo="Cúmulo de Plummer")

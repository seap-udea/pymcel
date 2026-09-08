import pymcel as pc
import numpy as np

N=4
masas, pos, vel = pc.condiciones_iniciales_coreografia(N=N)

pc.ncuerpos_rebound_visual_avanzada(
    masas=masas, 
    posiciones=pos, 
    velocidades=vel, 
    t_final=None,
    dt_grafico=0.01,
    limite_grafico=1.8,
    titulo=f"Coreografía de {N} Cuerpos (Simó 2001)",
    plot_3d=False,
    recentrado=False,
    trazos=True,
    integrator='ias15',epsilon=1e-12,
    #integrator='leapfrog',
    longitud_trazo=200,
    # salva_gif=f"coreografia_{N}.gif" # Descomentar para grabar
)
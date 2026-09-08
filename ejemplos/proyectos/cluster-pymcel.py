import pymcel as pc
import numpy as np
np.random.seed(42)

N_particulas = 50
M_tot = 1.0
a_plummer = 1.0

print(f"Generando condiciones iniciales para {N_particulas} partículas...")
masas, pos, vel = pc.condiciones_iniciales_plummer(N_particulas, M_tot, a_plummer)

pc.ncuerpos_rebound_visual_avanzada(
    masas=masas, 
    posiciones=pos, 
    velocidades=vel, 
    t_final=None, 
    dt_grafico=0.5,
    limite_grafico=30*a_plummer,
    titulo=f"Evolución de Cúmulo (N={N_particulas})",
    plot_3d=False,
    recentrado=True,
    trazos=0,
    # salva_gif='cluster_evolucion.gif'
)

import pymcel as pc
import numpy as np
np.random.seed(42)
    
M1 = 1.0
anillos1 = 10
estrellas_por_anillo = 10
masas1, pos1, vel1 = pc.condiciones_iniciales_toomre(
    M_central=M1, 
    anillos=anillos1, 
    estrellas_por_anillo=estrellas_por_anillo, 
    radio_minimo=0.2, 
    radio_maximo=1.0
)

M2 = 0.6  
anillos2 = 3
masas2, pos2, vel2 = pc.condiciones_iniciales_toomre(
    M_central=M2, 
    anillos=anillos2, 
    estrellas_por_anillo=int(estrellas_por_anillo/3), 
    radio_minimo=0.2, 
    radio_maximo=0.6,
    r_ini=[4.0, 2.0, 0.5],
    v_ini=[-0.6, 0.2, 0.0],
    angulo_ini=0.0
)

masas = np.concatenate([masas1, masas2])
posiciones = np.concatenate([pos1, pos2])
velocidades = np.concatenate([vel1, vel2])
N_total = len(masas)

print(f"Iniciando colisión de galaxias con {N_total} partículas...")

pc.ncuerpos_rebound_visual_avanzada(
    masas=masas, 
    posiciones=posiciones, 
    velocidades=velocidades, 
    t_final=None, 
    dt_grafico=0.3,
    limite_grafico=10.0,
    titulo="Colisión de Galaxias de Toomre (1972)",
    plot_3d=0,
    recentrado=False,
    trazos=0
)

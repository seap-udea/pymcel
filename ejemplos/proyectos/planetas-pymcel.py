import pymcel as pc
import numpy as np
np.random.seed(2)

M_central = 1.0
N_planetesimales = 10
masa_total_disco = 0.1 

print(f"Generando disco protoplanetario con {N_planetesimales} planetesimales...")
radio_minimo = 0.5
radio_maximo = 3.0
masas, pos, vel, radios = pc.condiciones_iniciales_planetesimales(
    M_central=M_central,
    N_planetesimales=N_planetesimales,
    masa_total_disco=masa_total_disco,
    radio_minimo=radio_minimo,
    radio_maximo=radio_maximo
)

radios[0]*=0.1
pc.ncuerpos_rebound_visual_avanzada(
    masas=masas, 
    posiciones=pos, 
    velocidades=vel, 
    t_final=None, 
    dt_grafico=0.1,
    limite_grafico=2*radio_maximo,
    titulo="Formación Planetaria (Acreción)",
    plot_3d=False, 
    recentrado=1,
    trazos=0,
    radios=radios,
    colisiones=True,
    tamanos_dinamicos=True,
    i_central=0,
    alpha_radio=1,
    grabar_posiciones=True
)

import numpy as np
import matplotlib.pyplot as plt
import rebound

import pymcel as pc
np.random.seed(42)

N_particulas = 10
masas, pos, vel = pc.condiciones_iniciales_plummer(N=N_particulas, masa_total=1.0, radio_escala=1.0)

sim = rebound.Simulation()
sim.G = 1.0 # Usamos unidades canónicas

for i in range(N_particulas):
    sim.add(m=masas[i], 
            x=pos[i, 0], y=pos[i, 1], z=pos[i, 2],
            vx=vel[i, 0], vy=vel[i, 1], vz=vel[i, 2])
sim.move_to_com()

plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

limite = 4.0
ax.set_xlim(-limite, limite)
ax.set_ylim(-limite, limite)
ax.set_zlim(-limite, limite)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(True)

x_data = [p.x for p in sim.particles]
y_data = [p.y for p in sim.particles]
z_data = [p.z for p in sim.particles]
dibujo, = ax.plot(x_data, y_data, z_data, 'o', markersize=3, color='navy', alpha=0.7)

dt_grafico = 0.2
while True and plt.fignum_exists(fig.number):
    
    sim.integrate(sim.t + dt_grafico)
    x_data = [p.x for p in sim.particles]
    y_data = [p.y for p in sim.particles]
    z_data = [p.z for p in sim.particles]
    
    dibujo.set_data(x_data, y_data)
    dibujo.set_3d_properties(z_data)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
plt.ioff()

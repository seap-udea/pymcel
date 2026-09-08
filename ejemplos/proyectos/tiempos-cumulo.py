"""
Tiempo de cruce y tiempo de relajación en un cúmulo estelar.

Este script demuestra los dos tiempos característicos fundamentales
de un sistema gravitacional autogravitante (cúmulo estelar):

  1. Tiempo de cruce (t_cr): tiempo que tarda una estrella típica
     en atravesar el sistema. Es la escala de tiempo dinámica.
         t_cr = r_h / σ

  2. Tiempo de relajación (t_relax): tiempo necesario para que los
     encuentros de dos cuerpos cambien significativamente la energía
     de una estrella. Es la escala de tiempo colisional.
         t_relax ≈ N / (8 ln N) × t_cr    (Chandrasekhar)

Para un cúmulo real con N ~ 10^5 estrellas, t_relax/t_cr ~ 2700.
Aquí usamos N pequeño por rapidez, pero la física es la misma.

Uso:
    python tiempos-cumulo.py
"""
import numpy as np
import matplotlib.pyplot as plt
import rebound
import pymcel as pc

np.random.seed(42)

# ============================================================
# 1. Parámetros del cúmulo (Plummer, unidades canónicas G=1)
# ============================================================
N = 50              # Número de partículas
M_total = 1.0       # Masa total
a = 1.0             # Radio de escala de Plummer

masas, pos, vel = pc.condiciones_iniciales_plummer(
    N=N, masa_total=M_total, radio_escala=a
)

# ============================================================
# 2. Tiempos característicos (teóricos)
# ============================================================
# Radio de medio masa del modelo de Plummer: r_h ≈ 1.305 a
r_h = 1.305 * a

# Dispersión de velocidades (equilibrio virial: 2K = -U)
# Para Plummer: σ² = G M / (6a)
sigma = np.sqrt(M_total / (6 * a))

# Tiempo de cruce
t_cr = r_h / sigma

# Tiempo de relajación (Chandrasekhar)
t_relax = N / (8 * np.log(N)) * t_cr

print("=" * 50)
print("PARÁMETROS DEL CÚMULO")
print("=" * 50)
print(f"  N partículas          = {N}")
print(f"  Masa total            = {M_total}")
print(f"  Radio de escala (a)   = {a}")
print(f"  Radio de medio masa   = {r_h:.3f}")
print(f"  Dispersión de vel.    = {sigma:.3f}")
print("=" * 50)
print("TIEMPOS CARACTERÍSTICOS")
print("=" * 50)
print(f"  Tiempo de cruce       t_cr    = {t_cr:.2f}")
print(f"  Tiempo de relajación  t_relax = {t_relax:.2f}")
print(f"  Cociente              t_relax / t_cr = {t_relax/t_cr:.1f}")
print("=" * 50)

# ============================================================
# 3. Configurar simulación en Rebound
# ============================================================
sim = rebound.Simulation()
sim.G = 1.0

for i in range(N):
    sim.add(m=masas[i],
            x=pos[i, 0], y=pos[i, 1], z=pos[i, 2],
            vx=vel[i, 0], vy=vel[i, 1], vz=vel[i, 2])
sim.move_to_com()

# ============================================================
# 4. Integrar y registrar la evolución
# ============================================================
t_total = 3 * t_relax
dt = t_cr / 5           # Muestrear ~5 veces por t_cr
n_pasos = int(t_total / dt)

print(f"\nIntegrando {n_pasos} pasos (t_total = {t_total:.1f} = {t_total/t_cr:.1f} t_cr)...")

# Almacenamiento
tiempos = np.zeros(n_pasos)
distancias = np.zeros((n_pasos, N))
energias = np.zeros((n_pasos, N))
velocidades = np.zeros((n_pasos, N))

for k in range(n_pasos):
    sim.integrate(sim.t + dt)
    tiempos[k] = sim.t

    # Extraer estado como arrays de numpy
    pos_arr = np.array([[p.x, p.y, p.z] for p in sim.particles])
    vel_arr = np.array([[p.vx, p.vy, p.vz] for p in sim.particles])
    m_arr = np.array([p.m for p in sim.particles])

    # Centro de masa
    com = np.sum(pos_arr * m_arr[:, np.newaxis], axis=0) / np.sum(m_arr)

    # Distancia al CM
    dr = pos_arr - com
    distancias[k] = np.sqrt(np.sum(dr**2, axis=1))

    # Magnitud de velocidad
    velocidades[k] = np.sqrt(np.sum(vel_arr**2, axis=1))

    # Energía específica (vectorizado con broadcasting)
    # diff[i,j,:] = r_i - r_j
    diff = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    dist_ij = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_ij, np.inf)

    E_cin = 0.5 * np.sum(vel_arr**2, axis=1)
    E_pot = -np.sum(m_arr[np.newaxis, :] / dist_ij, axis=1)
    energias[k] = E_cin + E_pot

    if (k + 1) % 5 == 0:
        print(f"  Paso {k+1}/{n_pasos}  (t = {sim.t:.1f})")

print("Integración completada.")

# Tiempo en unidades de t_cr
t_norm = tiempos / t_cr

# ============================================================
# 5. Gráficos pedagógicos
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
fig.suptitle(
    f"Tiempos característicos de un cúmulo (N={N})\n"
    f"$t_{{cr}} = {t_cr:.2f}$,  $t_{{relax}} = {t_relax:.2f}$  "
    f"($t_{{relax}}/t_{{cr}} = {t_relax/t_cr:.1f}$)",
    fontsize=13, fontweight='bold'
)

# --- Panel 1: Distancia de una partícula al centro ---
# Elegir una partícula que empieza cerca de r_h (representativa)
i_part = np.argmin(np.abs(distancias[0] - r_h))

ax1 = axes[0]
ax1.plot(t_norm, distancias[:, i_part], color='navy', linewidth=0.8,
         label=f'Partícula {i_part}')
ax1.axhline(r_h, color='crimson', linestyle='--', alpha=0.7,
            label=f'$r_h = {r_h:.2f}$')
ax1.axvline(1.0, color='seagreen', linestyle=':', linewidth=2, alpha=0.7,
            label=r'$1\, t_{cr}$')
ax1.set_ylabel("Distancia al centro de masa")
ax1.set_title(
    r"Tiempo de cruce: la partícula oscila con período $\sim t_{cr}$",
    fontsize=11
)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(alpha=0.3)

# --- Panel 2: Cambio de energía individual ---
ax2 = axes[1]
n_mostrar = min(10, N)
indices = np.linspace(0, N - 1, n_mostrar, dtype=int)
for idx in indices:
    dE = (energias[:, idx] - energias[0, idx]) / np.abs(energias[0, idx])
    ax2.plot(t_norm, dE, linewidth=0.7, alpha=0.7)

ax2.axvline(t_relax / t_cr, color='crimson', linestyle='--', linewidth=2,
            alpha=0.7, label=f'$t_{{relax}} = {t_relax/t_cr:.1f}\\, t_{{cr}}$')
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax2.set_ylabel(r"$\Delta E_i\, /\, |E_i(0)|$")
ax2.set_title(
    r"Tiempo de relajación: la energía individual cambia tras $\sim t_{relax}$",
    fontsize=11
)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(alpha=0.3)

# --- Panel 3: Dispersión de velocidades ---
ax3 = axes[2]
sigma_t = np.std(velocidades, axis=1)
sigma_norm = sigma_t / sigma_t[0]

ax3.plot(t_norm, sigma_norm, color='navy', linewidth=1.0)
ax3.axvline(1.0, color='seagreen', linestyle=':', linewidth=2, alpha=0.7,
            label=r'$t_{cr}$')
ax3.axvline(t_relax / t_cr, color='crimson', linestyle='--', linewidth=2,
            alpha=0.7, label=r'$t_{relax}$')
ax3.axhline(1.0, color='gray', linestyle='-', alpha=0.3)
ax3.set_xlabel(r"Tiempo  ($t\, /\, t_{cr}$)", fontsize=11)
ax3.set_ylabel(r"$\sigma_v(t)\, /\, \sigma_v(0)$")
ax3.set_title("Evolución de la dispersión de velocidades", fontsize=11)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("tiempos_cumulo.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nGráfico guardado en 'tiempos_cumulo.png'")

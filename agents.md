# PYMCEL Agent Guide

Guía completa para agentes de IA que trabajan con `pymcel` (Mecánica Celeste y Astrodinámica).

**Versión**: 0.9.X
**Objetivo**: Proporcionar contexto suficiente para que agentes ejecutores generen código reproducible sin necesidad de explorar documentación adicional.  
**Alcance**: Este archivo asume `pymcel` instalado vía `pip install pymcel` en un entorno Python 3.8+.

---

## 1. Instalación y Setup

### Instalación básica

```bash
pip install pymcel
```

### Instalación en desarrollo (para contribuidores)

```bash
git clone https://github.com/usuario/pymcel.git
cd pymcel
pip install -e .
```

### Dependencias principales

- `numpy`, `scipy`: Cálculo numérico.
- `matplotlib`, `plotly`: Visualización.
- `spiceypy`: Acceso a kernels SPICE.
- `astroquery`, `astropy`: Consultas astronómicas y constantes.
- `pandas`: Exportación de resultados.

### Primer test (verificar instalación)

```python
import pymcel as pc
print(pc.__version__)
# Muestra mensaje de bienvenida + versión
```

---

## 2. Quick Start: Ejemplos ejecutables de cada área

### 2.1 N-Cuerpos: Simulación de dos cuerpos binarios

Extraído de cuaderno 09-ncuerpos_general.ipynb:

```python
import numpy as np
import pymcel as pc

# Sistema de dos cuerpos binarios
sistema = [
    dict(m=1.0, r=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0]),
    dict(m=1.0, r=[1.0, 0.0, 0.0], v=[0.0, 1.0, 0.0]),
]

# Grid de tiempos
ts = np.linspace(0.0, 10.0, 200)

# Resolver (retorna posiciones, velocidades, constantes)
rs, vs, rps, vps, constantes = pc.ncuerpos_solucion(sistema, ts)

# Visualizar en 3D
fig = pc.plot_ncuerpos_3d(rps, vps)
```

**Qué hace:**
- `rs`: posiciones en marco inercial
- `vs`: velocidades en marco inercial
- `rps`: posiciones relativas al centro de masa
- `constantes`: diccionario con energía, momento angular, etc.

---

### 2.2 Ecuación de Kepler: Resolver anomalía excéntrica

Extraído de cuaderno 12-doscuerpos_ecuacion_kepler.ipynb (método Newton):

```python
from pymcel import kepler_newton

# Parámetros del problema
M = 0.5  # Anomalía media (radianes)
e = 0.3  # Excentricidad
delta = 1e-8  # Tolerancia

# Resolver E
E, err, n_iter = kepler_newton(M, e, G0=1.0, delta=delta)

print(f"E = {E} rad")
print(f"Error = {err:.2e}, iteraciones = {n_iter}")
```

**Alternativas rápidas:**
```python
# Método semicertificado (más rápido)
E, err, n_iter = pc.kepler_semianalitico(M, e)

# Método de series (para comparación)
E, err, n_iter = pc.kepler_eserie(M, e)
```

---

### 2.3 CRTBP: Constante de Jacobi y curvas de velocidad cero

Extraído de cuaderno 15-trescuerpos_constante_jacobi.ipynb:

```python
import numpy as np
from pymcel import crtbp_solucion, constante_jacobi

# Parámetro del sistema (cociente de masas reducido)
mu = 0.3

# Condiciones iniciales (marco rotante)
r0 = [0.8, 0.0, 0.0]
v0 = [0.0, 0.4, 0.0]
ts = np.linspace(0, 20, 500)

# Integrar trayectoria en CRTBP
rs_rot, vs_rot, rs_ine, vs_ine, r1, r2 = crtbp_solucion(mu, r0, v0, ts)

# Calcular constante de Jacobi
CJ = constante_jacobi(mu, rs_rot, vs_rot)
print(f"Constante de Jacobi: {CJ[0]:.6f}")

# Visualizar curvas de velocidad cero
import matplotlib.pyplot as plt
rango = 1 + mu
NG = 100
X, Y = np.meshgrid(np.linspace(-rango, rango, NG),
                     np.linspace(-rango, rango, NG))
Z = np.zeros_like(X)
v = 0.0  # Velocidad prescrita

CJ_map = (2*(1-mu)/np.sqrt((X+mu)**2+Y**2+Z**2) +
          2*mu/np.sqrt((X-1+mu)**2+Y**2+Z**2) +
          (X**2 + Y**2) - v**2)

fig, ax = plt.subplots()
c = ax.contourf(X, Y, CJ_map, levels=np.linspace(0, 5, 10))
plt.colorbar(c)
ax.set_xlabel('x'), ax.set_ylabel('y')
ax.set_title(f'Curvas de velocidad cero, μ={mu}')
```

---

### 2.4 Transferencia de Lambert: Trayectoria entre dos puntos

```python
import numpy as np
from pymcel import solucion_lambert

# Posiciones iniciales y finales
P1 = np.array([1.0, 0.0, 0.0])
P2 = np.array([0.0, 1.0, 0.0])

# Tiempo de vuelo
tf = 1.2

# Parámetro gravitacional
mu = 1.0

# Resolver (pro = órbita progresiva, retro = retrógrada)
V1, V2, info = solucion_lambert(P1, P2, tf, mu=mu, direccion='pro')

print(f"Velocidad inicial: {V1}")
print(f"Velocidad final: {V2}")
print(f"Info adicional: {info}")
```

---

### 2.5 Consulta de efemérides con Horizons (NASA)

```python
from pymcel import consulta_horizons

# Consultar posición y velocidad de Marte en época
datos = consulta_horizons(
    id='499',  # ID de Marte
    location='@0',  # Observatorio en el centro del Sol
    epochs={'start': '2025-01-01', 'stop': '2025-01-31', 'step': '1d'},
    datos='vectors',  # Posición y velocidad
    propiedades='default'  # Unidades estándar
)

print(datos)
# Retorna DataFrame con columnas de tiempo, x, y, z, vx, vy, vz, etc.
```

---

### 2.6 Conversión: Estado cartesiano ↔ Elementos orbitales

```python
import numpy as np
from pymcel import estado_a_elementos, elementos_a_estado

# Estado cartesiano
mu = 1.0
estado = np.array([1.0, 0.0, 0.0, 0.0, 0.73, 0.0])  # [x, y, z, vx, vy, vz]

# A elementos orbitales: (p, e, i, Ω, ω, f)
elementos = estado_a_elementos(mu, estado)
p, e, i, Omega, omega, f = elementos
print(f"p={p:.3f}, e={e:.3f}, i={i:.3f}°")

# De vuelta a cartesiano
estado_reconstruido = elementos_a_estado(mu, elementos)
print(f"Error de reconstrucción: {np.linalg.norm(estado - estado_reconstruido):.2e}")
```

---

### 2.7 Visualización: Sistema de N cuerpos en 3D con Plotly

```python
import numpy as np
from pymcel import ncuerpos_solucion, plot_ncuerpos_3d

# Tres cuerpos
sistema = [
    dict(m=1.0, r=[-0.5, 0.0, 0.0], v=[0.0, -0.5, 0.0]),
    dict(m=1.0, r=[0.5, 0.0, 0.0], v=[0.0, 0.5, 0.0]),
    dict(m=0.5, r=[0.0, 0.5, 0.2], v=[0.3, 0.0, 0.0]),
]

ts = np.linspace(0, 10, 300)
rs, vs, _, _, _ = ncuerpos_solucion(sistema, ts)

# Plot interactivo con Plotly
fig = plot_ncuerpos_3d(rs, vs, tipo='plotly')
fig.show()
```

---

## 3. Guía de decisión rápida (Decision Map)

| Necesidad | Función recomendada | Parámetros clave |
|-----------|---------------------|-----------------|
| **Efemérides reales (NASA)** | `consulta_horizons()` | `id`, `epochs`, `datos` |
| **Efemérides con kernels SPICE** | `consulta_spice()` | `id`, `epochs` |
| **Simular N cuerpos** | `ncuerpos_solucion()` | `sistema`, `ts` |
| **Integración manual N cuerpos** | `edm_ncuerpos()` | `Y`, `N`, `mus` |
| **Resolver ecuación de Kepler** | `kepler_newton()` | `M`, `e`, `delta` |
| **CRTBP dinámica** | `crtbp_solucion()` | `mu`, `r0`, `v0`, `ts` |
| **Jacobi constante** | `constante_jacobi()` | `mu`, `r`, `v` |
| **Transferencia orbital** | `solucion_lambert()` | `P1`, `P2`, `tf`, `mu` |
| **Plot 3D interactivo** | `plot_ncuerpos_3d()` con `tipo='plotly'` | `rs`, `vs`, `tipo` |
| **Conversión elementos** | `estado_a_elementos()` / `elementos_a_estado()` | `mu`, `estado`/`elementos` |

---

## 4. API categorizada

### 4.1 Datos, kernels y consultas astronómicas

**Kernels SPICE:**
```python
pc.descarga_kernels(basedir='./data/', overwrite=False)  # Descargar kernels
pc.carga_kernels(basedir='./data/')                       # Cargar kernels
pc.lista_kernels(basedir='./data/')                       # Listar disponibles
pc.obtiene_datos(basedir='./data/')                       # Copiar datos locales
```

**Consultas de efemérides:**
```python
# Horizons (remoto, flexible)
pc.consulta_horizons(id='399', location='@0', epochs=None, datos='vectors', propiedades='default')

# SPICE (local, rápido si kernels están cargados)
pc.consulta_spice(id='399', location='@0', epochs=None)

# Propiedades de cuerpos (masas, radios, etc.)
pc.consulta_propiedad(id='399', propiedad='masa', nvalues=1)
```

**Unidades canónicas:**
```python
pc.unidades_canonicas(UL=1.0, UM=1.0, UT=1.0, G=1)
```

---

### 4.2 N-Cuerpos y 2-Cuerpos

```python
# Integración de N cuerpos (nivel alto)
rs, vs, rps, vps, constantes = pc.ncuerpos_solucion(sistema, ts)

# Integración de N cuerpos (nivel bajo - ecuación diferencial)
from scipy.integrate import odeint
Y0 = pc.sistema_a_Y(sistema)
N, mus, Y0 = pc.sistema_a_Y(sistema)
solucion = odeint(pc.edm_ncuerpos, Y0, ts, args=(N, mus))
rs, vs = pc.solucion_a_estado(solucion, N, len(ts))

# Dos cuerpos
rs, vs = pc.doscuerpos_solucion(mu, r, v, ts)

# Exportar a pandas DataFrame
df = pc.ncuerpos_a_pandas(ts, rs, vs)
```

---

### 4.3 Ecuación de Kepler (múltiples métodos)

```python
# Newton (robusto, recomendado)
E, err, n_iter = pc.kepler_newton(M, e, G0=1.0, delta=1e-5)

# Semicertificado (rápido)
E, err, n_iter = pc.kepler_semianalitico(M, e)

# Series de Fourier (convergencia lenta para e grande)
E, err, n_iter = pc.kepler_eserie(M, e, orden=1)

# Bessel (especial)
E, err, n_iter = pc.kepler_bessel(M, e, delta)

# Métodos de raíz genéricos
E = pc.metodo_newton(f, x0=1.0, delta=1e-5, args=())
E = pc.metodo_laguerre(f, x0=1.0, delta=1e-5, args=(), eta=5)
```

---

### 4.4 CRTBP

```python
# Trayectoria en marco rotante e inercial
rs_rot, vs_rot, rs_ine, vs_ine, r1_ine, r2_ine = pc.crtbp_solucion(mu, r0, v0, ts)

# Constante de Jacobi (energía-like)
CJ = pc.constante_jacobi(mu, r_rot, v_rot)

# Puntos de Lagrange (L1, L2, L3 radican en x)
from scipy.optimize import fsolve
x_L1 = fsolve(pc.funcion_puntos_colineales, 0.5, args=(mu,))[0]

# Órbitas periódicas alrededor de Lagrange (2D y 3D)
pc.orbitas_crtbp(mu, r0, v0, T=100, Nt=1000, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
pc.orbitas_crtbp3d(mu, r0, v0, T=100, Nt=1000, ...)
```

---

### 4.5 Visualización

**Matplotlib (2D/3D estático):**
```python
import matplotlib.pyplot as plt

# Ejes proporcionales 2D
fig, ax = plt.subplots()
ax.plot(x, y)
pc.fija_ejes_proporcionales(ax, (x, y), margin=0.1)

# Ejes 3D proporcionales
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
pc.fija_ejes3d_proporcionales(ax, rangos=None)

# Plot N cuerpos directo
fig = pc.plot_ncuerpos_3d(rs, vs, tipo='matplotlib')
```

**Plotly (3D interactivo):**
```python
# Plot N cuerpos con Plotly
fig = pc.plot_ncuerpos_3d(rs, vs, tipo='plotly')

# Normalizar ejes
pc.axis_equal(fig)

# Esferas
pc.plotly_esfera(fig, R=1.0, sphereargs={'color': 'olive'})

# Campo de vectores
pc.plotly_campo_vectorial(fig, rs, vs)

fig.show()
```

**Cónicas:**
```python
pc.plot_elipse(e=0.5, a=10.0)
pc.plot_hiperbola(e=1.5, a=-10)
pc.dibuja_esfera(ax, centro=(0, 0, 0), radio=1)
```

---

### 4.6 Utilidades geométricas y matemáticas

```python
# Distancia haversine en esfera
dist = pc.haversine(lon1, lat1, lon2, lat2)

# Series de Stumpff C(z), S(z)
C = pc.C(z)
S = pc.S(z)

# Funciones de Kepler universal
pc.funcion_universal_kepler(x, M, e, q)
pc.funcion_universal_kepler_s(s, r0, rdot0, beta, mu, M)

# Intersección de circunferencias
pc.intersecta_circunferencias(x0, y0, r0, x1, y1, r1)
pc.intersecta_circunferencias3d(C1, r1, C2, r2, tol=1e-9)

# Rotación de puntos
pc.rota_puntos(R, x, y, z)
```

---

## 5. Constantes disponibles

Importa desde `pymcel.constantes`:

```python
from pymcel import constantes as const

# Constantes físicas (SI, de astropy)
const.G              # Constante gravitacional
const.c              # Velocidad luz
const.au             # Unidad astronómica
const.M_sun, const.M_earth, const.M_jup  # Masas planetarias
const.R_sun, const.R_earth, const.R_jup  # Radios
const.GM_sun, const.GM_earth, const.GM_jup  # Parámetros gravitacionales

# Tiempo
const.dia, const.year  # Día y año en segundos

# Parámetros SPICE DE441
const.mu_mercury, const.mu_venus, const.mu_earth
const.mu_mars, const.mu_jupiter, const.mu_saturn
const.mu_uranus, const.mu_neptune, const.mu_sun, const.mu_moon
```

**Ejemplo usando constantes:**
```python
from pymcel.constantes import GM_sun, au, year
import numpy as np

# Radio orbital de la Tierra
a_earth = au  # 1 AU

# Período de la Tierra (ley de Kepler)
T_earth = 2 * np.pi * np.sqrt(a_earth**3 / GM_sun)
print(f"Período: {T_earth / year:.3f} años")
```

---

## 6. Patrones comunes y buenas prácticas

### 6.1 Reproducibilidad

```python
import numpy as np
import pymcel as pc

# Fijar semilla si se usa aleatoriedad
np.random.seed(42)

# Fijar tolerancias explícitamente
tolerance = 1e-10
max_iterations = 1000

# Usar tiempos bien definidos
ts = np.linspace(t_ini, t_fin, N_pasos)

# Guardar parámetros para documentación
params = {
    'sistema': sistema,
    'tolerancia': tolerance,
    'N_pasos': N_pasos,
    'fecha_calculo': str(np.datetime64('today'))
}
```

### 6.2 Manejo de efemérides

```python
# Horizons es flexible en formato de época
epochs_1 = '2025-01-01'               # String ISO
epochs_2 = {'start': '2025-01-01', 'stop': '2025-12-31', 'step': '1d'}  # Dict
epochs_3 = [2460000.5, 2460100.5]    # JD directos
epochs_4 = 289.1                      # Número JD

# Siempre convertir unidades explícitamente si vienen de Horizons
df = pc.consulta_horizons(id='399', epochs=epochs_2, propiedades=[
    ('x', 'km'),
    ('y', 'km'),
    ('z', 'km'),
    ('vx', 'km/s'),
    ('vy', 'km/s'),
    ('vz', 'km/s'),
])
```

### 6.3 Validación de integraciones

```python
import numpy as np
from pymcel import ncuerpos_solucion

# Integrar
rs, vs, rps, vps, const = ncuerpos_solucion(sistema, ts)

# Verificar conservación de energía
E = const['E']
error_E = (E - E[0]) / E[0]
print(f"Error relativo en energía: {np.abs(error_E).max():.2e}")

# Verificar conservación de momento angular
L = const['L']
error_L = np.linalg.norm(L - L[0]) / np.linalg.norm(L[0])
print(f"Error relativo en L: {error_L:.2e}")

# Aceptar si error < 1e-6
if np.abs(error_E).max() > 1e-6:
    print("ADVERTENCIA: Energía no conservada bien. Reduce dt o aumenta tolerancia.")
```

### 6.4 Comparación de métodos de Kepler

```python
from pymcel import kepler_newton, kepler_semianalitico, kepler_eserie

M, e = 1.5, 0.4

# Resolver con diferentes métodos
E1, err1, n1 = kepler_newton(M, e, delta=1e-10)
E2, err2, n2 = kepler_semianalitico(M, e)
E3, err3, n3 = kepler_eserie(M, e, orden=2)

# Reportar
print(f"Newton:         E={E1:.10f}, iteraciones={n1}")
print(f"Semicertificado: E={E2:.10f}, iteraciones={n2}")
print(f"Series:         E={E3:.10f}, iteraciones={n3}")

# Diferencias
print(f"Diff E1-E2: {abs(E1-E2):.2e}")
print(f"Diff E1-E3: {abs(E1-E3):.2e}")
```

### 6.5 Exportar resultados

```python
import pandas as pd
import pymcel as pc

# N-cuerpos a pandas
rs, vs, _, _, _ = pc.ncuerpos_solucion(sistema, ts)
df = pc.ncuerpos_a_pandas(ts, rs, vs)

# Guardar
df.to_csv('resultados_ncuerpos.csv', index=False)

# Cargar
df_loaded = pd.read_csv('resultados_ncuerpos.csv')
```

---

## 7. Casos de uso por disciplina

### Astrodinámica orbital

1. **Transferencia Hohmann:** Usa `solucion_lambert()` con dos órbitas circulares.
2. **Órbita geoestacionaria:** Resuelve Kepler para ley 3, luego calcula elementos.
3. **Encuentro orbital:** Integra N-cuerpos con fuerzas perturbadoras.

### Mecánica celeste

1. **Problema de 3 cuerpos:** Usa `crtbp_solucion()` y analiza `constante_jacobi()`.
2. **Órbitas halo alrededor de Lagrange:** Resuelve puntos con `funcion_puntos_colineales()`.
3. **Estabilidad de sistemas binarios:** Integra con `ncuerpos_solucion()` y monitorea constantes.

### Astronomía observacional

1. **Observar asteroides:** `consulta_horizons()` con SBDB.
2. **Seguimiento satelital:** `consulta_spice()` con kernels DE430.
3. **Efemérides planetarias:** `consulta_horizons()` para campañas de observación.

---

## 8. Tips y precauciones

1. **El import de `pymcel` imprime un mensaje** de bienvenida. No asumir import silencioso en pipelines.

2. **Kernels SPICE:** Requieren llamadas explícitas a `descarga_kernels()` o `prepara_spice()` antes de consultas. No están integrados por defecto.

3. **Unidades:** Horizons devuelve en km y km/s. SPICE también. Convertir explícitamente si necesitas SI (m/s²) u otras.

4. **Tolerancias:** La mayoría de solucionesde Kepler y métodos iterativos usan `delta` como término de convergencia. Fijar en 1e-10 o menor para precisión de punto flotante.

5. **Marcos de referencia:** CRTBP distingue entre marco rotante y marco inercial. Tener cuidado con qué retorna.

6. **Visualización scalable:** Para >10,000 partículas o >1000 pasos temporales, evitar Matplotlib. Usar Plotly con `backend='webgl'` o exportar a HDF5.

7. **Reproducibilidad:** Fijar `np.random.seed()` si hay estocasticidad. Documentar parámetros de tolerancia y discretización.

---

## 9. Ejemplos avanzados

### 9.1 Órbita de Lissajous alrededor de L1 (Tierra-Sol)

```python
import numpy as np
from pymcel import crtbp_solucion, constante_jacobi, plot_ncuerpos_3d

# Parámetro masa Tierra-Sol en unidades canónicas
mu = 3.003e-6

# Punto L1 (calculado numéricamente)
from scipy.optimize import fsolve
def eq_L1(x):
    return (1-mu)/(x+mu)**2 - mu/(1-x-mu)**2 + x

x_L1 = fsolve(eq_L1, 0.9)[0]

# Órbita Lissajous pequeña alrededor de L1
r0 = [x_L1 + 0.01, 0.01, 0.0]
v0 = [0.0, 0.0005, 0.0]
ts = np.linspace(0, 20, 1000)

rs_rot, vs_rot, rs_ine, vs_ine, _, _ = crtbp_solucion(mu, r0, v0, ts)
CJ = constante_jacobi(mu, rs_rot, vs_rot)

print(f"L1 en x={x_L1:.6f}, CJ={CJ[0]:.6f}")

# Graficar
fig = plot_ncuerpos_3d(rs_rot, vs_rot, tipo='plotly')
fig.show()
```

### 9.2 Integración de perturbaciones solares en órbita lunar

```python
import numpy as np
from pymcel import ncuerpos_solucion

# Tierra, Luna, Sol (unidades canónicas SLR)
UL = 3.844e8  # m (distancia Tierra-Luna)
UM = 5.972e24  # kg (masa Tierra)
UT = np.sqrt(UL**3 / (6.674e-11 * UM))

GM_earth = 3.986e14 / (UL**3 / UT**2)
GM_moon = 4.904e12 / (UL**3 / UT**2)
GM_sun = 1.327e20 / (UL**3 / UT**2)

sistema = [
    dict(m=1.0, r=[0, 0, 0], v=[0, 0, 0]),  # Tierra
    dict(m=GM_moon/GM_earth, r=[1, 0, 0], v=[0, 1, 0]),  # Luna
    dict(m=GM_sun/GM_earth, r=[390, 0, 0], v=[0, -0.1, 0]),  # Sol (x lejano)
]

ts = np.linspace(0, 30, 500)
rs, vs, _, _, const = ncuerpos_solucion(sistema, ts)

# Luna (partícula 1) en marco Tierra
r_lunar = rs[1] - rs[0]
print(f"Distancia Tierra-Luna inicial: {np.linalg.norm(r_lunar[0]):.3f} UA")
print(f"Distancia Tierra-Luna final: {np.linalg.norm(r_lunar[-1]):.3f} UA")
```

### 9.3 Transferencia Lambert multiarco

```python
import numpy as np
from pymcel import solucion_lambert

# Tres puntos de paso: Tierra -> Venus -> Mercurio
P_earth = np.array([1.0, 0.0, 0.0])
P_venus = np.array([0.723, 0.0, 0.0])
P_merc = np.array([0.387, 0.0, 0.0])

# Transferencias
tf1 = 1.0  # Tierra -> Venus
tf2 = 0.5  # Venus -> Mercurio

V1_t, V2_t, _ = solucion_lambert(P_earth, P_venus, tf1, mu=1.0, direccion='pro')
V1_v, V2_v, _ = solucion_lambert(P_venus, P_merc, tf2, mu=1.0, direccion='pro')

print(f"Impulso Tierra -> Venus: {np.linalg.norm(V1_t):.3f}")
print(f"Impulso Venus -> Mercurio: {np.linalg.norm(V1_v):.3f}")
```

---

## 10. Contacto y referencias

- **Repositorio**: https://github.com/usuario/pymcel
- **Documentación oficial**: Incluida en repositorio bajo `docs/`
- **Cuadernos ejemplos**: `ejemplos/cuadernos-libro/` contiene 22 notebooks temáticos
- **Issues y contribuciones**: GitHub issues tracker

---

## Apéndice: Almacén de fragmentos de código

### Setup estándar para agent

```python
import numpy as np
import matplotlib.pyplot as plt
import pymcel as pc
from pymcel.constantes import G, au, year, GM_sun
from scipy.integrate import odeint
```

### Función auxiliar: Integrator robusto con validación

```python
def integrar_n_cuerpos_robusto(sistema, t_ini, t_fin, n_pasos=1000, 
                               tol_energia=1e-6, verbose=True):
    """
    Integra N-cuerpos con validación automática.
    """
    ts = np.linspace(t_ini, t_fin, n_pasos)
    rs, vs, rps, vps, const = pc.ncuerpos_solucion(sistema, ts)
    
    E = const['E']
    error_E = np.abs((E - E[0]) / E[0]).max()
    
    if verbose:
        print(f"Integración: {n_pasos} pasos, dt={ts[1]-ts[0]:.3e}")
        print(f"Error energía: {error_E:.2e}")
        if error_E > tol_energia:
            print(f"  ADVERTENCIA: Error > {tol_energia}")
    
    return rs, vs, rps, vps, const, {'error_E': error_E}
```

---

**Última actualización**: Marzo 2026  
**Versión de pymcel**: 0.9.10  
**Estado**: Producción

"""
Constantes físicas y astronómicas.

Notas:

- Todas las constantes están en unidades SI.
- Las constantes de astropy están en unidades SI, pero se pueden convertir a otras unidades usando la función
    astropy.constants.convert.
- Las constantes de astropy están en el módulo astropy.constants.
- Los parámetros gravitacionales vienen del kernel SPICE DE441.
"""

#############################################################
# CONSTANTES DE ASTROPY
#############################################################
lista = [
    'G', 'N_A', 'R', 'Ryd', 'a0', 'alpha', 'atm', 
    'b_wien', 'c', 'e', 'eps0', 'g0', 'h', 'hbar', 'k_B', 'm_e', 'm_n', 'm_p', 'mu0', 'muB', 
    'sigma_T', 'sigma_sb', 'u', 'GM_earth', 'GM_jup', 'GM_sun', 'L_bol0', 'L_sun', 'M_earth', 'M_jup', 
    'M_sun', 'R_earth', 'R_jup', 'R_sun', 'au', 'kpc', 'pc', 
]
exec('from astropy.constants import ' + ', '.join(lista))
for constante in lista:
    exec(f"{constante} = {constante}.value")

#############################################################
# OTRAS CONSTANTES
#############################################################
lista += ['año','día',]
día = 24 * 60 * 60
año = 365.25 * día
dia = día
yr = año

#############################################################
# CONSTANTES ACTUALIZADAS
#############################################################
mu_mercury = 22031.868551e9  # m^3/s^2, SPICE Kernels DE441
mu_venus = 324858.592000e9   # m^3/s^2, SPICE Kernels DE441
mu_earth = 398600.435507e9   # m^3/s^2, SPICE Kernels DE441
mu_emb = 403503.235625e9  # m^3/s^2, SPICE Kernels DE441
mu_mars = 42828.375816e9     # m^3/s^2, SPICE Kernels DE441
mu_jupiter = 126712764.100000e9  # m^3/s^2, SPICE Kernels DE441
mu_saturn = 37940584.841800e9    # m^3/s^2, SPICE Kernels DE441
mu_uranus = 5794556.400000e9     # m^3/s^2, SPICE Kernels DE441
mu_neptune = 6836527.100580e9    # m^3/s^2, SPICE Kernels DE441
mu_pluto = 975.500000e9          # m^3/s^2, SPICE Kernels DE441
mu_sun = 132712440041.279419e9   # m^3/s^2, SPICE Kernels DE441
mu_moon = 4902.800118e9          # m^3/s^2, SPICE Kernels DE441

lista += [
    'mu_mercury', 'mu_venus', 'mu_earth', 'mu_emb', 
    'mu_mars', 'mu_jupiter', 'mu_saturn', 'mu_uranus', 
    'mu_neptune', 'mu_pluto', 'mu_sun', 'mu_moon'
]

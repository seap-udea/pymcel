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
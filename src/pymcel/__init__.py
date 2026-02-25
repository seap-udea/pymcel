from pymcel.version import *
import numpy as np
import os
import re
import requests
import glob
import sys
import types
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy import sin, cos
from spiceypy import mxv
import spiceypy as spy
from numpy import zeros_like, pi, arccos, linspace
from numpy import pi
from numpy import arccos
from numpy import linspace,pi
from numpy import sin,cos,tan
import matplotlib.pyplot as plt
from numpy import zeros,floor
from numpy import array,concatenate
from numpy.linalg import norm
from scipy.integrate import odeint
from numpy import zeros,cross
from numpy import subtract
from numpy import hstack
from numpy import sin,cos,sinh,cosh,tan,tanh
from numpy import sqrt,arctan,arctanh
from numpy import tan
from numpy import arctan
from numpy import dot
from spiceypy import rotate,mxv,vcrss
try:
    from scipy.misc import derivative
except:
    from scipy.differentiate import derivative   
from scipy.integrate import quad
from scipy.special import jv
import math
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import pandas as pd
from matplotlib import patches
from pymcel import constantes
from plotly import graph_objs as go

def _welcome():
    """Muestra un mensaje de bienvenida al importar PyMCel.

    Examples
    --------
    >>> import pymcel  # dispara el mensaje de bienvenida
    """
    print(f"Bienvenido a PyMCel v{version} ¡al infinito y más allá!")
_welcome()

#Root directory
try:
    FILE=__file__
    ROOTDIR=os.path.abspath(os.path.dirname(FILE))
except:
    FILE=""
    ROOTDIR=os.path.abspath('')

def unidades_canonicas(UL=None, UM=None, UT=None, G=1):
    """Calcula unidades canonicas consistentes con un valor de G.

    Parameters
    ----------
    UL : float, optional
        Unidad de longitud en SI (valor del metro canonico).
    UM : float, optional
        Unidad de masa en SI (valor del kilogramo canonico).
    UT : float, optional
        Unidad de tiempo en SI (valor del segundo canonico).
    G : float, optional
        Valor deseado de G en el sistema canonico (adimensional).

    Returns
    -------
    tuple
        `(UL, UM, UT, Gc)` con unidades canonicas en SI y el valor de
        la constante gravitacional `Gc` en el sistema canonico.

    Examples
    --------
    >>> UL, UM, UT, Gc = unidades_canonicas(UL=1.0e3, UM=1.0)

    Elaborado por
    -------------
        GPT-5.2-Codex, prompt por Jorge I. Zuluaga
        Pruebas y Codigo adaptado por Jorge I. Zuluaga
    """
    provided = [UL is not None, UM is not None, UT is not None]
    if sum(provided) < 2:
        raise ValueError("Debes proporcionar al menos dos de UL, UM, UT.")
    if G is None or G <= 0:
        raise ValueError("El valor de G debe ser positivo.")

    G_si = constantes.G

    if UL is not None and UM is not None and UT is not None:
        Gc = G_si * (UM * UT**2 / UL**3)
        return UL, UM, UT, Gc

    if UL is None:
        UL = (G_si * UM * UT**2 / G) ** (1.0 / 3.0)
    elif UM is None:
        UM = G * UL**3 / (G_si * UT**2)
    else:
        UT = math.sqrt(G * UL**3 / (G_si * UM))

    return UL, UM, UT, G

def ubica_archivos(path,basedir=None):
    """Obtiene la ruta absoluta de un archivo de datos del paquete.

    Parameters
    ----------
    path : str
        Nombre del archivo dentro del directorio `data` del paquete.
    basedir : str, optional
        Directorio base donde esta el paquete instalado. Si es `None`,
        se usa el directorio del modulo.

    Returns
    -------
    str
        Ruta absoluta al archivo de datos.

    Examples
    --------
    >>> ubica_archivos("kernels.txt")
    """
    if basedir is None:
        basedir = ROOTDIR
    return os.path.join(basedir,'data',path);

def descarga_kernel(url,filename=None,overwrite=False,basedir=None,verbose=False):
    """Descarga un kernel SPICE al directorio de datos del paquete.

    Parameters
    ----------
    url : str
        URL del kernel a descargar.
    filename : str, optional
        Nombre de archivo destino. Si es `None`, se toma del URL.
    overwrite : bool, optional
        Si `True`, reescribe un archivo existente.
    basedir : str, optional
        Directorio base donde esta el paquete instalado.
    verbose : bool, optional
        Si `True`, imprime mensajes de avance.

    Examples
    --------
    >>> descarga_kernel("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp")
    """
    if not filename:
        filename=url.split("/")[-1]
    if filename == 'kernels.txt':
        return
    qdata=False
    if 'data:' in filename:
        filename=url.split(":")[1]
        qdata=True
    
    if verbose:print(f"Descargando kernel '{filename}' en '{basedir}'...")
    if not os.path.exists(ubica_archivos(filename,basedir)) or overwrite:
        if qdata:
            os.system(f"cp -rf {ROOTDIR}/data/{filename} {basedir}/data/")
        else:
            response = requests.get(url)
            open(ubica_archivos(filename,basedir),"wb").write(response.content)
        if verbose:print("Hecho.")
    else:
        if verbose:print(f"El kernel '{filename}' ya fue descargado")
        
def descarga_kernels(basedir='pymcel/',overwrite=False,verbose=False):
    """Descarga todos los kernels listados por el paquete.

    Parameters
    ----------
    basedir : str, optional
        Directorio base donde se guardan los kernels.
    overwrite : bool, optional
        Si `True`, reescribe kernels existentes.
    verbose : bool, optional
        Si `True`, imprime mensajes de avance.

    Examples
    --------
    >>> pc.descarga_kernels()
    """
    descarga_kernel("https://raw.githubusercontent.com/seap-udea/pymcel/main/src/pymcel/data/kernels.desc",
                    overwrite=overwrite,basedir=basedir)
    f=open(ubica_archivos("kernels.desc"),"r")
    kernel_dir = basedir+"/data/" 
    if not os.path.exists(kernel_dir):
        if verbose:print(f"Creando el directorio con los kernels {kernel_dir}...")
        os.makedirs(kernel_dir)
    for line in f:
        url=line.strip()
        descarga_kernel(url,basedir=basedir,overwrite=overwrite,verbose=verbose)

def carga_kernels(basedir='pymcel/', verbose=False):
    """Carga los kernels SPICE descargados en el sistema.

    Parameters
    ----------
    basedir : str, optional
        Directorio base donde se guardan los kernels.
    verbose : bool, optional
        Si `True`, imprime mensajes de avance.

    Examples
    --------
    >>> carga_kernels(verbose=True)
    """
    if not os.path.isfile(ubica_archivos("kernels",basedir)):
        descarga_kernels(basedir=basedir)
    if verbose:print(f"Cargando todos los kernels de SPICE...")
    try:
        spy.furnsh([
            ubica_archivos("kernels.txt",basedir)
        ])
        if verbose:print(f"El entorno está listo para usar los datos de SPICE.")
    except Exception as e:
        print(f"Error al cargar los kernels: {e}")

def lista_kernels(basedir='pymcel/'):
    """Lista los kernels disponibles en el directorio de datos.

    Parameters
    ----------
    basedir : str, optional
        Directorio base donde se guardan los kernels.

    Returns
    -------
    list[str]
        Rutas a los archivos encontrados.

    Examples
    --------
    >>> lista_kernels()
    """
    print("Para descargar todos los kernels use: pymcel.descarga_kernels(). Para descargar un kernel específico use pymcel.descarga_kernel(<url>)")
    return glob.glob(ubica_archivos("*",basedir))

def obtiene_datos(basedir='pymcel/'):
    """Copia los datos del paquete al directorio de trabajo.

    Parameters
    ----------
    basedir : str, optional
        Directorio base donde se guardan los datos.

    Examples
    --------
    >>> pc.obtiene_datos()
    """
    # Descarga todos los kernels para trabajar con SPICE
    descarga_kernels()

    # Obtiene los datos a partir del directorio de instalación de PYMCEL
    datadir = f"{ROOTDIR}/data"
    if os.path.isdir(datadir):
        if not os.path.isdir("pymcel/data"):
            os.mkdir("pymcel/data")
        print("Copiando archivos de datos...")
        os.system(f"cp -rf {datadir}/*.* pymcel/data/")    

def consulta_horizons(id='399',location='@0',epochs=None,datos='vectors',propiedades='default'):
    """Realiza una consulta en Horizons usando astroquery.

    Parameters
    ----------
    id : str, optional
        Identificador del cuerpo en Horizons.
    location : str, optional
        Ubicacion del observador (p. ej. '@0').
    epochs : str | list | dict | float | int, optional
        Epoca(s) de consulta. Puede ser una fecha, lista de fechas o
        un diccionario con `start`, `stop`, `step`.
    datos : {'vectors', 'elements', 'ephemeris'}, optional
        Tipo de datos a solicitar.
    propiedades : list | 'default', optional
        Propiedades a extraer y sus unidades, por ejemplo
        `[('x','m'),('y','m'),('z','m')]`. Si es 'default', usa un
        conjunto por defecto segun `datos`.

    Returns
    -------
    tabla : astropy.table.Table
        Tabla de resultados de Horizons.
    ts : numpy.ndarray | float
        Tiempos en JD de la consulta.
    salida : pandas.DataFrame | numpy.ndarray
        Datos convertidos a las unidades solicitadas.

    Examples
    --------
    >>> epochs = '2024-01-01 12:00:00'
    >>> tabla, ts, salida = pc.consulta_horizons(
    ...     id='399', location='@0', datos='elements',
    ...     propiedades='elementos', epochs=epochs
    ... )
    >>> epochs = ['2024-01-01 12:00:00', '2024-01-02 12:00:00']
    >>> tabla, ts, salida = pc.consulta_horizons(
    ...     id='399', location='@0', datos='elements',
    ...     propiedades='elementos', epochs=epochs
    ... )
    >>> epochs = dict(start='2024-01-01 12:00:00', stop='2024-01-02 12:00:00', step='1d')
    >>> propiedades = [('a','km'),('incl','deg')]
    >>> tabla, ts, salida = pc.consulta_horizons(
    ...     id='399', location='@0', datos='elements',
    ...     propiedades=propiedades, epochs=epochs
    ... )
    """

    # Verifica cuál es la información solicitada
    if propiedades == 'default':
        if datos == 'vectors':
            propiedades = [
                ('x','m'),('y','m'),('z','m'),
                ('vx','m/s'),('vy','m/s'),('vz','m/s')
            ]
        elif datos == 'elements':
            propiedades = [
                ('a','m'),('e','--'),('incl','deg'),
                ('Omega','deg'),('w','deg'),('nu','deg'),
                ('M','deg'),('P','d'),('n','deg/d'),('Tp_jd','d')
            ]
        else:
            propiedades = None

    # verifica el formato de las épocas 
    if isinstance(epochs,dict):
        # Mantiene el formato original
        epochs = epochs

    elif isinstance(epochs,(list,pd.core.series.Series,np.ndarray)):
        if isinstance(epochs,list):
            lista = []
            for epoch in epochs:
                if isinstance(epoch,str):
                    time = Time(epoch).jd
                else:
                    time = epoch
                lista += [time]
            epochs = lista
        else:
            # Mantiene el formato original
            epochs = epochs
    elif isinstance(epochs,str):
        # En este caso es una fecha individual
        epochs = [Time(epochs).jd]

    elif isinstance(epochs,(float,int)):
        # En este caso es un valor individual en JD
        epochs = [epochs]

    else:
        raise ValueError("El formato de epochs no es reconocido")

    # Realizamos el query y obtenemos la tabla
    query = Horizons(id=id, location=location, epochs=epochs)
    tabla = eval(f"query.{datos}()")
    data = tabla.to_pandas()

    tiempos = np.array(data.datetime_jd)

    # Extraemos las propiedades
    if propiedades is not None:
        # Obtenemos las unidades de las columnas
        unidades = dict()
        for columna in tabla.columns:
            unidades[columna] = tabla[columna].unit
    
        # Convertirmos a las unidades solicitadas
        salida = dict()
        for item in propiedades:
            propiedad = item[0]
            unidad = item[1]
            try:
                unit = unidades[propiedad]
            except KeyError:
                raise KeyError(f"Propiedad '{propiedad}' no reconocida. Las columnas son: {unidades.keys()}")

            if unidad == '--':
                factor = 1
            else:
                factor = 1*unidades[propiedad].to(unidad)
            salida[propiedad] = data[propiedad]*factor
        salida = pd.DataFrame(salida)
    else:
        salida = data

    if len(salida) < 2:
        salida = np.array(salida.iloc[0])
        tiempos = tiempos[0]

    return tabla, tiempos, salida

def prepara_spice(verbose=True):
    """Prepara el entorno SPICE descargando y cargando kernels.

    Parameters
    ----------
    verbose : bool, optional
        Si `True`, imprime mensajes de avance.

    Examples
    --------
    >>> prepara_spice(verbose=True)
    """
    # Obtiene kernels si no se han descargado
    if not os.path.isfile('pymcel/data/kernels.txt'):
        descarga_kernels(verbose=verbose)

    # Carga todos los kernels
    if verbose:
        print(f"Cargando todos los kernels de SPICE...")
    spy.furnsh([
        'pymcel/data/kernels.txt'
    ])
    if verbose:
        print(f"El entorno está listo para usar los datos de SPICE.")

def consulta_spice(id='399', location='@0', epochs=None):
    """Consulta vectores de estado desde SPICE usando kernels cargados.

    El kernel por defecto (DE430) tiene una precision de ~1 km para la Tierra.

    Parameters
    ----------
    id : str, optional
        Identificador del cuerpo.
    location : str, optional
        Identificador del observador.
    epochs : str | list | dict | float | int, optional
        Epoca(s) de consulta. Puede ser una fecha o una lista de fechas.

    Returns
    -------
    data : numpy.ndarray
        Vector(es) de estado en SI.
    ts : numpy.ndarray
        Epocas en JD.
    df : pandas.DataFrame | numpy.ndarray
        DataFrame con columnas x, y, z, vx, vy, vz.

    Examples
    --------
    >>> data, ts, df = consulta_spice(id='399', location='@0', epochs='2024-01-01 12:00:00')
    """
    # verifica el formato de las épocas
    if isinstance(epochs,dict):
        # Mantiene el formato original
        time_start = Time(epochs['start'],scale='tt').jd
        time_stop = Time(epochs['stop'],scale='tt').jd
        time_step = epochs['step']
        match = re.match(r"(\d+)([a-zA-Z]+)",time_step)
        if match:
            number = int(match.group(1))  # Extract number part
            letters = match.group(2)      # Extract letter part
            tstep = number
            if letters == 'd':
              deltat = 1
            elif letters == 'h':
              deltat = 1/24
            elif letters == 'm':
              deltat = 1/24/60
            elif letters == 's':
              deltat = 1/24/60/60
        else:
            raise ValueError(f"El paso provisto '{time_step}' no es reconocido")

        epochs = np.arange(time_start,time_stop+deltat*tstep/2,deltat*tstep)

    elif isinstance(epochs,(list,pd.core.series.Series,np.ndarray)):
        if isinstance(epochs,list):
            lista = []
            for epoch in epochs:
                if isinstance(epoch,str):
                    time = Time(epoch,scale='tt').jd
                else:
                    time = epoch
                lista += [time]
            epochs = lista
        else:
            # Mantiene el formato original
            epochs = epochs
    elif isinstance(epochs,str):
        # En este caso es una fecha individual
        epochs = [Time(epochs,scale='tt').jd]

    # Carga todos los kernels
    if not os.path.isfile('pymcel/data/kernels.txt'):
        descarga_kernels(verbose=False)
        prepara_spice()

    ets = []
    for epoch in epochs:
        et = spy.unitim(epoch,'JDTDB','ET')
        ets += [ et ]

    Xs = []
    for et in ets:
        X,tl = spy.spkezr(id,et,'ECLIPJ2000','None',location.replace('@',''))
        Xs += [X*1000]

    if len(ets)>1:
        data = np.array(Xs)
        df = pd.DataFrame(data,columns=['x','y','z','vx','vy','vz'])
    else:
        data = Xs[0]
        df = Xs[0]

    return data, np.array(epochs), df

def consulta_propiedad(id='399',propiedad='masa',nvalues=1):
    """Obtiene propiedades de los kernels TPC.

    Parameters
    ----------
    id : str, optional
        Identificador del cuerpo.
    propiedad : str, optional
        Nombre de la propiedad (por defecto 'masa').
    nvalues : int, optional
        Numero de valores solicitados para propiedades no escalares.

    Returns
    -------
    float | numpy.ndarray
        Valor(es) de la propiedad solicitada.

    Examples
    --------
    >>> masa_tierra = consulta_propiedad(id='399', propiedad='masa')
    """
    # Obtiene kernels si no se han descargado
    if not os.path.isfile('pymcel/data/kernels.txt'):
        descarga_kernels()
    # Carga todos los kernels
    spy.furnsh([
        'pymcel/data/kernels.txt'
    ])

    # Carga la propiedad
    if propiedad == 'masa':
        # La masa es especial porque necesita factor G
        valor = spy.bodvrd(id,'GM',1)[1][0]/(constantes.G*1e-9)
    else:
        valor = spy.bodvrd(id,propiedad,nvalues)[1]

    return valor

def fija_ejes_proporcionales(ax,values=(),margin=0,xcm=None,ycm=None,xmin=None,ymin=None):
    """Ajusta los ejes para mantener proporciones en 2D.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje donde se aplica el ajuste.
    values : tuple, optional
        Tupla con datos que definen el rango (se convierten a arrays).
    margin : float, optional
        Margen relativo alrededor del grafico.
    xcm, ycm : float, optional
        Centro del rango en x e y. Si es `None`, se usa el centro de los datos.
    xmin, ymin : float, optional
        Fuerza el limite inferior en x o y, manteniendo el rango actual.

    Returns
    -------
    tuple
        Tupla con limites `(xlims, ylims)`.

    Examples
    --------
    >>> fija_ejes_proporcionales(ax, rs)
    >>> xrango, yrango = fija_ejes_proporcionales(ax, valores, xcm=0)
    """
    
    #values
    vals=np.array([])
    for value in values:
        vals=np.append(vals,np.array(value).flatten())
    #Center of values
    rcm=vals.mean()
    vals=vals-rcm

    if xcm is None:
        xcm=rcm
    if ycm is None:
        ycm=rcm
    
    fig=ax.figure
    bbox=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width,height=bbox.width,bbox.height
    fx=width/height
    fy=1
    if fx<1:
        factor=fy
        fy=(1+margin)*1/fx
        fx=(1+margin)*factor
    else:
        fx*=(1+margin)
        fy*=(1+margin)

    max_value=np.abs(vals).max()
    ax.set_xlim((xcm-fx*max_value,xcm+fx*max_value))
    ax.set_ylim((ycm-fy*max_value,ycm+fy*max_value))

    if xmin is not None:
        xinf,xsup=ax.get_xlim()
        dx=xsup-xinf
        ax.set_xlim((xmin,xmin+dx))

    if ymin is not None:
        yinf,ysup=ax.get_ylim()
        dy=ysup-yinf
        ax.set_ylim((ymin,ymin+dy))

    return ax.get_xlim(),ax.get_ylim()

def encuentra_rangos(rs):
    """Calcula rangos en x, y, z para datos 3D.

    Parameters
    ----------
    rs : numpy.ndarray
        Arreglo 2D o 3D con coordenadas `(x, y, z)`.

    Returns
    -------
    tuple
        `(xlims, ylims, zlims)` con limites centrados y de igual escala.

    Examples
    --------
    >>> rangos = encuentra_rangos(rs)
    """
    cube = len(rs.shape)

    if cube == 2:
        x_limits = rs[:,0].min(),rs[:,0].max()
        y_limits = rs[:,1].min(),rs[:,1].max()
        z_limits = rs[:,2].min(),rs[:,2].max()    
    else:
        x_limits = rs[:,:,0].min(),rs[:,:,0].max()
        y_limits = rs[:,:,1].min(),rs[:,:,1].max()
        z_limits = rs[:,:,2].min(),rs[:,:,2].max()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    return ([x_middle - plot_radius, x_middle + plot_radius],
            [y_middle - plot_radius, y_middle + plot_radius],
            [z_middle - plot_radius, z_middle + plot_radius])

def fija_ejes3d_proporcionales(ax,rangos=None):
    """Ajusta los ejes en 3D para que tengan la misma escala.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje 3D de matplotlib.
    rangos : tuple, optional
        Rangos `(xlims, ylims, zlims)` precomputados.

    Returns
    -------
    tuple
        Limites `(xlims, ylims, zlims)` aplicados al eje.

    Examples
    --------
    >>> fija_ejes3d_proporcionales(ax)
    """
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    if rangos == None:
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
    else:
        x_range = abs(rangos[0][1] - rangos[0][0])
        x_middle = (rangos[0][1] + rangos[0][0])/2
        y_range = abs(rangos[1][1] - rangos[1][0])
        y_middle = (rangos[1][1] + rangos[1][0])/2
        z_range = abs(rangos[2][1] - rangos[2][0])
        z_middle = (rangos[2][1] + rangos[2][0])/2

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.55*max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    return ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()

def plot_ncuerpos_3d(rs,vs=None,tipo='matplotlib',**opciones):
    """Grafica trayectorias 3D de un sistema de N cuerpos.

    Parameters
    ----------
    rs : numpy.ndarray
        Posiciones con forma `(N, Nt, 3)`.
    vs : numpy.ndarray, optional
        Velocidades (no se usan para el grafico, se mantiene por API).
    tipo : {'matplotlib', 'plotly'}, optional
        Motor de graficacion.
    **opciones
        Opciones del trazado (matplotlib o plotly segun `tipo`).

    Returns
    -------
    matplotlib.figure.Figure | plotly.graph_objects.Figure
        Figura generada.

    Examples
    --------
    >>> fig = plot_ncuerpos_3d(rps, vps)
    """

    #Número de partículas
    N=rs.shape[0]

    if tipo == 'matplotlib':

        opciones_defecto = dict(lw=1)
        opciones_defecto.update(opciones)

        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')

        for i in range(N):
            ax.plot(rs[i,:,0],rs[i,:,1],rs[i,:,2],**opciones_defecto);

        fija_ejes3d_proporcionales(ax);
        fig.tight_layout();
        plt.show();
        return fig

    elif tipo == 'plotly':

        opciones_defecto = dict(
            mode='lines',
            name='Cuerpo',
            marker=dict(),
            line=dict(),
        )
        opciones_defecto.update(opciones)

        try:
            import plotly.graph_objects as go
        except:
            print("Debes instalar primero plotly en tu sistema: pip install -Uq plotly")
            return None

        fig = go.Figure()
        for i in range(N):
            xs = rs[i,:,0]
            ys = rs[i,:,1]
            zs = rs[i,:,2]
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode=opciones_defecto['mode'],
                    name=opciones_defecto['name']+f" {i}",
                    marker=opciones_defecto['marker'],
                )
            )
        rmin = rs.min()
        rmax = rs.max()

        rangos = encuentra_rangos(rs)
        fig['layout']['scene']['aspectmode'] = 'cube'
        for i,axis in enumerate(['xaxis','yaxis','zaxis']):
            fig['layout']['scene'][axis]['range'] = rangos[i]
        fig.show()
    else:
        raise AssertionError(f"Tipo de gráfico '{tipo}' no reconocido")

    return fig

from scipy.interpolate import interp1d
def plot_doscuerpos_3d(rs,vs=None,tipo='matplotlib',ts=None,**opciones):
    """Grafica la trayectoria 3D relativa de dos cuerpos.

    Parameters
    ----------
    rs : numpy.ndarray
        Posiciones con forma `(Nt, 3)`.
    vs : numpy.ndarray, optional
        Velocidades asociadas (no se usan para el grafico).
    tipo : {'matplotlib', 'plotly'}, optional
        Motor de graficacion.
    ts : numpy.ndarray, optional
        Tiempos para interpolacion suave si se proporciona.
    **opciones
        Opciones del trazado (matplotlib o plotly segun `tipo`).

    Returns
    -------
    matplotlib.figure.Figure | plotly.graph_objects.Figure
        Figura generada.

    Examples
    --------
    >>> fig = plot_doscuerpos_3d(rs, vs, tipo='matplotlib')
    """

    #Número de partículas
    N=rs.shape[0]

    if ts is not None:
        xfun = interp1d(ts,rs[:,0],kind='cubic')
        yfun = interp1d(ts,rs[:,1],kind='cubic')
        zfun = interp1d(ts,rs[:,2],kind='cubic')
        tss = np.linspace(ts[0],ts[-1],10*len(ts))
        rs = np.array([[xfun(t),yfun(t),zfun(t)] for t in tss])
        print(rs.shape)
        
    if tipo == 'matplotlib':  
    
        opciones_defecto = dict(color='k',lw=1)
        opciones_defecto.update(opciones)
    
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')

        ax.plot(rs[:,0],rs[:,1],rs[:,2],**opciones_defecto);

        rangos = encuentra_rangos(rs)
        fija_ejes3d_proporcionales(ax,rangos);

        fig.tight_layout();
        plt.show();
        return fig

    elif tipo == 'plotly':
        
        opciones_defecto = dict(
            mode='lines',
            name='Vector relativo',
            marker=dict(color='Black'),
            line=dict(color='Black'),
        )
        opciones_defecto.update(opciones)

        try:
            import plotly.graph_objects as go
        except:
            print("Debes instalar primero plotly en tu sistema: pip install -Uq plotly")
            return None

        fig = go.Figure()
        xs = rs[:,0]
        ys = rs[:,1]
        zs = rs[:,2]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode=opciones_defecto['mode'],
                name=opciones_defecto['name'],
                marker=opciones_defecto['marker'],
            )
        )
        rmin = rs.min()
        rmax = rs.max()

        rangos = encuentra_rangos(rs)
        fig['layout']['scene']['aspectmode'] = 'cube'
        for i,axis in enumerate(['xaxis','yaxis','zaxis']):
            fig['layout']['scene'][axis]['range'] = rangos[i]
        fig.show()
    else:
        raise AssertionError(f"Tipo de gráfico '{tipo}' no reconocido")

    return fig

def haversine(lon1, lat1, lon2, lat2):
    """Calcula la distancia angular entre dos puntos sobre una esfera.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : float
        Longitudes y latitudes en grados.

    Returns
    -------
    float
        Distancia angular en grados.

    Examples
    --------
    >>> haversine(-75.6, 6.2, -74.1, 4.6)
    """
    
    lon1, lat1, lon2, lat2 = map(np.radians,[lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(c)

def calcula_discriminante(a,b,c):
    """Calcula el discriminante de un polinomio cuadratico.

    Parameters
    ----------
    a, b, c : float
        Coeficientes del polinomio :math:`ax^2 + bx + c`.

    Returns
    -------
    float
        Discriminante :math:`b^2 - 4ac`.

    Examples
    --------
    >>> calcula_discriminante(1, -2, -3)
    """
    disc=b**2-4*a*c
    return disc

def coeficientes_fourier(funcion,T,k,args=()):
    """Calcula coeficientes de Fourier para una funcion periodica.

    Parameters
    ----------
    funcion : callable
        Funcion a expandir, `f(t, *args)`.
    T : float
        Periodo de la funcion.
    k : int
        Numero de armonicos.
    args : tuple, optional
        Argumentos adicionales para `funcion`.

    Returns
    -------
    tuple
        Listas `(As, Bs)` con coeficientes coseno y seno.

    Examples
    --------
    >>> As, Bs = coeficientes_fourier(f, T, k)
    """
    #Parametro omega
    w=2*pi/T
    
    #Determina los coeficientes en t:
    f=lambda t:funcion(t,*args)
    As=[2*quad(f,0,T,args=args)[0]/T]
    Bs=[0]
    for n in range(1,k+1):
        f_cos_n=lambda t:funcion(t,*args)*cos(n*w*t)
        As+=[2*quad(f_cos_n,0,T)[0]/T]
        f_sin_n=lambda t:funcion(t,*args)*sin(n*w*t)
        Bs+=[2*quad(f_sin_n,0,T)[0]/T]
    
    return As,Bs

def rota_puntos(R,x,y,z):
    """Rota un conjunto de puntos con una matriz de rotacion.

    Parameters
    ----------
    R : array-like
        Matriz de rotacion 3x3.
    x, y, z : array-like
        Coordenadas de los puntos.

    Returns
    -------
    tuple
        Coordenadas rotadas `(xp, yp, zp)`.

    Examples
    --------
    >>> xps, yps, zps = rota_puntos(Rz, xs, ys, zs)
    """
    N=len(x)
    xp=zeros_like(x)
    yp=zeros_like(y)
    zp=zeros_like(z)
    for i in range(N):
        xp[i],yp[i],zp[i]=mxv(R,[x[i],y[i],z[i]])
    return xp,yp,zp

def polinomio_segundo_grado(coeficientes,x,y):
    """Evalua un polinomio de segundo grado en x e y.

    Parameters
    ----------
    coeficientes : array-like
        Coeficientes `(A, B, C, D, E, F)`.
    x, y : array-like
        Variables independientes.

    Returns
    -------
    numpy.ndarray
        Valores del polinomio :math:`Ax^2 + Bxy + Cy^2 + Dx + Ey + F`.

    Examples
    --------
    >>> Pxpsyps = polinomio_segundo_grado(coeficientes, xps, yps)
    """
    A,B,C,D,E,F=coeficientes
    P=A*x**2+B*x*y+C*y**2+D*x+E*y+F
    return P

def puntos_conica(p,e,df=0.1):
    """Genera puntos de una conica en coordenadas cartesianas.

    Parameters
    ----------
    p : float
        Semilatus rectum.
    e : float
        Excentricidad.
    df : float, optional
        Separacion angular en radianes para evitar singularidades.

    Returns
    -------
    tuple
        Arreglos `(xs, ys, zs)` con los puntos de la conica.

    Examples
    --------
    >>> xs, ys, zs = puntos_conica(p, e)
    """

    #Compute fmin,fmax
    if e<1:
        fmin=-pi
        fmax=pi
    elif e>1:
        psi=arccos(1/e)
        fmin=-pi+psi+df
        fmax=pi-psi-df
    else:
        fmin=-pi+df
        fmax=pi-df
            
    #Valores del ángulo
    fs=linspace(fmin,fmax,500)

    #Distancias 
    rs=p/(1+e*cos(fs))

    #Coordenadas
    xs=rs*cos(fs)
    ys=rs*sin(fs)
    zs=zeros_like(xs)
    
    return xs,ys,zs

def conica_de_elementos(p=10.0,e=0.8,i=0.0,Omega=0.0,omega=0.0,
                        df=0.1,
                        elev=30,azim=60,
                        figreturn=False):
    """Grafica una conica 3D a partir de elementos orbitales clasicos.

    Parameters
    ----------
    p, e, i, Omega, omega : float
        Elementos orbitales (angulos en grados).
    df : float, optional
        Separacion angular para evitar singularidades.
    elev, azim : float, optional
        Angulos de vista del grafico 3D.
    figreturn : bool, optional
        Si `True`, devuelve la figura.

    Returns
    -------
    matplotlib.figure.Figure | None
        Figura generada si `figreturn=True`.

    Examples
    --------
    >>> fig = conica_de_elementos(p, e, i*180/pi, W*180/pi, w*180/pi, figreturn=True)
    """

    #Convierte elementos angulares en radianes
    p=float(p)
    e=float(e)
    i=float(i)*pi/180
    Omega=float(Omega)*pi/180
    omega=float(omega)*pi/180
    
    #Compute fmin,fmax
    if e<1:
        fmin=-pi
        fmax=pi
    elif e>1:
        psi=arccos(1/e)
        fmin=-pi+psi+df
        fmax=pi-psi-df
    else:
        fmin=-pi+df
        fmax=pi-df
            
    #Valores del ángulo
    fs=linspace(fmin,fmax,500)

    #Distancia al periapsis
    q=p/(1+e)

    #Distancia al foco
    rs=p/(1+e*cos(fs))

    #Coordenadas
    xs=rs*(cos(Omega)*cos(omega+fs)-cos(i)*sin(Omega)*sin(omega+fs))
    ys=rs*(sin(Omega)*cos(omega+fs)+cos(i)*cos(Omega)*sin(omega+fs))
    zs=rs*(cos(fs)*sin(omega)*sin(i)+sin(fs)*cos(omega)*sin(i))
    
    #Posición del periapsis (f=0)
    xp=q*(cos(Omega)*cos(omega)-cos(i)*sin(Omega)*sin(omega))
    yp=q*(sin(Omega)*cos(omega)+cos(i)*cos(Omega)*sin(omega))
    zp=q*sin(omega)*sin(i)
    
    #Posición del nodo ascendente
    rn=p/(1+e*cos(omega))
    xn=rn*cos(Omega)
    yn=rn*sin(Omega)
    zn=0
    
    #Gráfico

    plt.close("all")
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')

    #Gráfica de los puntos originales
    ax.plot(xs,ys,zs,'b-')
    
    #Posición del periapsis
    ax.plot([0,xp],[0,yp],[0,zp],'r-')

    #Posición del nodo ascendente
    ax.plot([0,xn],[0,yn],[0,zn],'g-')

    #Fija punto de vista
    ax.view_init(elev=elev,azim=azim)
    
    #Decoración
    xrange,yrange,zrange=fija_ejes3d_proporcionales(ax);

    ax.set_title(f"Cónica con:"+rf"$p={p:.2f}$, $e={e:.2f}$, "+rf"$i={i*180/pi:.2f}$, "+rf"$\Omega={Omega*180/pi:.1f}$, "+rf"$\omega={Omega*180/pi:.1f}$")
    
    #Dibuja Ejes
    ax.plot([0,xrange[1]],[0,0],[0,0],'k-')
    ax.plot([0,0],[0,yrange[1]],[0,0],'k-')
    ax.plot([0,0],[0,0],[0,zrange[1]],'k-')
    ax.text(xrange[1],0,0,"$x$",ha='left',va='top')
    ax.text(0,yrange[1],0,"$y$",ha='left',va='top')
    ax.text(0,0,zrange[1],"$z$",ha='left',va='bottom')

    fig.tight_layout();
    
    if figreturn:return fig

def ncuerpos_a_pandas(ts,rs,vs):
    """Convierte una solucion N-cuerpos a un DataFrame de pandas.

    Parameters
    ----------
    ts : numpy.ndarray
        Arreglo de tiempos.
    rs : numpy.ndarray
        Posiciones con forma `(N, Nt, 3)`.
    vs : numpy.ndarray
        Velocidades con forma `(N, Nt, 3)`.

    Returns
    -------
    pandas.DataFrame
        Tabla con columnas `Particula`, `tiempo`, `x`, `y`, `z`, `vx`, `vy`, `vz`.

    Examples
    --------
    >>> df = ncuerpos_a_pandas(ts, rs, vs)
    """
    Np = len(rs[:,0,0])
    Nt = len(rs[0,:,0])
    tabla = np.zeros((Np*Nt, 8))
    tabla[:,1] = np.concatenate((ts,ts,ts))
    for i in range(Np):
        tabla[i*Nt:(i+1)*Nt,0]=i
        tabla[i*Nt:(i+1)*Nt,2:5]=rs[i,:,:]
        tabla[i*Nt:(i+1)*Nt,5:8]=vs[i,:,:]
        df = pd.DataFrame(tabla, columns=['Partícula','tiempo','x','y','z','vx','vy','vz'])
    return df

def edm_ncuerpos(Y,t,N=2,mus=[]):    
    """Ecuaciones de movimiento para N cuerpos (forma directa).

    Parameters
    ----------
    Y : numpy.ndarray
        Vector de estado concatenado `(r1, r2, ..., v1, v2, ...)`.
    t : float
        Tiempo.
    N : int, optional
        Numero de cuerpos.
    mus : list, optional
        Masas (o parametros gravitacionales) de cada cuerpo.

    Returns
    -------
    numpy.ndarray
        Derivada temporal `dY/dt`.

    Examples
    --------
    >>> dY = edm_ncuerpos(Y, t, N=3, mus=mus)
    """
    dYdt=zeros(6*N)

    #Primer conjunto de ecuaciones
    dYdt[:3*N]=Y[3*N:]
    
    #Segundo conjunto de ecuaciones
    for k in range(3*N,6*N):
        l=k%3
        i=int(floor((k-3*N)/3))
        for j in range(N):
            if j==i:continue
            rij=(Y[3*i]-Y[3*j])**2+                (Y[3*i+1]-Y[3*j+1])**2+                (Y[3*i+2]-Y[3*j+2])**2
            dYdt[k]+=-mus[j]*(Y[3*i+l]-Y[3*j+l])/rij**1.5
            
    return dYdt

def edm_ncuerpos_eficiente(Y,t,N=2,mus=[]):
    """Ecuaciones de movimiento N-cuerpos (version eficiente).

    Esta rutina fue mejorada por Simon Echeverri, Astronomia UdeA.

    Parameters
    ----------
    Y : numpy.ndarray
        Vector de estado concatenado.
    t : float
        Tiempo.
    N : int, optional
        Numero de cuerpos.
    mus : list, optional
        Masas (o parametros gravitacionales) de cada cuerpo.

    Returns
    -------
    list
        Derivadas concatenadas en formato plano.

    Examples
    --------
    >>> dY = edm_ncuerpos_eficiente(Y, t, N=3, mus=mus)
    """

    dY=Y[3*N:]
    mj=np.array(mus).reshape(-1,1)
    
    r=Y[:3*N].reshape(N,3)
    dydt=np.zeros((N,3))
    for i in range(N):
        g=(r[i]-r) 
        c=mj*g 
        c=np.delete(c,i,0) 
        g=np.delete(g,i,0)
        rij3=(np.linalg.norm(g,axis=1)**3).reshape(-1,1) 
        A=c/rij3
        dydt[i]=-sum(A)
    return [*dY,*(dydt.flatten())] 

def sistema_a_Y(sistema):
    """Convierte un sistema de particulas a un vector de estado.

    Parameters
    ----------
    sistema : list[dict]
        Lista de particulas con claves `m`, `r`, `v`.

    Returns
    -------
    tuple
        `(N, mus, Y0s)` con numero de particulas, masas y estado inicial.

    Examples
    --------
    >>> N, mus, Y0s = sistema_a_Y(sistema)
    """
    mus=[]
    r0s=[]
    v0s=[]
    N=0
    for particula in sistema:
        m=particula['m']
        if m>0:
            mus+=[m]
            r0s+=list(particula["r"])
            v0s+=list(particula["v"])
            N+=1
    Y0s=array(r0s+v0s)
    mus=array(mus)
    return N,mus,Y0s

def solucion_a_estado(solucion,Nparticulas,Ntiempos):
    """Convierte una solucion plana a arreglos de posiciones y velocidades.

    Parameters
    ----------
    solucion : numpy.ndarray
        Solucion plana con todas las coordenadas.
    Nparticulas : int
        Numero de particulas.
    Ntiempos : int
        Numero de tiempos.

    Returns
    -------
    tuple
        `(rs, vs)` con forma `(N, Nt, 3)`.

    Examples
    --------
    >>> rs, vs = solucion_a_estado(solucion, N, Nt)
    """
    rs=zeros((Nparticulas,Ntiempos,3))
    vs=zeros((Nparticulas,Ntiempos,3))
    for i in range(Nparticulas):
        rs[i]=solucion[:,3*i:3*i+3]
        vs[i]=solucion[:,3*Nparticulas+3*i:3*Nparticulas+3*i+3]
    return rs,vs

def ncuerpos_solucion(sistema,ts):
    """Resuelve el problema N-cuerpos y calcula constantes.

    Parameters
    ----------
    sistema : list[dict]
        Lista de particulas con `m`, `r`, `v`. Cada particula es un
        diccionario con:

        - `m`: masa (o parametro gravitacional)
        - `r`: posicion inicial como iterable de 3 componentes
        - `v`: velocidad inicial como iterable de 3 componentes
    ts : numpy.ndarray
        Tiempos de integracion.

        Returns
        -------
        tuple
                `(rs, vs, rps, vps, constantes)` donde:

                - `rs`, `vs`: arreglos con forma `(N, Nt, 3)` para posiciones y
                    velocidades absolutas.
                - `rps`, `vps`: arreglos con forma `(N, Nt, 3)` para posiciones y
                    velocidades relativas al centro de masa.
                - `constantes`: diccionario con constantes de movimiento y series
                    asociadas (`M`, `RCM`, `PCM`, `L`, `K`, `U`, `E`).

    Examples
    --------
    >>> sistema = [
    ...     dict(m=1.0, r=[-0.5, 0.0, 0.0], v=[0.0, -0.5, 0.0]),
    ...     dict(m=1.0, r=[ 0.5, 0.0, 0.0], v=[0.0,  0.5, 0.0]),
    ... ]
    >>> ts = np.linspace(0, 10, 1000)
    >>> rs, vs, rps, vps, constantes = ncuerpos_solucion(sistema, ts)
    """
    #Condiciones iniciales
    N,mus,Y0s=sistema_a_Y(sistema)
    
    #Masa total
    M=sum(mus)
    
    #Número de tiempos
    Nt=len(ts)
    
    #Solución
    solucion=odeint(edm_ncuerpos_eficiente,Y0s,ts,args=(N,mus))
    
    #Extracción de las posiciones y velocidades
    rs,vs=solucion_a_estado(solucion,N,Nt)
    
    #Calcula las constantes de movimiento
    PCM=zeros(3)
    for i in range(N):
        PCM=PCM+mus[i]*vs[i,0,:]

    #Posición del CM como función del tiempo    
    RCM=zeros((Nt,3))
    for i in range(N):
        RCM=RCM+mus[i]*rs[i,:,:]
    RCM/=M

    #Momento angular
    L=zeros(3)
    for i in range(N):
        L=L+mus[i]*cross(rs[i,0,:],vs[i,0,:])

    #Posiciones y velocidades relativas al centro de masa    
    rps=rs-RCM
    vps=subtract(vs,PCM/M)
    
    #Energía total
    K=zeros(Nt)
    U=zeros(Nt)
    for i in range(N):
        K=K+0.5*mus[i]*norm(vps[i,:,:],axis=1)**2
        for j in range(N):
            if i==j:continue
            rij=norm(rps[i,:,:]-rps[j,:,:],axis=1)
            U+=-0.5*mus[i]*mus[j]/rij
    E=K[0]+U[0]
    
    #Constantes
    constantes=dict(M=M,
                    RCM=RCM,PCM=PCM,
                    L=L,K=K,U=U,E=E)
        
    #Devuelve las posiciones y velocidades
    return rs,vs,rps,vps,constantes

def edm_dos_cuerpos(Y,t,mu):
    """Ecuaciones de movimiento del problema de dos cuerpos.

    Parameters
    ----------
    Y : numpy.ndarray
        Vector de estado `(r, v)` concatenado.
    t : float
        Tiempo.
    mu : float
        Parametro gravitacional.

    Returns
    -------
    numpy.ndarray
        Derivadas concatenadas `(dr/dt, dv/dt)`.

    Examples
    --------
    >>> dY = edm_dos_cuerpos(Y, t, mu)
    """
    r = Y[:3]
    v = Y[3:]
    drdt = v
    dvdt = -mu*r/np.linalg.norm(r)**3
    return np.concatenate([drdt,dvdt])

def doscuerpos_solucion(mu,r,v,ts):
    """Integra el problema de dos cuerpos.

    Parameters
    ----------
    mu : float
        Parametro gravitacional.
    r, v : array-like
        Posicion y velocidad inicial.
    ts : numpy.ndarray
        Tiempos de integracion.

    Returns
    -------
    tuple
        `(rs, vs)` con posiciones y velocidades.

    Examples
    --------
    >>> rs, vs = doscuerpos_solucion(mu, r, v, ts)
    """
    X0 = np.concatenate([r,v])
    solucion = odeint(edm_dos_cuerpos,X0,ts,args=(mu,))
    rs = solucion[:,:3]
    vs = solucion[:,3:]
    return rs,vs

def funcion_kepler(G,M=0,e=0):
    """Evalua la ecuacion de Kepler y sus derivadas.

    Parameters
    ----------
    G : float
        Anomalia excéntrica (eliptica/hiperbolica).
    M : float, optional
        Anomalia media.
    e : float, optional
        Excentricidad.

    Returns
    -------
    tuple
        `(k, kp, kpp)` con funcion y derivadas.

    Examples
    --------
    >>> ks, kps, kpps = funcion_kepler(Gs, M, e)
    """
    #Parametro sigma
    sigma=+1 if e<1 else -1
    #Funciones cG, sG
    from numpy import cos,cosh,sin,sinh
    cG=cos(G) if e<1 else cosh(G)
    sG=sin(G) if e<1 else sinh(G)
    #Función de Kepler
    k=sigma*(G-e*sG)-M
    #Primera derivada
    kp=sigma*(1-e*cG)
    #Segunda derivada
    kpp=e*sG
    return k,kp,kpp

def kepler_kepler(M,e,E0=1.0,delta=1e-5):
    """Resuelve la ecuacion de Kepler por iteracion simple.

    Parameters
    ----------
    M : float
        Anomalia media.
    e : float
        Excentricidad.
    E0 : float, optional
        Valor inicial.
    delta : float, optional
        Tolerancia.

    Returns
    -------
    tuple
        `(E, error, ni)` solucion, error relativo e iteraciones.

    Examples
    --------
    >>> E, error, ni = kepler_kepler(M, e, E0, 1e-8)
    """
    #Valor inicial de la anomalía excéntrica
    E=E0
    #Valor inicial del error relativo
    Dn=1
    #Contador de iteraciones
    ni=0
    while Dn>delta:
        #"En" es igual al último valor de E
        En=E
        #Regla de iteración
        from math import sin
        Mn=En-e*sin(En)
        en=M-Mn
        E=En+en
        #Valor promedio
        Emed=(E+En)/2
        #Error relativo
        Dn=abs(en/M)
        #Conteo de iteraciones
        ni+=1
    return Emed,Dn,ni

def kepler_newton(M,e,G0=1,delta=1e-5):
    """Resuelve la ecuacion de Kepler usando Newton.

    Parameters
    ----------
    M : float
        Anomalia media.
    e : float
        Excentricidad.
    G0 : float, optional
        Valor inicial.
    delta : float, optional
        Tolerancia.

    Returns
    -------
    tuple
        `(G, error, ni)` solucion, error relativo e iteraciones.

    Examples
    --------
    >>> E, error, ni = kepler_newton(M, e, E0, 1e-8)
    """
    #Valor inicial de la anomalía excéntrica
    Gn=G0
    #Valor inicial del error relativo
    Dn=1
    #Contador de iteraciones
    ni=0
    while Dn>delta:
        #Inicializa el valor de En
        G=Gn
        #Función de Kepler y de su primera derivada en G
        from pymcel import funcion_kepler
        k,kp,kpp=funcion_kepler(G,M,e)
        #Nuevo valor (regla de iteración)
        Gn=G-k/kp
        #Valor medio
        Gmed=(G+Gn)/2
        #Criterio de convergencia
        en=Gn-G
        Dn=abs(en/Gmed)
        ni+=1
    return Gmed,Dn,ni

def kepler_aproximacion(M,e,orden=1):
    """Aproxima la solucion de Kepler por serie truncada.

    Parameters
    ----------
    M : float
        Anomalia media.
    e : float
        Excentricidad.
    orden : int, optional
        Orden de aproximacion (1, 2, 3).

    Returns
    -------
    tuple
        `(E, error, 1)` con aproximacion y error relativo.

    Examples
    --------
    >>> E1, error1, _ = kepler_aproximacion(M, e, orden=1)
    """
    from math import sin
    
    #Formula de acuerdo al orden de aproximacion
    if orden==1:
        E=M+e*sin(M)
    elif orden==2:
        E=M+e*sin(M)+0.5*e**2*sin(2*M)
    elif orden==3:
        E=M+(e-1./8*e**3)*sin(M)+0.5*e**2*sin(2*M)+3./8*e**3*sin(3*M)
        
    #Estimación el error relativo
    Ma=E-e*sin(E)
    Dn=abs(Ma-M)/M
    
    return E,Dn,1

def propaga_estado(sistema,t0,t,verbose=0):
    """Propaga el estado de un sistema de dos cuerpos a un tiempo t.

    Parameters
    ----------
    sistema : list[dict]
        Dos particulas con claves `m`, `r`, `v`.
    t0 : float
        Tiempo inicial.
    t : float
        Tiempo final.
    verbose : int, optional
        Si es mayor que 0, imprime detalles del calculo.

    Returns
    -------
    tuple
        `(r1, v1, r2, v2, r, v)` estados en el tiempo `t`.

    Examples
    --------
    >>> r1, v1, r2, v2, rvec, vvec = propaga_estado(sistema, t0, t)
    """
    

    #Condiciones iniciales
    m1=sistema[0]["m"]
    r1_0=sistema[0]["r"]
    v1_0=sistema[0]["v"]

    m2=sistema[1]["m"]
    r2_0=sistema[1]["r"]
    v2_0=sistema[1]["v"]

    if verbose:
        print(f"r1_0 = {r1_0}, v1_0 = {v1_0}")
        print(f"r2_0 = {r2_0}, v2_0 = {v2_0}")

    Mtot=m1+m2

    #En unidades canónicas G=1
    mu=Mtot

    #Paso 1: estado del centro de masa
    r_CM_0=(m1*r1_0+m2*r2_0)/Mtot
    v_CM_0=(m1*v1_0+m2*v2_0)/Mtot
    if verbose:print(f"r_CM_0 = {r_CM_0}, v_CM_0 = {v_CM_0}")
        
    #Paso 2: Condiciones iniciales relativas
    r_0=r1_0-r2_0
    v_0=v1_0-v2_0
    if verbose:print(f"r_0 = {r_0}, v_0 = {v_0}")

    #Paso 3: Constantes de movimiento 
    hvec=cross(r_0,v_0)
    evec=cross(v_0,hvec)/mu-r_0/norm(r_0)
    if verbose:print(f"hvec = {hvec}, evec = {evec}")

    #Paso 4 y 5: Elementos orbitales
    p,e,i,W,w,f0=estado_a_elementos(mu,hstack((r_0,v_0)))

    if verbose:
        print(f"Elementos: {p}, {e}, {i*180/pi}, {W*180/pi}, {w*180/pi}, {f0*180/pi}")
    
    #Paso 6: Anomalía media inicial
    if e==1:
        tanf02=tan(f0/2)
        #Ecuación de Halley
        M0=0.5*(tanf02**3+3*tanf02)
    else:
        sigma=+1 if e<1 else -1
        s=sin if e<1 else sinh
        c=cos if e<1 else cosh
        ta=tan if e<1 else tanh
        at=arctan if e<1 else arctanh
        #Anomalía excéntrica
        G0=2*at(sqrt(sigma*(1-e)/(1+e))*tan(f0/2))

        #Ecuación de Kepler
        M0=sigma*(G0-e*s(G0))
        
    if verbose:print(f"M0 = {M0*180/pi}")


    #Paso 7: Anomalía media en t
    if e==1:
        n=3*sqrt(mu/p**3)
    else:
        a=p/(1-e**2)
        n=sqrt(mu/abs(a)**3)
    M=M0+n*(t-t0)
    if verbose:print(f"n = {n}, M = {M*180/pi}")

    #Paso 8: Anomalía verdadera en t:
    if e==1:
        y=(M+sqrt(M**2+1))**(1./3)
        f=2*arctan(y-1/y)
    else:
        G,error,ni=kepler_newton(M,e,M,1e-14)
        f=2*arctan(sqrt((1+e)/(sigma*(1-e)))*ta(G/2))

    if verbose:print(f"f = {f*180/pi}")
        
    #Paso 9: de elementos a estado 
    x=elementos_a_estado(mu,array([p,e,i,W,w,f]))
    r=x[:3]
    v=x[3:]

    if verbose:
        print(f"r = {r}, v = {v}")
        print(f"h = {cross(r,v)}")

    #Paso 10: estado en el sistema de referencia original
    v_CM=v_CM_0
    r_CM=r_CM_0+v_CM_0*(t-t0)
    if verbose:print(f"r_CM = {r_CM}, v_CM = {v_CM}")

    r1=r_CM+(m2/Mtot)*r
    v1=v_CM+(m2/Mtot)*v
    
    r2=r_CM-(m1/Mtot)*r
    v2=v_CM-(m1/Mtot)*v
    
    #Variables requeridas para comparaciones
    if verbose:
        print(f"f0={f0};f={f};r={norm(r)};r0={norm(r_0)};rdot0={dot(r_0,v_0)/norm(r_0)}")

    return r1,v1,r2,v2,r,v

def funcion_universal_kepler(x,M,e,q):
    """Ecuacion universal de Kepler y sus derivadas.

    Parameters
    ----------
    x : float
        Variable universal.
    M : float
        Anomalia media.
    e : float
        Excentricidad.
    q : float
        Distancia al periapsis.

    Returns
    -------
    tuple
        `(k, kp, kpp)` funcion y derivadas.

    Examples
    --------
    >>> k, kp, kpp = funcion_universal_kepler(x, M, e, q)
    """
    #Parametro alga
    alfa=(1-e)/q
    #Funcion universal de Kepler
    k=q*x+e*x**3*serie_stumpff(alfa*x**2,3)-M
    kp=q+e*x**2*serie_stumpff(alfa*x**2,2)
    kpp=q+e*x*serie_stumpff(alfa*x**2,1)
    return k,kp,kpp

def funcion_universal_kepler_s(s,r0,rdot0,beta,mu,M):
    """Ecuacion universal de Kepler en la variable s.

    Parameters
    ----------
    s : float
        Variable universal.
    r0 : float
        Distancia inicial.
    rdot0 : float
        Derivada radial inicial.
    beta : float
        Parametro auxiliar.
    mu : float
        Parametro gravitacional.
    M : float
        Anomalia media equivalente.

    Returns
    -------
    tuple
        `(k, kp, kpp)` funcion y derivadas.

    Examples
    --------
    >>> k, kp, kpp = funcion_universal_kepler_s(s, r0, rdot0, beta, mu, M)
    """
    #Variable auxiliar
    u=beta*s**2
    #Series de Stumpff requeridas
    c0=serie_stumpff(u,0)
    s1c1=s*serie_stumpff(u,1)
    s2c2=s**2*serie_stumpff(u,2)
    s3c3=s**3*serie_stumpff(u,3)
    #Ecuación universal de Kepler en s y sus derivadas
    k=r0*s1c1+r0*rdot0*s2c2+mu*s3c3-M
    kp=r0*c0+r0*rdot0*s1c1+mu*s2c2
    kpp=(mu-r0*beta)*s1c1+r0*rdot0*c0
    return k,kp,kpp

def propaga_f_g(mu,rvec0,vvec0,t0,t,delta=1e-14,verbose=False):
    """Propaga un estado usando las funciones de Lagrange f y g.

    Parameters
    ----------
    mu : float
        Parametro gravitacional.
    rvec0, vvec0 : array-like
        Estado inicial.
    t0, t : float
        Tiempo inicial y final.
    delta : float, optional
        Tolerancia del solucionador.
    verbose : bool, optional
        Si `True`, imprime detalles.

    Returns
    -------
    tuple
        `(s, f, g, dotf, dotg, rvec, vvec)`.

    Examples
    --------
    >>> s, f, g, dotf, dotg, rvec, vvec = propaga_f_g(mu, rvec0, vvec0, t0, t, verbose=True)
    """

    #Calcular r0, rdot0
    r0=norm(rvec0)
    rdot0=dot(rvec0,vvec0)/r0
    
    #Calcula el valor del parámetro beta
    hvec=cross(rvec0,vvec0)
    h=norm(hvec)
    e=norm(cross(vvec0,hvec)/mu-rvec0/norm(rvec0))
    p=h**2/mu
    q=p/(1+e)
    beta=mu*(1-e)/q

    #Equivalente a la anomalía media
    M=t-t0
    
    #Resuelve la ecuación universal de Kepler en s
    sn=M/r0

    s,error,ni=metodo_laguerre(funcion_universal_kepler_s,
                               x0=sn,args=(r0,rdot0,beta,mu,M),delta=1e-15)
    
    #Variable auxiliar
    u=beta*s**2
    #Series de Stumpff requeridas
    s1c1=s*serie_stumpff(u,1)
    s2c2=s**2*serie_stumpff(u,2)
    s3c3=s**3*serie_stumpff(u,3)
    
    #Calcula las funciones f,g
    f=1-(mu/r0)*s2c2
    g=M-mu*s3c3
    
    #Calcula r
    rvec=rvec0*f+vvec0*g
    r=norm(rvec)
    
    #Calcula las funciones f',g'
    dotf=-(mu/(r*r0))*s1c1
    dotg=1-(mu/r)*s2c2
    
    #Calcula v
    vvec=rvec0*dotf+vvec0*dotg
    
    return s,f,g,dotf,dotg,rvec,vvec

def edm_crtbp(Y,t,alfa):
    """Ecuaciones de movimiento del CRTBP (3D).

    Parameters
    ----------
    Y : numpy.ndarray
        Vector de estado `(r, v)` concatenado.
    t : float
        Tiempo.
    alfa : float
        Parametro de masa reducido.

    Returns
    -------
    numpy.ndarray
        Derivadas concatenadas `dY/dt`.

    Examples
    --------
    >>> dY = edm_crtbp(Y, t, alfa)
    """

    r=Y[:3]
    v=Y[3:]
    
    #Vectores relativos
    r1=r-array([-alfa,0,0])
    r2=r-array([1-alfa,0,0])
    ez=array([0,0,1])
    
    #Aceleraciones
    g1=-(1-alfa)*r1/norm(r1)**3
    g2=-alfa*r2/norm(r2)**3
    acen=-cross(ez,cross(ez,r))
    acor=-2*cross(ez,v)
    a=g1+g2+acen+acor

    dYdt=concatenate((v,a))
    return dYdt

def crtbp_solucion(alfa,ro,vo,ts):
    """Integra el CRTBP y devuelve estados en marcos rotante e inercial.

    Parameters
    ----------
    alfa : float
        Parametro de masa reducido.
    ro, vo : array-like
        Posicion y velocidad inicial.
    ts : numpy.ndarray
        Tiempos de integracion.

    Returns
    -------
    tuple
                `(rs_rot, vs_rot, rs_ine, vs_ine, r1_ine, r2_ine)` donde:

                - `rs_rot`, `vs_rot`: posiciones y velocidades del tercer cuerpo en el
                    marco rotante, con forma `(Nt, 3)`.
                - `rs_ine`, `vs_ine`: posiciones y velocidades del tercer cuerpo en el
                    marco inercial, con forma `(Nt, 3)`.
                - `r1_ine`, `r2_ine`: posiciones inerciales de las dos masas primarias
                    (cuerpos masivos) a lo largo del tiempo, con forma `(Nt, 3)`.

    Examples
    --------
    >>> rs_rot, vs_rot, rs_ine, vs_ine, r1_ine, r2_ine = crtbp_solucion(alfa, ro, vo, ts)
    """
    #Condiciones iniciales
    Yo=concatenate((array(ro),array(vo)))

    #Solución
    Ys=odeint(edm_crtbp,Yo,ts,args=(alfa,))
    rs_rot=Ys[:,:3]
    vs_rot=Ys[:,3:]
    
    #Transformación al sistema inercial de coordenadas
    rs_ine=zeros_like(rs_rot)
    vs_ine=zeros_like(vs_rot)
    r1_ine=zeros_like(rs_rot)
    r2_ine=zeros_like(rs_rot)
    ez=array([0,0,1])
    
    for i in range(len(ts)):
        #Transformar al sistema inercial
        R=rotate(-ts[i],3)
        rs_ine[i]=mxv(R,rs_rot[i])
        vs_ine[i]=mxv(R,vs_rot[i]+vcrss(ez,rs_rot[i]))
        #Posición de las partículas masivas
        r1_ine[i]=array([-alfa*cos(ts[i]),-alfa*sin(ts[i]),0])
        r2_ine[i]=array([(1-alfa)*cos(ts[i]),(1-alfa)*sin(ts[i]),0])
        
    return rs_rot,vs_rot,rs_ine,vs_ine,r1_ine,r2_ine

def constante_jacobi(alfa,r,vel):
    """Calcula la constante de Jacobi para el CRTBP.

    Parameters
    ----------
    alfa : float
        Parametro de masa reducido.
    r : array-like
        Posiciones (N, 3).
    vel : array-like
        Velocidades (N, 3).

    Returns
    -------
    numpy.ndarray
        Valores de la constante de Jacobi.

    Examples
    --------
    >>> CJ = constante_jacobi(alfa, rs, vs)
    """
    r=array(r)
    vel=array(vel)
    
    #Valor de x, y, z
    x=r[:,0]
    y=r[:,1]
    z=r[:,2]
    
    #Rapidez
    v=norm(vel,axis=1)
    
    #Posiciones relativas
    r1=sqrt((x+alfa)**2+y**2+z**2)
    r2=sqrt((x-1+alfa)**2+y**2+z**2)
    
    #Valor de la constante
    CJ=2*(1-alfa)/r1+2*alfa/r2+(x**2+y**2)-v**2
    return CJ

def funcion_puntos_colineales(x,alfa):
    """Funcion auxiliar para puntos colineales del CRTBP.

    Parameters
    ----------
    x : float | numpy.ndarray
        Coordenada x.
    alfa : float
        Parametro de masa reducido.

    Returns
    -------
    float | numpy.ndarray
        Valor de la funcion.

    Examples
    --------
    >>> fs = funcion_puntos_colineales(xs, alfa)
    """
    x1=-alfa
    x2=1-alfa
    f=(1-alfa)*(x-x1)/abs(x-x1)**3+alfa*(x-x2)/abs(x-x2)**3-x
    return f

def orbitas_crtbp(alfa,ro,vo,
                  T=100,Nt=1000,
                  xlim=(-1.5,1.5),ylim=(-1.5,1.5),
                  xL=0,yL=0,
                 ):
    """Grafica orbitas del CRTBP en el plano.

    Parameters
    ----------
    alfa : float
        Parametro de masa reducido.
    ro, vo : array-like
        Condiciones iniciales.
    T : float, optional
        Tiempo total.
    Nt : int, optional
        Numero de pasos.
    xlim, ylim : tuple, optional
        Limites del grafico.
    xL, yL : float, optional
        Punto de Lagrange a marcar.

    Returns
    -------
    matplotlib.figure.Figure
        Figura generada.

    Examples
    --------
    >>> fig = orbitas_crtbp(alfa, ro, vo)
    """
    #Tiempos de integración
    ts=linspace(0,T,Nt)
    #Solución numérica a la ecuación de movimiento
    solucion=crtbp_solucion(alfa,ro,vo,ts)
    #Posiciones y velocidades en el sistema rotante
    rs=solucion[0]
    vs=solucion[1]
    #Gráfico
    fig=plt.figure(figsize=(5,5))
    ax=fig.gca()
    ax.plot(rs[:,0],rs[:,1],'k-')
    ax.plot([-alfa],[0],'ro',ms=10)
    ax.plot([1-alfa],[0],'bo',ms=5)
    #Punto de Lagrange
    ax.plot([xL],[yL],'r+',ms=10)
    #Decoración
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()
    return fig

def orbitas_crtbp3d(alfa,ro,vo,
                  T=100,Nt=1000,
                  xlim=(-1.5,1.5),ylim=(-1.5,1.5),zlim=(-1.5,1.5),
                  xL=0,yL=0,zL=0,
                  elevation=10,azimuth=-80
                 ):
    """Grafica orbitas del CRTBP en 3D.

    Parameters
    ----------
    alfa : float
        Parametro de masa reducido.
    ro, vo : array-like
        Condiciones iniciales.
    T : float, optional
        Tiempo total.
    Nt : int, optional
        Numero de pasos.
    xlim, ylim, zlim : tuple, optional
        Limites del grafico.
    xL, yL, zL : float, optional
        Punto de Lagrange a marcar.
    elevation, azimuth : float, optional
        Angulos de vista.

    Returns
    -------
    matplotlib.figure.Figure
        Figura generada.

    Examples
    --------
    >>> fig = orbitas_crtbp3d(alfa, ro, vo)
    """
    #Tiempos de integración
    ts=linspace(0,T,Nt)
    #Solución numérica a la ecuación de movimiento
    solucion=crtbp_solucion(alfa,ro,vo,ts)
    #Posiciones y velocidades en el sistema rotante
    rs=solucion[0]
    vs=solucion[1]
    #Gráfico
    fig=plt.figure(figsize=(5,5))
    ax=fig.gca(projection='3d')
    ax.plot(rs[:,0],rs[:,1],rs[:,2],'k-')
    ax.plot([-alfa],[0],[0],'ro',ms=10)
    ax.plot([1-alfa],[0],[0],'bo',ms=5)
    ax.plot([xL],[yL],[zL],'r+',ms=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elevation,azimuth)
    fig.tight_layout()
    return fig

def accion_hamilton(lagrangiano,q0,eta,epsilon,t1,t2,**opciones_de_L):
    """Evalua el funcional de accion para una variacion dada.

    Parameters
    ----------
    lagrangiano : callable
        Funcion L(q, dqdt, t, **opciones_de_L).
    q0 : callable
        Trayectoria base.
    eta : callable
        Variacion.
    epsilon : float
        Amplitud de la variacion.
    t1, t2 : float
        Intervalo de integracion.
    **opciones_de_L
        Parametros adicionales del lagrangiano.

    Returns
    -------
    float
        Valor de la accion.

    Examples
    --------
    >>> S = accion_hamilton(lagrangiano_pendulo_simple, q0, eta, epsilon, t1, t2)
    """
    
    #Definimos las función con su variación
    q=lambda t:q0(t,**opciones_de_L)+epsilon*eta(t,**opciones_de_L)
    
    #La derivada de q la calculamos con derivative
    dqdt=lambda t:derivative(q,t,0.01)
        
    #Lagrangiano del péndulo simple
    Lsistema=lambda t:lagrangiano(q(t),dqdt(t),t,**opciones_de_L)

    #El funcional es la integral definida del integrando
    integral=quad(Lsistema,t1,t2)
    S=integral[0]
    
    return S

def Vfuerza(r,**parametros):
    """Potencial de fuerza central general.

    Parameters
    ----------
    r : float | numpy.ndarray
        Radio.
    **parametros
        `mu` y `n` para :math:`V=-mu/r^n`.

    Returns
    -------
    float | numpy.ndarray
        Potencial.

    Examples
    --------
    >>> Vs = Vfuerza(rs, mu=mu, n=n)
    """
    V=-parametros["mu"]/r**parametros["n"]
    return V

def Vcen(r,**parametros):
    """Potencial centrifugo para un momento angular h.

    Parameters
    ----------
    r : float | numpy.ndarray
        Radio.
    **parametros
        `h` momento angular.

    Returns
    -------
    float | numpy.ndarray
        Potencial centrifugo.

    Examples
    --------
    >>> Vcens = Vcen(rs, h=h)
    """
    V=parametros["h"]**2/(2*r**2)
    return V

def Veff(r,Vf,**parametros):
    """Potencial efectivo :math:`V = V_f + V_{cen}`.

    Parameters
    ----------
    r : float | numpy.ndarray
        Radio.
    Vf : callable
        Potencial de fuerza central.
    **parametros
        Parametros para `Vf` y `Vcen`.

    Returns
    -------
    float | numpy.ndarray
        Potencial efectivo.

    Examples
    --------
    >>> Veffs = Veff(rs, Vfuerza, h=h, mu=mu, n=n)
    """
    V=Vf(r,**parametros)+Vcen(r,**parametros)
    return V

def estado_a_elementos(mu,x):
    """Convierte estado cartesiano a elementos orbitales clasicos.

    Parameters
    ----------
    mu : float
        Parametro gravitacional.
    x : array-like
        Vector `(r, v)` concatenado.

    Returns
    -------
    tuple
        `(p, e, i, W, w, f)`.

    Examples
    --------
    >>> p, e, i, W, w, f = estado_a_elementos(mu, x)
    """
    #Posición y velocidad del sistema relativo
    rvec=x[:3]
    vvec=x[3:]
    
    #Momento angular relativo específico
    hvec=cross(rvec,vvec)
    h=norm(hvec)
    #Vector excentricidad
    r=norm(rvec)
    evec=cross(vvec,hvec)/mu-rvec/r
    #Vector nodo ascendente
    nvec=cross([0,0,1],hvec)
    n=norm(nvec)
    
    #Semilatus rectum y excentricidad
    p=h**2/mu
    e=norm(evec)

    #Orientación
    i=arccos(hvec[2]/h)

    Wp=arccos(nvec[0]/n)
    W=Wp if nvec[1]>=0 else 2*pi-Wp

    wp=arccos(dot(nvec,evec)/(e*n))
    w=wp if evec[2]>=0 else 2*pi-wp

    fp=arccos(dot(rvec,evec)/(r*e))
    f=fp if dot(rvec,vvec)>0 else 2*pi-fp
    
    return p,e,i,W,w,f

def elementos_a_estado(mu,elementos):
    """Convierte elementos orbitales clasicos a estado cartesiano.

    Parameters
    ----------
    mu : float
        Parametro gravitacional.
    elementos : array-like
        `(p, e, i, W, w, f)`.

    Returns
    -------
    numpy.ndarray
        Vector `(x, y, z, vx, vy, vz)`.

    Examples
    --------
    >>> x = elementos_a_estado(mu, array([p, e, i, W, w, f]))
    """
    #Extrae elementos
    p,e,i,W,w,f=elementos
    
    #Calcula momento angular relativo específico
    h=sqrt(mu*p)
    
    #Calcula r
    r=p/(1+e*cos(f))
    
    #Posición
    x=r*(cos(W)*cos(w+f)-cos(i)*sin(W)*sin(w+f))
    y=r*(sin(W)*cos(w+f)+cos(i)*cos(W)*sin(w+f))
    z=r*sin(i)*sin(w+f)
    
    #Velocidad
    muh=mu/h

    vx=muh*(-cos(W)*sin(w+f)-cos(i)*sin(W)*cos(w+f))       -muh*e*(cos(W)*sin(w)+cos(w)*cos(i)*sin(W))
    vy=muh*(-sin(W)*sin(w+f)+cos(i)*cos(W)*cos(w+f))       +muh*e*(-sin(W)*sin(w)+cos(w)*cos(i)*cos(W))
    vz=muh*(sin(i)*cos(w+f)+e*cos(w)*sin(i))

    return array([x,y,z,vx,vy,vz])

def metodo_newton(f,x0=1,delta=1e-5,args=()):
    """Metodo de Newton para resolver f(x)=0.

    Parameters
    ----------
    f : callable
        Funcion que devuelve `(f, f')`.
    x0 : float, optional
        Valor inicial.
    delta : float, optional
        Tolerancia.
    args : tuple, optional
        Argumentos adicionales para `f`.

    Returns
    -------
    tuple
        `(x, error, ni)`.

    Examples
    --------
    >>> x, error, ni = metodo_newton(funcion_kepler, x0=E0, delta=1e-8, args=(M, e))
    """
    #Valor inicial de la anomalía excéntrica
    xn=x0
    #Valor inicial del error relativo
    Dn=1
    #Contador de iteraciones
    ni=0
    while Dn>delta:
        #Inicializa el valor de En
        x=xn
        #Nuevo valor (regla de iteración)
        xn=x-f(x,*args)[0]/f(x,*args)[1]
        #Valor medio
        xmed=(x+xn)/2
        #Criterio de convergencia
        en=xn-x
        Dn=abs(en/xmed)
        ni+=1
    return xmed,Dn,ni

def metodo_laguerre(f,x0=1,delta=1e-5,args=(),eta=5):
    """Metodo de Laguerre para resolver f(x)=0.

    Parameters
    ----------
    f : callable
        Funcion que devuelve `(f, f', f'')`.
    x0 : float, optional
        Valor inicial.
    delta : float, optional
        Tolerancia.
    args : tuple, optional
        Argumentos adicionales para `f`.
    eta : int, optional
        Parametro del metodo.

    Returns
    -------
    tuple
        `(x, error, ni)`.

    Examples
    --------
    >>> E, error, ni = metodo_laguerre(funcion_kepler, x0=E0, delta=1e-8, args=(M, e))
    """
    #Varifica que el valor inicial sea apropiado
    disc=-1
    mi=0
    #Valor inicial de la anomalía excéntrica
    xn=x0
    #Valor inicial del error relativo
    Dn=1
    #Contador de iteraciones
    ni=0
    while Dn>delta:
        #Inicializa el valor de En
        x=xn
        disc=-1
        mi=0
        while disc<0:
            mi+=1
            #Valor de la función y sus derivadas
            y,yp,ypp=f(x,*args)
            #Discriminante
            disc=(eta-1)**2*yp**2-eta*(eta-1)*y*ypp
            eta=eta-1 if disc<0 else eta
        #Raiz del discriminante
        raiz_disc=sqrt(disc)
        #Signo en el denominador
        sgn=+1 if abs(yp+raiz_disc)>abs(yp-raiz_disc) else -1
        #Valor de en
        en=eta*y/(yp+sgn*raiz_disc)
        #Nuevo valor (regla de iteración)
        xn=x-en
        #Valor medio
        xmed=(x+xn)/2
        #Criterio de convergencia
        en=xn-x
        Dn=abs(en/xmed)
        ni+=1
    return xmed,Dn,ni+mi-1

def kepler_semianalitico(M,e):
    """Solucion semianalitica de la ecuacion de Kepler.

    Parameters
    ----------
    M : float
        Anomalia media.
    e : float
        Excentricidad.

    Returns
    -------
    tuple
        `(E, error, ni)`.

    Examples
    --------
    >>> E, error, ni = kepler_semianalitico(M, e)
    """
    from math import sin,cos,pi
    
    #Casos extremos
    if M==0 or M==2*pi or e==1:return M,0,0
    Minp=M
    
    Ecorr=0;Esgn=1.0
    if M>pi:
        M=2*pi-M
        Ecorr=2*pi
        Esgn=-1.0
    
    #Circunferencia
    if e==0:return Ecorr+Esgn*M,0,0
        
    a=(1-e)*3/(4*e+0.5);
    b=-M/(4*e+0.5);
    y=(b**2/4 +a**3/27)**0.5;
    x=(-0.5*b+y)**(1./3)-(0.5*b+y)**(1./3);
    w=x-0.078*x**5/(1 + e);
    E=M+e*(3*w-4*w**3);

    #Corrección por Newton
    sE=sin(E)
    cE=cos(E)

    f=(E-e*sE-M);
    fd=1-e*cE;
    f2d=e*sE;
    f3d=-e*cE;
    f4d=e*sE;
    E=E-f/fd*(1+f*f2d/(2*fd*fd)+              f*f*(3*f2d*f2d-fd*f3d)/(6*fd**4)+              (10*fd*f2d*f3d-15*f2d**3-fd**2*f4d)*              f**3/(24*fd**6))

    #Corrección por Newton
    f=(E-e*sE-M);
    fd=1-e*cE;
    f2d=e*sE;
    f3d=-e*cE;
    f4d=e*sE;
    E=E-f/fd*(1+f*f2d/(2*fd*fd)+              f*f*(3*f2d*f2d-fd*f3d)/(6*fd**4)+              (10*fd*f2d*f3d-15*f2d**3-fd**2*f4d)*              f**3/(24*fd**6))
    
    E=Ecorr+Esgn*E
    
    #Error relativo
    Mnum=E-e*sin(E)
    Dn=abs(Mnum-Minp)/Minp
    
    return E,Dn,1

def kepler_eserie(M,e,delta=0,orden=1):
    """Solucion de Kepler por serie de potencias en e.

    Parameters
    ----------
    M : float
        Anomalia media.
    e : float
        Excentricidad.
    delta : float, optional
        Tolerancia.
    orden : int, optional
        Orden de la serie si `delta=0`.

    Returns
    -------
    tuple
        `(E, error, n)`.

    Examples
    --------
    >>> E8, error8, ni = kepler_eserie(M, e, orden=8)
    """
    nfac=1
    En=M
    Dn=1
    n=0
    condicion=Dn>delta if delta>0 else n<=orden
    while condicion:
        n+=1
        E=En
        prefactor=e**n/2**(n-1)
        kmax=int(math.floor(n/2))
        sgn=-1
        #Los factoriales se calculan así para mayor eficiencia
        nfac=nfac*n if n>0 else 1
        kfac=1
        nkfac=1
        termino=0
        for k in range(kmax+1):
            sgn*=-1
            kfac=kfac*k if k>0 else 1
            nkfac=nkfac/(n-k+1) if k>0 else nfac
            ank=sgn/(kfac*nkfac)*(n-2*k)**(n-1)
            termino+=ank*math.sin((n-2*k)*M)
        dE=prefactor*termino
        En+=dE
        Dn=abs(dE/En)
        #La condicion depende de si se pasa o no la tolerancia
        condicion=Dn>delta if delta>0 else n<orden
    return En,Dn,n

def kepler_bessel(M,e,delta):
    """Solucion de Kepler usando expansion en funciones de Bessel.

    Parameters
    ----------
    M : float
        Anomalia media.
    e : float
        Excentricidad.
    delta : float
        Tolerancia.

    Returns
    -------
    tuple
        `(E, error, n)`.

    Examples
    --------
    >>> E, error, ni = kepler_bessel(M, e, 1e-8)
    """
    Dn=1
    n=1
    En=M
    while Dn>delta:
        E=En
        dE=(2./n)*jv(n,n*e)*math.sin(n*M)
        En+=dE
        Emed=(E+En)/2
        Dn=abs(dE/Emed)
        n+=1
    return En,Dn,n

def serie_stumpff(t,k,N=15):
    """Calcula la serie de Stumpff :math:`c_k(t)`.

    Parameters
    ----------
    t : float
        Argumento de la serie.
    k : int
        Orden.
    N : int, optional
        Numero de terminos.

    Returns
    -------
    float
        Valor de la serie.

    Examples
    --------
    >>> c0 = serie_stumpff(t, 0)
    """
    sk=lambda n:t/((2*n+k+1)*(2*n+k+2))*(1-sk(n+1)) if n<N else 0
    return (1-sk(0))/math.factorial(k)

def plot_elipse(e=0.5,a=10.0):
    """Grafica una elipse en el plano.

    Parameters
    ----------
    e : float, optional
        Excentricidad, debe ser menor que 1.
    a : float, optional
        Semieje mayor, debe ser positivo.

    Returns
    -------
    None
        Muestra la figura con `matplotlib`.

    Raises
    ------
    ValueError
        Si `e >= 1` o `a <= 0`.

    Examples
    --------
    >>> plot_elipse(e=0.3, a=5)
    """

    e=float(e)
    a=float(a)
    if e>1:
        raise ValueError("La excentricidad de una elipse debe ser menor que 1")
    if a<0:
        raise ValueError("El semieje mayor de una elipse debe ser positivo")

    b=a*sqrt(1-e**2)

    #Distancia foco-centro
    c=a*e

    #Máximo valor de x
    xcmax=a

    #Valores de x en los que graficaremos
    xcs=linspace(-a,a,100)

    #Ecuaciones de las cónicas referidas al apside
    ycs_cir=a*sqrt(1-xcs**2/a**2)
    ycs=b*sqrt(1-xcs**2/a**2)

    #Gráfica
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(6,6))
    ax=fig.gca()

    ax.plot(xcs,ycs_cir,'k--')
    ax.plot(xcs,-ycs_cir,'k--')
    ax.plot(xcs,ycs,'r')
    ax.plot(xcs,-ycs,'r')

    #Graficar la posición del foco y el centro
    ax.plot([-c],[0],'bx',markersize=10)
    ax.plot([0],[0],'ko',markersize=5)

    #Decoración
    ax.grid()
    ax.set_title(f"Elipse con $a = {a}$, $e={e}$")

    #Fijamos la misma escala en los ejes
    plt.axis("equal")
    plt.show()

#Definimos el algoritmo como una rutina
def plot_hiperbola(e=1.5,a=-10):
    """Grafica una hipérbola en el plano.

    Parameters
    ----------
    e : float, optional
        Excentricidad, debe ser mayor que 1.
    a : float, optional
        Semieje mayor, debe ser negativo.

    Returns
    -------
    None
        Muestra la figura con `matplotlib`.

    Raises
    ------
    ValueError
        Si `e <= 1` o `a >= 0`.

    Examples
    --------
    >>> plot_hiperbola(e=1.7, a=-8)
    """

    e=float(e)
    a=float(a)

    if e<1:
        raise ValueError("La excentricidad de una hipérbola debe ser mayor que 1")
    if a>0:
        raise ValueError("El semieje mayor de una hipérbola debe ser negativo")

    #Semieje menor
    beta=abs(a)*sqrt(e**2-1)

    #Semilatus rectum
    p=a*(1-e**2)

    #Posición del foco
    q=p/(1+e)
    F=abs(a)+q

    #Máximo valor de x
    xcmax=3*abs(a)

    #Valores de x en los que graficaremos
    xcs=linspace(abs(a),xcmax,100)

    #Ecuaciones de las cónicas referidas al apside
    ycs=beta*sqrt(xcs**2/a**2-1)

    #Ecuación de las asintotas
    xas=linspace(0,xcmax,100)
    ycs_asi=beta*xas/abs(a)

    #Gráfica
    fig=plt.figure(figsize=(6,6))
    ax=fig.gca()

    ax.plot(xas,ycs_asi,'k--')
    ax.plot(xas,-ycs_asi,'k--')
    ax.plot(xcs,ycs,'r')
    ax.plot(xcs,-ycs,'r')

    #Graficar la posición del foco y el vértice
    ax.plot([F],[0],'bx',markersize=10)
    ax.plot([0],[0],'ko',markersize=5)

    #Decoración
    ax.grid()
    ax.set_title(f"Hipérbola con $a = {a}$, $e={e}$")

    #Fijamos la misma escala en los ejes
    plt.axis("equal")
    plt.show()

def intersecta_circunferencias(x0, y0, r0, x1, y1, r1):
    """Calcula la intersección entre dos circunferencias:

    Parametros:
        Circunferencia 1: x0, y0, r0
        Circunferencia 2: x1, y1, r1

    Retorna:
        Puntos de intersección: (x3, y3), (x4, y4)

    Notas:
        Adaptado de: https://stackoverflow.com/a/55817881
    """

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return (x3, y3, x4, y4)

def dibuja_esfera(ax, centro=(0,0,0), radio=1, **kwargs):
    """Dibuja una esfera en un axis en 3d

    Examples
    --------
    >>> # Esfera en 3d
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> dibuja_esfera(ax, centro=(1,1,0), radio=0.2, color='b', alpha=0.5)
    >>> ax.axis('equal')
    >>>
    >>> # Esfera en 2d
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> dibuja_esfera(ax, centro=(1,1), radio=0.2, color='b', alpha=0.5)
    >>> ax.axis('equal')

    Notas:
        Adaptado de: https://stackoverflow.com/q/31768031
    """
    if ax.name != '3d':
        s = patches.Circle(centro[:2], radius=radio, fill=True, **kwargs)
        ax.add_patch(s)
    else:
        phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
        x = centro[0] + radio*np.sin(phi)*np.cos(theta)
        y = centro[1] + radio*np.sin(phi)*np.sin(theta)
        z = centro[2] + radio*np.cos(phi)
        s = ax.plot_surface(x, y, z, **kwargs)

    return s

def intersecta_circunferencias3d(C1, r1, C2, r2, tol=1e-9):
    """Encuentra los puntos de intersección de dos circunferencias en el espacio 3D.

    Se asume que las circunferencias yacen en el plano definido por el origen (0,0,0)
    y los centros de las circunferencias, C1 y C2.

    Args:
        C1 (np.ndarray): Centro de la primera circunferencia (vector 3D numpy).
        r1 (float): Radio de la primera circunferencia. Debe ser positivo.
        C2 (np.ndarray): Centro de la segunda circunferencia (vector 3D numpy).
        r2 (float): Radio de la segunda circunferencia. Debe ser positivo.
        tol (float): Tolerancia para comparaciones de punto flotante.

    Returns:
        list[np.ndarray]: Una lista que contiene los puntos de intersección
                          (como arrays numpy 3D). Devuelve una lista vacía si no
                          hay intersecciones, si las circunferencias son idénticas,
                          o si el plano está mal definido (O, C1, C2 colineales).
                          Devuelve una lista con un punto si las circunferencias
                          son tangentes.

        Examples
        --------
        >>> C1 = np.array([0.2, 0.3, 0.0])
        >>> C2 = np.array([0.9, 0.2, 0.1])
        >>> I1, I2 = intersecta_circunferencias3d(C1, 1.0, C2, 1.0)
        >>> I1, I2
        (array([ 0.40954474, -0.67138648,  0.11180031]),
         array([ 0.69045526,  1.17138648, -0.01180031]))
        >>> np.linalg.norm(C1 - I1), np.linalg.norm(C1 - I2)
        np.float64(0.9999999999999999), np.float64(0.9999999999999998)
        >>> np.linalg.norm(C2 - I1), np.linalg.norm(C2 - I2)
        np.linalg.norm(C2 - I1), np.linalg.norm(C2 - I2)
        >>> np.cross(np.cross(C1, C2), np.cross(C1, (I1 - C1)))
        array([-8.67361738e-19, -8.67361738e-19,  0.00000000e+00])

    Elaborado por:
      Gemini 2.5 Pro, prompt por Jorge I. Zuluaga
      Pruebas y Código adaptado por Jorge I. Zuluaga

    """
    origin = np.array([0, 0, 0])
    producto_cruz = np.cross(C1, C2)
    if np.linalg.norm(producto_cruz) < tol:
        print(f"El origen (0,0,0), C1 ({C1}), y C2 ({C2}) son colineales. "
                "No se puede definir un plano único según el método especificado.")
        return (origin, origin) # No hay intersección

    # Cálculo de Distancia entre Centros
    d_vec = C2 - C1 
    d = np.linalg.norm(d_vec) 

    # Comprobación de Posibilidad de Intersección
    suma_radios = r1 + r2
    dif_radios = abs(r1 - r2)
    if d > suma_radios + tol:
      print(f"No hay intersección porque los puntos están (d = {d}) más lejos que r1+r2 ({suma_radios})")
      return (origin, origin) # No hay intersección

    if d < dif_radios - tol:
      print(f"No hay intersección porque los puntos están (d = {d}) más cerca que r1 - r2 ({dif_radios}) ")
      return (origin, origin) # No hay intersección


    # Cálculo de Parámetros 'a' y 'h' (Basado en Solución 2D)
    """
    'a' es la distancia desde C1 al punto medio del segmento de intersección,
    medido a lo largo de la línea que une C1 y C2.
    Se deriva de la ley de cosenos o restando las ecuaciones de las circunferencias.
    Ecuación: r2^2 = h^2 + (d-a)^2 ; r1^2 = h^2 + a^2
    Restando: r1^2 - r2^2 = a^2 - (d-a)^2 = a^2 - (d^2 - 2ad + a^2) = 2ad - d^2
    => 2ad = r1^2 - r2^2 + d^2 => a = (r1^2 - r2^2 + d^2) / (2d)
    """
    a = (r1**2 - r2**2 + d**2) / (2 * d)

    """
    'h^2' es el cuadrado de la mitad de la longitud del segmento de intersección
    (perpendicular a la línea C1-C2). Se deriva de r1^2 = a^2 + h^2.
    """
    h_cuadrado = r1**2 - a**2

    # Manejo de imprecisiones numéricas cerca de la tangencia (h^2 debería ser >= 0)
    if h_cuadrado < 0 and abs(h_cuadrado) < tol:
         h_cuadrado = 0 # Forzar tangencia si es negativo pero muy cercano a cero
    elif h_cuadrado < 0:
         # Si es significativamente negativo, algo falló (no debería ocurrir si d está en el rango)
         print(f"Advertencia: h^2 es negativo ({h_cuadrado}) a pesar de pasar el chequeo de distancia. No hay intersección real.")
         return (origin, origin) # No hay intersección real

    # h es la semi-longitud de la cuerda común
    h = np.sqrt(h_cuadrado) 

    # Definición de la Base Ortogonal 3D en el Plano O-C1-C2
    """
    Necesitamos una base ortonormal {ex, ey} dentro del plano O-C1-C2
    donde ex va de C1 a C2 y ey es perpendicular a ex.

    ex: vector unitario en la dirección de C1 a C2
    
    ey: vector unitario en el plano O-C1-C2, perpendicular a ex.
    La normal al plano O-C1-C2 es n = C1 x C2 (calculado como producto_cruz).
    ey debe ser ortogonal a n y a ex. Usamos el producto cruz: ey = normalize(n x ex).
    El producto cruz n x ex da un vector perpendicular a n y a ex,
    por lo tanto, yace en el plano (por ser perp. a n) y es perp. a ex.
    """
    ex = d_vec / d
    ey_no_normalizado = np.cross(producto_cruz, ex)
    norma_ey = np.linalg.norm(ey_no_normalizado)

    # Comprobación de seguridad (no debería ser cero si O, C1, C2 no son colineales)
    if norma_ey < tol:
        print(f"Fallo al calcular el vector ortogonal ey. Verifique los vectores de entrada.")
        return (origin, origin) # No hay intersección

    ey = ey_no_normalizado / norma_ey # Normalizamos para obtener el vector unitario

    """
    Cálculo de los Puntos de Intersección en 3D
    P_medio: Es el punto sobre el segmento C1-C2 a distancia 'a' de C1.
    Es la proyección de los puntos de intersección sobre la línea C1-C2.
    """
    P_medio = C1 + a * ex

    """
    Los puntos de intersección (P1, P2) se obtienen moviéndose +/- h
    en la dirección perpendicular ey desde P_medio.
    """
    P1 = P_medio + h * ey
    P2 = P_medio - h * ey

    # Devolver Resultados
    if np.isclose(h, 0, atol=tol):
        # Caso tangente: h es (casi) cero, P1 y P2 coinciden en P_medio.
        print(f"Las circunferencias son tangentes")
        return [P_medio, P_medio]
    else:
        # Caso secante: dos puntos de intersección distintos.
        return [P1, P2]

def plotly_esfera(pfig,R,sphereargs=dict()):
    """Gráfica una esfera en plotly

    Examples
    --------
    >>> R = 3
    >>> fig = go.Figure()
    >>> pc.plotly_esfera(fig, R, sphereargs=dict(colorscale='Blues'))
    >>> fig.show()
    """

    # Opciones por defecto
    sphereargs_default=dict(
        colorscale='Blues',
        opacity=0.5,
        showscale=False
    )
    sphereargs_default.update(sphereargs)

    # Coordenadas
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = R * np.outer(np.cos(u), np.sin(v))
    y_sphere = R * np.outer(np.sin(u), np.sin(v))
    z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))

    # Agregar la superficie de la Tierra
    pfig.add_trace(go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        **sphereargs_default
    ))

    pfig.update_layout(
        scene=dict(
            xaxis=dict(nticks=5, range=[-2*R,2*R]),
            yaxis=dict(nticks=5, range=[-2*R,2*R]),
            zaxis=dict(nticks=5, range=[-2*R,2*R]),
            aspectmode='cube',

        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

def plotly_campo_vectorial(pfig,rs,vs,
                           lineargs=dict(),
                           scatterargs=dict(),
                           coneargs=dict()):
    """Gráfica un campo vectorial en plotly

    Examples
    --------
    >>> R = 3
    >>> deg = np.pi / 180
    >>> phis = np.linspace(0, 2 * np.pi, 10)
    >>> thetas = 60 * deg * np.ones_like(phis)
    >>> xs = np.array([
    ...     R * np.sin(thetas) * np.cos(phis),
    ...     R * np.sin(thetas) * np.sin(phis),
    ...     R * np.cos(thetas),
    ... ]).T
    >>> us = np.array([
    ...     np.sin(thetas) * np.cos(phis),
    ...     np.sin(thetas) * np.sin(phis),
    ...     np.cos(thetas),
    ... ]).T
    >>> fig = go.Figure()
    >>> pc.plotly_esfera(fig, R, sphereargs=dict(colorscale='Blues'))
    >>> pc.plotly_campo_vectorial(fig, xs, us)
    >>> fig.add_trace(
    ...     go.Scatter3d(
    ...         x=xs[:, 0], y=xs[:, 1], z=xs[:, 2],
    ...         mode='markers', name='Puntos'
    ...     )
    ... )
    >>> fig.show()
    """

    # Opciones por defecto
    lineargs_default=dict(color='blue',width=5)
    lineargs_default.update(lineargs)

    scatterargs_default=dict(showlegend=False)
    scatterargs_default.update(scatterargs)

    coneargs_default=dict(showscale=False,sizemode='absolute')
    coneargs_default.update(coneargs)

    for i in range(len(rs)):
        pfig.add_trace(go.Scatter3d(
            x=[rs[i,0], rs[i,0] + vs[i,0]],
            y=[rs[i,1], rs[i,1] + vs[i,1]],
            z=[rs[i,2], rs[i,2] + vs[i,2]],
            mode='lines',
            line=lineargs_default,
            **scatterargs_default,
        ))
        pfig.add_trace(go.Cone(
            x=[rs[i,0] + vs[i,0]],
            y=[rs[i,1] + vs[i,1]],
            z=[rs[i,2] + vs[i,2]],
            u=[vs[i,0]],
            v=[vs[i,1]],
            w=[vs[i,2]],
            colorscale=[
                [0,lineargs_default['color']],
                [1,lineargs_default['color']]
            ],
            **coneargs_default
        ))

from urllib import request
import json
def obtiene_elementos_asteroide(id, verbose=True):
    """Obtiene los elementos orbitales de un asteroide, y las covarianzas
    de sus elementos

    Examples
    --------
    >>> promedios, covarianza = obtiene_elementos_asteroide('2024yr4', verbose=True)

    Notas:
      - Basado en el código de Leonard Gómez, Astronomía UdeA (2022)
    """
    url = f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={id}&cov=mat&full-prec=true"
    if verbose:
      print(f"Descargando los datos para {id.upper()} de {url}...")
    html=request.urlopen(url)
    json_data=json.loads(html.read().decode())

    t0=float(json_data["orbit"]["epoch"])
    if verbose:
      print(f"Epoca de los elementos (JDTDB): {t0}")

    if verbose:
      print(f"Extrayendo la matriz de covarianza...")
    Cov=np.array(json_data["orbit"]["covariance"]["data"],dtype=float)
    Cov_label=json_data["orbit"]["covariance"]["labels"]
    t=float(json_data["orbit"]["epoch"])

    if verbose:
      print(f"Extrayendo los elementos y sus errores...")
    nlen=len(json_data["orbit"]["elements"])
    elnames=[]
    elements=dict()
    for i in range(nlen):
      element=json_data["orbit"]["elements"][i]
      elements[element["name"]]=dict()
      for prop in element.keys():
        try:
          elements[element["name"]][prop]=float(element[prop])
        except:
          pass

    for elname in elements.keys():
      element=elements[elname]
      print(f"Elemento {elname} = {element['value']:.7e} +/- {element['sigma']:.7e}")

    means=[elements['e']['value'],elements['q']['value'],elements['tp']['value'],elements['om']['value'],elements['w']['value'],elements['i']['value']]
    if verbose:
      print(f"Orden de elementos orbitales: {Cov_label}")

    return t0,means,Cov

def axis_equal(fig):
    """Ajusta los ejes 3D de una figura de plotly para que tengan la misma escala.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figura con trazas `Scatter3d`.

    Returns
    -------
    None
        Modifica `fig` in-place.

    Examples
    --------
    >>> fig = plot_ncuerpos_3d(rps, vps)
    >>> axis_equal(fig)
    """
    # Calculate the range for all axes using fig.data, excluding the sphere data
    x_data = [trace.x for trace in fig.data if isinstance(trace, go.Scatter3d)]
    y_data = [trace.y for trace in fig.data if isinstance(trace, go.Scatter3d)]
    z_data = [trace.z for trace in fig.data if isinstance(trace, go.Scatter3d)]
    if len(x_data)==0:
        return

    x_data = np.concatenate(x_data)
    y_data = np.concatenate(y_data)
    z_data = np.concatenate(z_data)

    max_range = np.array([x_data.max()-x_data.min(), y_data.max()-y_data.min(), z_data.max()-z_data.min()]).max() / 2.0
    mid_x = (x_data.max()+x_data.min()) * 0.5
    mid_y = (y_data.max()+y_data.min()) * 0.5
    mid_z = (z_data.max()+z_data.min()) * 0.5

    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range]))
    )

def fig_body(R, mesh3d_opts=dict(), lighting_opts=dict()):
    """Crea una figura de plotly con una esfera de radio R
    """
        
    # Opciones por defecto
    mesh3d_opts_def = dict(color='lightblue', opacity=0.5, alphahull=0)
    mesh3d_opts_def.update(mesh3d_opts)
    lighting_opts_def = dict(ambient=0.5, specular=1.0)
    lighting_opts_def.update(lighting_opts)

    # Esfera
    d=np.pi/32
    theta, phi = np.mgrid[0:np.pi+d:d, 0:2*np.pi:d]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = R*np.vstack([x.ravel(), y.ravel(), z.ravel()])
    x, y, z = points
    fig = go.Figure(data=[
        go.Mesh3d(x=x, y=y, z=z, **mesh3d_opts_def)
    ])
    fig.update_traces(lighting=lighting_opts_def)
    return fig

def C(z):
    """Funcion de Stumpff :math:`C(z)`.

    Parameters
    ----------
    z : float
        Argumento de la funcion.

    Returns
    -------
    float
        Valor de :math:`C(z)`.

    Examples
    --------
    >>> C(0.1)
    """

    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        return 1 / 2

def S(z):
    """Funcion de Stumpff :math:`S(z)`.

    Parameters
    ----------
    z : float
        Argumento de la funcion.

    Returns
    -------
    float
        Valor de :math:`S(z)`.

    Examples
    --------
    >>> S(0.1)
    """

    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
    elif z < 0:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3
    else:
        return 1 / 6

def solucion_lambert(P1, P2, tf, mu=1, direccion='pro', tol=1e-6, maxiter=10000):
    """Resuelve el problema de Lambert para unir dos puntos en un tiempo dado.

    Adaptado de: https://github.com/iscoooooo/Porkchop-Plot-Generator.
    Las formulas estan basadas en "Orbital Mechanics for Engineering Students"
    (Howard D. Curtis, 4th ed.).

    Parameters:
    P1, P2 : np.array
        Posiciones iniciales
    tf : float
        Tiempo de vuelo
    mu = 1 : float
        Parámetro gravitacional.
    tol = 1e-6 : int, optional
        Tolerancia para el solucionador de Newton
    maxiter = 10000: int, optional
        Máximo número de iteraciones para el solucionador de Newton
    direccion = 'pro' : str, optional
        'pro' para la órbita prograda y 'retro' para la órbita retrograda

    Returns:
    V1, V2 : np.array
        Velocidades inicial y final

    orbita: dict
        z: Variable universal (z<0 para órbita hiperbólica)
        elts: Elementos orbitales de SPICE (q, e, I, Omega, omega, M, t, mu)
    Examples
    --------
    >>> V1, V2, info = solucion_lambert(P1, P2, tf, mu=mu, direccion='pro')
    """

    from scipy.optimize import newton

    # Distancias
    r1 = np.linalg.norm(P1)
    r2 = np.linalg.norm(P2)

    # Ángulo del triángulo espacial
    theta = np.arccos(np.dot(P1, P2) / (r1 * r2))
    cross12 = np.cross(P1, P2)
    if direccion == 'pro':
        if cross12[2] < 0:
            theta = 2 * np.pi - theta
    elif direccion == 'retro':
        if cross12[2] >= 0:
            theta = 2 * np.pi - theta
    else:
        raise ValueError("Debe indicar la dirección de movimiento ('pro' de P1 a P2 o 'retro' de P1 a P2 por el lado opuesto).")

    # Función auxiliar
    A = np.sin(theta) * np.sqrt(r1 * r2 / (1 - np.cos(theta)))

    # Funciones auxiliares
    def y(z):
        return r1 + r2 + A * (z * S(z) - 1) / np.sqrt(C(z))

    def F(z):
        return (y(z) / C(z)) ** 1.5 * S(z) + A * np.sqrt(y(z)) - np.sqrt(mu) * tf

    def dFdz(z):
        if z == 0:
            return np.sqrt(2) / 40 * y(0) ** 1.5 + A / 8 * (np.sqrt(y(0)) + A * np.sqrt(1 / 2 / y(0)))
        return (y(z) / C(z)) ** 1.5 * (1 / 2 / z * (C(z) - 3 * S(z) / 2 / C(z)) + 3 * S(z) ** 2 / 4 / C(z)) + A / 8 * (3 * S(z) / C(z) * np.sqrt(y(z)) + A * np.sqrt(C(z) / y(z)))

    # Busca el valor inicial de z para resolver por Newton
    z = 0.1
    while F(z) < 0:
        z += 0.1
        if z > 1e6:
            raise ValueError("No pude encontrar un valor inicial apropiado para z.")

    # Resuelve la ecuación de variable universal
    z = newton(F, z, tol=tol, maxiter=maxiter)

    # Determina si encontro una solución
    solved = not (np.isnan(z) or np.isinf(z))

    # Encuentra las velocidades usando las funciones de Lagrange f, g
    if solved:
        f = 1 - y(z) / r1
        fdot = (np.sqrt(mu) / (r1 * r2)) * np.sqrt(y(z) / C(z)) * (z * S(z) - 1)
        g = A * np.sqrt(y(z) / mu)
        gdot = 1 - y(z) / r2

        # Compute the velocities V1 & V2
        V1 = 1 / g * (P2 - f * P1)
        V2 = 1 / g * (gdot * P2 - P1)

        elts = spy.oscelt(list(P1) + list(V1), 0, mu)
        orbit_info = dict(z=z, elts=elts)
        return V1, V2, orbit_info

    print('El método de Lambert no convergió')
    return np.array([0, 0, 0]), np.array([0, 0, 0]), dict(z=0, elts=[0] * 8)


# Compatibilidad con versiones anteriores a la 0.6.31
def _alias_module(module_name, symbols):
    """Registra un submodulo de compatibilidad en `sys.modules`.

    Parameters
    ----------
    module_name : str
        Nombre completo del submodulo (por ejemplo, 'pymcel.plot').
    symbols : dict
        Simbolos que se exponen en el submodulo.

    Examples
    --------
    >>> _alias_module('pymcel.plot', {'plot_ncuerpos_3d': plot_ncuerpos_3d})
    """
    mod = types.ModuleType(module_name)
    mod.__dict__.update(symbols)
    sys.modules[module_name] = mod
    setattr(sys.modules[__name__], module_name.rsplit('.', 1)[-1], mod)

_public_symbols = {k: v for k, v in globals().items() if not k.startswith('_')}
_plot_symbols = {k: v for k, v in _public_symbols.items() if k.startswith('plot_') or k in (
    'fija_ejes3d_proporcionales',
    'fija_ejes_proporcionales',
    'encuentra_rangos',
    'axis_equal',
    'fig_body',
    'plotly_esfera',
    'plotly_campo_vectorial',
)}
_extra_symbols = {k: v for k, v in _public_symbols.items() if k in ('C', 'S', 'solucion_lambert')}

_alias_module('pymcel.plot', _plot_symbols)
_alias_module('pymcel.extra', _extra_symbols)
_alias_module('pymcel.export', _public_symbols)


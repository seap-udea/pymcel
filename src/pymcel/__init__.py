#############################################################
# PAQUETES REQUERIDOS
#############################################################
from pymcel.version import *
import numpy as np
import os
import requests
import glob
import sys
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy import sin, cos
from spiceypy import mxv
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
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.special import jv
import math
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import pandas as pd

from pymcel import constantes

print("Paquete pymcel cargado. Versión:",version)

#############################################################
# UTILIDADES DEL SISTEMA
#############################################################
#Root directory
try:
    FILE=__file__
    ROOTDIR=os.path.abspath(os.path.dirname(FILE))
except:
    FILE=""
    ROOTDIR=os.path.abspath('')

#############################################################
# KERNELS DE SPICE
#############################################################
def ubica_archivos(path,basedir=None):
    """
        Get the full path of the `datafile` which is one of the datafiles provided with the package.
        
        Parameters:
            datafile: Name of the data file, string.
            
        Return:
            Full path to package datafile in the python environment.
            
    """
    if basedir is None:
        basedir = ROOTDIR
    return os.path.join(basedir,'data',path);

def descarga_kernel(url,filename=None,overwrite=False,basedir=None):
    """
    Descarga kernels de SPICE a la ubicación del paquete.

    Ejemplo:
    https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp
    """
    if not filename:
        filename=url.split("/")[-1]
    if filename == 'kernels':
        return
    qdata=False
    if 'data:' in filename:
        filename=url.split(":")[1]
        qdata=True
    
    print(f"Descargando kernel '{filename}' en '{basedir}'...")
    if not os.path.exists(ubica_archivos(filename,basedir)) or overwrite:
        if qdata:
            os.system(f"cp -rf {ROOTDIR}/data/{filename} {basedir}/data/")
        else:
            response = requests.get(url)
            open(ubica_archivos(filename,basedir),"wb").write(response.content)
        print("Hecho.")
    else:
        print(f"El kernel '{filename}' ya fue descargado")
        
def descarga_kernels(basedir='pymcel/',overwrite=False):
    """
    Descarga todos los kernels utiles para pymcel
    """
    descarga_kernel("https://raw.githubusercontent.com/seap-udea/pymcel/main/src/pymcel/data/kernels",
                    overwrite=overwrite,basedir=basedir)
    f=open(ubica_archivos("kernels"),"r")
    kernel_dir = basedir+"/data/" 
    if not os.path.exists(kernel_dir):
        print(f"Creando el directorio con los kernels {kernel_dir}...")
        os.makedirs(kernel_dir)
    for line in f:
        url=line.strip()
        descarga_kernel(url,basedir=basedir,overwrite=overwrite)

def lista_kernels(basedir='pymcel/'):
    print("Para descargar todos los kernels use: pymcel.descarga_kernels(). Para descargar un kernel específico use pymcel.descarga_kernel(<url>)")
    return glob.glob(ubica_archivos("*",basedir))

def obtiene_datos(basedir='pymcel/'):
    # Descarga todos los kernels para trabajar con SPICE
    descarga_kernels()

    # Obtiene los datos a partir del directorio de instalación de PYMCEL
    datadir = f"{ROOTDIR}/data"
    if os.path.isdir(datadir):
        if not os.path.isdir("pymcel/data"):
            os.mkdir("pymcel/data")
        print("Copiando archivos de datos...")
        os.system(f"cp -rf {datadir}/*.* pymcel/data/")    

#############################################################
# RUTINAS DE BASES DE DATOS
#############################################################
def consulta_horizons(id='399',location='@0',epochs=None,datos='vectors',propiedades='default'):
    """Realiza una consulta en Horizons

    Opciones:
        id, location, epochs: entradas comunes de Horizons.
        Se puede pasar una epoca como una única fecha o una lista de fechas.

        datos: datos requeridos. Valores aceptados: 'vectors', 'elements', 'ephemeris'

        propiedades: lista con las propiedades que se quieren extraer.
        
            Ejemplo: propiedades = propiedades = [('x','m'),('y','m'),('z','m')]
            
            Si propiedades se pasa como 'default' se extraen las propiedades por defecto
            de acuerdo con 'datos' así:
                'vectors': vector de estado en SI.
                'elements': elementos orbitales clásicos (a, e, I, W, w, f, M, P, to)

    Salida:
        Devuelve tres cosas:
        tabla: la tabla de astropy con el resultado del query.
        ts: lista de tiempos en JD de la consulta.
        salida: pandas dataframe con los resultados de la consulta.


    Ejemplos:

        Una llamada básica: 

            epochs = '2024-01-01 12:00:00'
            tabla, ts, salida = pc.consulta_horizons(
                id='399',location='@0',datos='elements',
                propiedades='elementos',epochs=epochs
            )

        Otra:
            epochs = ['2024-01-01 12:00:00','2024-01-02 12:00:00']
            tabla, ts, salida = pc.consulta_horizons(
                id='399',location='@0',datos='elements',
                propiedades='elementos',epochs=epochs
            )

        Se pueden pedir propiedades específicas:
        
            epochs = dict(start='2024-01-01 12:00:00', stop='2024-01-02 12:00:00',step='1d')
            propiedades=[('a','km'),('incl','deg')]
            tabla, ts, salida = pc.consulta_horizons(
                id='399',location='@0',datos='elements',
                propiedades=propiedades,epochs=epochs
            )
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

#############################################################
# RUTINAS DE GRAFICACIÓN
#############################################################
def fija_ejes_proporcionales(ax,values=(),margin=0,xcm=None,ycm=None,xmin=None,ymin=None):
    """Ajusta los ejes para hacerlos proporcionales de acuerdo a un
    conjunto de valores.

    Normalmente esta tarea es realizada ax.set_aspect('equal','box')
    pero este comando solo se puede ejecutar después de que se han
    graficado los datos.  Esta rutina se puede ejecutar antes, si se
    pasan (como una tupla), todos los datos que van en el gráfico.

    Args:
      ax (matplotlib.axes): axes de matplotlib.

    Keyword Args:
      values (tuple): tupla de datos.
          Los datos deben corresponder a objetos de que puedan
          convertirse en arreglos de numpy. Si no se pasa nada se
          usan los valores en axes.

      margin (float): margen alrededor del gráfico.
          En unidades del (ancho o alto del mismo)

    Returns:
      (xlims,ylims) (tuple,tuple): Límites en x e y.

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
    """Encuentra los rangos de los datos en un conjunto de datos
    tridimensionales.

    Args:
      rs (array): arreglo de numpy con los datos.
          Los datos deben ser tridimensionales.

    Returns:
      (xlims,ylims,zlims) (tuple,tuple,tuple): Límites en x, y, z.

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
    """Ajusta los ejes en 3d para hacelos proporcionales.

    Hace que los ejes de un gráfico en 3d tengan la misma escala, de
    modo que las esferas aparezcan como esferas, los cubos como cubos
    y así sucesivamente.

    Esta es una de las soluciones alternativas para los comandos de
    matplotlib ax.set_aspect('equal') and ax.axis('equal') que no
    funcionan en 3D.

    Args:
      ax (matplotlib.axes): axis de matplotlib.
          Este debe ser el axis donde esta la figura.
    
    References: 
      tomado originalmente de https://stackoverflow.com/a/31364297

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

#############################################################
# RUTINAS GENERALES ÚTILES
#############################################################
def haversine(lon1, lat1, lon2, lat2):
    """Calcula la distancia angular entre dos puntos sobre una esfera
    una vez se han especificado los valores de la longitud y latitud
    de los puntos.

    La rutina usa la formula de Haversine.

    Tomado de: https://stackoverflow.com/a/29546836    
    """
    
    lon1, lat1, lon2, lat2 = map(np.radians,[lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(c)

def calcula_discriminante(a,b,c):
    disc=b**2-4*a*c
    return disc

def coeficientes_fourier(funcion,T,k,args=()):
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
    N=len(x)
    xp=zeros_like(x)
    yp=zeros_like(y)
    zp=zeros_like(z)
    for i in range(N):
        xp[i],yp[i],zp[i]=mxv(R,[x[i],y[i],z[i]])
    return xp,yp,zp

def polinomio_segundo_grado(coeficientes,x,y):
    A,B,C,D,E,F=coeficientes
    P=A*x**2+B*x*y+C*y**2+D*x+E*y+F
    return P

def puntos_conica(p,e,df=0.1):

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

    ax.set_title(f"Cónica con:"+                 f"$p={p:.2f}$, $e={e:.2f}$, "+                 f"$i={i*180/pi:.2f}$, "+                 f"$\Omega={Omega*180/pi:.1f}$, "+                 f"$\omega={Omega*180/pi:.1f}$"
            )
    
    #Dibuja Ejes
    ax.plot([0,xrange[1]],[0,0],[0,0],'k-')
    ax.plot([0,0],[0,yrange[1]],[0,0],'k-')
    ax.plot([0,0],[0,0],[0,zrange[1]],'k-')
    ax.text(xrange[1],0,0,"$x$",ha='left',va='top')
    ax.text(0,yrange[1],0,"$y$",ha='left',va='top')
    ax.text(0,0,zrange[1],"$z$",ha='left',va='bottom')

    fig.tight_layout();
    
    if figreturn:return fig

def edm_ncuerpos(Y,t,N=2,mus=[]):    
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
    """Esta rutina fue mejorada por Simón Echeverri, Astronomía UdeA
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
    rs=zeros((Nparticulas,Ntiempos,3))
    vs=zeros((Nparticulas,Ntiempos,3))
    for i in range(Nparticulas):
        rs[i]=solucion[:,3*i:3*i+3]
        vs[i]=solucion[:,3*Nparticulas+3*i:3*Nparticulas+3*i+3]
    return rs,vs

def ncuerpos_solucion(sistema,ts):
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
    r = Y[:3]
    v = Y[3:]
    drdt = v
    dvdt = -mu*r/np.linalg.norm(r)**3
    return np.concatenate([drdt,dvdt])

def doscuerpos_solucion(mu,r,v,ts):
    X0 = np.concatenate([r,v])
    solucion = odeint(edm_dos_cuerpos,X0,ts,args=(mu,))
    rs = solucion[:,:3]
    vs = solucion[:,3:]
    return rs,vs

def funcion_kepler(G,M=0,e=0):
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
    
    ########################################################
    # Preparación del cálculo
    ########################################################

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

    ########################################################
    # Aquí viene la predicción
    ########################################################

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
    #Parametro alga
    alfa=(1-e)/q
    #Funcion universal de Kepler
    k=q*x+e*x**3*serie_stumpff(alfa*x**2,3)-M
    kp=q+e*x**2*serie_stumpff(alfa*x**2,2)
    kpp=q+e*x*serie_stumpff(alfa*x**2,1)
    return k,kp,kpp

def funcion_universal_kepler_s(s,r0,rdot0,beta,mu,M):
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
    x1=-alfa
    x2=1-alfa
    f=(1-alfa)*(x-x1)/abs(x-x1)**3+alfa*(x-x2)/abs(x-x2)**3-x
    return f

def orbitas_crtbp(alfa,ro,vo,
                  T=100,Nt=1000,
                  xlim=(-1.5,1.5),ylim=(-1.5,1.5),
                  xL=0,yL=0,
                 ):
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
    V=-parametros["mu"]/r**parametros["n"]
    return V

def Vcen(r,**parametros):
    V=parametros["h"]**2/(2*r**2)
    return V

def Veff(r,Vf,**parametros):
    V=Vf(r,**parametros)+Vcen(r,**parametros)
    return V

def estado_a_elementos(mu,x):
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
    sk=lambda n:t/((2*n+k+1)*(2*n+k+2))*(1-sk(n+1)) if n<N else 0
    return (1-sk(0))/math.factorial(k)

def plot_elipse(e=0.5,a=10.0):

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

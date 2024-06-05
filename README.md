# PymCel

## Utilidades de Mecánica Celeste

[![version](https://img.shields.io/pypi/v/pymcel?color=blue)](https://pypi.org/project/pymcel/)
[![downloads](https://img.shields.io/pypi/dw/pymcel)](https://pypi.org/project/pymcel/)
[![license](https://img.shields.io/pypi/l/pymcel)](https://pypi.org/project/pymcel/)
[![implementation](https://img.shields.io/pypi/implementation/pymcel)](https://pypi.org/project/pymcel/)
[![pythonver](https://img.shields.io/pypi/pyversions/pymcel)](https://pypi.org/project/pymcel/)

El paquete `pymcel` contiene un conjunto de utilidades que pueden usarse para la enseñanza de (o la investigación en) Mecánica Celeste o Astrodinámica.

Las utilidades que contienen el paquete fueron originalmente desarrolladas como parte del libro [**Mecánica Celeste: teoría, algoritmos y problemas**](https://www.libreriadelau.com/bw-mecanica-celeste-teoria-algoritmos-y-problemas-u-de-antioquia-fisica/p) del profesor Jorge I. Zuluaga de la Universidad de Antioquia. Muchos de los códigos incluídos en el libro están disponibles en [la sección de ejemplos del repositorio en `GitHub`](https://github.com/seap-udea/pymcel/tree/main/ejemplos/cuadernos-libro) del paquete.

## Descarga e instala

`pymcel` esta disponible en `PyPI`, https://pypi.org/project/pymcel/. Para instalar solo debe ejecutar:

```
   pip install -U pymcel
```

Si usted prefiere puede descargar e instalar directamente desde las [fuentes](https://pypi.org/project/pymcel/#files).

## Para empezar

Para empezar a usar el paquete basta que lo importe:

```python
import pymcel as pc
```

El siguiente código, por ejemplo, integra las ecuaciones de movimiento de una partícula en el CRTBP (problema circular restringido de los tres cuerpos):

```python
Nt=300
ts=linspace(0,10,Nt)
alfa=0.3
ro=[1.0,0.0,0.0]
vo=[0.0,0.45,0.0]
rs_rot,vs_rot,rs_ine,vs_ine,r1_ine,r2_ine=pc.crtbp_solucion(alfa,ro,vo,ts)
```

Un gráfico de la trayectoria de la partícula, y de la posición de los cuerpos más masivos, tanto en el sistema de referencia rotante, como en el sistema de referencia inercial se puede realizar con este código:

```python
import matplotlib.pyplot as plt

fig,axs=plt.subplots(1,2,figsize=(8,4))

# Sistema de referencia rotante
ax=axs[0]
ax.plot(rs_rot[:,0],rs_rot[:,1],'k-')
ax.plot([-alfa],[0],'ro',ms=10)
ax.plot([1-alfa],[0],'bo',ms=5)
ax.set_title("Sistema Rotante")
ax.grid()
ax.axis('equal')	

# Sistema de referencia inercial
ax=axs[1]
ax.plot(rs_ine[:,0],rs_ine[:,1],'k-')
ax.plot(r1_ine[:,0],r1_ine[:,1],'r-')
ax.plot(r2_ine[:,0],r2_ine[:,1],'b-')
ax.set_title("Sistema Inercial")
ax.grid()
ax.axis('equal')

plt.show()
```

<p align="center"><img src="https://github.com/seap-udea/pymcel/blob/main/ejemplos/figuras/crtbp-ejemplo.png?raw=true" alt="Ejemplo de CRTBP"/></p>

## Ejemplos de uso y códigos en el libro

Es también interesante consultar el [repositorio en `GitHub`](http://github.com/seap-udea/pymcel) del paquete, donde además de las fuentes, encontrará, entre otras cosas utiles, [cuadernos de ejemplos y tutoriales](https://github.com/seap-udea/pymcel/tree/main/ejemplos) sobre el uso del paquete.

Muchos de estos cuadernos incorporan los códigos que vienen con el libro y pueden ser ejecutados por comodidad por docentes y estudiantes en `Google Colaboratory`.

## ¿Qué hay de nuevo?

Para una lista detallada de las características más nuevas introducidas en el paquete con la última versión vea el archivo [What's new](https://github.com/seap-udea/pymcel/blob/master/WHATSNEW.md).

------------

Este paquete ha sido diseñado y escrito originalmente por Jorge I. Zuluaga (C) 2023, 2024

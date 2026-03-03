# PymCel

## Utilidades de Mecánica Celeste

[![version](https://img.shields.io/pypi/v/pymcel?color=blue)](https://pypi.org/project/pymcel/)
[![downloads](https://img.shields.io/pypi/dw/pymcel)](https://pypi.org/project/pymcel/)
[![license](https://img.shields.io/pypi/l/pymcel)](https://pypi.org/project/pymcel/)
[![implementation](https://img.shields.io/pypi/implementation/pymcel)](https://pypi.org/project/pymcel/)
[![pythonver](https://img.shields.io/pypi/pyversions/pymcel)](https://pypi.org/project/pymcel/)
[![docs](https://readthedocs.org/projects/pymcel/badge/?version=latest)](https://pymcel.readthedocs.io/es/latest/)
[![book](https://img.shields.io/badge/Libro-Mecanica%20Celeste-0b7285)](https://libros.udea.edu.co/index.php/editorial_udea/catalog/book/345)
[![buy](https://img.shields.io/badge/Donde%20conseguirlo-Librerias%20en%20linea-0b7285)](https://www.libreriadelau.com/mecanica-celeste-u-de-antioquia-fisica/p)
[![orcid](https://img.shields.io/badge/ORCID-0000--0002--6140--3116-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-6140-3116)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18849743.svg)](https://doi.org/10.5281/zenodo.18849743)

El paquete `pymcel` contiene un conjunto de utilidades que pueden usarse para la enseñanza de (o la investigación en) Mecánica Celeste o Astrodinámica.

Las utilidades que contienen el paquete fueron originalmente desarrolladas como parte del libro [**Mecánica Celeste: teoría, algoritmos y problemas**](https://libros.udea.edu.co/index.php/editorial_udea/catalog/book/345) del profesor Jorge I. Zuluaga de la Universidad de Antioquia. Muchos de los códigos incluídos en el libro están disponibles en [la sección de ejemplos del repositorio en `GitHub`](https://github.com/seap-udea/pymcel/tree/main/ejemplos/cuadernos-libro) del paquete. El libro puede conseguirse [en PDF y en papel con la editorial de la Universidad de Antioquia](https://libros.udea.edu.co/index.php/editorial_udea/catalog/book/345) (normalmente envíos solo dentro de Colombia) o internacionalmente en [librerías en línea](https://www.libreriadelau.com/mecanica-celeste-u-de-antioquia-fisica/p) ([aquí también](https://www.buscalibre.com.co/libro-mecanica-celeste-teoria-algoritmos-y-problemas/9789585011953/p/62242977?afiliado=74c874bfb5a8145d7c1b)).

<a href="https://libros.udea.edu.co/index.php/editorial_udea/catalog/book/345" target="_blank">
<p align="center"><img src="https://github.com/seap-udea/pymcel/blob/main/ejemplos/figuras/mcel-jorge-zuluaga-2024.png?raw=true" alt="Portada del Libro"/></p>
</a>

En este sitio encontrará además un documento con las [*Fe de Erratas* del libro](https://github.com/seap-udea/pymcel/blob/main/ejemplos/cuadernos-libro/mcel_zuluaga-00-FeDeErratas.ipynb), en el que encontrarán algunas correcciones puntuales a defectos que se fueron con la primera edición.

## Descarga e instalación

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

Muchos de estos cuadernos incorporan los códigos que vienen con el libro **Mecánica Celeste: teoría, algoritmos y problemas** y pueden ser ejecutados por comodidad por docentes y estudiantes en `Google Colab`.

## Uso de `agents.md` con asistentes de IA (VS Code, Cursor, Antigravity)

Este repositorio incluye un archivo `agents.md` con contexto técnico de `pymcel` para que un asistente de IA entienda:

- Qué hace el paquete y cómo instalarlo.
- Qué funciones usar según el problema (N-cuerpos, Kepler, CRTBP, Lambert, efemérides).
- Ejemplos ejecutables para empezar desde cero.
- Buenas prácticas de unidades, tolerancias, kernels SPICE y validación.

### Recomendación general

1. Abra la carpeta del proyecto donde trabajará.
2. Asegúrese de que `agents.md` esté en la raíz del proyecto (o copie ese archivo al proyecto activo).
3. En el primer mensaje al asistente, pídale explícitamente usar ese archivo como contexto.

Prompt sugerido:

```text
Usa el archivo agents.md de este proyecto como contexto principal para trabajar con pymcel.
Antes de proponer código, sigue sus convenciones de API, unidades y ejemplos.
```

### En VS Code

- Abra el repositorio/folder en VS Code.
- Abra el chat del asistente.
- Mencione en su primer prompt que use `agents.md` como referencia.
- Cuando sea posible, adjunte o cite el archivo en la conversación.

### En Cursor

- Abra el proyecto en Cursor.
- Inicie el chat del agente dentro del workspace.
- Indique que tome `agents.md` como guía de trabajo para `pymcel`.
- Si usa reglas/proyecto-contexto, incluya el contenido de `agents.md` allí.

### En Antigravity (u otros entornos con agentes)

- Cargue el repositorio o el directorio de trabajo.
- Adjunte `agents.md` como documento de contexto del agente, o péguelo en la configuración de instrucciones del proyecto.
- Pida al agente que siga explícitamente esa guía para generar y validar código con `pymcel`.

### Si instala `pymcel` en otra máquina

Puede descargar solo el archivo guía y reutilizarlo en su proyecto:

```bash
curl -L https://raw.githubusercontent.com/seap-udea/pymcel/main/agents.md -o agents.md
```

Luego, abra su entorno de desarrollo y use el prompt sugerido para que el agente trabaje con contexto correcto.

## Como citar PyMCel

Si usa `pymcel` en un trabajo academico, puede citar:

**El libro (recomendado para metodología y teoría):**

```bibtex
@book{jorge2024mecanica,
   title={MECANICA CELESTE; TEORIA, ALGORITMOS Y PROBLEMAS.},
   author={JORGE, I ZULUAGA},
   year={2024},
   publisher={UNIVERSIDAD DE ANTIOQUIA}
}
```

**El software (para el package en sí):**

```bibtex
@software{zuluaga2026pymcel,
  author = {Zuluaga, Jorge I.},
  title = {pymcel: Utilidades de Mecánica Celeste y Astrodinámica},
  year = {2026},
  doi = {10.5281/zenodo.18849743},
  url = {https://doi.org/10.5281/zenodo.18849743}
}
```

## ¿Qué hay de nuevo?

Para una lista detallada de las características más nuevas introducidas en el paquete con la última versión vea el archivo [What's new](https://github.com/seap-udea/pymcel/blob/main/WHATSNEW.md).

------------
Este paquete ha sido diseñado y escrito originalmente por Jorge I. Zuluaga (C) 2023-Presente

# Pymcel

En este archivo encontraran una lista (no exhaustiva) de las
características introducidas en el paquete.

## ¿Qué hay de nuevo?

- **Versiones 0.9.x (beta)**:

  - Mejor experiencia de uso desde instalación con `pip`, con ejemplos y
    documentación más consistentes para comenzar rápido.
  - Notebooks del libro actualizados y sincronizados para que los ejemplos
    funcionen de forma más uniforme entre `ejemplos/` y `docs/examples/`.
  - Flujo de kernels SPICE más estable y predecible (descarga/carga/listado),
    reduciendo fricción al trabajar con efemérides y datos astronómicos.
  - Mejora de compatibilidad en ejemplos e imports, simplificando el uso para
    usuarios nuevos y evitando patrones legacy en los cuadernos.
  - Ajustes de empaquetado y dependencias para instalaciones más robustas en
    distintos entornos Python.
  - Documentación práctica para trabajar con asistentes de IA:
    - `agents.md`: guía completa para que un agente pueda usar `pymcel` de
      forma correcta y empezar desde cero en otra máquina.
  - Registro del software en Zenodo con DOI permanente (10.5281/zenodo.18849743)
    para facilitar citación académica y aumentar impacto científico del paquete.
  - Integración nativa con `rebound` para simulaciones de N-cuerpos en tiempo real, 
    usando la nueva función `ncuerpos_rebound_tiempo_real` que incluye despliegue 
    gráfico interactivo 2D/3D con Matplotlib, soporte para exportar estados 
    y detección de partículas ligadas.
  - Nuevas rutinas avanzadas para generar condiciones iniciales de diversos sistemas astrofísicos:
    - `condiciones_iniciales_plummer`: Cúmulos estelares esféricos.
    - `condiciones_iniciales_toomre`: Colisiones de galaxias (modelo de Toomre 1972) con herramientas paramétricas como `trasladar_y_rotar_sistema`.
    - `condiciones_iniciales_planetesimales`: Discos protoplanetarios enfocados en simular acreción planetaria escalando tamaños con el radio de Hill.
    - `condiciones_iniciales_coreografia`: Generación de datos iniciales precisos para coreografías gravitacionales (Simó, 2001) para N=3, 4 y 5.
  - Mayor personalización numérica y visual en `ncuerpos_rebound_tiempo_real`: selección del motor de integración interno (p. ej. el integrador de alta precisión adaptativo `ias15` para sistemas caóticos y su parámetro `epsilon`), control explícito del tamaño de la estela visual (`longitud_trazo`) y función automatizada para volcar la animación generada en un GIF.
  - Mejoras de mantenimiento que impactan estabilidad general del paquete y
    calidad de releases en la rama 0.9.x.

- **Versiones 0.6.x**:
  - El libro ha sido publicado. Puede conseguirse en formato electrónico 
    [aquí](https://www.buscalibre.com.co/libro-mecanica-celeste-teoria-algoritmos-y-problemas/9789585011953/p/62242977?afiliado=74c874bfb5a8145d7c1b)


- **Versiones 0.5.x**:

  - Primera versión que será liberada con el libro.
  - Se incluyen notebooks de ejemplos.
  - Se incluyen notebooks con los códigos del libro.
  
- **Versiones 0.1.x**:

  - Primer release del paquete.

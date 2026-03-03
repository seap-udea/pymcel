# pymcel v0.9.16 — Beta: mejoras de documentación, notebooks y flujo de trabajo con agentes IA

## Acerca de pymcel

**pymcel** es una biblioteca de Python para **Mecánica Celeste y Astrodinámica** diseñada para facilitar:

- **Problema de N-cuerpos**: Simulación gravitacional de múltiples cuerpos con integración numérica y conservación de constantes de movimiento.
- **Problema restringido circular de 3 cuerpos (CRTBP)**: Análisis de trayectorias en marcos rotantes e inerciales, constante de Jacobi, puntos de Lagrange y órbitas halo.
- **Ecuación de Kepler**: Múltiples métodos de resolución (Newton, semicertificado, series, Bessel) para distintos regímenes de excentricidad.
- **Transferencias orbitales**: Problema de Lambert con trayectorias prograde/retrogradas y manejo de órbitas especiales.
- **Efemérides astronómicas**: Consultas remotas via Horizons (NASA JPL) y locales con kernels SPICE, incluyendo planetas, lunas, asteroides y satélites.
- **Conversión de elementos orbitales**: Transformaciones entre estado cartesiano y elementos clásicos (p, e, i, Ω, ω, f).
- **Visualización**: Gráficos 2D/3D con Matplotlib y visualizaciones interactivas con Plotly.

Ideal para estudiantes, investigadores y profesionales en astronomía, astrofísica, ingeniería aeroespacial y ciencias planetarias.

## Release notes (resumen)

- Consolidación del refactoring y mejoras de estabilidad en la rama `0.9.x`.
- Notebooks del libro sincronizados y normalizados para uso más consistente.
- Ajustes en flujo de kernels/datos SPICE para reducir fricción en configuración.
- Mejoras de compatibilidad en ejemplos e imports, eliminando patrones legacy.
- Actualizaciones de empaquetado y dependencias para instalaciones más robustas.
- Nueva guía para agentes IA mediante `agents.md`, con instrucciones prácticas para VS Code, Cursor y Antigravity.
- Actualización de documentación de cambios en `WHATSNEW.md`.

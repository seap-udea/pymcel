from pymcel import *

def plot_ncuerpos_3d(rs,vs,tipo='matplotlib',**opciones):

    #Número de partículas
    N=rs.shape[0]

    if tipo == 'matplotlib':  
      fig=plt.figure()
      ax=fig.add_subplot(111,projection='3d')

      for i in range(N):
          ax.plot(rs[i,:,0],rs[i,:,1],rs[i,:,2],**opciones);

      fija_ejes3d_proporcionales(ax);
      fig.tight_layout();
      plt.show();
      return fig

    elif tipo == 'plotly':

      if 'plotly' not in sys.modules:
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
                mode='lines',
                name=f"Cuerpo {i}"
            )
        )
      rmin = rs.min()
      rmax = rs.max()

      fig['layout']['scene']['aspectmode'] = 'cube'
      for axis in 'xaxis','yaxis','zaxis':
        fig['layout']['scene'][axis]['range'] = [rmin,rmax]
      fig.show()

    else:
      raise AssertionError(f"Tipo de gráfico '{tipo}' no reconocido")

    return fig

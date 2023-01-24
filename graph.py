import plotly.express as px
from plotly.figure_factory import create_scatterplotmatrix
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import sqrt, ceil


def box (df, x, y):
    return px.box(df, x=x, y=y)


def pairplot (df, cols_to_use = None, index=None, title = None, diag='histogram', 
              graph_size = 800 , size = 5, colormap=None):
    """
    diag: 'histogram', 'box'
    """
    if cols_to_use:
        df_use = df[cols_to_use]
    else:
        df_use = df
    fig = create_scatterplotmatrix(df_use, diag=diag, index=index, size=size, 
                                    colormap=colormap)
    fig.update_layout( title=title, width=graph_size,height=graph_size)
    
    return fig


def histo_many (df, num_cols, num_bins, graph_size):
    cols = df.columns
    num_vars = len(cols) 
    num_rows = ceil(num_vars / num_cols)

    fig = make_subplots( rows=num_rows, cols=num_cols, 
                         subplot_titles = df.columns)

    cnt_var = 0

    for row in range(1, num_rows+1):
        for col in range (1, num_cols+1):
            cnt_var+=1
            if cnt_var > num_vars:
                break
            
            data = go.Histogram(x=df[cols[cnt_var-1]], nbinsx=num_bins)
            fig.add_trace(data, row=row, col=col)

    fig.update_layout(width=graph_size, height=graph_size, showlegend=False)

    return fig


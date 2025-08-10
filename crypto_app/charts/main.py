
import plotly.graph_objects as go
def render_main_chart(df, series, title="Simulation de plusieurs stratégies"):
    fig = go.Figure()
    for name, y in series:
        yplot = df['price'] if y is None else y
        fig.add_trace(go.Scatter(x=df.index, y=yplot, mode='lines', name=name))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Valeur portefeuille ($)")
    return fig

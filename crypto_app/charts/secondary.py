import plotly.graph_objects as go

def render_trades_chart(df, buys_d, buys_p, sells_d, sells_p, price_col="price", title="Trades"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col], mode="lines", name="Prix"))

    if buys_d:
        fig.add_trace(go.Scatter(
            x=buys_d, y=buys_p, mode="markers", name="Achat",
            marker=dict(symbol="triangle-up", size=10),
            hovertemplate="Achat<br>%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>"
        ))
    if sells_d:
        fig.add_trace(go.Scatter(
            x=sells_d, y=sells_p, mode="markers", name="Vente",
            marker=dict(symbol="triangle-down", size=10),
            hovertemplate="Vente<br>%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>"
        ))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Prix")
    return fig

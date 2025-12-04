import streamlit as st
import pandas as pd
import statsmodels.api as sm
import duckdb
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(
    layout='wide',
    page_title='Dashboard',
    page_icon='📊'
    )

@st.cache_data
def load_data():
    data = pd.read_csv('df_games_limpio.csv')
    return data

@st.cache_data
def top_consolas(df:pd.DataFrame):

    filtro_consolas = df.groupby('platform')['total_sales'].sum().sort_values(ascending=False)[:3]
    filtro_consolas = filtro_consolas.reset_index(drop=0)

    col1, col2 = st.columns(2)

    with col1:

        with st.expander('Preview Top Platforms'):
            st.dataframe(filtro_consolas,hide_index=1)

        query_1 = duckdb.sql(
            f"""
            SELECT
                year_of_release,
                platform,
                SUM (total_sales) AS total
            FROM
                df
            WHERE
                platform IN ('PS2','X360','PS3') AND year_of_release BETWEEN 2000 AND 2016
            GROUP BY
                year_of_release,
                platform
            ORDER BY
                total DESC;
            """
            ).df()

        fig_1 = px.bar(query_1,x='year_of_release',y='total',color='platform',barmode='group',title='Top 20th Century')
        fig_1.update_layout(legend=dict(orientation="h",y=-0.3,x=0.5,xanchor="center"))
        st.plotly_chart(fig_1)


    with col2:

        with st.expander('Preview Top Old Platforms'):
            st.dataframe(
                data=
                duckdb.sql(f"""
                    SELECT
                        platform,
                        SUM (total_sales) AS total
                    FROM
                        df
                    WHERE
                        year_of_release BETWEEN 1985 AND 2000
                    GROUP BY
                        platform
                    ORDER BY
                        total DESC
                    LIMIT
                        4;
                    """)
            ,hide_index=1)

        query_2 = duckdb.sql(
            f"""
            SELECT
                year_of_release,
                platform,
                SUM (total_sales) AS total
            FROM
                df
            WHERE
                platform IN ('GB','N64','SNES','PS') AND year_of_release BETWEEN 1985 AND 2000
            GROUP BY
                year_of_release,
                platform
            ORDER BY
                total DESC;
            """
            )
        fig_2 = px.bar(query_2,x='year_of_release',y='total',color='platform',barmode='group',title='Top Last Century')
        fig_2.update_layout(legend=dict(orientation="h",y=-0.3,x=0.5,xanchor="center"))
        st.plotly_chart(fig_2)

def current_tendency(df:pd.DataFrame):
    filtrado = df[((df['year_of_release'] >= 2006)  & (df['year_of_release']<=2016))]
    current_platforms = filtrado.groupby('platform')['total_sales'].sum().sort_values(ascending=False)[:5].reset_index()

    with st.expander(label='Current Platforms Data'):
        st.dataframe(current_platforms,hide_index=1)

    query_1 = duckdb.sql(f"""
    SELECT
        year_of_release,
        platform,
        SUM (total_sales) AS total
    FROM
        df
    WHERE
        year_of_release BETWEEN 2006 AND 2016 AND platform IN ('X360','PS3','Wii','DS','PS4')
    GROUP BY
        year_of_release,
        platform
    ORDER BY
        year_of_release;
    """)
    fig_1 = px.line(query_1,x='year_of_release',y='total',color='platform',markers=0,title='Market Tendency')
    fig_1.update_layout(legend=dict(orientation="h",y=-0.3,x=0.5,xanchor="center"))

    st.plotly_chart(fig_1)

def top_platforms_distribution(df:pd.DataFrame):

    query_1 = duckdb.sql(f"""
    SELECT
        platform,
        SUM (total_sales) AS total,
        AVG (total_sales) AS mean
    FROM
        df
    GROUP BY
        platform
    ORDER BY
        total DESC
    LIMIT
        8;
    """)

    query_2 = duckdb.sql(f"""
    SELECT
        platform,
        total_sales,
        AVG (total_sales) AS mean
    FROM
        df
    WHERE
        platform IN ('PS2','X360','PS3','Wii','DS','PS','GBA','PS4') AND total_sales BETWEEN 0 AND 40
    GROUP BY
        platform,
        total_sales
    ORDER BY
        mean DESC;
    """)
    fig_1 = px.strip(query_2,x='total_sales',y='platform',color='platform',title='Top Platforms Sales Distribution')
    st.plotly_chart(fig_1)

def top_platform_correlation(df:pd.DataFrame):
    query_1 = duckdb.sql(f"""
    SELECT
        platform,
        name,
        critic_score,
        user_score,
        total_sales
    FROM
        df
    WHERE 
        platform == 'PS4' AND critic_score <> -1 AND user_score <> -1
    ORDER BY
        total_sales DESC;
    """).df()

    Y = query_1['total_sales']

    # 1. Regresión para Critic Score vs Sales
    X_critico = query_1['critic_score']
    
    X_critico = sm.add_constant(X_critico)
    modelo_critico = sm.OLS(Y, X_critico).fit()
    Y_pred_critico = modelo_critico.predict(X_critico)

    # 2. Regresión para User Score vs Sales
    X_usuario = query_1['user_score']

    X_usuario = sm.add_constant(X_usuario)
    modelo_usuario = sm.OLS(Y, X_usuario).fit()
    Y_pred_usuario = modelo_usuario.predict(X_usuario)

    # CREACION DE GRAFICOS CON go.Scatter()
    trace_line_critico = go.Scatter(x=query_1['critic_score'], y=Y_pred_critico,mode='lines', name='Critic Tendency',line=dict(color='Blue', width=2))
    trace_line_usuario = go.Scatter(x=query_1['user_score'], y=Y_pred_usuario,mode='lines', name='User Tendency',line=dict(color='Red', width=2))
    trace_0 = go.Scatter(x=query_1['critic_score'],y=query_1['total_sales'],mode='markers',marker=dict(color='Cyan'))
    trace_1 = go.Scatter(x=query_1['user_score'],y=query_1['total_sales'],mode='markers',marker=dict(color='Yellow'))

    fig = make_subplots(rows=2,cols=1)
    fig.append_trace(trace_0,1,1)
    fig.append_trace(trace_line_critico, 1, 1)
    fig.append_trace(trace_1,2,1)
    fig.append_trace(trace_line_usuario, 2, 1)

    fig.update_layout(title_text='Score Sales Correlation PS4',legend=dict(orientation="h",y=-0.3,x=0.5,xanchor="center"))

    # Etiquetas de titulos para los graficos segun su indice en el layout:
    fig.update_xaxes(title_text="Critic Score", row=1, col=1)
    fig.update_xaxes(title_text="User Score", row=2, col=1)
    fig.update_yaxes(title_text="Sales", row=1, col=1)
    fig.update_yaxes(title_text="Sales", row=2, col=1)

    st.plotly_chart(fig)

# Interfaz
st.header(':material/analytics: Videogame Platforms Throughout Years Dashboard')

df = load_data()

with st.expander('Original Data Preview'):
    st.dataframe(df)

col1, col2 = st.columns([1.8,2.2])
with col1:
    current_tendency(df)
    top_platforms_distribution(df)
with col2:
    top_consolas(df)
    top_platform_correlation(df)

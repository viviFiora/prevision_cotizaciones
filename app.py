import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

lista_tickers = ["MELI","BMA.BA","GGAL.BA","PAMP.BA","YPFD.BA","TECO2.BA","CEPU.BA","TXR.BA","LOMA.BA","DESP.BA","GLOBD.BA","SUPVD.BA","BPAT.BA","GCLA.BA","HAVA.BA"]

tickers = {
    "MELI": "Mercado Libre, Inc.",
    "BMA.BA": "Banco Macro S.A.",
    "GGAL.BA": "Grupo Financiero Galicia S.A.",
    "PAMP.BA": "Pampa Energía S.A.",
    "YPFD.BA": "YPF Sociedad Anónima",
    "TECO2.BA": "Telecom Argentina S.A.",
    "CEPU.BA": "Central Puerto S.A.",
    "TXR.BA": "Ternium S.A.",
    "LOMA.BA": "Loma Negra Compañía Industrial Argentina Sociedad Anónima",
    "DESP.BA": "Despegar.com, Corp.",
    "GLOBD.BA": "Globant S.A.",
    "SUPVD.BA": "Grupo Supervielle S.A.",
    "BPAT.BA": "Banco Patagonia S.A.",
    "GCLA.BA": "Grupo Clarín S.A.",
    "HAVA.BA": "Havanna Holding S.A."
}

@st.cache_data(ttl=86400)
def cargar_datos(ticket, fecha_inicial, fecha_final):
    try:
        df = yf.Ticker(ticket).history(start=fecha_inicial.strftime("%Y-%m-%d"),
                                       end=fecha_final.strftime("%Y-%m-%d"))
        return df
    except Exception as e:
        st.error(f"Error al obtener datos de {ticket}: {e}")
        return None

def preveer_datos(df, periodo):
    df.reset_index(inplace=True)
    df = df.loc[:,["Date","Close"]]
    df["Date"] = df["Date"].dt.tz_localize(None)
    df.rename(columns={"Date":"ds", "Close":"y"}, inplace=True)

    modelo = Prophet()
    modelo.fit(df)

    fecha_futura = modelo.make_future_dataframe(periods=int(periodo) * 30)
    previsiones = modelo.predict(fecha_futura)

    return modelo, previsiones

st.set_page_config(
    page_title="Analisis predictiva",
    page_icon="📈"
)

st.image("logo invertir en bolsa.jpg")
st.markdown("""
# Analisis Predictiva
### Preveer valores de acciones de empresas Argentinas en la Bolsa
""")
st.divider()

with st.sidebar:
    st.header("Menu")
    ticket_seleccionado = st.selectbox("Seleccione la accion:", lista_tickers)
    fecha_inicial = st.date_input("Fecha inicial",value=date(2015,1,1))
    fecha_final = st.date_input("Fecha final del periodo")
    meses = st.number_input("Indique la cantidad de meses a preveer", min_value=1, max_value=24,value=6)

datos = cargar_datos(ticket_seleccionado,fecha_inicial,fecha_final)

if datos is not None and not datos.empty::
    st.header(f"Datos de la empresa - {tickers[ticket_seleccionado]}")
    st.dataframe(datos)

    st.subheader("Variacion del periodo seleccionado")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos.index, y=datos["Close"], name="Close"))
    st.plotly_chart(fig)

    st.divider()

    st.header(f"Predicción para los proximos: {meses} meses")
    modelo, prevision = preveer_datos(datos, meses)
    figura_prevision = plot_plotly(modelo, prevision, xlabel="periodo", ylabel="valor")
    st.plotly_chart(figura_prevision)

else:
    st.warning("Ningun resultado encontrado en el periodo seleccionado")

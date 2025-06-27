
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("modelo_entrenado.pkl")
columnas = joblib.load("columnas_modelo.pkl")

st.set_page_config(page_title="Simulador de Precios", layout="wide")
st.title("💰 Simulador de Precio Óptimo por Cliente")

def predecir_precio(dataframe):
    df_completo = dataframe.copy()
    for col in columnas:
        if col not in df_completo.columns:
            df_completo[col] = 0
    df_completo = df_completo[columnas]
    pred = model.predict(df_completo)
    return pred

with st.expander("📝 Introducir un cliente manualmente"):
    col1, col2, col3 = st.columns(3)

    with col1:
        frecuencia = st.slider("Frecuencia de compra anual", 0, 100, 10)
        volumen = st.slider("Volumen de compras (€)", 0, 10000, 1000)
        promociones = st.slider("Promociones aplicadas (€)", 0, 1000, 100)

    with col2:
        resistencia = st.slider("Resistencia a subida de precio (0 a 1)", 0.0, 1.0, 0.5)
        coste = st.number_input("Coste del producto (€)", value=100)
        rentabilidad = st.number_input("Rentabilidad esperada (%)", value=20)

    with col3:
        tipo = st.selectbox("Tipo de cliente", ["A", "B", "C"])
        sector = st.selectbox("Sector", ["Hostelería", "Ferretería", "Peluquería", "Particular"])
        zona = st.selectbox("Zona/Ubicación", ["Norte", "Sur", "Centro", "Este", "Oeste"])
        competencia = st.slider("Precio de la competencia (€)", 50, 300, 120)

    if st.button("📊 Calcular precio óptimo"):
        input_manual = pd.DataFrame([{
            "frecuencia_compra": frecuencia,
            "volumen_compras": volumen,
            "promociones": promociones,
            "resistencia_precio": resistencia,
            "coste_producto": coste,
            "rentabilidad_esperada": rentabilidad,
            "tipo_cliente": tipo,
            "sector": sector,
            "zona": zona,
            "precio_competencia": competencia
        }])
        input_manual = pd.get_dummies(input_manual)
        resultado = predecir_precio(input_manual)
        st.success(f"💸 Precio óptimo estimado: **{resultado[0]:.2f} €**")

st.divider()
st.subheader("📂 O subir un archivo con múltiples clientes")
archivo = st.file_uploader("Sube un archivo .csv o .xlsx con los datos de clientes")

if archivo:
    try:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith(".xlsx"):
            df = pd.read_excel(archivo)
        else:
            st.warning("Formato no soportado. Usa .csv o .xlsx")
            st.stop()

        df = pd.get_dummies(df)
        predicciones = predecir_precio(df)
        df["precio_estimado"] = predicciones
        st.success("✅ Predicciones realizadas")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar resultados", data=csv, file_name="predicciones_precios.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

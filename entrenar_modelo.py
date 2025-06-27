import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Simular un conjunto de datos
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'frecuencia_compra_anual': np.random.randint(1, 20, n),
    'precio_historico_promedio': np.random.uniform(80, 150, n),
    'volumen_compras': np.random.randint(1, 50, n),
    'promociones_aplicadas': np.random.randint(0, 5, n),
    'resistencia_precio': np.random.uniform(0, 1, n),
    'coste_producto': np.full(n, 100),
    'rentabilidad_esperada': np.random.uniform(0.1, 0.5, n),
    'tipo_cliente': np.random.choice(['A', 'B', 'C'], n),
    'sector': np.random.choice(['Ferretería', 'Hostelería', 'Particular', 'Peluquería'], n),
    'precio_competencia': np.random.uniform(90, 160, n),
    'zona': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n),
})

# Codificar variables categóricas
df_encoded = pd.get_dummies(df, columns=['tipo_cliente', 'sector', 'zona'], drop_first=True)

# Variable objetivo simulada (precio óptimo)
df_encoded['precio_optimo'] = (
    df_encoded['precio_historico_promedio'] * 0.3 +
    df_encoded['volumen_compras'] * 0.5 +
    df_encoded['resistencia_precio'] * 20 +
    df_encoded['precio_competencia'] * 0.2 +
    df_encoded['rentabilidad_esperada'] * 100 +
    np.random.normal(0, 5, n)
)

# Entrenar el modelo
X = df_encoded.drop(columns=['precio_optimo'])
y = df_encoded['precio_optimo']

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X, y)

# Guardar modelo y columnas
joblib.dump(modelo, "modelo_entrenado.pkl")
joblib.dump(X.columns.tolist(), "columnas_modelo.pkl")

print("✅ Archivos modelo_entrenado.pkl y columnas_modelo.pkl generados correctamente.")


import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# ==================== ESTILO PERSONALIZADO ====================
st.set_page_config(page_title="Dashboard Zurich", layout="wide")

st.markdown("""
    <style>
        h1 {text-align: center; color: #0a5275;}
        .stApp {background-color: #f9f9f9;}
        .sidebar .sidebar-content {background-color: #eaf4f4;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📊 Dashboard Analítico - Airbnb en <span style='color:#0a5275;'>Zurich</span></h1>", unsafe_allow_html=True)
st.divider()

# ============================================================
# INFORMACIÓN SOBRE ZURICH Y MAPA
# ============================================================

st.subheader("📍 Información y Ubicación de Zurich")

with st.expander("📝 Acerca de Zurich"):
    st.markdown("""
    **Zurich** es la ciudad más grande de Suiza, conocida por su papel en el sector financiero, así como por su alta calidad de vida y paisajes impresionantes.

    - 📌 **Población**: ~430,000 habitantes  
    - 🌍 **Idioma principal**: Alemán, aunque también se hablan inglés y otros idiomas
    - 🏛️ **Atracciones principales**: Lago de Zurich, Iglesia de San Pedro, Museo Nacional Suizo
    - 🌤️ **Clima**: Continental, con veranos suaves e inviernos fríos
    """)

st.markdown("### 🗺️ Mapa de Zurich")

zurich_coords = {"lat": 47.3769, "lon": 8.5417}

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=zurich_coords["lat"],
        longitude=zurich_coords["lon"],
        zoom=11,
        pitch=40,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=[zurich_coords],
            get_position="[lon, lat]",
            get_color="[0, 0, 255, 160]",
            get_radius=500,
        ),
    ],
))

# ============================================================
# ETAPA 1 - CARGA Y PREPARACIÓN DE DATOS
# ============================================================

@st.cache_resource
def load_data():
    df = pd.read_csv("listings.csv")

    # 1. Convertir 'price' a tipo numérico
    if 'price' in df.columns:
        df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

    # 2. Rellenar valores nulos
    numeric_cols_all = df.select_dtypes(include=['float', 'int']).columns
    df[numeric_cols_all] = df[numeric_cols_all].fillna(df[numeric_cols_all].median())

    categorical_cols_all = df.select_dtypes(include='object').columns
    df[categorical_cols_all] = df[categorical_cols_all].fillna('No especificado')

    # 3. Identificar variables categóricas útiles (para visualización)
    cat_vars = df[categorical_cols_all].nunique()
    categorical_cols = cat_vars[cat_vars <= 50].index.tolist()

    # 4. Variables numéricas útiles
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()

    return df, numeric_cols, categorical_cols

# ✅ Cargar los datos y variables útiles
df, numeric_cols, categorical_cols = load_data()

# ============================================================
# MENÚ DE NAVEGACIÓN PRINCIPAL POR ETAPA
# ============================================================

st.sidebar.title("🧭 Menú del Dashboard")
etapa = st.sidebar.radio("Seleccione una etapa:", [
    "Etapa 1: Carga y preparación",
    "Etapa 2: Análisis univariado",
    "Etapa 3: Visualización explicativa",
    "Etapa 4: Modelado predictivo"
])

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Vista rápida del dataset")

if st.sidebar.checkbox("Mostrar primeros registros"):
    st.sidebar.dataframe(df.head())

if st.sidebar.checkbox("Mostrar estadísticas"):
    st.sidebar.dataframe(df.describe().T)

# ============================================================
# ETAPA 2 - ANÁLISIS UNIVARIADO
# ============================================================

if etapa == "Etapa 2: Análisis univariado":
    st.header("2️⃣ Análisis Univariado")
    st.info("Solo se muestran variables categóricas con pocas categorías. Esto evita gráficos innecesarios o ilegibles y mejora la experiencia del análisis.")

    selected_col = st.selectbox("Seleccione una variable categórica:", categorical_cols)
    if selected_col:
        vc_df = df[selected_col].value_counts().reset_index()
        vc_df.columns = [selected_col, 'count']

        st.plotly_chart(px.bar(vc_df,
                               x=selected_col, y='count',
                               labels={selected_col: selected_col.capitalize(), 'count': 'Frecuencia'},
                               title=f"Distribución de {selected_col}"))

        st.plotly_chart(px.pie(df, names=selected_col, title=f"Distribución porcentual de {selected_col}"))

# ============================================================
# ETAPA 3 - VISUALIZACIÓN EXPLICATIVA
# ============================================================

if etapa == "Etapa 3: Visualización explicativa":
    st.header("3️⃣ Visualización Explicativa")

    st.info("En esta etapa solo se incluyen variables numéricas, ya que los gráficos como heatmaps y boxplots requieren datos cuantitativos para mostrar relaciones estadísticas.")

    st.subheader("📈 Heatmap de Correlaciones")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Boxplot")
    num_box = st.selectbox("Seleccione variable numérica para boxplot", numeric_cols)
    cat_box = st.selectbox("Seleccione variable categórica para agrupar", categorical_cols)
    st.plotly_chart(px.box(df, x=cat_box, y=num_box, title=f"{num_box} por {cat_box}"))

    with st.expander("ℹ️ ¿Por qué no aparecen todas las variables?"):
        st.markdown("""
        Las variables que se muestran en cada etapa del análisis están filtradas en función de su tipo de dato y utilidad estadística:

        - 🔢 Variables numéricas: se usan en correlaciones y modelos.
        - 🔠 Variables categóricas: se usan en gráficos de distribución.
        - ⚠️ Columnas con muchos nulos o demasiadas categorías se excluyen para evitar errores o gráficos poco claros.
        """)

# ============================================================
# ETAPA 4 - MODELADO PREDICTIVO
# ============================================================

if etapa == "Etapa 4: Modelado predictivo":
    st.header("4️⃣ Modelado Predictivo")
    st.info("Las variables disponibles para modelado han sido filtradas para incluir únicamente valores numéricos y sin datos faltantes. Esto garantiza que los algoritmos puedan aplicarse correctamente.")

    # --- Regresión Lineal Simple ---
    st.subheader("📌 Regresión Lineal Simple")
    x_simple = st.selectbox("Variable independiente (X)", numeric_cols)
    y_simple = st.selectbox("Variable dependiente (Y)", numeric_cols, index=0, key="simple_y")

    if x_simple and y_simple:
        X = df[[x_simple]]
        y = df[y_simple]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)

        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**Coeficiente:** {model.coef_[0]:.2f}")
        st.write(f"**Intercepto:** {model.intercept_:.2f}")
        fig = px.scatter(x=X_test[x_simple], y=y_test, trendline="ols", title="Regresión Lineal Simple")
        st.plotly_chart(fig)

    # --- Regresión Lineal Múltiple ---
st.subheader("📌 Regresión Lineal Múltiple")

# Asegurar que solo se usen columnas numéricas válidas
valid_numeric_cols = df[numeric_cols].select_dtypes(include=['number']).columns.tolist()
multivars = st.multiselect("Seleccione variables independientes (X)", valid_numeric_cols, default=valid_numeric_cols[1:])
y_multi = st.selectbox("Variable dependiente (Y)", valid_numeric_cols, index=0, key="multi_y")

if multivars and y_multi:
    X_multi = df[multivars]
    y = df[y_multi]

    # Eliminar filas con valores NaN o infinitos
    combined = pd.concat([X_multi, y], axis=1).replace([float("inf"), float("-inf")], pd.NA).dropna()

    if len(combined) < 5:
        st.warning("⚠️ No hay suficientes datos válidos para realizar la regresión múltiple. Intente seleccionar otras variables.")
    else:
        X_multi = combined[multivars]
        y = combined[y_multi]

        X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)
        model_multi = LinearRegression().fit(X_train, y_train)
        y_pred = model_multi.predict(X_test)
        mse_multi = mean_squared_error(y_test, y_pred)

        st.write(f"**MSE:** {mse_multi:.2f}")
        st.write("**Coeficientes:**")
        for var, coef in zip(multivars, model_multi.coef_):
            st.write(f"{var}: {coef:.2f}")


    # --- Regresión Logística ---
    st.subheader("📌 Regresión Logística")
    df['high_price'] = (df['price'] > df['price'].median()).astype(int)
    x_log_vars = st.multiselect("Seleccione variables para clasificación", ['accommodates', 'number_of_reviews', 'availability_365'], default=['accommodates'])

    if x_log_vars:
        X = df[x_log_vars]
        y = df['high_price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.write(f"**Precisión del modelo:** {acc:.2%}")
        st.write("**Coeficientes:**")
        for var, coef in zip(x_log_vars, model.coef_[0]):
            st.write(f"{var}: {coef:.2f}")

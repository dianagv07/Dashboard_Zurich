
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# ==================== CONFIGURACION ====================
st.set_page_config(page_title="Dashboard Zurich", layout="wide")

# ==================== BIENVENIDA ====================
if 'inicio' not in st.session_state:
    st.session_state.inicio = True

if st.session_state.inicio:
    st.markdown("""
        <style>.bienvenida { text-align: center; margin-top: 50px; }</style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="bienvenida"><h1>📍 Dashboard de Airbnb en Zurich</h1></div>', unsafe_allow_html=True)
    st.image("imagenes/zurich.jpg", width=300, caption="Ciudad de Zurich")
    if st.button("🚀 Comenzar"):
        st.session_state.inicio = False
        st.rerun()
    st.stop()

# ==================== CARGA DE DATOS ====================
@st.cache_resource
def load_data():
    df = pd.read_csv("listings.csv")
    if 'price' in df.columns:
        df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df[df.select_dtypes(include=['float', 'int']).columns] = df.select_dtypes(include=['float', 'int']).fillna(df.median(numeric_only=True))
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('No especificado')
    cat_vars = df.select_dtypes(include='object').nunique()
    categorical_vars = cat_vars[cat_vars <= 50].index.tolist()
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    return df, numeric_cols, categorical_vars

df, numeric_cols, categorical_vars = load_data()

# ==================== MENU ====================
st.sidebar.title("🧭 Menú del Dashboard")
etapa = st.sidebar.radio("Seleccione una etapa:", [
    "Etapa 0: Información de Zurich",
    "Etapa 1: Carga y preparación",
    "Etapa 2: Análisis univariado",
    "Etapa 3: Visualización explicativa",
    "Etapa 4: Modelado predictivo"
])

# ==================== ETAPA 0 ====================
if etapa == "Etapa 0: Información de Zurich":
    st.header("0️⃣ Información General de Zurich")
    with st.expander("📝 Acerca de Zurich"):
        st.markdown("""
        **Zurich** es la ciudad más grande de Suiza, conocida por su papel en el sector financiero y su alta calidad de vida.

        - 📌 **Población**: ~430,000 habitantes  
        - 🌍 **Idioma principal**: Alemán  
        - 🏛️ **Atracciones**: Lago de Zurich, Iglesia de San Pedro, Museo Nacional Suizo  
        - 🌤️ **Clima**: Continental, con veranos suaves e inviernos fríos
        """)

    locations_df = df[['neighbourhood_cleansed', 'latitude', 'longitude', 'price']].dropna()
    neigh_summary = locations_df.groupby('neighbourhood_cleansed').agg({
        'latitude': 'median', 'longitude': 'median', 'price': 'mean'
    }).reset_index()

    q1 = neigh_summary['price'].quantile(0.33)
    q2 = neigh_summary['price'].quantile(0.66)

    def precio_color(precio):
        if precio <= q1:
            return [0, 200, 0, 160]
        elif precio <= q2:
            return [255, 215, 0, 160]
        else:
            return [200, 0, 0, 160]

    neigh_summary['color'] = neigh_summary['price'].apply(precio_color)
    neigh_summary['tooltip'] = neigh_summary.apply(lambda row:
        f"<b>{row['neighbourhood_cleansed']}</b><br>💰 Precio promedio: €{row['price']:.2f}", axis=1)

    scatter_layer = pdk.Layer("ScatterplotLayer", data=neigh_summary,
        get_position=["longitude", "latitude"], get_color="color",
        get_radius=800, pickable=True)

    text_layer = pdk.Layer("TextLayer", data=neigh_summary,
        get_position=["longitude", "latitude"], get_text="neighbourhood_cleansed",
        get_size=15, get_color=[0, 0, 0], get_angle=0, get_alignment_baseline="'bottom'")

    view_state = pdk.ViewState(latitude=47.3769, longitude=8.5417, zoom=11, pitch=30)

    st.pydeck_chart(pdk.Deck(
        layers=[scatter_layer, text_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip={"html": "{tooltip}", "style": {"color": "black", "backgroundColor": "white"}}
    ))

    st.markdown("### 🗂️ Simbología de Precios")
    st.markdown(f"""
    - 🟢 **Barato**: ≤ €{q1:.2f}  
    - 🟡 **Medio**: €{q1:.2f} – €{q2:.2f}  
    - 🔴 **Caro**: > €{q2:.2f}
    """)

# ==================== ETAPA 1 ====================
elif etapa == "Etapa 1: Carga y preparación":
    st.header("1️⃣ Carga y Preparación de Datos")
    st.subheader("📄 Primeros registros del dataset")
    st.dataframe(df.head(10))
    st.subheader("📊 Estadísticas generales")
    st.dataframe(df.describe().T.style.highlight_max(axis=0))

# ==================== ETAPA 2 ====================
elif etapa == "Etapa 2: Análisis univariado":
    st.header("2️⃣ Análisis Univariado de Variables Categóricas")
    var_cat = st.selectbox("Seleccione variable categórica", categorical_vars)
    df_count = df[var_cat].value_counts().reset_index()
    df_count.columns = [var_cat, "Frecuencia"]
    st.subheader(f"Distribución de: {var_cat}")
    st.bar_chart(df_count.set_index(var_cat))

# ==================== ETAPA 3 ====================
elif etapa == "Etapa 3: Visualización explicativa":
    st.header("3️⃣ Visualización Interactiva y Análisis Explicativo")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔥 Heatmap de Correlaciones")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    with col2:
        st.markdown("### 📦 Boxplot por categoría")
        var = st.selectbox("Variable numérica", numeric_cols)
        cat = st.selectbox("Categoría agrupadora", categorical_vars)
        fig = px.box(df, x=cat, y=var, title=f"Distribución de {var} por {cat}")
        st.plotly_chart(fig)

# ==================== ETAPA 4 ====================
elif etapa == "Etapa 4: Modelado predictivo":
    st.header("4️⃣ Modelado Predictivo")

    st.subheader("📈 Regresión Lineal Simple")
    x_simple = st.selectbox("Variable independiente (X)", numeric_cols)
    y_simple = st.selectbox("Variable dependiente (Y)", numeric_cols, key="rls")
    X = df[[x_simple]].dropna()
    y = df[y_simple].dropna()
    X = X.loc[y.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"Coeficiente: {model.coef_[0]:.2f}, Intercepto: {model.intercept_:.2f}")

    st.subheader("📈 Regresión Logística")
    df['high_price'] = (df['price'] > df['price'].median()).astype(int)
    X_log = df[['latitude', 'longitude', 'accommodates']].dropna()
    y_log = df['high_price']
    X_log = X_log.loc[y_log.index]
    X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
    log_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Precisión: {acc:.2%}")

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import folium
from streamlit_folium import st_folium
import json
import time
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PREDICCIÓN DE SEVERIDAD DE DAÑOS",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 1. Modelo de ML que construimos y exportamos---

MODELO_PRINCIPAL = "lgboost_binary.joblib"

NOMBRE_MODELO_AMIGABLE = "LightGBM Binary Classifier"

# --- 2. Función para Cargar el Modelo Principal ---
@st.cache_resource 
def cargar_modelo_principal(path):
    with st.spinner(f"Cargando el modelo {NOMBRE_MODELO_AMIGABLE}..."):
        if not os.path.exists(path):
            st.error(f"Error: El archivo del modelo '{path}' no se encontró. Asegúrate de que esté en la carpeta raíz de tu aplicación.")
            st.stop() 
        try:
            model = joblib.load(path)
            st.success(f"Modelo '{NOMBRE_MODELO_AMIGABLE}' cargado exitosamente.")
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo '{path}': {e}")
            st.stop()

# --- Cargar el modelo principal al inicio de la aplicación ---
modelo_actual = cargar_modelo_principal(MODELO_PRINCIPAL)

# --- Cargar los valores únicos (asumiendo que unique_values_for_streamlit.json existe) ---
@st.cache_data
def load_unique_values(path='unique_values_for_streamlit.json'):
    if not os.path.exists(path):
        st.error(f"Error: No se encontró el archivo de valores únicos en '{path}'.")
        st.warning("Asegúrate de haber ejecutado 'generate_unique_values.py' y que el archivo esté en la misma carpeta.")
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error al cargar los valores únicos desde '{path}': {e}")
        return None

unique_values_data = load_unique_values()

# --- Título y resto de la Interfaz de Streamlit ---
st.title("PREDICCIÓN DE SEVERIDAD DE DAÑOS EN EDIFICACIONES POR TERREMOTOS")

# --- Crear pestañas para modo Individual vs. Batch CSV ---
tab1, tab2 = st.tabs(["Predicción Individual", "Predicción por CSV"])

# --- Pestaña 1: Predicción Individual (tu código actual) ---
with tab1:
    st.header("Sección de Predicción de Daños")
    # ... copia aquí todo tu bloque de inputs + botón "Obtener Predicción de Daño" ...
    # Diccionario para almacenar los inputs del usuario
    user_inputs = {}

    if unique_values_data:

        # 1. 'count_floors_pre_eq'
        user_inputs['count_floors_pre_eq'] = st.selectbox(
            "Número de Pisos (antes del sismo):",
            options=sorted([x for x in unique_values_data.get('count_floors_pre_eq', [1,2,3]) if x is not None]),
            index=0 if 1 not in unique_values_data.get('count_floors_pre_eq', [1,2,3]) else unique_values_data['count_floors_pre_eq'].index(1)
        )

        # 2. 'age_building'
        valid_ages = [x for x in unique_values_data.get('age_building', [0, 10, 20]) if x is not None]
        min_age, max_age, default_age = (min(valid_ages), max(valid_ages), int(np.median(valid_ages))) if valid_ages else (0, 100, 10)
        user_inputs['age_building'] = st.slider(
            "Antigüedad del Edificio (años):",
            min_value=min_age, max_value=max_age, value=default_age, step=1
        )

        # 3. 'plinth_area_sq_ft'
        valid_areas = [x for x in unique_values_data.get('plinth_area_sq_ft', [100, 300, 500]) if x is not None]
        min_area, max_area, default_area = (min(valid_areas), max(valid_areas), int(np.median(valid_areas))) if valid_areas else (0, 1000, 300)
        user_inputs['plinth_area_sq_ft'] = st.slider(
            "Área del Zócalo (pies cuadrados):",
            min_value=min_area, max_value=max_area, value=default_area, step=1
        )

        # 4. 'height_ft_pre_eq'
        valid_heights = [x for x in unique_values_data.get('height_ft_pre_eq', [10, 20, 30]) if x is not None]
        min_height, max_height, default_height = (min(valid_heights), max(valid_heights), int(np.median(valid_heights))) if valid_heights else (0, 80, 15)
        user_inputs['height_ft_pre_eq'] = st.slider(
            "Altura del Edificio (pies, antes del sismo):",
            min_value=min_height, max_value=max_height, value=default_height, step=1
        )

        # 5. 'land_surface_condition'
        user_inputs['land_surface_condition'] = st.selectbox(
            "Condición de la Superficie del Terreno:",
            options=sorted([x for x in unique_values_data.get('land_surface_condition', ['Flat', 'Moderate slope', 'Steep slope']) if x is not None])
        )

        # 6. 'foundation_type'
        user_inputs['foundation_type'] = st.selectbox(
            "Tipo de Cimentación:",
            options=sorted([x for x in unique_values_data.get('foundation_type', ['Other', 'Mud mortar-Stone/Brick']) if x is not None])
        )

        # 7. 'roof_type'
        user_inputs['roof_type'] = st.selectbox(
            "Tipo de Techo:",
            options=sorted([x for x in unique_values_data.get('roof_type', ['Bamboo/Timber-Light roof', 'RCC/RB/RBC']) if x is not None])
        )

        # 8. 'ground_floor_type'
        user_inputs['ground_floor_type'] = st.selectbox(
            "Tipo de Planta Baja:",
            options=sorted([x for x in unique_values_data.get('ground_floor_type', ['Mud', 'RC']) if x is not None])
        )

        # 9. Binary 'has_superstructure_...' features (Checkboxes)
        st.subheader("Características de la Superestructura:")
        binary_cols = [
            'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
            'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
            'has_superstructure_timber', 'has_superstructure_bamboo',
            'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
            'has_superstructure_other'
        ]
        for col in binary_cols:
            display_name = col.replace('has_superstructure_', '').replace('_', ' ').title()
            user_inputs[col] = int(st.checkbox(f"Tiene superestructura de {display_name}", value=False))

        # 10. 'count_families'
        user_inputs['count_families'] = st.selectbox(
            "Número de Familias:",
            options=sorted([x for x in unique_values_data.get('count_families', [0, 1, 2]) if x is not None]),
            index=0 if 1 not in unique_values_data.get('count_families', [0,1,2]) else unique_values_data['count_families'].index(1)
        )

        expected_features = [
            'count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq',
            'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
            'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
            'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
            'has_superstructure_timber', 'has_superstructure_bamboo',
            'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
            'has_superstructure_other',
            'count_families'
        ]

        if 'land_surface_condition' in unique_values_data and user_inputs['land_surface_condition'] is not None:
            land_surface_condition_map = {val: i for i, val in enumerate(sorted([x for x in unique_values_data['land_surface_condition'] if x is not None]))}
            user_inputs['land_surface_condition'] = land_surface_condition_map.get(user_inputs['land_surface_condition'], -1)

        if 'foundation_type' in unique_values_data and user_inputs['foundation_type'] is not None:
            foundation_type_map = {val: i for i, val in enumerate(sorted([x for x in unique_values_data['foundation_type'] if x is not None]))}
            user_inputs['foundation_type'] = foundation_type_map.get(user_inputs['foundation_type'], -1)

        if 'roof_type' in unique_values_data and user_inputs['roof_type'] is not None:
            roof_type_map = {val: i for i, val in enumerate(sorted([x for x in unique_values_data['roof_type'] if x is not None]))}
            user_inputs['roof_type'] = roof_type_map.get(user_inputs['roof_type'], -1)

        if 'ground_floor_type' in unique_values_data and user_inputs['ground_floor_type'] is not None:
            ground_floor_type_map = {val: i for i, val in enumerate(sorted([x for x in unique_values_data['ground_floor_type'] if x is not None]))}
            user_inputs['ground_floor_type'] = ground_floor_type_map.get(user_inputs['ground_floor_type'], -1)

        # Preparar los datos para la predicción
        try:
            data_for_prediction = [user_inputs[feature] for feature in expected_features]
            final_input_array = np.array([data_for_prediction])
        except KeyError as ke:
            st.error(f"Error: La característica '{ke}' no se encontró en los inputs proporcionados o en la lista 'expected_features'.")
            st.info("Verifica que todos los widgets están generando inputs para las características esperadas y que 'expected_features' esté completo y en el orden correcto.")
            st.stop()
        except Exception as e:
            st.error(f"Error general al preparar los datos para la predicción: {e}")
            st.stop()

        if st.button("Obtener Predicción de Daño"):
            try:
                with st.spinner("Realizando predicción..."):
                    time.sleep(1.2)  # Tiempo de loading
                    prediccion = modelo_actual.predict(final_input_array)
                
                st.success("Predicción realizada correctamente ✅")
                
                grados_dano = {0: "Bajo", 1: "Alto"}
                resultado_amigable = grados_dano.get(prediccion[0], f"Grado Desconocido: {prediccion[0]}")

                st.markdown(f"La predicción del **nivel de daño** es: **{resultado_amigable}**")

                if hasattr(modelo_actual, 'predict_proba'):
                    proba = modelo_actual.predict_proba(final_input_array)[0]
                    st.metric("Probabilidad de Daño Alto", f"{proba[1]*100:.2f}%")
                    st.metric("Probabilidad de Daño Bajo", f"{proba[0]*100:.2f}%")

            except Exception as e:
                st.error(f"Ocurrió un error al intentar predecir: {e}")
                st.info("Asegúrate de que los datos de entrada preprocesados coincidan con el número y tipo de características que el modelo espera.")
                st.info(f"Forma del input enviado al modelo: {final_input_array.shape}")

    else:
        st.warning("No se pudo cargar los valores únicos. Los widgets de entrada no son dinámicos y se usarán placeholders. No se podrá realizar predicciones.")

        st.slider("Número de Pisos (Ejemplo):", min_value=1, max_value=8, value=3)
        st.slider("Antigüedad del Edificio (Ejemplo):", min_value=0, max_value=60, value=20)
        st.slider("Área del Zócalo (Ejemplo):", min_value=0, max_value=1000, value=300)
        st.slider("Altura del Edificio (Ejemplo):", min_value=0, max_value=80, value=15)
        st.selectbox("Condición del Terreno (Ejemplo):", options=['Flat', 'Moderate slope', 'Steep slope'])
        st.selectbox("Tipo de Cimentación (Ejemplo):", options=['Other', 'Mud mortar-Stone/Brick', 'Cement-Stone/Brick', 'Bamboo/Timber', 'RC'])
        st.selectbox("Tipo de Techo (Ejemplo):", options=['Bamboo/Timber-Light roof', 'Bamboo/Timber-Heavy roof', 'RCC/RB/RBC'])
        st.selectbox("Tipo de Planta Baja (Ejemplo):", options=['Mud', 'Brick/Stone', 'RC', 'Timber', 'Other'])
        for col_name in ['Adobe Mud', 'Mud Mortar Stone', 'Stone Flag', 'Cement Mortar Stone', 'Mud Mortar Brick', 'Cement Mortar Brick', 'Timber', 'Bamboo', 'RC Non Engineered', 'RC Engineered', 'Other']:
            st.checkbox(f"Tiene superestructura de {col_name} (Ejemplo)", value=False)
        st.number_input("Número de Familias (Ejemplo):", min_value=0, max_value=10, value=1)

# --- Pestaña 2: Batch CSV ---
with tab2:
    st.header("Predicción Masiva desde CSV")
    st.write("Sube un archivo CSV con la misma estructura de columnas que el modelo espera:")
    uploaded_file = st.file_uploader("Selecciona tu CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Se cargaron {len(df)} registros.")
            
            # ——— Preprocesamiento ———
            # 1. Mapeo de categorías a índices
            # (Usar los mismos mapas que en individual)
            def map_categorical(col_name, series):
                opciones = sorted([x for x in unique_values_data[col_name] if x is not None])
                m = {val: i for i, val in enumerate(opciones)}
                return series.map(m).fillna(-1).astype(int)
            
            # Aplica mapeo a cada variable categórica
            df["land_surface_condition"] = map_categorical("land_surface_condition", df["land_surface_condition"])
            df["foundation_type"]         = map_categorical("foundation_type", df["foundation_type"])
            df["roof_type"]               = map_categorical("roof_type", df["roof_type"])
            df["ground_floor_type"]       = map_categorical("ground_floor_type", df["ground_floor_type"])
            
            # 2. Asegurarse de tener todas las columnas binarias
            binary_cols = [
                'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
                'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
                'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
                'has_superstructure_timber', 'has_superstructure_bamboo',
                'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
                'has_superstructure_other'
            ]
            for col in binary_cols:
                if col not in df:
                    df[col] = 0  # o NaN según prefieras

            # 3. Asegurar orden de columnas
            expected_features = [
                'count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq',
                'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type'
            ] + binary_cols + ['count_families']

            X = df[expected_features].values

            # ——— Predict ———
            preds = modelo_actual.predict(X)
            proba = modelo_actual.predict_proba(X)

            # ——— Construir tabla de resultados ———
            resultados = df.copy()
            resultados["pred_damage_grade"] = preds
            resultados["pred_damage_label"] = resultados["pred_damage_grade"].map({0:"Bajo",1:"Alto"})
            resultados["prob_bajo_%"] = (proba[:,0]*100).round(2)
            resultados["prob_alto_%"] = (proba[:,1]*100).round(2)

            # Mostrar tabla interactiva
            st.dataframe(resultados[[
                *expected_features,
                "pred_damage_label", "prob_bajo_%", "prob_alto_%"
            ]])

            # Opción para descargar resultados
            csv = resultados.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar resultados como CSV", csv, "predicciones.csv", "text/csv")

        except Exception as e:
            st.error(f"Error procesando el CSV: {e}")


# --- Información Adicional en la barra lateral ---
st.sidebar.markdown("---")
st.sidebar.info("**Analisis con Machine Learning**\n\nMaestria en Ingenieria de Informacion\nUniversidad de los Andes")

st.sidebar.title("Equipo de Trabajo")

# --- Integrantes / equipo de trabajo ---

st.sidebar.info("👨🏻‍💻 **Juan Velasquez**\n\nData Scientist")
st.sidebar.info("👨🏻‍💻 **Diego Rodríguez**\n\nData Engineer")
st.sidebar.info("👨🏻‍💻 **Pablo Diaz**\n\n Data Engineer")
st.sidebar.info("👨🏻‍💻 **Eduardo Arcos**\n\n Data & ML Architect")
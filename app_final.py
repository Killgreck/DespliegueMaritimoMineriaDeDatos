# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para Predicción de Tráfico Portuario
Modelo: Gradient Boosting para clasificar tráfico público vs privado
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Configuración de la página
st.set_page_config(
    page_title="Predicción Tráfico Portuario",
    page_icon="🚢",
    layout="wide"
)

# Título principal
st.title('🚢 Predicción de Tráfico Portuario Marítimo en Colombia')
st.markdown("### Clasificación: Tráfico Público vs Privado")

# Función para cargar el modelo
@st.cache_resource
def load_model():
    try:
        # Cargar el modelo Gradient Boosting
        model = joblib.load('campeon_secuencial_Gradient_Boosting.joblib')
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

# Función para obtener las categorías exactas del entrenamiento
@st.cache_data
def get_training_categories():
    
    categories = {
        'zona_portuaria': [
            'CARTAGENA', 'BARRANQUILLA', 'BUENAVENTURA', 'SANTA MARTA', 
            'GUAJIRA', 'G. MORROSQUILLO', 'SAN ANDRES', 'TURBO', 
            'BARRANCABERMEJA', 'Z.P. RIO MAGDALENA', 'CIENAGA', 'TUMACO'
        ],
        'tipo_carga': [
            'GRANEL LIQUIDO', 'GENERAL', 'GRANEL SOLIDO DIFER. DE CARBON', 
            'CONTENEDORES', 'CARBON AL GRANEL'
        ],
        'sociedad_portuaria': [
            'COMPA�IA DE PUERTOS ASOCIADOS S.A.',
            'SOCIEDAD PORTUARIA REGIONAL DE SANTA MARTA SA',
            'SOCIEDAD PORTUARIA REGIONAL DE BARRANQUILLA S.A.',
            'SOCIEDAD PORTUARIA REGIONAL DE BUENAVENTURA S.A',
            'PALERMO SOCIEDAD PORTUARIA S.A',
            'PUERTO DE MAMONAL S.A.',
            'CERREJON ZONA NORTE S.A.',
            'SOCIEDAD PUERTO INDUSTRIAL AGUADULCE S.A.',
            'SOCIEDAD PORTUARIA PUERTO BAHIA',
            'SAN ANDRES PORT SOCIETY SA'
        ]
    }
    return categories

# Función para preprocesar datos exactamente como en el entrenamiento
def preprocess_data_exact(data):
   
    try:
        # Hacer una copia de los datos
        df_processed = data.copy()

    

        # 2. VARIABLES CATEGÓRICAS - Convertir a dummies (como en el entrenamiento)
        categorical_columns = ['zona_portuaria', 'sociedad_portuaria', 'tipo_carga']

        # Crear dummies para cada variable categórica
        df_dummies = pd.get_dummies(df_processed[categorical_columns], 
                                   prefix=categorical_columns, 
                                   drop_first=False)

        # 3. VARIABLES NUMÉRICAS - Normalizar (como en el entrenamiento)
        # Según el notebook, las variables numéricas finales son:
        numerical_columns = [
            'exportacion', 'importacion', 'transbordo', 'transito_internacional',
            'fluvial', 'cabotaje', 'movilizaciones_a_bordo', 'mes_vigencia'
        ]

        df_numerical = df_processed[numerical_columns].copy()

        # Aplicar normalización con StandardScaler
        scaler = StandardScaler()
        df_numerical_scaled = pd.DataFrame(
            scaler.fit_transform(df_numerical),
            columns=numerical_columns,
            index=df_numerical.index
        )

        # 4. COMBINAR datos numéricos normalizados con dummies
        df_final = pd.concat([df_numerical_scaled, df_dummies], axis=1)

        return df_final, scaler

    except Exception as e:
        st.error(f"Error en preprocesamiento: {e}")
        return None, None

# Cargar modelo
model = load_model()

if model is not None:
    st.success("✅ Modelo cargado exitosamente")

    # Sidebar para información
    st.sidebar.header("ℹ️ Información del Modelo")
    st.sidebar.info("""
    **Modelo:** Gradient Boosting Classifier

    **Preprocesamiento (exacto del entrenamiento):**
    - Variables eliminadas: transitoria, anno_vigencia
    - Variables numéricas: Normalizadas (StandardScaler)
    - Variables categóricas: One-Hot Encoding (dummies)

    **Variables finales:** 8 numéricas + dummies categóricas
    """)

    # Obtener categorías del entrenamiento
    categories = get_training_categories()

    # Crear formulario de entrada
    st.header("📊 Ingrese los datos para la predicción")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Información Portuaria")

        # Zona portuaria (categorías exactas del entrenamiento)
        zona_portuaria = st.selectbox(
            'Zona Portuaria',
            categories['zona_portuaria'],
            help="Seleccione la zona donde se encuentra el puerto"
        )

        # Sociedad portuaria (top 10 del entrenamiento)
        sociedad_portuaria = st.selectbox(
            'Sociedad Portuaria',
            categories['sociedad_portuaria'],
            help="Seleccione la sociedad portuaria operadora"
        )

        # Tipo de carga (categorías exactas del entrenamiento)
        tipo_carga = st.selectbox(
            'Tipo de Carga',
            categories['tipo_carga'],
            help="Seleccione el tipo de carga a movilizar"
        )

    with col2:
        st.subheader("Volúmenes de Tráfico (Toneladas)")

        # Variables numéricas según el entrenamiento
        exportacion = st.number_input(
            'Exportación',
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Volumen de exportación en toneladas"
        )

        importacion = st.number_input(
            'Importación',
            min_value=0.0,
            value=4669.0,  # Mediana del entrenamiento
            step=1000.0,
            help="Volumen de importación en toneladas"
        )

        transbordo = st.number_input(
            'Transbordo',
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Volumen de transbordo en toneladas"
        )

        transito_internacional = st.number_input(
            'Tránsito Internacional',
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Volumen de tránsito internacional en toneladas"
        )

    # Más variables numéricas
    st.subheader("📦 Otros Volúmenes")
    col3, col4, col5 = st.columns(3)

    with col3:
        fluvial = st.number_input(
            'Fluvial',
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Volumen fluvial en toneladas"
        )

    with col4:
        cabotaje = st.number_input(
            'Cabotaje',
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Volumen de cabotaje en toneladas"
        )

    with col5:
        movilizaciones_a_bordo = st.number_input(
            'Movilizaciones a Bordo',
            min_value=0.0,
            value=0.0,
            step=10.0,
            help="Movilizaciones a bordo"
        )

    # Información temporal
    st.subheader("📅 Información Temporal")
    mes_vigencia = st.selectbox(
        'Mes',
        list(range(1, 13)),
        index=5,  # Junio (mediana del entrenamiento)
        help="Mes de la operación"
    )

    # Mostrar información de preprocesamiento
    with st.expander("🔧 Ver detalles del preprocesamiento (exacto del entrenamiento)"):
        st.write("**Variables numéricas que serán normalizadas:**")
        st.write("- exportacion, importacion, transbordo, transito_internacional")
        st.write("- fluvial, cabotaje, movilizaciones_a_bordo, mes_vigencia")
        st.write("**Variables categóricas convertidas a dummies:**")
        st.write("- zona_portuaria, sociedad_portuaria, tipo_carga")
        st.write("**Variables eliminadas (como en entrenamiento):**")
        st.write("- transitoria, anno_vigencia")

    # Botón de predicción
    if st.button('🔮 Realizar Predicción', type="primary"):
        try:
            # Crear DataFrame con los datos ingresados
            datos_entrada = {
                'zona_portuaria': [zona_portuaria],
                'sociedad_portuaria': [sociedad_portuaria],
                'tipo_carga': [tipo_carga],
                'exportacion': [exportacion],
                'importacion': [importacion],
                'transbordo': [transbordo],
                'transito_internacional': [transito_internacional],
                'fluvial': [fluvial],
                'cabotaje': [cabotaje],
                'movilizaciones_a_bordo': [movilizaciones_a_bordo],
                'mes_vigencia': [mes_vigencia]
            }

            df_entrada = pd.DataFrame(datos_entrada)

            # Mostrar datos originales
            st.subheader("📋 Datos Ingresados")
            st.dataframe(df_entrada)

            # Aplicar preprocesamiento exacto del entrenamiento
            df_procesado, scaler_usado = preprocess_data_exact(df_entrada)

            if df_procesado is not None:
                # Mostrar datos preprocesados
                with st.expander("🔍 Ver datos preprocesados (exacto como entrenamiento)"):
                    st.write("**Datos después del preprocesamiento:**")
                    st.dataframe(df_procesado)
                    st.write(f"**Forma de los datos:** {df_procesado.shape}")
                    st.write(f"**Columnas generadas:** {list(df_procesado.columns)}")

                # Realizar predicción
                prediccion = model.predict(df_procesado)[0]
                probabilidades = model.predict_proba(df_procesado)[0]

                # Mostrar resultados
                st.header("🎯 Resultados de la Predicción")

                col5, col6 = st.columns(2)

                with col5:
                    if prediccion == 1:
                        st.success("### 🏛️ TRÁFICO PÚBLICO")
                        probabilidad_publico = probabilidades[1]
                        st.info(f"**Probabilidad:** {probabilidad_publico:.2%}")
                        st.progress(probabilidad_publico)
                    else:
                        st.warning("### 🏢 TRÁFICO PRIVADO")
                        probabilidad_privado = probabilidades[0]
                        st.info(f"**Probabilidad:** {probabilidad_privado:.2%}")
                        st.progress(probabilidad_privado)

                with col6:
                    st.subheader("📊 Distribución de Probabilidades")
                    prob_df = pd.DataFrame({
                        'Tipo': ['Privado', 'Público'],
                        'Probabilidad': [probabilidades[0], probabilidades[1]]
                    })
                    st.bar_chart(prob_df.set_index('Tipo'))

                # Información adicional
                st.subheader("📈 Análisis Adicional")

                # Mostrar todas las probabilidades
                st.write("**Probabilidades detalladas:**")
                col7, col8 = st.columns(2)
                with col7:
                    st.metric("Tráfico Privado", f"{probabilidades[0]:.4f}", f"{probabilidades[0]:.2%}")
                with col8:
                    st.metric("Tráfico Público", f"{probabilidades[1]:.4f}", f"{probabilidades[1]:.2%}")

                # Información contextual
                st.info("""
                **Interpretación:**
                - **Tráfico Público:** Operado por entidades estatales o con participación del Estado
                - **Tráfico Privado:** Operado por empresas privadas

                **Distribución en entrenamiento:**
                - Público: 6,830 registros (82.3%)
                - Privado: 1,467 registros (17.7%)

                Esta predicción puede ayudar en la planificación de recursos y regulaciones portuarias.
                """)

        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            st.info("Verifique que el modelo esté correctamente cargado y los datos sean válidos.")

else:
    st.error("❌ No se pudo cargar el modelo. Verifique que el archivo 'campeon_secuencial_Gradient_Boosting.joblib' esté en el directorio.")
    st.info("📁 Asegúrese de que el archivo del modelo esté en la misma carpeta que esta aplicación.")

# Footer
st.markdown("---")
st.markdown("**Desarrollado para la predicción de tráfico portuario marítimo en Colombia**")
st.markdown("*Modelo basado en Gradient Boosting Classifier con preprocesamiento exacto del entrenamiento*")
st.markdown("*Basado en metodología CRISP-DM*")

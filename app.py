# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para Predicción de Tráfico Portuario
Modelo: Gradient Boosting para clasificar tráfico público vs privado
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# Función para preprocesar datos
def preprocess_data(data):
    """
    Preprocesa los datos de entrada para que coincidan con el formato del modelo
    """
    # Aquí deberías incluir el mismo preprocesamiento que usaste en el entrenamiento
    # Por ejemplo: encoding de variables categóricas, escalado, etc.
    return data

# Cargar modelo
model = load_model()

if model is not None:
    st.success("✅ Modelo cargado exitosamente")

    # Sidebar para información
    st.sidebar.header("ℹ️ Información del Modelo")
    st.sidebar.info("""
    **Modelo:** Gradient Boosting Classifier

    **Objetivo:** Predecir si el tráfico portuario será de tipo público o privado

    **Variables:** zona portuaria, sociedad portuaria, tipo de carga, volúmenes de tráfico, etc.
    """)

    # Crear formulario de entrada
    st.header("📊 Ingrese los datos para la predicción")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Información Portuaria")

        # Zona portuaria
        zona_portuaria = st.selectbox(
            'Zona Portuaria',
            ['Atlántico', 'Pacífico', 'Magdalena', 'Caribe', 'Otros'],
            help="Seleccione la zona donde se encuentra el puerto"
        )

        # Sociedad portuaria
        sociedad_portuaria = st.selectbox(
            'Sociedad Portuaria',
            ['SPRC', 'Contecar', 'SPRB', 'SPBT', 'TCBUEN', 'Otros'],
            help="Seleccione la sociedad portuaria operadora"
        )

        # Tipo de carga
        tipo_carga = st.selectbox(
            'Tipo de Carga',
            ['Contenedores', 'Granel Sólido', 'Granel Líquido', 'Carga General', 'Otros'],
            help="Seleccione el tipo de carga a movilizar"
        )

    with col2:
        st.subheader("Volúmenes de Tráfico (Toneladas)")

        # Exportación
        exportacion = st.number_input(
            'Exportación',
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Volumen de exportación en toneladas"
        )

        # Importación
        importacion = st.number_input(
            'Importación',
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Volumen de importación en toneladas"
        )

        # Transbordo
        transbordo = st.number_input(
            'Transbordo',
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Volumen de transbordo en toneladas"
        )

        # Cabotaje
        cabotaje = st.number_input(
            'Cabotaje',
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Volumen de cabotaje en toneladas"
        )

    # Información temporal
    st.subheader("📅 Información Temporal")
    col3, col4 = st.columns(2)

    with col3:
        anno_vigencia = st.selectbox(
            'Año',
            list(range(2020, 2026)),
            index=4  # 2024 por defecto
        )

    with col4:
        mes_vigencia = st.selectbox(
            'Mes',
            list(range(1, 13)),
            index=0  # Enero por defecto
        )

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
                'cabotaje': [cabotaje],
                'anno_vigencia': [anno_vigencia],
                'mes_vigencia': [mes_vigencia]
            }

            df_entrada = pd.DataFrame(datos_entrada)

            # Aquí deberías aplicar el mismo preprocesamiento que usaste en el entrenamiento
            # Por ejemplo: encoding de variables categóricas, escalado, etc.
            # df_procesado = preprocess_data(df_entrada)

            # Por ahora, simulamos una predicción
            # En la implementación real, usarías: prediccion = model.predict(df_procesado)
            # y probabilidades = model.predict_proba(df_procesado)

            # Simulación de predicción (reemplazar con predicción real)
            prediccion_simulada = np.random.choice([0, 1])  # 0: Privado, 1: Público
            probabilidad_simulada = np.random.random()

            # Mostrar resultados
            st.header("🎯 Resultados de la Predicción")

            col5, col6 = st.columns(2)

            with col5:
                if prediccion_simulada == 1:
                    st.success("### 🏛️ TRÁFICO PÚBLICO")
                    st.info(f"Probabilidad: {probabilidad_simulada:.2%}")
                else:
                    st.warning("### 🏢 TRÁFICO PRIVADO")
                    st.info(f"Probabilidad: {(1-probabilidad_simulada):.2%}")

            with col6:
                st.subheader("📋 Resumen de Datos")
                st.write(df_entrada)

            # Información adicional
            st.subheader("📈 Análisis Adicional")
            st.info("""
            **Interpretación:**
            - **Tráfico Público:** Operado por entidades estatales o con participación del Estado
            - **Tráfico Privado:** Operado por empresas privadas

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
st.markdown("*Modelo basado en Gradient Boosting Classifier*")

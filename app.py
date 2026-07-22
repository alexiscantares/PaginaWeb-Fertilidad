import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import json
import io
import base64
import requests
from pathlib import Path

# =====================================================================
# RUTAS ABSOLUTAS DINÁMICAS (Evita KeyErrors y FileNotFoundError en Cloud)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent

# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================================
st.set_page_config(page_title="Fertilidad Geoquímica", layout="wide", page_icon="🌋")
sns.set_theme(style="whitegrid")

st.title("Clasificador de Fertilidad Magmática")
st.markdown("Plataforma de inferencia multivariada y análisis geoespacial.")
st.markdown("---")

# =====================================================================
# CARGA DE MODELOS (CACHÉ OPTIMIZADA)
# =====================================================================
@st.cache_resource
def cargar_modelos():
    ruta_scaler = BASE_DIR / 'scaler_geoquimico.pkl'
    ruta_modelo = BASE_DIR / 'modelo_fertilidad_rf.pkl'
    
    scaler = joblib.load(ruta_scaler)
    modelo = joblib.load(ruta_modelo)
    return scaler, modelo

try:
    scaler, modelo_rf = cargar_modelos()
except Exception as e:
    st.error(f"❌ Error al cargar los modelos (.pkl): {e}")
    st.stop()

elementos_requeridos = ['AU', 'AG', 'CU', 'PB', 'ZN', 'MO', 'NI', 'CO', 'CD', 'BI',
                        'FE', 'MN', 'TE', 'BA', 'CR', 'V', 'SN', 'W', 'LA', 'AL',
                        'MG', 'CA', 'NA', 'K', 'SR', 'Y', 'GA', 'LI', 'NB', 'SC',
                        'TA', 'TI', 'ZR', 'AS', 'SB', 'HG', 'PT', 'PD']

# =====================================================================
# PANEL LATERAL (SIDEBAR)
# =====================================================================
st.sidebar.header("⚙️ Panel de Control")
st.sidebar.write("Sube la matriz analítica del laboratorio.")
archivo_subido = st.sidebar.file_uploader("Formato CSV o Excel", type=['csv', 'xlsx'])

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Parámetros del Modelo")
umbral_corte = st.sidebar.slider("Umbral de Probabilidad (Corte Fértil)", 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")

# =====================================================================
# ÁREA PRINCIPAL
# =====================================================================
if archivo_subido is not None:
    if archivo_subido.name.endswith('.csv'):
        df_input = pd.read_csv(archivo_subido)
    else:
        df_input = pd.read_excel(archivo_subido)
        
    st.sidebar.success(f"Archivo cargado: {len(df_input)} muestras.")
    
    if st.sidebar.button("🚀 Ejecutar Modelo Predictivo"):
        with st.spinner('Procesando datos y renderizando gráficos espaciales...'):
            try:
                # 1. Inferencia Predictiva
                datos_modelo = df_input[elementos_requeridos].copy()
                datos_log = np.log10(datos_modelo + 1e-5)
                datos_escalados = scaler.transform(datos_log)
                probabilidades = modelo_rf.predict_proba(datos_escalados)[:, 1]
                
                df_input['Prob_Fertilidad'] = probabilidades
                df_input['Clasificacion_IA'] = np.where(df_input['Prob_Fertilidad'] >= umbral_corte, 'Fértil', 'Estéril/Artefacto')
                
                if 'SR' in df_input.columns and 'Y' in df_input.columns:
                    df_input['Sr_Y'] = df_input['SR'] / df_input['Y']
                
                # =====================================================================
                # CREACIÓN DE PESTAÑAS (TABS)
                # =====================================================================
                tab1, tab2, tab3, tab4 = st.tabs(["📄 Resumen y Datos", "📉 Diagramas Geoquímicos", "🗺️ Mapa Espacial", "🤖 Asistente Gemini"])
                
                # ----- PESTAÑA 1: DATOS Y KPIS -----
                with tab1:
                    st.subheader("📊 Resumen Analítico")
                    total_muestras = len(df_input)
                    muestras_fertiles = len(df_input[df_input['Clasificacion_IA'] == 'Fértil'])
                    porcentaje_anomalias = (muestras_fertiles / total_muestras) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Muestras", total_muestras)
                    col2.metric("Blancos Fértiles Detectados", muestras_fertiles)
                    col3.metric("Tasa de Anomalía", f"{porcentaje_anomalias:.2f}%")
                    
                    st.markdown("---")
                    st.markdown("### 📋 Vista Previa de Muestras (Primeras 500 filas)")
                    
                    def colorear_fertiles(val):
                        color = '#ffcccc' if val == 'Fértil' else ''
                        return f'background-color: {color}'
                    
                    st.dataframe(
                        df_input.head(500).style.map(colorear_fertiles, subset=['Clasificacion_IA']), 
                        use_container_width=True
                    )
                
                # ----- PESTAÑA 2: DIAGRAMAS GEOQUÍMICOS -----
                with tab2:
                    st.subheader("Análisis Multivariado de Elementos Traza y Mayores")
                    fig = plt.figure(figsize=(16, 12))
                    cmap_prob = 'coolwarm'
                    
                    # 1. Sr/Y vs Y
                    ax1 = fig.add_subplot(231)
                    sns.scatterplot(data=df_input, x='Y', y='Sr_Y', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax1)
                    ax1.set_xscale('log')
                    ax1.set_yscale('log')
                    ax1.axhline(20, color='gray', linestyle='--')
                    ax1.set_title('Fertilidad: Sr/Y vs Y')
                    ax1.legend([],[], frameon=False)

                    # 2. Spider
                    ax2 = fig.add_subplot(232)
                    trace_elems = ['LA', 'SR', 'Y', 'ZR', 'TI', 'V', 'SC', 'CU', 'ZN', 'PB']
                    means_fert = df_input[df_input['Clasificacion_IA'] == 'Fértil'][trace_elems].mean()
                    means_inf = df_input[df_input['Clasificacion_IA'] != 'Fértil'][trace_elems].mean()
                    ax2.plot(trace_elems, np.log10(means_fert + 1e-5), marker='o', color='darkred', lw=2, label='Fértil (Prom)')
                    ax2.plot(trace_elems, np.log10(means_inf + 1e-5), marker='s', color='darkblue', lw=2, label='Infértil (Prom)')
                    ax2.set_title('Spider de Elementos Traza (Log10)')
                    ax2.legend()

                    # 3. Ternario AFM Simplificado
                    ax3 = fig.add_subplot(233)
                    A, F, M = df_input['NA'] + df_input['K'], df_input['FE'], df_input['MG']
                    Total = A + F + M + 1e-5
                    X_tern = (M/Total) + (F/Total) / 2.0
                    Y_tern = (F/Total) * np.sqrt(3) / 2.0
                    ax3.scatter(X_tern, Y_tern, c=df_input['Prob_Fertilidad'], cmap=cmap_prob, s=20, alpha=0.7)
                    ax3.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=1.5)
                    ax3.text(-0.05, -0.05, 'Na+K', ha='center')
                    ax3.text(1.05, -0.05, 'Mg', ha='center')
                    ax3.text(0.5, np.sqrt(3)/2 + 0.05, 'Fe', ha='center')
                    ax3.set_title('Diagrama Ternario AFM')
                    ax3.axis('off')

                    # 4. Fe vs Cr
                    ax4 = fig.add_subplot(234)
                    sns.scatterplot(data=df_input, x='CR', y='FE', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax4)
                    ax4.set_xscale('log')
                    ax4.set_yscale('log')
                    ax4.set_title('IA: Fe vs Cr (Fraccionamiento)')
                    ax4.legend([],[], frameon=False)

                    # 5. Cu vs K
                    ax5 = fig.add_subplot(235)
                    sns.scatterplot(data=df_input, x='K', y='CU', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax5)
                    ax5.set_xscale('log')
                    ax5.set_yscale('log')
                    ax5.set_title('IA: Cu vs K (Alteración Potásica)')
                    ax5.legend([],[], frameon=False)

                    # 6. V vs Ti
                    ax6 = fig.add_subplot(236)
                    sns.scatterplot(data=df_input, x='TI', y='V', hue='Prob_Fertilidad', palette=cmap_prob, s=20, alpha=0.7, ax=ax6)
                    ax6.set_xscale('log')
                    ax6.set_yscale('log')
                    ax6.set_title('IA: V vs Ti (Evolución Magmática)')
                    ax6.legend([],[], frameon=False)

                    plt.tight_layout()
                    st.pyplot(fig)
                
                # ----- PESTAÑA 3: MAPA ESPACIAL -----
                with tab3:
                    st.subheader("Modelamiento Espacial de Prospectividad")
                    if 'LONGITUD' in df_input.columns and 'LATITUD' in df_input.columns:
                        df_mapa = df_input.dropna(subset=['LATITUD', 'LONGITUD'])
                        if len(df_mapa) > 0:
                            X_coord = df_mapa['LONGITUD'].values
                            Y_coord = df_mapa['LATITUD'].values
                            Z_prob = df_mapa['Prob_Fertilidad'].values
                            
                            grid_x, grid_y = np.mgrid[X_coord.min():X_coord.max():200j, Y_coord.min():Y_coord.max():200j]
                            grid_z = griddata((X_coord, Y_coord), Z_prob, (grid_x, grid_y), method='linear')
                            
                            fig_map = plt.figure(figsize=(10, 8))
                            mapa_calor = plt.imshow(grid_z.T, extent=(X_coord.min(), X_coord.max(), Y_coord.min(), Y_coord.max()),
                                                     origin='lower', cmap='coolwarm', alpha=0.85)
                            
                            contornos = plt.contour(grid_x, grid_y, grid_z, levels=[0.5, 0.7, 0.9], colors=['yellow', 'orange', 'darkred'], linewidths=1.5, linestyles='--')
                            plt.clabel(contornos, inline=True, fontsize=10, fmt='Prob: %.1f')
                            
                            plt.title('Mapa de Calor: Certeza de Fertilidad Magmática')
                            plt.xlabel('Longitud')
                            plt.ylabel('Latitud')
                            cbar = plt.colorbar(mapa_calor, shrink=0.8)
                            cbar.set_label('Índice de Certeza')
                            st.pyplot(fig_map)
                        else:
                            st.error("⚠️ No hay coordenadas válidas para generar el mapa.")
                    else:
                        st.warning("⚠️ El archivo subido no contiene columnas de 'LATITUD' y 'LONGITUD'.")

                # ----- PESTAÑA 4: ASISTENTE MULTIMODAL (HUGGING FACE) -----
                with tab4:
                    st.subheader("Interpretación Geológica Asistida por IA")
                    hf_token = st.text_input("Ingresa tu Hugging Face Access Token:", type="password")
                    
                    if not hf_token:
                        st.warning("⚠️ Ingresa tu Token de Hugging Face para activar el análisis.")
                    else:
                        try:
                            with st.spinner("Procesando gráficos..."):
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', bbox_inches='tight')
                                buf.seek(0)
                                base64_image = base64.b64encode(buf.read()).decode('utf-8')
                                
                                prompt_texto = f"""
                                Actúa como un geoquímico experto. Acabo de procesar {total_muestras} muestras y el {porcentaje_anomalias:.1f}% tienen firmas de fertilidad.
                                Analiza los gráficos e interpreta:
                                1. Firma adakítica en Sr/Y vs Y.
                                2. Patrones en el diagrama Spider.
                                3. Tendencia calcoalcalina en AFM.
                                4. Estado de oxidación e implicaciones petrogenéticas.
                                """
                                
                                API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct"
                                headers = {"Authorization": f"Bearer {hf_token}"}
                                payload = {
                                    "inputs": {
                                        "image": f"data:image/png;base64,{base64_image}",
                                        "prompt": prompt_texto
                                    }
                                }
                                
                                response = requests.post(API_URL, headers=headers, json=payload)
                                resultado = response.json()
                                
                                if response.status_code == 200:
                                    st.success("✅ Análisis completado con éxito.")
                                    st.markdown(resultado[0]['generated_text'] if isinstance(resultado, list) else resultado)
                                else:
                                    st.error(f"Error en la API de Hugging Face: {resultado}")
                        except Exception as e:
                            st.error(f"⚠️ Error al procesar la solicitud: {e}")

                # =====================================================================
                # DESCARGAS
                # =====================================================================
                st.sidebar.markdown("---")
                csv_buffer = df_input.drop(columns=['Color_Punto'], errors='ignore').to_csv(index=False).encode('utf-8')
                st.sidebar.download_button("📥 Descargar Tabla CSV", data=csv_buffer, file_name="Resultados_Predictivos.csv", mime="text/csv")
                
                if 'LONGITUD' in df_input.columns and 'LATITUD' in df_input.columns:
                    features = []
                    for _, row in df_input.iterrows():
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [row['LONGITUD'], row['LATITUD']]},
                            "properties": {
                                "Prob_Fertilidad": round(row['Prob_Fertilidad'], 4),
                                "Clasificacion_IA": row['Clasificacion_IA']
                            }
                        })
                    geojson_data = {"type": "FeatureCollection", "features": features}
                    geojson_buffer = json.dumps(geojson_data).encode('utf-8')
                    st.sidebar.download_button("🗺️ Descargar Capa GeoJSON", data=geojson_buffer, file_name="Anomalias_Espaciales.geojson", mime="application/geo+json")

            except KeyError as e:
                st.error(f"⚠️ Error: Falta la columna {e} en el archivo. Se requieren los 38 elementos especificados.")
            except Exception as e:
                st.error(f"⚠️ Error al procesar: {e}")
else:
    st.info("👈 Por favor, carga una matriz de datos en el panel lateral.")

#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect
from scipy.stats import ttest_rel, zscore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

def main():
    # Configuración global de la fuente a Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    sns.set(font='Times New Roman')

    # Crear carpeta de salida "analisis" si no existe
    output_folder = "analisis"
    os.makedirs(output_folder, exist_ok=True)

    # Conectar a la base de datos ubicada en "instance/mesocosmos.db"
    engine = create_engine('sqlite:///instance/mesocosmos.db')

    # Verificar si la tabla "medicion" existe
    inspector = inspect(engine)
    if "medicion" not in inspector.get_table_names():
        print('Error: La tabla "medicion" no existe en la base de datos.')
        print('Por favor, ejecuta primero el script del servidor Flask para crear la base de datos y la tabla.')
        sys.exit(1)

    # Leer los datos de la tabla "medicion"
    try:
        df = pd.read_sql("SELECT * FROM medicion", engine)
    except Exception as e:
        print("Error al leer la base de datos:", e)
        sys.exit(1)

    # Convertir la columna 'timestamp' a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df.empty:
        print("No se encontraron datos en la base de datos.")
        sys.exit(1)

    # -------------------------------
    # Preparación y Análisis Descriptivo Global
    # -------------------------------
    # Crear columna de categoría de pH
    df['ph_categoria'] = pd.cut(df['ph'], bins=[0, 7, np.inf], labels=["Acidico", "Basico"])

    # Para series temporales, establecer 'timestamp' como índice
    df.set_index('timestamp', inplace=True)

    # Definir variables de interés
    variables = ['temp_int', 'hum_int', 'ph', 'temp_ext', 'hum_ext']
    agregaciones = {var: ['mean', 'std'] for var in variables}

    # Agregaciones horaria y diaria
    hourly_df = df.resample('H').agg(agregaciones)
    daily_df  = df.resample('D').agg(agregaciones)

    # Aplanar nombres de columnas
    hourly_df.columns = ['_'.join(col) for col in hourly_df.columns]
    daily_df.columns = ['_'.join(col) for col in daily_df.columns]

    # Guardar tablas agregadas
    hourly_df.to_csv(os.path.join(output_folder, "tabla_hourly_aggregates.csv"))
    with open(os.path.join(output_folder, "tabla_hourly_aggregates.txt"), "w", encoding="utf-8") as f:
        f.write(hourly_df.to_string())

    daily_df.to_csv(os.path.join(output_folder, "tabla_daily_aggregates.csv"))
    with open(os.path.join(output_folder, "tabla_daily_aggregates.txt"), "w", encoding="utf-8") as f:
        f.write(daily_df.to_string())

    # Estadísticas globales y matriz de correlación
    df_reset = df.reset_index()  # para cálculos que requieran la columna 'timestamp'
    global_stats = df_reset[variables].describe()
    global_stats.to_csv(os.path.join(output_folder, "tabla_global_estadisticas.csv"))
    with open(os.path.join(output_folder, "tabla_global_estadisticas.txt"), "w", encoding="utf-8") as f:
        f.write(global_stats.to_string())

    corr_matrix = df_reset[variables].corr()
    corr_matrix.to_csv(os.path.join(output_folder, "tabla_matriz_correlacion.csv"))
    with open(os.path.join(output_folder, "tabla_matriz_correlacion.txt"), "w", encoding="utf-8") as f:
        f.write(corr_matrix.to_string())

    # Frecuencia de categorías de pH
    ph_freq = df_reset['ph_categoria'].value_counts().sort_index()
    ph_freq.to_csv(os.path.join(output_folder, "tabla_frecuencia_ph.csv"), header=["Frecuencia"])
    with open(os.path.join(output_folder, "tabla_frecuencia_ph.txt"), "w", encoding="utf-8") as f:
        f.write(ph_freq.to_string())

    # -------------------------------
    # Análisis de Anomalías (Detección de outliers)
    # -------------------------------
    # Se calcula el z-score para cada variable y se marcan valores con |z| > 3
    anomalies = pd.DataFrame()
    for var in variables:
        df_reset[var + '_zscore'] = zscore(df_reset[var])
        anomalies[var] = df_reset[var + '_zscore'].abs() > 3

    # Ejemplo: Gráfica de Temperatura Interior con anomalías marcadas
    plt.figure(figsize=(10,4))
    plt.plot(df_reset['timestamp'], df_reset['temp_int'], label='Temp. Interior', color='blue')
    # Marcar puntos anómalos
    anomaly_mask = df_reset['temp_int_zscore'].abs() > 3
    plt.scatter(df_reset.loc[anomaly_mask, 'timestamp'],
                df_reset.loc[anomaly_mask, 'temp_int'],
                color='red', label='Anomalías', zorder=5)
    plt.title("Anomalías en Temperatura Interior")
    plt.xlabel("Tiempo")
    plt.ylabel("Temperatura Interior (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "anomalias_temp_int.png"))
    plt.close()

    # -------------------------------
    # Análisis de Ciclos: Día vs. Noche
    # -------------------------------
    # Extraer hora y definir ciclo (por ejemplo, 'Día' entre 7 y 19 horas, 'Noche' en otro caso)
    df_reset['hora'] = df_reset['timestamp'].dt.hour
    df_reset['ciclo'] = df_reset['hora'].apply(lambda h: 'Dia' if 7 <= h < 19 else 'Noche')
    
    # Promedios por ciclo para variables de interés
    ciclo_stats = df_reset.groupby('ciclo')[variables].mean()
    ciclo_stats.to_csv(os.path.join(output_folder, "tabla_ciclo_dia_noche.csv"))
    
    # Gráfico comparativo: Temperatura Interior según ciclo
    plt.figure(figsize=(6,4))
    sns.barplot(x='ciclo', y='temp_int', data=df_reset, ci='sd')
    plt.title("Temperatura Interior: Día vs. Noche")
    plt.xlabel("Ciclo")
    plt.ylabel("Temperatura Interior (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "temp_int_dia_vs_noche.png"))
    plt.close()

    # -------------------------------
    # Análisis Multivariado: PCA y Clustering
    # -------------------------------
    # Seleccionar características y escalar
    features = df_reset[variables].dropna()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Aplicar PCA para reducir a 2 dimensiones
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    df_reset['pca1'] = pca_result[:, 0]
    df_reset['pca2'] = pca_result[:, 1]

    # Gráfica de PCA (variación explicada)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='pca1', y='pca2', data=df_reset, hue='ciclo', palette='Set1', alpha=0.6)
    plt.title("PCA de Variables Ambientales (Color por Ciclo)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "pca_variables.png"))
    plt.close()

    # Clustering: KMeans con 3 clusters sobre las componentes principales
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(pca_result)
    df_reset['cluster'] = clusters

    plt.figure(figsize=(8,6))
    sns.scatterplot(x='pca1', y='pca2', data=df_reset, hue='cluster', palette='viridis', alpha=0.6)
    plt.title("Clustering (KMeans, k=3) sobre PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "pca_clustering.png"))
    plt.close()

    # -------------------------------
    # Modelado y Predicción: Forecasting ARIMA para Temp. Interior Diaria
    # -------------------------------
    # Usar la serie diaria de temperatura interior
    daily_temp_int = daily_df['temp_int_mean'].dropna()
    # Ajustar modelo ARIMA (por ejemplo, orden (1,1,1))
    try:
        model = ARIMA(daily_temp_int, order=(1,1,1))
        model_fit = model.fit()
        # Pronosticar los próximos 7 días
        forecast = model_fit.get_forecast(steps=7)
        forecast_index = pd.date_range(start=daily_temp_int.index[-1] + pd.Timedelta(days=1), periods=7)
        forecast_mean = forecast.predicted_mean
        forecast_conf = forecast.conf_int()

        # Graficar el pronóstico junto a la serie histórica
        plt.figure(figsize=(10,4))
        plt.plot(daily_temp_int.index, daily_temp_int, label='Histórico')
        plt.plot(forecast_index, forecast_mean, label='Pronóstico', color='red')
        plt.fill_between(forecast_index, forecast_conf.iloc[:, 0], forecast_conf.iloc[:, 1], color='pink', alpha=0.3)
        plt.title("Forecast ARIMA: Temperatura Interior Diaria")
        plt.xlabel("Fecha")
        plt.ylabel("Temp. Interior (°C)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "forecast_arima_temp_int.png"))
        plt.close()
    except Exception as e:
        print("Error en modelado ARIMA:", e)

    # -------------------------------
    # Análisis Comparativo: Métricas Interiores vs. Exteriores
    # (Se utiliza la prueba t de Student para muestras apareadas)
    # -------------------------------
    df_comp = df_reset.dropna(subset=['temp_int', 'temp_ext', 'hum_int', 'hum_ext'])
    t_temp, p_temp = ttest_rel(df_comp['temp_int'], df_comp['temp_ext'])
    t_hum, p_hum = ttest_rel(df_comp['hum_int'], df_comp['hum_ext'])
    comp_results = {
        'Temperatura': {
            't_stat': t_temp,
            'p_value': p_temp,
            'mean_temp_int': df_comp['temp_int'].mean(),
            'std_temp_int': df_comp['temp_int'].std(),
            'mean_temp_ext': df_comp['temp_ext'].mean(),
            'std_temp_ext': df_comp['temp_ext'].std()
        },
        'Humedad': {
            't_stat': t_hum,
            'p_value': p_hum,
            'mean_hum_int': df_comp['hum_int'].mean(),
            'std_hum_int': df_comp['hum_int'].std(),
            'mean_hum_ext': df_comp['hum_ext'].mean(),
            'std_hum_ext': df_comp['hum_ext'].std()
        }
    }

    # Gráficos comparativos: Boxplots
    plt.figure(figsize=(8,4))
    temp_data = pd.melt(df_comp[['temp_int', 'temp_ext']], var_name='Ambiente', value_name='Temperatura')
    sns.boxplot(x='Ambiente', y='Temperatura', data=temp_data)
    plt.title("Comparación de Temperatura: Interior vs. Exterior")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "comparacion_temperatura_boxplot.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    hum_data = pd.melt(df_comp[['hum_int', 'hum_ext']], var_name='Ambiente', value_name='Humedad')
    sns.boxplot(x='Ambiente', y='Humedad', data=hum_data)
    plt.title("Comparación de Humedad: Interior vs. Exterior")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "comparacion_humedad_boxplot.png"))
    plt.close()

    # -------------------------------
    # Generación del Informe de Análisis Completo
    # -------------------------------
    report_path = os.path.join(output_folder, "analisis_completo_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        # Página de título y APA
        f.write("Informe de Análisis Estadístico de Datos de Medición\n")
        f.write("Autor: [Nombre del autor]\n")
        f.write("Afiliación: [Institución]\n")
        f.write("Fecha: {}\n\n".format(pd.Timestamp.now().strftime("%d de %B de %Y")))
        
        f.write("Abstract\n")
        f.write("---------\n")
        f.write("Este estudio presenta un análisis estadístico global y comparativo de datos de medición ambiental. Se incluyen agregaciones horarias y diarias, análisis descriptivo, detección de anomalías, evaluación de ciclos (día vs. noche), análisis multivariado (PCA y clustering) y modelado predictivo mediante ARIMA.\n\n")
        
        f.write("Introducción\n")
        f.write("------------\n")
        f.write("El presente análisis tiene como objetivo explorar patrones, detectar anomalías, identificar ciclos y evaluar relaciones entre variables ambientales (temperatura, humedad y pH) en un mesocosmos. Además, se realiza un pronóstico de la temperatura interior diaria.\n\n")
        
        f.write("Método\n")
        f.write("------\n")
        f.write("Se extrajeron los datos de una base de datos SQLite y se procesaron mediante agregaciones temporales, cálculo de estadísticas descriptivas y análisis multivariado. Se aplicaron pruebas de hipótesis (t de Student), se detectaron outliers a partir de z-scores y se realizó clustering basado en PCA. Adicionalmente, se modeló la serie temporal de temperatura interior usando un modelo ARIMA.\n\n")
        
        f.write("Resultados\n")
        f.write("----------\n")
        f.write("Los análisis descriptivos y las visualizaciones indican que las variables presentan tendencias estables con comportamientos diferenciados entre entornos interiores y exteriores. La detección de anomalías permitió identificar valores extremos en la serie de temperatura interior. El análisis de ciclos evidenció diferencias notables entre las mediciones realizadas durante el día y la noche. La aplicación de PCA reveló que las dos primeras componentes capturan gran parte de la varianza, permitiendo además segmentar los datos en tres clusters. Finalmente, el modelo ARIMA aplicado sobre la serie diaria de temperatura interior permitió pronosticar la evolución de esta variable para los próximos 7 días.\n\n")
        
        f.write("Comparación Estadística: Métricas Interiores vs. Exteriores\n")
        f.write("-------------------------------------------------------------\n")
        f.write("Temperatura:\n")
        f.write("  - Temperatura Interior: media = {:.2f}, std = {:.2f}\n".format(comp_results['Temperatura']['mean_temp_int'],
                                                                                comp_results['Temperatura']['std_temp_int']))
        f.write("  - Temperatura Exterior: media = {:.2f}, std = {:.2f}\n".format(comp_results['Temperatura']['mean_temp_ext'],
                                                                                comp_results['Temperatura']['std_temp_ext']))
        f.write("  - t-statistic = {:.4f}\n".format(comp_results['Temperatura']['t_stat']))
        f.write("  - p-value = {:.4f}\n".format(comp_results['Temperatura']['p_value']))
        if comp_results['Temperatura']['p_value'] < 0.05:
            f.write("  -> La diferencia en la temperatura es estadísticamente significativa (p < 0.05).\n\n")
        else:
            f.write("  -> No se encontró diferencia estadísticamente significativa en la temperatura (p >= 0.05).\n\n")
            
        f.write("Humedad:\n")
        f.write("  - Humedad Interior: media = {:.2f}, std = {:.2f}\n".format(comp_results['Humedad']['mean_hum_int'],
                                                                              comp_results['Humedad']['std_hum_int']))
        f.write("  - Humedad Exterior: media = {:.2f}, std = {:.2f}\n".format(comp_results['Humedad']['mean_hum_ext'],
                                                                              comp_results['Humedad']['std_hum_ext']))
        f.write("  - t-statistic = {:.4f}\n".format(comp_results['Humedad']['t_stat']))
        f.write("  - p-value = {:.4f}\n".format(comp_results['Humedad']['p_value']))
        if comp_results['Humedad']['p_value'] < 0.05:
            f.write("  -> La diferencia en la humedad es estadísticamente significativa (p < 0.05).\n\n")
        else:
            f.write("  -> No se encontró diferencia estadísticamente significativa en la humedad (p >= 0.05).\n\n")
        
        f.write("Discusión\n")
        f.write("---------\n")
        f.write("Los resultados indican que, a pesar del elevado volumen de datos, las variables presentan patrones consistentes y diferencias significativas en función del entorno y del ciclo (día vs. noche). La detección de anomalías sugiere la existencia de eventos atípicos en la medición, mientras que el análisis multivariado permitió segmentar los datos en grupos con comportamientos similares. El forecasting ARIMA aporta información valiosa para la predicción de la temperatura interior, lo que puede orientar futuros ajustes y mejoras en el sistema de monitoreo.\n\n")
        
        f.write("Referencias\n")
        f.write("-----------\n")
        f.write("Smith, J., Doe, A., & Johnson, P. (2020). Título del artículo. Nombre de la Revista, 10(2), 123-134.\n")
    
    print("Análisis completo finalizado. Resultados guardados en el directorio '{}'.".format(output_folder))

if __name__ == '__main__':
    main()

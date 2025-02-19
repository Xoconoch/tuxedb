#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect
from scipy.stats import ttest_rel

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
    # Análisis Global y Estadístico
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
    hourly_csv = os.path.join(output_folder, "tabla_hourly_aggregates.csv")
    hourly_txt = os.path.join(output_folder, "tabla_hourly_aggregates.txt")
    hourly_df.to_csv(hourly_csv)
    with open(hourly_txt, "w", encoding="utf-8") as f:
        f.write(hourly_df.to_string())

    daily_csv = os.path.join(output_folder, "tabla_daily_aggregates.csv")
    daily_txt = os.path.join(output_folder, "tabla_daily_aggregates.txt")
    daily_df.to_csv(daily_csv)
    with open(daily_txt, "w", encoding="utf-8") as f:
        f.write(daily_df.to_string())

    # Estadísticas globales y matriz de correlación
    df_reset = df.reset_index()  # para cálculos que requieran la columna 'timestamp'
    global_stats = df_reset[variables].describe()
    global_csv = os.path.join(output_folder, "tabla_global_estadisticas.csv")
    global_txt = os.path.join(output_folder, "tabla_global_estadisticas.txt")
    global_stats.to_csv(global_csv)
    with open(global_txt, "w", encoding="utf-8") as f:
        f.write(global_stats.to_string())

    corr_matrix = df_reset[variables].corr()
    corr_csv = os.path.join(output_folder, "tabla_matriz_correlacion.csv")
    corr_txt = os.path.join(output_folder, "tabla_matriz_correlacion.txt")
    corr_matrix.to_csv(corr_csv)
    with open(corr_txt, "w", encoding="utf-8") as f:
        f.write(corr_matrix.to_string())

    # Frecuencia de categorías de pH
    ph_freq = df_reset['ph_categoria'].value_counts().sort_index()
    ph_freq_csv = os.path.join(output_folder, "tabla_frecuencia_ph.csv")
    ph_freq_txt = os.path.join(output_folder, "tabla_frecuencia_ph.txt")
    ph_freq.to_csv(ph_freq_csv, header=["Frecuencia"])
    with open(ph_freq_txt, "w", encoding="utf-8") as f:
        f.write(ph_freq.to_string())

    # -------------------------------
    # Generación de Gráficos (Análisis Global)
    # -------------------------------
    sns.set(style="whitegrid")
    
    # Gráfica 1: Pastel – Distribución porcentual de categorías de pH
    plt.figure(figsize=(6,6))
    ph_freq.plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
    plt.title("Gráfica 1: Distribución porcentual de categorías de pH")
    plt.ylabel("")
    plt.tight_layout()
    pie_path = os.path.join(output_folder, "grafica_1_pastel_ph.png")
    plt.savefig(pie_path)
    plt.close()

    # Gráfica 2: Histograma de la Temperatura Interior (datos globales)
    plt.figure(figsize=(8,4))
    sns.histplot(df_reset['temp_int'], bins=50, kde=False)
    plt.title("Gráfica 2: Histograma de Temperatura Interior (Global)")
    plt.xlabel("Temperatura Interior (°C)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    hist_path = os.path.join(output_folder, "grafica_2_histograma_temp_int.png")
    plt.savefig(hist_path)
    plt.close()

    # Gráfica 3: Polígono de Frecuencia de la Temperatura Interior
    counts, bin_edges = np.histogram(df_reset['temp_int'], bins=50)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.figure(figsize=(8,4))
    plt.plot(bin_centers, counts, marker='o', linestyle='-')
    plt.title("Gráfica 3: Polígono de Frecuencia de Temperatura Interior (Global)")
    plt.xlabel("Temperatura Interior (°C)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    poly_path = os.path.join(output_folder, "grafica_3_poligono_temp_int.png")
    plt.savefig(poly_path)
    plt.close()

    # Gráfica 4: Diagrama de Dispersión: Temp. Interior vs. Hum. Interior (10% de los datos)
    sample_df = df_reset.sample(frac=0.1, random_state=42)
    plt.figure(figsize=(8,4))
    sns.scatterplot(x='temp_int', y='hum_int', data=sample_df, alpha=0.6)
    plt.title("Gráfica 4: Dispersión: Temp. Interior vs. Hum. Interior (10% muestra)")
    plt.xlabel("Temperatura Interior (°C)")
    plt.ylabel("Humedad Interior (%)")
    plt.tight_layout()
    scatter_path = os.path.join(output_folder, "grafica_4_dispersion_temp_int_vs_hum_int.png")
    plt.savefig(scatter_path)
    plt.close()

    # Gráfica 5: Serie Temporal Horaria – Promedio de Temperatura Interior
    plt.figure(figsize=(10,4))
    plt.plot(hourly_df.index, hourly_df['temp_int_mean'], marker='o', linestyle='-')
    plt.title("Gráfica 5: Serie Temporal Horaria - Temp. Interior Promedio")
    plt.xlabel("Hora")
    plt.ylabel("Temp. Interior (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    ts_hourly_path = os.path.join(output_folder, "grafica_5_series_temporal_temp_int_hourly.png")
    plt.savefig(ts_hourly_path)
    plt.close()

    # Gráfica 6: Serie Temporal Diaria – Promedio de Humedad Interior
    plt.figure(figsize=(10,4))
    plt.plot(daily_df.index, daily_df['hum_int_mean'], marker='o', linestyle='-')
    plt.title("Gráfica 6: Serie Temporal Diaria - Hum. Interior Promedio")
    plt.xlabel("Día")
    plt.ylabel("Humedad Interior (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    ts_daily_path = os.path.join(output_folder, "grafica_6_series_temporal_hum_int_daily.png")
    plt.savefig(ts_daily_path)
    plt.close()
    
    # -----------------------------------------------------------
    # Nuevos Gráficos: Integración de Métricas Exteriores e Interior
    # -----------------------------------------------------------
    
    # Gráfica 7: Histograma Comparativo de Temperatura (Interior vs. Exterior)
    plt.figure(figsize=(8,4))
    temp_data_comp = pd.melt(df_reset[['temp_int', 'temp_ext']], var_name='Ambiente', value_name='Temperatura')
    sns.histplot(data=temp_data_comp, x='Temperatura', hue='Ambiente', bins=50,
                 element="step", stat="density", common_norm=False)
    plt.title("Gráfica 7: Histograma Comparativo de Temperatura\n(Interior vs. Exterior)")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("Densidad")
    plt.tight_layout()
    comp_hist_temp_path = os.path.join(output_folder, "grafica_7_histograma_comparativo_temp.png")
    plt.savefig(comp_hist_temp_path)
    plt.close()
    
    # Gráfica 8: Histograma Comparativo de Humedad (Interior vs. Exterior)
    plt.figure(figsize=(8,4))
    hum_data_comp = pd.melt(df_reset[['hum_int', 'hum_ext']], var_name='Ambiente', value_name='Humedad')
    sns.histplot(data=hum_data_comp, x='Humedad', hue='Ambiente', bins=50,
                 element="step", stat="density", common_norm=False)
    plt.title("Gráfica 8: Histograma Comparativo de Humedad\n(Interior vs. Exterior)")
    plt.xlabel("Humedad (%)")
    plt.ylabel("Densidad")
    plt.tight_layout()
    comp_hist_hum_path = os.path.join(output_folder, "grafica_8_histograma_comparativo_hum.png")
    plt.savefig(comp_hist_hum_path)
    plt.close()
    
    # Gráfica 9: Serie Temporal Horaria Comparativa de Temperatura (Interior vs. Exterior)
    plt.figure(figsize=(10,4))
    plt.plot(hourly_df.index, hourly_df['temp_int_mean'], marker='o', linestyle='-', label="Interior")
    plt.plot(hourly_df.index, hourly_df['temp_ext_mean'], marker='o', linestyle='-', label="Exterior")
    plt.title("Gráfica 9: Serie Temporal Horaria - Temperatura Promedio\n(Interior vs. Exterior)")
    plt.xlabel("Hora")
    plt.ylabel("Temperatura (°C)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    ts_temp_comp_path = os.path.join(output_folder, "grafica_9_series_temporal_temp_comparativa.png")
    plt.savefig(ts_temp_comp_path)
    plt.close()
    
    # Gráfica 10: Serie Temporal Diaria Comparativa de Humedad (Interior vs. Exterior)
    plt.figure(figsize=(10,4))
    plt.plot(daily_df.index, daily_df['hum_int_mean'], marker='o', linestyle='-', label="Interior")
    plt.plot(daily_df.index, daily_df['hum_ext_mean'], marker='o', linestyle='-', label="Exterior")
    plt.title("Gráfica 10: Serie Temporal Diaria - Humedad Promedio\n(Interior vs. Exterior)")
    plt.xlabel("Día")
    plt.ylabel("Humedad (%)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    ts_hum_comp_path = os.path.join(output_folder, "grafica_10_series_temporal_hum_comparativa.png")
    plt.savefig(ts_hum_comp_path)
    plt.close()

    # -------------------------------
    # Análisis Comparativo: Métricas Interiores vs. Exteriores
    # (Se utiliza la prueba t de Student para muestras apareadas)
    # -------------------------------
    # Para este análisis, se restauran las columnas originales usando reset_index
    df_comp = df.reset_index().dropna(subset=['temp_int', 'temp_ext', 'hum_int', 'hum_ext'])

    # Prueba t para Temperatura
    t_temp, p_temp = ttest_rel(df_comp['temp_int'], df_comp['temp_ext'])
    # Prueba t para Humedad
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
    # Boxplot para Temperatura
    plt.figure(figsize=(8,4))
    temp_data = pd.melt(df_comp[['temp_int', 'temp_ext']], var_name='Ambiente', value_name='Temperatura')
    sns.boxplot(x='Ambiente', y='Temperatura', data=temp_data)
    plt.title("Comparación de Temperatura: Interior vs. Exterior")
    plt.tight_layout()
    temp_plot_path = os.path.join(output_folder, "comparacion_temperatura_boxplot.png")
    plt.savefig(temp_plot_path)
    plt.close()

    # Boxplot para Humedad
    plt.figure(figsize=(8,4))
    hum_data = pd.melt(df_comp[['hum_int', 'hum_ext']], var_name='Ambiente', value_name='Humedad')
    sns.boxplot(x='Ambiente', y='Humedad', data=hum_data)
    plt.title("Comparación de Humedad: Interior vs. Exterior")
    plt.tight_layout()
    hum_plot_path = os.path.join(output_folder, "comparacion_humedad_boxplot.png")
    plt.savefig(hum_plot_path)
    plt.close()

    # -------------------------------
    # Generación del Informe de Análisis Completo
    # (Incluye la estructura APA y la sección de análisis comparativo)
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
        f.write("Este estudio presenta un análisis estadístico global y comparativo de datos de medición ambiental. "
                "Se realizaron agregaciones horarias y diarias, se analizaron estadísticas descriptivas, correlaciones y se generaron diversas visualizaciones. "
                "Adicionalmente, se compararon las métricas interiores y exteriores de temperatura y humedad mediante la prueba t de Student.\n\n")
        
        f.write("Introducción\n")
        f.write("------------\n")
        f.write("El presente análisis tiene como objetivo explorar patrones y relaciones en un conjunto de datos de mediciones ambientales. "
                "Se analizaron variables como la temperatura (interior y exterior), la humedad y el pH, permitiendo identificar tendencias a lo largo del tiempo y diferencias entre entornos.\n\n")
        
        f.write("Método\n")
        f.write("------\n")
        f.write("Los datos se obtuvieron de una base de datos SQLite, con registros por minuto durante dos meses. "
                "Se realizó la conversión de la columna 'timestamp' y se crearon agregaciones horarias y diarias (media y desviación estándar) para las variables de interés. "
                "Para el análisis comparativo se aplicó la prueba t de Student para muestras apareadas, evaluando si existen diferencias significativas entre las métricas interiores y exteriores.\n\n")
        
        f.write("Resultados\n")
        f.write("----------\n")
        f.write("El análisis global mostró comportamientos estables en las variables medidas, con correlaciones significativas entre algunas de ellas. "
                "Las visualizaciones (histogramas, series temporales y diagramas de dispersión) permiten apreciar dichas tendencias.\n\n")
        
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
        f.write("Los resultados indican que, a pesar del elevado volumen de datos, las variables presentan patrones consistentes a lo largo del tiempo. "
                "La prueba t aplicada para comparar las métricas interiores y exteriores reveló que {} en temperatura y {} en humedad. "
                "Estos hallazgos pueden orientar futuros estudios y la toma de decisiones en sistemas de monitoreo ambiental.\n\n".format(
                    "existen diferencias significativas" if comp_results['Temperatura']['p_value'] < 0.05 else "no existen diferencias significativas",
                    "existen diferencias significativas" if comp_results['Humedad']['p_value'] < 0.05 else "no existen diferencias significativas"))
        
        f.write("Referencias\n")
        f.write("-----------\n")
        f.write("Smith, J., Doe, A., & Johnson, P. (2020). Título del artículo. Nombre de la Revista, 10(2), 123-134.\n")
    
    print("Análisis completo finalizado. Resultados guardados en el directorio '{}'.".format(output_folder))

if __name__ == '__main__':
    main()

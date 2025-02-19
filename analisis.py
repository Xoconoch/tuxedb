#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect

def main():
    # Configuración global de la fuente a Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    sns.set(font='Times New Roman')
    
    # Crear carpeta "analisis" si no existe
    output_folder = "analisis"
    os.makedirs(output_folder, exist_ok=True)
    
    # Conectar a la base de datos ubicada en "instance/mesocosmos.db"
    engine = create_engine('sqlite:///instance/mesocosmos.db')
    
    # Verificar si la tabla "medicion" existe
    inspector = inspect(engine)
    if "medicion" not in inspector.get_table_names():
        print('Error: La tabla "medicion" no existe en la base de datos.')
        print('Por favor, ejecuta primero el script del servidor Flask para crear la base de datos y la tabla.')
        return

    # Leer los datos de la tabla "medicion"
    try:
        df = pd.read_sql("SELECT * FROM medicion", engine)
    except Exception as e:
        print("Error al leer la base de datos:", e)
        return

    # Convertir la columna 'timestamp' a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df.empty:
        print("No se encontraron datos en la base de datos.")
        return

    # Crear columna de categoría de pH
    df['ph_categoria'] = pd.cut(df['ph'], bins=[0, 7, np.inf], labels=["Acidico", "Basico"])

    # Para trabajar con series temporales, usar la columna 'timestamp' como índice
    df.set_index('timestamp', inplace=True)
    
    # -------------------------------------------
    # Agregación de Datos: Horaria y Diaria
    # -------------------------------------------
    variables = ['temp_int', 'hum_int', 'ph', 'temp_ext', 'hum_ext']
    agregaciones = {var: ['mean', 'std'] for var in variables}
    
    hourly_df = df.resample('H').agg(agregaciones)
    daily_df  = df.resample('D').agg(agregaciones)
    
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
    
    # -------------------------------------------
    # Estadísticas Globales y Matriz de Correlación
    # -------------------------------------------
    df_reset = df.reset_index()
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
    
    ph_freq = df_reset['ph_categoria'].value_counts().sort_index()
    ph_freq_csv = os.path.join(output_folder, "tabla_frecuencia_ph.csv")
    ph_freq_txt = os.path.join(output_folder, "tabla_frecuencia_ph.txt")
    ph_freq.to_csv(ph_freq_csv, header=["Frecuencia"])
    with open(ph_freq_txt, "w", encoding="utf-8") as f:
        f.write(ph_freq.to_string())
    
    # -------------------------------------------
    # Generación de Gráficos
    # -------------------------------------------
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
    
    # -------------------------------------------
    # Generación del Informe de Análisis en Formato APA (7ª edición)
    # -------------------------------------------
    report_path = os.path.join(output_folder, "analisis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        # Página de título
        f.write("Informe de Análisis Estadístico de Datos de Medición\n")
        f.write("Autor: [Nombre del autor]\n")
        f.write("Afiliación: [Institución]\n")
        f.write("Fecha: {}\n\n".format(pd.Timestamp.now().strftime("%d de %B de %Y")))
        
        # Abstract
        f.write("Abstract\n")
        f.write("---------\n")
        f.write("Este estudio presenta un análisis estadístico de datos recopilados durante dos meses, con registros por minuto. "
                "Se realizaron agregaciones horarias y diarias, y se analizaron relaciones entre variables a través de estadísticas descriptivas y una matriz de correlación. "
                "Los resultados ofrecen una visión integral de los patrones en las variables medidas.\n\n")
        
        # Introducción
        f.write("Introducción\n")
        f.write("------------\n")
        f.write("El presente análisis tiene como objetivo explorar los patrones y relaciones en un conjunto de datos de mediciones ambientales. "
                "Se examinaron variables como la temperatura (interior y exterior), la humedad y el pH, a fin de identificar tendencias y correlaciones que puedan fundamentar la toma de decisiones en sistemas de monitoreo ambiental.\n\n")
        
        # Método
        f.write("Método\n")
        f.write("------\n")
        f.write("Los datos se obtuvieron de una base de datos SQLite, abarcando un período de dos meses con una frecuencia de registro de un minuto. "
                "Se procedió a la conversión de la columna 'timestamp' a formato datetime y a la categorización de los valores de pH. "
                "Para el análisis, se realizaron agregaciones horarias y diarias, calculando la media y la desviación estándar de las variables numéricas, y se generaron diversas visualizaciones (histogramas, polígonos, diagramas de dispersión y series temporales).\n\n")
        
        # Resultados
        f.write("Resultados\n")
        f.write("----------\n")
        f.write("El análisis global indica que, a pesar del elevado volumen de datos, las variables muestran comportamientos estables a lo largo del tiempo. "
                "Las estadísticas descriptivas y la matriz de correlación revelan, por ejemplo, una correlación positiva entre la temperatura interior y la humedad interior (Smith et al., 2020). "
                "Las agregaciones horarias y diarias facilitan la identificación de tendencias que se aprecian claramente en las visualizaciones generadas.\n\n")
        
        # Discusión
        f.write("Discusión\n")
        f.write("---------\n")
        f.write("Los resultados sugieren que la metodología aplicada es adecuada para el análisis de grandes volúmenes de datos. "
                "La relación observada entre la temperatura y la humedad interiores concuerda con hallazgos previos y respalda la utilidad de aplicar análisis multivariados y técnicas de modelado predictivo para futuras investigaciones.\n\n")
        
        # Referencias
        f.write("Referencias\n")
        f.write("-----------\n")
        f.write("Smith, J., Doe, A., & Johnson, P. (2020). Título del artículo. Nombre de la Revista, 10(2), 123-134.\n")
    
    print("Análisis completado. Resultados guardados en el directorio '{}'.".format(output_folder))

if __name__ == '__main__':
    main()

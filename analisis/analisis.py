#!/usr/bin/env python3
"""
Script de Análisis del Experimento Mesocosmos.

Este script se conecta a una base de datos SQLite (por defecto: ./mesocosmos.db), obtiene datos de la
tabla "medicion" y realiza lo siguiente:
  - Ordena los datos cronológicamente según la columna `timestamp`.
  - Elimina registros posteriores al 2025-04-23 09:30.
  - Determina la duración total del experimento (usando los timestamps inicial y final tras el filtrado).
  - Divide los datos en dos periodos basados en una fecha fija (2025-04-02 09:00:23):
       Grupo Abierto (control): Mesocosmos abierto (registros hasta y incluyendo la fecha de corte)
       Grupo Sellado (experimental): Mesocosmos sellado (registros posteriores a la fecha de corte)
  - Hipótesis 1: Compara el pH del sustrato entre periodos mediante prueba t-independiente de Welch.
  - Hipótesis 2: Compara la temperatura y la humedad internas entre periodos mediante pruebas t-independientes.
  - Hipótesis 3: Evalúa la relación entre la humedad interna y ambiental usando:
      * Prueba de normalidad Shapiro-Wilk.
      * Correlación Pearson o Spearman.
      * Distance correlation y mutual information.
      * Ajuste polinómico iterativo, deteniéndose cuando la mejora en R² sea menor al 1 %.
  - Genera gráficos en PNG y un informe en Markdown bajo el directorio "./report".

Uso:
    python mesocosmos_analysis.py [--db RUTA_A_DB]
"""
import numba

# Desactivar caché en numba para evitar inconsistencias
_original_njit = numba.njit
def no_cache_njit(*args, **kwargs):
    kwargs['cache'] = False
    return _original_njit(*args, **kwargs)
numba.njit = no_cache_njit

import os
import sqlite3
import pandas as pd
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import dcor  # Para distance correlation
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

OUTPUT_DIR = "./report"
# Fecha fija para dividir los periodos
SPLIT_TIMESTAMP = pd.to_datetime("2025-04-02 09:00:23.726423")
# Fecha máxima para incluir datos en el análisis
MAX_TIMESTAMP = pd.to_datetime("2025-04-23 09:30:00")

def ensure_output_dir(directory=OUTPUT_DIR):
    """Crea el directorio de salida si no existe."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(db_path="./mesocosmos.db"):
    """
    Carga la tabla "medicion" desde la base SQLite, ordena por timestamp
    y filtra registros posteriores a MAX_TIMESTAMP.
    """
    try:
        conn = sqlite3.connect(db_path)
        query = (
            "SELECT id, timestamp, temp_int, hum_int, ph, temp_ext, hum_ext "
            "FROM medicion"
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print("Error al conectar o leer la base de datos:", e)
        sys.exit(1)

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
    except Exception as e:
        print("Error al procesar timestamps:", e)
        sys.exit(1)

    # Filtrar registros posteriores a la fecha máxima
    if df['timestamp'].max() > MAX_TIMESTAMP:
        cnt = (df['timestamp'] > MAX_TIMESTAMP).sum()
        print(f"Advertencia: se eliminarán {cnt} registros posteriores a {MAX_TIMESTAMP}.")
    df = df[df['timestamp'] <= MAX_TIMESTAMP]

    return df

def split_data(df):
    """
    Separa el DataFrame usando la fecha fija SPLIT_TIMESTAMP.
    Registra advertencia si SPLIT_TIMESTAMP está fuera del rango de datos.
    """
    inicio = df['timestamp'].iloc[0]
    fin = df['timestamp'].iloc[-1]
    if SPLIT_TIMESTAMP < inicio or SPLIT_TIMESTAMP > fin:
        print(f"Advertencia: la fecha de corte {SPLIT_TIMESTAMP} está fuera del rango de datos ({inicio} - {fin}).")
    control_df = df[df['timestamp'] <= SPLIT_TIMESTAMP]
    experimental_df = df[df['timestamp'] > SPLIT_TIMESTAMP]
    return control_df, experimental_df, inicio, fin

def hypothesis1_ph(control_df, experimental_df):
    mean_control = control_df['ph'].mean()
    mean_experimental = experimental_df['ph'].mean()
    t_stat, p_val = stats.ttest_ind(
        control_df['ph'].dropna(),
        experimental_df['ph'].dropna(),
        equal_var=False
    )
    conclusion = (
        "Rechazar H₀ (diferencia significativa)" 
        if p_val < 0.05 else
        "No rechazar H₀ (sin diferencia significativa)"
    )
    return {
        'mean_control': mean_control,
        'mean_experimental': mean_experimental,
        't_stat': t_stat,
        'p_val': p_val,
        'conclusion': conclusion
    }

def hypothesis2_internal_conditions(control_df, experimental_df):
    results = {}
    for var in ['temp_int', 'hum_int']:
        data_c = control_df[var].dropna()
        data_e = experimental_df[var].dropna()
        t_stat, p_val = stats.ttest_ind(data_c, data_e, equal_var=False)
        conclusion = (
            "Rechazar H₀ (diferencia significativa)" 
            if p_val < 0.05 else
            "No rechazar H₀ (sin diferencia significativa)"
        )
        results[var] = {
            'mean_control': data_c.mean(),
            'mean_experimental': data_e.mean(),
            't_stat': t_stat,
            'p_val': p_val,
            'conclusion': conclusion
        }
    return results

def hypothesis3_humidity_correlation(control_df, experimental_df):
    results = {}
    for group_name, df_group in [('abierto', control_df), ('sellado', experimental_df)]:
        paired = df_group[['hum_int', 'hum_ext']].dropna()
        if len(paired) < 3:
            results[group_name] = {'error': 'Datos insuficientes'}
            continue
        x, y = paired['hum_ext'], paired['hum_int']
        try:
            p_x = stats.shapiro(x)[1]
            p_y = stats.shapiro(y)[1]
            normal = (p_x > 0.05 and p_y > 0.05)
        except:
            normal = False
        if normal:
            method, (coef, p_val) = 'Pearson', stats.pearsonr(x, y)
        else:
            method, (coef, p_val) = 'Spearman', stats.spearmanr(x, y)
        dist_corr = dcor.distance_correlation(x, y)
        mi = mutual_info_regression(x.values.reshape(-1,1), y.values)[0]
        best_degree = 1
        poly = PolynomialFeatures(degree=best_degree)
        Xp = poly.fit_transform(x.values.reshape(-1,1))
        model = LinearRegression().fit(Xp, y)
        best_r2 = r2_score(y, model.predict(Xp))
        while best_degree < 10:
            deg = best_degree + 1
            poly_new = PolynomialFeatures(degree=deg)
            Xp_new = poly_new.fit_transform(x.values.reshape(-1,1))
            m_new = LinearRegression().fit(Xp_new, y)
            r2_new = r2_score(y, m_new.predict(Xp_new))
            if (r2_new - best_r2) / best_r2 < 0.01:
                break
            best_degree, best_r2, model = deg, r2_new, m_new
        results[group_name] = {
            'n': len(paired),
            'linear_analysis': {'method': method, 'coef': coef, 'p_val': p_val},
            'non_linear_analysis': {
                'dist_corr': dist_corr,
                'mutual_info': mi,
                'best_degree': best_degree,
                'best_r2': best_r2,
                'model': model
            }
        }
    return results

def generate_graphs(df, control_df, experimental_df, h3_results):
    ensure_output_dir()
    split_time = SPLIT_TIMESTAMP

    # Hipótesis 1 – Serie de tiempo y caja
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df['timestamp'], df['ph'], label='pH')
    ax.axvline(split_time, linestyle='--', label='Fecha de corte')
    ax.set_xlabel('Fecha y hora')
    ax.set_ylabel('pH')
    ax.set_title('Serie de tiempo: pH del sustrato')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hipotesis1_tiempo.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(8,6))
    ax.boxplot([control_df['ph'].dropna(), experimental_df['ph'].dropna()],
               labels=['Abierto','Sellado'])
    ax.set_ylabel('pH')
    ax.set_title('pH del sustrato por periodo')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hipotesis1_caja.png'))
    plt.close()

    # Hipótesis 2 – Condiciones internas
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    axes[0].boxplot([control_df['temp_int'].dropna(), experimental_df['temp_int'].dropna()],
                    labels=['Abierto','Sellado'])
    axes[0].set_title('Temperatura interna (°C)')
    axes[1].boxplot([control_df['hum_int'].dropna(), experimental_df['hum_int'].dropna()],
                    labels=['Abierto','Sellado'])
    axes[1].set_title('Humedad interna (%)')
    plt.suptitle('Comparativa de condiciones internas')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'hipotesis2_cajas.png'))
    plt.close()

    # Hipótesis 3 – Ajustes polinómicos y residuos
    plt.figure(figsize=(12,6))
    for i, (label, grp) in enumerate([('Abierto', control_df), ('Sellado', experimental_df)], start=1):
        paired = grp[['hum_int','hum_ext']].dropna()
        if len(paired) < 3:
            continue
        x, y = paired['hum_ext'], paired['hum_int']
        result = h3_results[label.lower()]['non_linear_analysis']
        deg = result['best_degree']
        model = result['model']
        poly = PolynomialFeatures(degree=deg)
        Xp = poly.fit_transform(x.values.reshape(-1,1))

        plt.subplot(2,2,i)
        plt.scatter(x, y, alpha=0.6)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = model.predict(poly.transform(xs.reshape(-1,1)))
        plt.plot(xs, ys, linestyle='--')
        plt.xlabel('Humedad ambiental (%)')
        plt.ylabel('Humedad interna (%)')
        plt.title(f'{label}: Ajuste polinómico (grado {deg})')

        plt.subplot(2,2,i+2)
        residuos = y - model.predict(Xp)
        plt.scatter(x, residuos, alpha=0.6)
        plt.axhline(0, linestyle='--')
        plt.xlabel('Humedad ambiental (%)')
        plt.ylabel('Residuos')
        plt.title(f'{label}: Residuos')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hipotesis3_analisis.png'))
    plt.close()

def generate_report(total_duration, total_records,
                    control_df, experimental_df, h1, h2, h3, df):
    ensure_output_dir()
    lines = []

    # Resumen
    lines.append('# Informe de Análisis del Experimento Mesocosmos\n')
    lines.append('## Resumen del Experimento')
    lines.append(f'- **Duración:** {total_duration}')
    lines.append(f'- **Total de registros:** {total_records}')
    lines.append(f'- **Fecha de corte:** {SPLIT_TIMESTAMP}')
    lines.append(f'- **Registros (Abierto):** {len(control_df)}')
    lines.append(f'- **Registros (Sellado):** {len(experimental_df)}\n')

    # Hipótesis 1
    lines.append('## Hipótesis 1 – pH del Sustrato')
    lines.append(f'- pH medio (Abierto): {h1["mean_control"]:.3f}')
    lines.append(f'- pH medio (Sellado): {h1["mean_experimental"]:.3f}')
    lines.append(f'- Estadístico t: {h1["t_stat"]:.3f}')
    lines.append(f'- Valor p: {h1["p_val"]:.3f}')
    lines.append(f'- Conclusión: {h1["conclusion"]}\n')
    lines.append('**Gráficos:** hipotesis1_tiempo.png, hipotesis1_caja.png\n')

    # Hipótesis 2
    lines.append('## Hipótesis 2 – Condiciones Internas')
    for var, title in [('temp_int','Temperatura interna (°C)'), ('hum_int','Humedad interna (%)')]:
        r = h2[var]
        lines.append(f'### {title}')
        lines.append(f'- Media (Abierto): {r["mean_control"]:.3f}')
        lines.append(f'- Media (Sellado): {r["mean_experimental"]:.3f}')
        lines.append(f'- Estadístico t: {r["t_stat"]:.3f}')
        lines.append(f'- Valor p: {r["p_val"]:.3f}')
        lines.append(f'- Conclusión: {r["conclusion"]}\n')
    lines.append('**Gráficos:** hipotesis2_cajas.png\n')

    # Hipótesis 3
    lines.append('## Hipótesis 3 – Relación de Humedad Interna vs Ambiental')
    for grp, label in [('abierto','Abierto'), ('sellado','Sellado')]:
        r = h3[grp]
        if 'error' in r:
            lines.append(f'**{label}:** {r["error"]}\n')
            continue
        la = r['linear_analysis']
        nla = r['non_linear_analysis']
        # Construir la ecuación polinómica
        model = nla['model']
        coefs = model.coef_
        intercept = model.intercept_
        degree = nla['best_degree']
        # Asumimos que el primer coef corresponde a la potencia cero (sesgo), que ignoramos
        terms = [f"{intercept:.3f}"]
        for i in range(1, degree+1):
            coef_i = coefs[i] if i < len(coefs) else 0.0
            terms.append(f"{coef_i:.3f}·x^{i}")
        equation = " + ".join(terms)

        lines.append(f'### {label} (n={r["n"]})')
        lines.append(f'- Correlación {la["method"]}: coef={la["coef"]:.2f}, p={la["p_val"]:.3f}')
        lines.append(f'- Distance correlation: {nla["dist_corr"]:.2f}')
        lines.append(f'- Mutual information: {nla["mutual_info"]:.2f}')
        lines.append(f'- Mejor grado polinómico: {degree}, R²={nla["best_r2"]:.2f}')
        lines.append(f'- **Ecuación ajustada:** _y_ = {equation}\n')
    lines.append('**Gráficos:** hipotesis3_analisis.png\n')

    # Evolución diaria de métricas
    lines.append('## Evolución de métricas diarias')
    daily = df.set_index('timestamp').resample('D').agg({
        'ph':'mean',
        'temp_int':'mean',
        'hum_int':'mean',
        'temp_ext':'mean',
        'hum_ext':'mean'
    }).dropna()
    lines.append('| Fecha      | pH medio | Temp Int (°C) | Hum Int (%) | Temp Ext (°C) | Hum Ext (%) |')
    lines.append('|:----------:|:--------:|:-------------:|:-----------:|:-------------:|:-----------:|')
    for fecha, row in daily.iterrows():
        fecha_str = fecha.strftime('%Y-%m-%d')
        lines.append(
            f'| {fecha_str} | {row["ph"]:.3f}   | {row["temp_int"]:.2f}        | {row["hum_int"]:.2f}      | {row["temp_ext"]:.2f}        | {row["hum_ext"]:.2f}      |'
        )
    lines.append('')

    report_path = os.path.join(OUTPUT_DIR, 'reporte.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(
        description='Script de Análisis del Experimento Mesocosmos'
    )
    parser.add_argument(
        '--db', type=str, default='./mesocosmos.db',
        help='Ruta al archivo de base de datos SQLite'
    )
    args = parser.parse_args()

    df = load_data(args.db)
    if df.empty:
        print('Sin datos en la base de datos.')
        sys.exit(1)

    control_df, experimental_df, inicio, fin = split_data(df)
    duracion = f"{inicio} a {fin}"
    total = len(df)

    h1 = hypothesis1_ph(control_df, experimental_df)
    h2 = hypothesis2_internal_conditions(control_df, experimental_df)
    h3 = hypothesis3_humidity_correlation(control_df, experimental_df)

    generate_graphs(df, control_df, experimental_df, h3)
    generate_report(duracion, total, control_df, experimental_df, h1, h2, h3, df)

    print('Análisis completo. Salida en el directorio ./report')

if __name__ == '__main__':
    main()

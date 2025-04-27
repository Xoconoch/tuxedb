#!/usr/bin/env python3
"""
Script de Análisis del Experimento Mesocosmos.

Este script se conecta a una base de datos SQLite (por defecto: ./mesocosmos.db), obtiene datos de la
tabla "medicion" y realiza lo siguiente:
  - Ordena los datos cronológicamente según la columna `timestamp`.
  - Filtra los registros hasta el 2025-04-23 09:30.
  - Determina la duración total del experimento (usando los timestamps inicial y final tras el filtro).
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
# Fecha límite de datos
END_TIMESTAMP = pd.to_datetime("2025-04-23 09:30:00")

def ensure_output_dir(directory=OUTPUT_DIR):
    """Crea el directorio de salida si no existe."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(db_path="./mesocosmos.db"):
    """
    Carga la tabla "medicion" desde la base SQLite y ordena por timestamp.
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
    mean_c = control_df['ph'].mean()
    mean_e = experimental_df['ph'].mean()
    t_stat, p_val = stats.ttest_ind(
        control_df['ph'].dropna(),
        experimental_df['ph'].dropna(),
        equal_var=False
    )
    concl = "Diferencia significativa" if p_val < 0.05 else "Sin diferencia significativa"
    return {'mean_control': mean_c, 'mean_experimental': mean_e,
            't_stat': t_stat, 'p_val': p_val, 'conclusion': concl}

def hypothesis2_internal_conditions(control_df, experimental_df):
    results = {}
    for var in ['temp_int', 'hum_int']:
        data_c = control_df[var].dropna()
        data_e = experimental_df[var].dropna()
        t_stat, p_val = stats.ttest_ind(data_c, data_e, equal_var=False)
        concl = "Diferencia significativa" if p_val < 0.05 else "Sin diferencia significativa"
        results[var] = {
            'mean_control': data_c.mean(),
            'mean_experimental': data_e.mean(),
            't_stat': t_stat,
            'p_val': p_val,
            'conclusion': concl
        }
    return results

def hypothesis3_humidity_correlation(control_df, experimental_df):
    results = {}
    for name, grp in [('abierto', control_df), ('sellado', experimental_df)]:
        paired = grp[['hum_ext','hum_int']].dropna()
        if len(paired) < 3:
            results[name] = {'error': 'Datos insuficientes'}
            continue
        x, y = paired['hum_ext'], paired['hum_int']
        try:
            px, py = stats.shapiro(x)[1], stats.shapiro(y)[1]
            normal = px>0.05 and py>0.05
        except:
            normal = False
        if normal:
            method, (coef, p_val) = 'Pearson', stats.pearsonr(x,y)
        else:
            method, (coef, p_val) = 'Spearman', stats.spearmanr(x,y)
        dist_corr = dcor.distance_correlation(x,y)
        mi = mutual_info_regression(x.values.reshape(-1,1), y.values)[0]
        best_deg, best_r2 = 1, None
        poly = PolynomialFeatures(degree=1)
        Xp = poly.fit_transform(x.values.reshape(-1,1))
        model = LinearRegression().fit(Xp,y)
        best_r2 = r2_score(y, model.predict(Xp))
        while best_deg < 10:
            deg = best_deg+1
            poly_n = PolynomialFeatures(degree=deg)
            Xn = poly_n.fit_transform(x.values.reshape(-1,1))
            m2 = LinearRegression().fit(Xn,y)
            r2n = r2_score(y, m2.predict(Xn))
            if (r2n-best_r2)/best_r2 < 0.01:
                break
            best_deg, best_r2, model = deg, r2n, m2
        results[name] = {
            'linear': {'method': method, 'coef': coef, 'p_val': p_val},
            'non_linear': {
                'dist_corr': dist_corr,
                'mutual_info': mi,
                'best_degree': best_deg,
                'best_r2': best_r2
            }
        }
    return results

def generate_graphs(df, control_df, experimental_df, h3_results):
    ensure_output_dir()
    # Divisor fijo en gráficos
    split_time = SPLIT_TIMESTAMP
    # Hipótesis 1
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df['timestamp'], df['ph'], label='pH')
    ax.axvline(split_time, color='red', linestyle='--', label='Fecha de corte')
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

    # Hipótesis 2
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

    # Hipótesis 3
    plt.figure(figsize=(12,6))
    for i, (label, grp) in enumerate([('Abierto', control_df), ('Sellado', experimental_df)], start=1):
        paired = grp[['hum_int','hum_ext']].dropna()
        if len(paired) < 3: continue
        x, y = paired['hum_ext'], paired['hum_int']
        deg = h3_results[label.lower()]['non_linear_analysis']['best_degree']
        model = h3_results[label.lower()]['non_linear_analysis']['model']
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
        res = y - model.predict(Xp)
        plt.scatter(x, res, alpha=0.6)
        plt.axhline(0, linestyle='--')
        plt.xlabel('Hum. ambiental (%)')
        plt.ylabel('Residuos')
        plt.title(f'{label}: Residuos')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hipotesis3_analisis.png'))
    plt.close()

def generate_report(duration, total, ctrl, exp, h1, h2, h3):
    ensure_output_dir()
    lines = [
        '# Informe de Análisis del Experimento Mesocosmos',
        '',
        '## Resumen del Experimento',
        f'- **Duración:** {duration}',
        f'- **Total de registros:** {total}',
        f'- **Fecha de corte:** {SPLIT_TIMESTAMP}',
        f'- **Registros (Abierto):** {len(ctrl)}',
        f'- **Registros (Sellado):** {len(exp)}',
        '',
        '## Hipótesis 1 – pH del Sustrato',
        '',
        '| Periodo | pH medio | t | p | Conclusión |',
        '|:-------:|:--------:|:-:|:-:|:----------:|',
        f'| Abierto | {h1["mean_control"]:.3f} | {h1["t_stat"]:.3f} | {h1["p_val"]:.3f} | {h1["conclusion"]} |',
        f'| Sellado | {h1["mean_experimental"]:.3f} |  |  |  |',
        '',
        '## Hipótesis 2 – Condiciones Internas',
        '',
        '| Variable            | Periodo | Media   | t        | p       | Conclusión             |',
        '|:-------------------:|:-------:|:-------:|:--------:|:-------:|:----------------------:|',
        f'| Temperatura interna | Abierto | {h2["temp_int"]["mean_control"]:.3f} | {h2["temp_int"]["t_stat"]:.3f} | {h2["temp_int"]["p_val"]:.3f} | {h2["temp_int"]["conclusion"]} |',
        f'| Temperatura interna | Sellado | {h2["temp_int"]["mean_experimental"]:.3f} |  |  |  |',
        f'| Humedad interna     | Abierto | {h2["hum_int"]["mean_control"]:.3f} | {h2["hum_int"]["t_stat"]:.3f} | {h2["hum_int"]["p_val"]:.3f} | {h2["hum_int"]["conclusion"]} |',
        f'| Humedad interna     | Sellado | {h2["hum_int"]["mean_experimental"]:.3f} |  |  |  |',
        '',
        '## Hipótesis 3 – Relación Humedad Interna vs Ambiental',
        '',
        '| Periodo  | ρ (Spearman) | p (ρ)    | Dist. Corr. | Mutual Info | Mejor grado | R²   |',
        '|:--------:|:------------:|:--------:|:-----------:|:-----------:|:-----------:|:----:|',
        f'| Abierto  | {h3["abierto"]["linear"]["coef"]:.2f}       | {h3["abierto"]["linear"]["p_val"]:.3f} | {h3["abierto"]["non_linear"]["dist_corr"]:.2f}      | {h3["abierto"]["non_linear"]["mutual_info"]:.2f}      | {h3["abierto"]["non_linear"]["best_degree"]}           | {h3["abierto"]["non_linear"]["best_r2"]:.2f} |',
        f'| Sellado  | {h3["sellado"]["linear"]["coef"]:.2f}       | {h3["sellado"]["linear"]["p_val"]:.3f} | {h3["sellado"]["non_linear"]["dist_corr"]:.2f}      | {h3["sellado"]["non_linear"]["mutual_info"]:.2f}      | {h3["sellado"]["non_linear"]["best_degree"]}           | {h3["sellado"]["non_linear"]["best_r2"]:.2f} |',
        '',
        '**Gráficos:** hipotesis1_tiempo.png, hipotesis1_caja.png, hipotesis2_cajas.png, hipotesis3_analisis.png'
    ]
    with open(os.path.join(OUTPUT_DIR, 'reporte.md'), 'w') as f:
        f.write('\n'.join(lines))

def main():
    parser = argparse.ArgumentParser(description='Script de Análisis del Experimento Mesocosmos')
    parser.add_argument('--db', type=str, default='./mesocosmos.db',
                        help='Ruta al archivo de base de datos SQLite')
    args = parser.parse_args()

    df = load_data(args.db)
    # Filtrar hasta el 23 de abril de 2025 a las 09:30
    df = df[df['timestamp'] <= END_TIMESTAMP]
    if df.empty:
        print('Sin datos tras el filtro de fecha.')
        sys.exit(1)

    control_df, experimental_df, inicio, fin = split_data(df)
    duration = f"{inicio} a {fin}"
    total = len(df)

    h1 = hypothesis1_ph(control_df, experimental_df)
    h2 = hypothesis2_internal_conditions(control_df, experimental_df)
    h3 = hypothesis3_humidity_correlation(control_df, experimental_df)

    generate_graphs(df, control_df, experimental_df, h3)
    generate_report(duration, total, control_df, experimental_df, h1, h2, h3)

    print('Análisis completo. Salida en el directorio ./report')

if __name__ == '__main__':
    main()

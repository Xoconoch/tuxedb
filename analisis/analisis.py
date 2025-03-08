#!/usr/bin/env python3
"""
Mesocosmos Experiment Analysis Script

This script connects to an SQLite database (default: ./mesocosmos.db), retrieves data from the
"medicion" table, and performs the following tasks:
  - Orders the data chronologically by the timestamp column.
  - Determines the total duration of the experiment (using the first and last timestamps).
  - Splits the data into two halves:
       Control Group: Mesocosmos open (first half)
       Experimental Group: Mesocosmos sealed (second half)
  - Hypothesis 1: Compares substrate pH between groups using an independent t-test.
  - Hypothesis 2: Compares internal temperature and humidity between groups via separate t-tests.
  - Hypothesis 3: Assesses the relationship between internal and ambient humidity using
                  normality tests and an appropriate correlation (Pearson or Spearman).
  - Generates several graphs as PNG images.
  - Generates a report (in Markdown format) summarizing the analysis.

All output (graphs and report) will be saved under the "./report" directory.

Usage:
    python mesocosmos_analysis.py [--db PATH_TO_DB]

Author: Your Name
Date: YYYY-MM-DD
"""
import numba

# Save the original njit decorator
_original_njit = numba.njit

# Define a new njit that forces caching to be disabled
def no_cache_njit(*args, **kwargs):
    kwargs['cache'] = False
    return _original_njit(*args, **kwargs)

# Override njit in the numba module
numba.njit = no_cache_njit


import os
import sqlite3
import pandas as pd
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import dcor  # New dependency for distance correlation
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Define the directory to store report and graphs
OUTPUT_DIR = "./report"

def ensure_output_dir(directory=OUTPUT_DIR):
    """
    Ensures the output directory exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(db_path="./mesocosmos.db"):
    """
    Connects to the SQLite database and loads the 'medicion' table into a pandas DataFrame.
    The DataFrame is sorted chronologically based on the 'timestamp' column.
    """
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT id, timestamp, temp_int, hum_int, ph, temp_ext, hum_ext FROM medicion"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print("Error connecting to database or retrieving data:", e)
        sys.exit(1)
    
    try:
        # Convert timestamp column to datetime and sort chronologically
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
    except Exception as e:
        print("Error processing timestamps:", e)
        sys.exit(1)
    
    return df

def split_data(df):
    """
    Splits the DataFrame into two halves based on the experiment duration.
    The midpoint (by time) is used to separate:
      - control_df: records up to and including the midpoint (mesocosmos open)
      - experimental_df: records after the midpoint (mesocosmos sealed)
    Returns both subsets along with the start and end times.
    """
    start_time = df['timestamp'].iloc[0]
    end_time = df['timestamp'].iloc[-1]
    midpoint = start_time + (end_time - start_time) / 2

    control_df = df[df['timestamp'] <= midpoint]
    experimental_df = df[df['timestamp'] > midpoint]
    
    return control_df, experimental_df, start_time, end_time

def hypothesis1_ph(control_df, experimental_df):
    """
    Hypothesis 1 – Variations in Substrate pH:
      - Computes the mean substrate pH for both control and experimental groups.
      - Applies an independent t-test (using Welch’s correction) to compare the means.
      - Null Hypothesis (H₀): Mean pH in control equals mean pH in experimental.
    Returns a dictionary with the computed means, t-test statistic, p-value, and conclusion.
    """
    mean_ph_control = control_df['ph'].mean()
    mean_ph_experimental = experimental_df['ph'].mean()
    
    # Perform independent t-test (Welch's t-test)
    t_stat, p_val = stats.ttest_ind(control_df['ph'].dropna(), experimental_df['ph'].dropna(), equal_var=False)
    conclusion = "Reject H₀ (significant difference)" if p_val < 0.05 else "Fail to reject H₀ (no significant difference)"
    
    return {
        "mean_ph_control": mean_ph_control,
        "mean_ph_experimental": mean_ph_experimental,
        "t_stat": t_stat,
        "p_val": p_val,
        "conclusion": conclusion
    }

def hypothesis2_internal_conditions(control_df, experimental_df):
    """
    Hypothesis 2 – Influence of Ambient Temperature on Internal Conditions:
      - For internal temperature (temp_int) and internal humidity (hum_int), computes the mean values
        for control and experimental groups.
      - Performs separate independent t-tests.
      - Null Hypothesis (H₀): The means for the two groups are equal.
    Returns a dictionary with results for both comparisons.
    """
    results = {}
    
    # Internal Temperature comparison
    temp_control = control_df['temp_int'].dropna()
    temp_experimental = experimental_df['temp_int'].dropna()
    t_stat_temp, p_val_temp = stats.ttest_ind(temp_control, temp_experimental, equal_var=False)
    conclusion_temp = "Reject H₀ (significant difference)" if p_val_temp < 0.05 else "Fail to reject H₀ (no significant difference)"
    results['temp_int'] = {
        "mean_control": temp_control.mean(),
        "mean_experimental": temp_experimental.mean(),
        "t_stat": t_stat_temp,
        "p_val": p_val_temp,
        "conclusion": conclusion_temp
    }
    
    # Internal Humidity comparison
    hum_control = control_df['hum_int'].dropna()
    hum_experimental = experimental_df['hum_int'].dropna()
    t_stat_hum, p_val_hum = stats.ttest_ind(hum_control, hum_experimental, equal_var=False)
    conclusion_hum = "Reject H₀ (significant difference)" if p_val_hum < 0.05 else "Fail to reject H₀ (no significant difference)"
    results['hum_int'] = {
        "mean_control": hum_control.mean(),
        "mean_experimental": hum_experimental.mean(),
        "t_stat": t_stat_hum,
        "p_val": p_val_hum,
        "conclusion": conclusion_hum
    }
    
    return results

def hypothesis3_humidity_correlation(control_df, experimental_df):
    """
    Enhanced Hypothesis 3 Analysis:
    Performs comprehensive linear and non-linear correlation analysis between
    internal and ambient humidity for both groups.
    """
    results = {}
    for group_name, df_group in zip(["control", "experimental"], [control_df, experimental_df]):
        paired = df_group[['hum_int', 'hum_ext']].dropna()
        if len(paired) < 3:  # Minimum for meaningful analysis
            results[group_name] = {"error": "Insufficient data"}
            continue
            
        hum_int = paired['hum_int']
        hum_ext = paired['hum_ext']
        
        # Basic statistics
        stats_results = {
            "n_samples": len(paired),
            "hum_int_mean": hum_int.mean(),
            "hum_ext_mean": hum_ext.mean()
        }
        
        # Traditional correlation analysis
        try:
            shapiro_int = stats.shapiro(hum_int)
            shapiro_ext = stats.shapiro(hum_ext)
            normality = shapiro_int[1] > 0.05 and shapiro_ext[1] > 0.05
        except Exception as e:
            normality = False
            
        if normality:
            corr_method = "Pearson"
            corr_coef, p_val = stats.pearsonr(hum_int, hum_ext)
        else:
            corr_method = "Spearman"
            corr_coef, p_val = stats.spearmanr(hum_int, hum_ext)
        
        # Non-linear correlation metrics
        distance_corr = dcor.distance_correlation(hum_int, hum_ext)
        mi = mutual_info_regression(hum_ext.values.reshape(-1, 1), hum_int.values)[0]
        
        # Polynomial regression analysis
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(hum_ext.values.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, hum_int)
        poly_r2 = r2_score(hum_int, model.predict(X_poly))
        
        results[group_name] = {
            **stats_results,
            "linear_analysis": {
                "method": corr_method,
                "coef": corr_coef,
                "p_value": p_val,
                "normality_test": (shapiro_int[1], shapiro_ext[1])
            },
            "non_linear_analysis": {
                "distance_correlation": distance_corr,
                "mutual_information": mi,
                "polynomial_r2": poly_r2
            }
        }
    
    return results


def generate_graphs(df, control_df, experimental_df, h3_results):
    """
    Generates and saves graphical outputs for the analyses under the OUTPUT_DIR.
    
    Hypothesis 1:
      - Time series plot of pH over the experiment duration with a line indicating the split.
      - Boxplot comparing pH distributions between control and experimental groups.
      
    Hypothesis 2:
      - Side-by-side boxplots for internal temperature and internal humidity.
      
    Hypothesis 3:
      - Scatter plots (with regression lines) showing the relationship between internal and ambient humidity
        for control and experimental groups.
    """
    # Ensure output directory exists
    ensure_output_dir()
    
    # --- Hypothesis 1: pH Time Series and Boxplot ---
    # Time series plot for pH with a vertical line at the split (using the last timestamp of control group)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['timestamp'], df['ph'], label='pH', color='blue')
    midpoint = control_df['timestamp'].iloc[-1]
    ax.axvline(x=midpoint, color='red', linestyle='--', label='Group Split')
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("pH")
    ax.set_title("Time Series of pH with Control/Experimental Split")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hypothesis1_timeseries.png"))
    plt.close()
    
    # Boxplot of pH for control vs experimental
    fig, ax = plt.subplots(figsize=(8, 6))
    data_to_plot = [control_df['ph'].dropna(), experimental_df['ph'].dropna()]
    ax.boxplot(data_to_plot, labels=['Control', 'Experimental'])
    ax.set_ylabel("pH")
    ax.set_title("Boxplot of Substrate pH by Group")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hypothesis1_boxplot.png"))
    plt.close()
    
    # --- Hypothesis 2: Side-by-Side Boxplots for Internal Conditions ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Boxplot for internal temperature
    axs[0].boxplot([control_df['temp_int'].dropna(), experimental_df['temp_int'].dropna()],
                   labels=['Control', 'Experimental'])
    axs[0].set_title("Internal Temperature")
    axs[0].set_ylabel("Temperature (°C)")
    # Boxplot for internal humidity
    axs[1].boxplot([control_df['hum_int'].dropna(), experimental_df['hum_int'].dropna()],
                   labels=['Control', 'Experimental'])
    axs[1].set_title("Internal Humidity")
    axs[1].set_ylabel("Humidity (%)")
    plt.suptitle("Boxplots of Internal Temperature and Humidity")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "hypothesis2_boxplots.png"))
    plt.close()

    # --- Enhanced Hypothesis 3 Visualizations ---
    plt.figure(figsize=(12, 6))
    groups = {'Control': control_df, 'Experimental': experimental_df}
    
    for idx, (group_name, group_df) in enumerate(groups.items(), 1):
        paired = group_df[['hum_int', 'hum_ext']].dropna()
        if len(paired) < 3:
            continue
            
        x = paired['hum_ext']
        y = paired['hum_int']
        
        # Main scatter plot with polynomial fit
        plt.subplot(2, 2, idx)
        plt.scatter(x, y, alpha=0.6)
        
        # Polynomial regression line
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(x.values.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, y)
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = model.predict(poly.transform(x_vals.reshape(-1, 1)))
        plt.plot(x_vals, y_vals, 'r--', label='Quadratic Fit')
        
        plt.xlabel('Ambient Humidity')
        plt.ylabel('Internal Humidity')
        plt.title(f'{group_name} Group\nDistance Corr: {h3_results[group_name.lower()]["non_linear_analysis"]["distance_correlation"]:.2f}')
        plt.legend()

        # Residual plot
        plt.subplot(2, 2, idx+2)
        residuals = y - model.predict(X_poly)
        plt.scatter(x, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Ambient Humidity')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hypothesis3_analysis.png"))
    plt.close()

def generate_report(total_duration, total_records, control_df, experimental_df, h1_results, h2_results, h3_results):
    """
    Generates a Markdown report summarizing:
      - Experiment duration and record counts.
      - Data split between control and experimental groups.
      - Results (statistics, test details, and conclusions) for each hypothesis.
      - References to the generated graphs.
      
    The report is saved as 'report.md' under the OUTPUT_DIR.
    """
    # Ensure output directory exists
    ensure_output_dir()
    
    report_lines = []
    report_lines.append("# Mesocosmos Experiment Analysis Report")
    report_lines.append("")
    report_lines.append("## Experiment Overview")
    report_lines.append(f"- **Total Duration:** {total_duration}")
    report_lines.append(f"- **Total Records:** {total_records}")
    report_lines.append(f"- **Control Group (Open) Records:** {len(control_df)}")
    report_lines.append(f"- **Experimental Group (Sealed) Records:** {len(experimental_df)}")
    report_lines.append("")
    
    # Hypothesis 1
    report_lines.append("## Hypothesis 1 – Variations in Substrate pH")
    report_lines.append(f"- **Mean pH (Control):** {h1_results['mean_ph_control']:.3f}")
    report_lines.append(f"- **Mean pH (Experimental):** {h1_results['mean_ph_experimental']:.3f}")
    report_lines.append(f"- **t-statistic:** {h1_results['t_stat']:.3f}")
    report_lines.append(f"- **p-value:** {h1_results['p_val']:.3f}")
    report_lines.append(f"- **Conclusion:** {h1_results['conclusion']}")
    report_lines.append("")
    report_lines.append("**Graphical Outputs:**")
    report_lines.append("- Time Series Plot: `hypothesis1_timeseries.png`")
    report_lines.append("- Boxplot: `hypothesis1_boxplot.png`")
    report_lines.append("")
    
    # Hypothesis 2
    report_lines.append("## Hypothesis 2 – Influence of Ambient Temperature on Internal Conditions")
    temp_res = h2_results['temp_int']
    report_lines.append("### Internal Temperature")
    report_lines.append(f"- **Mean (Control):** {temp_res['mean_control']:.3f}")
    report_lines.append(f"- **Mean (Experimental):** {temp_res['mean_experimental']:.3f}")
    report_lines.append(f"- **t-statistic:** {temp_res['t_stat']:.3f}")
    report_lines.append(f"- **p-value:** {temp_res['p_val']:.3f}")
    report_lines.append(f"- **Conclusion:** {temp_res['conclusion']}")
    report_lines.append("")
    hum_res = h2_results['hum_int']
    report_lines.append("### Internal Humidity")
    report_lines.append(f"- **Mean (Control):** {hum_res['mean_control']:.3f}")
    report_lines.append(f"- **Mean (Experimental):** {hum_res['mean_experimental']:.3f}")
    report_lines.append(f"- **t-statistic:** {hum_res['t_stat']:.3f}")
    report_lines.append(f"- **p-value:** {hum_res['p_val']:.3f}")
    report_lines.append(f"- **Conclusion:** {hum_res['conclusion']}")
    report_lines.append("")
    report_lines.append("**Graphical Outputs:**")
    report_lines.append("- Boxplots for Internal Conditions: `hypothesis2_boxplots.png`")
    report_lines.append("")

    # Enhanced Hypothesis 3 section
    report_lines.append("## Hypothesis 3 – Humidity Relationship Analysis")
    report_lines.append("### Comprehensive Linear and Non-Linear Analysis")
    
    for group in ['control', 'experimental']:
        res = h3_results[group]
        if 'error' in res:
            report_lines.append(f"**{group.capitalize()} Group:** {res['error']}")
            continue
            
        report_lines.append(f"### {group.capitalize()} Group (n={res['n_samples']})")
        report_lines.append("- **Linear Analysis:**")
        report_lines.append(f"  - Method: {res['linear_analysis']['method']}")
        report_lines.append(f"  - Coefficient: {res['linear_analysis']['coef']:.2f}")
        report_lines.append(f"  - p-value: {res['linear_analysis']['p_value']:.3f}")
        
        report_lines.append("- **Non-Linear Analysis:**")
        report_lines.append(f"  - Distance Correlation: {res['non_linear_analysis']['distance_correlation']:.2f}")
        report_lines.append(f"  - Mutual Information: {res['non_linear_analysis']['mutual_information']:.2f}")
        report_lines.append(f"  - Polynomial R²: {res['non_linear_analysis']['polynomial_r2']:.2f}")
        
        report_lines.append("- **Interpretation:**")
        if res['non_linear_analysis']['distance_correlation'] > 0.5:
            report_lines.append("  - Strong non-linear relationship detected")
        elif res['non_linear_analysis']['distance_correlation'] > 0.3:
            report_lines.append("  - Moderate non-linear relationship detected")
        else:
            report_lines.append("  - Weak non-linear relationship")
        
        report_lines.append("")

    report_lines.append("**Graphical Outputs:**")
    report_lines.append("- Comprehensive Analysis: `hypothesis3_analysis.png`")

    report_content = "\n".join(report_lines)
    try:
        with open(os.path.join(OUTPUT_DIR, "report.md"), "w") as report_file:
            report_file.write(report_content)
    except Exception as e:
        print(f"Error writing report file: {e}")

def main():
    # Parse command-line arguments (e.g., path to the SQLite database)
    parser = argparse.ArgumentParser(description="Mesocosmos Experiment Analysis Script")
    parser.add_argument("--db", type=str, default="./mesocosmos.db", help="Path to the SQLite database file")
    args = parser.parse_args()
    
    # Load and prepare data
    df = load_data(args.db)
    if df.empty:
        print("No data found in the database.")
        sys.exit(1)
    
    control_df, experimental_df, start_time, end_time = split_data(df)
    total_duration = f"{start_time} to {end_time}"
    total_records = len(df)
    
    # Perform hypothesis tests
    h1_results = hypothesis1_ph(control_df, experimental_df)
    h2_results = hypothesis2_internal_conditions(control_df, experimental_df)
    h3_results = hypothesis3_humidity_correlation(control_df, experimental_df)
    
    # Generate graphical outputs
    generate_graphs(df, control_df, experimental_df, h3_results)
    
    # Generate and save the Markdown report
    generate_report(total_duration, total_records, control_df, experimental_df, h1_results, h2_results, h3_results)
    
    print("Analysis complete. All outputs are saved in the './report' directory.")

if __name__ == "__main__":
    main()

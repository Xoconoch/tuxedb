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
  - Generates a report (in Markdown format, saved as report.md) summarizing the analysis.

Usage:
    python mesocosmos_analysis.py [--db PATH_TO_DB]

Author: Your Name
Date: YYYY-MM-DD
"""

import sqlite3
import pandas as pd
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

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
    Hypothesis 3 – Relationship Between Internal and Ambient Humidity:
      - For each group (control and experimental), extracts internal (hum_int) and ambient (hum_ext) humidity.
      - Assesses normality using the Shapiro–Wilk test.
      - If both variables are normally distributed (p > 0.05), uses Pearson correlation;
        otherwise, uses Spearman correlation.
    Returns a dictionary with the correlation method, coefficient, and p-value for each group.
    """
    results = {}
    for group_name, df_group in zip(["control", "experimental"], [control_df, experimental_df]):
        # Drop any rows with missing values in hum_int or hum_ext
        paired = df_group[['hum_int', 'hum_ext']].dropna()
        hum_int = paired['hum_int']
        hum_ext = paired['hum_ext']
        
        # Normality tests (Shapiro-Wilk)
        try:
            shapiro_int = stats.shapiro(hum_int)
        except Exception as e:
            shapiro_int = (None, None)
        try:
            shapiro_ext = stats.shapiro(hum_ext)
        except Exception as e:
            shapiro_ext = (None, None)
        
        # Choose correlation method based on normality
        if shapiro_int[1] is not None and shapiro_ext[1] is not None and shapiro_int[1] > 0.05 and shapiro_ext[1] > 0.05:
            corr_method = "Pearson"
            corr_coef, p_val = stats.pearsonr(hum_int, hum_ext)
        else:
            corr_method = "Spearman"
            corr_coef, p_val = stats.spearmanr(hum_int, hum_ext)
        
        results[group_name] = {
            "corr_method": corr_method,
            "corr_coef": corr_coef,
            "p_val": p_val,
            "shapiro_hum_int": shapiro_int,
            "shapiro_hum_ext": shapiro_ext
        }
    
    return results

def generate_graphs(df, control_df, experimental_df, h3_results):
    """
    Generates and saves graphical outputs for the analyses.
    
    Hypothesis 1:
      - Time series plot of pH over the experiment duration with a line indicating the split.
      - Boxplot comparing pH distributions between control and experimental groups.
      
    Hypothesis 2:
      - Side-by-side boxplots for internal temperature and internal humidity.
      
    Hypothesis 3:
      - Scatter plots (with regression lines) showing the relationship between internal and ambient humidity
        for control and experimental groups.
    """
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
    plt.savefig("hypothesis1_timeseries.png")
    plt.close()
    
    # Boxplot of pH for control vs experimental
    fig, ax = plt.subplots(figsize=(8, 6))
    data_to_plot = [control_df['ph'].dropna(), experimental_df['ph'].dropna()]
    ax.boxplot(data_to_plot, labels=['Control', 'Experimental'])
    ax.set_ylabel("pH")
    ax.set_title("Boxplot of Substrate pH by Group")
    plt.tight_layout()
    plt.savefig("hypothesis1_boxplot.png")
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
    plt.savefig("hypothesis2_boxplots.png")
    plt.close()
    
    # --- Hypothesis 3: Scatter Plots for Humidity Correlation ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    groups = {'Control': control_df, 'Experimental': experimental_df}
    for ax, (group_name, group_df) in zip(axs, groups.items()):
        paired = group_df[['hum_int', 'hum_ext']].dropna()
        x = paired['hum_ext']
        y = paired['hum_int']
        ax.scatter(x, y, label=group_name, color='green')
        # Add regression line if there is enough data
        if len(x) > 1:
            slope, intercept, r_value, p_val, std_err = stats.linregress(x, y)
            x_vals = np.array([x.min(), x.max()])
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, color='red', linestyle='--', label='Regression line')
        ax.set_xlabel("Ambient Humidity")
        ax.set_ylabel("Internal Humidity")
        ax.set_title(f"{group_name} Group")
        ax.legend()
    plt.suptitle("Scatter Plot: Internal vs Ambient Humidity")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("hypothesis3_scatter.png")
    plt.close()

def generate_report(total_duration, total_records, control_df, experimental_df, h1_results, h2_results, h3_results):
    """
    Generates a Markdown report summarizing:
      - Experiment duration and record counts.
      - Data split between control and experimental groups.
      - Results (statistics, test details, and conclusions) for each hypothesis.
      - References to the generated graphs.
      
    The report is saved as 'report.md'.
    """
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
    
    # Hypothesis 3
    report_lines.append("## Hypothesis 3 – Relationship Between Internal and Ambient Humidity")
    for group in ['control', 'experimental']:
        res = h3_results[group]
        report_lines.append(f"### {group.capitalize()} Group")
        report_lines.append(f"- **Correlation Method:** {res['corr_method']}")
        report_lines.append(f"- **Correlation Coefficient:** {res['corr_coef']:.3f}")
        report_lines.append(f"- **p-value:** {res['p_val']:.3f}")
        if res['p_val'] < 0.05:
            report_lines.append("- **Interpretation:** The relationship is statistically significant.")
        else:
            report_lines.append("- **Interpretation:** The relationship is not statistically significant.")
        report_lines.append("")
    report_lines.append("**Graphical Outputs:**")
    report_lines.append("- Scatter Plots: `hypothesis3_scatter.png`")
    
    # Write the report to a Markdown file
    with open("report.md", "w") as f:
        f.write("\n".join(report_lines))
    print("Report generated and saved as report.md")

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
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()

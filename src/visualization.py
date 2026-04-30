from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_load_curve(result_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(result_df["hour"], result_df["load_kw"], marker="o", label="Original Load")
    plt.plot(result_df["hour"], result_df["grid_power_kw"], marker="o", label="Grid Power After ESS")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.title("Load Curve Before and After Energy Storage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "load_curve.png", dpi=200)
    plt.close()


def plot_soc_curve(result_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(result_df["hour"], result_df["soc"] * 100, marker="o")
    plt.xlabel("Hour")
    plt.ylabel("SOC (%)")
    plt.title("Energy Storage SOC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "soc_curve.png", dpi=200)
    plt.close()


def plot_charge_discharge_curve(result_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(result_df["hour"], result_df["charge_kw"], label="Charge Power")
    plt.bar(result_df["hour"], -result_df["discharge_kw"], label="Discharge Power")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.title("Charge and Discharge Power Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "charge_discharge_curve.png", dpi=200)
    plt.close()


def generate_all_charts(result_df: pd.DataFrame, output_dir: Path):
    plot_load_curve(result_df, output_dir)
    plot_soc_curve(result_df, output_dir)
    plot_charge_discharge_curve(result_df, output_dir)
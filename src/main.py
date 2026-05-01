from pathlib import Path
from copy import deepcopy

import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT_DIR / "config" / "project_config.yaml"
LOAD_PATH = ROOT_DIR / "data" / "load_profile_24h.csv"
PRICE_PATH = ROOT_DIR / "data" / "electricity_price.csv"
OUTPUT_DIR = ROOT_DIR / "output"


# ============================================================
# 2. 基础读取函数
# ============================================================

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data() -> pd.DataFrame:
    load_df = pd.read_csv(LOAD_PATH)
    price_df = pd.read_csv(PRICE_PATH)

    data = pd.merge(load_df, price_df, on="hour", how="left")

    if data["price"].isna().any():
        raise ValueError("电价数据存在缺失，请检查 electricity_price.csv 是否覆盖 0-23 小时。")

    return data


# ============================================================
# 3. 电价敏感性处理
# ============================================================

def apply_price_scenario(data: pd.DataFrame, peak_price: float, valley_price: float) -> pd.DataFrame:
    """
    作用：
    用于敏感性分析，替换峰段和谷段电价。

    不改变 flat 平段电价。
    """
    new_data = data.copy()

    new_data.loc[new_data["price_type"] == "peak", "price"] = peak_price
    new_data.loc[new_data["price_type"] == "valley", "price"] = valley_price

    return new_data


# ============================================================
# 4. 储能运行策略模拟
# ============================================================

def simulate_strategy(data: pd.DataFrame, ess: dict, strategy: dict) -> pd.DataFrame:
    """
    作用：
    模拟储能系统 24 小时运行策略。

    当前策略：
    - 谷段充电；
    - 峰段放电；
    - 平段不动作；
    - SOC 上下限保护；
    - 防逆流约束。
    """

    power_kw = float(ess["power_kw"])
    capacity_kwh = float(ess["capacity_kwh"])
    soc_min = float(ess["soc_min"])
    soc_max = float(ess["soc_max"])
    soc = float(ess["initial_soc"])
    efficiency = float(ess["efficiency"])

    charge_type = strategy["charge_price_type"]
    discharge_type = strategy["discharge_price_type"]
    anti_reverse_power = bool(strategy["anti_reverse_power"])

    records = []

    for _, row in data.iterrows():
        hour = int(row["hour"])
        load_kw = float(row["load_kw"])
        price_type = row["price_type"]
        price = float(row["price"])

        charge_kw = 0.0
        discharge_kw = 0.0
        action = "idle"

        available_charge_kwh = max((soc_max - soc) * capacity_kwh, 0)
        available_discharge_kwh = max((soc - soc_min) * capacity_kwh, 0)

        # 谷段充电
        if price_type == charge_type and available_charge_kwh > 0:
            charge_kw = min(power_kw, available_charge_kwh)
            soc += charge_kw * efficiency / capacity_kwh
            action = "charge"

        # 峰段放电
        elif price_type == discharge_type and available_discharge_kwh > 0:
            discharge_limit = min(power_kw, available_discharge_kwh)

            # 防逆流：放电功率不能超过当前负荷
            if anti_reverse_power:
                discharge_limit = min(discharge_limit, load_kw)

            discharge_kw = discharge_limit
            soc -= discharge_kw / efficiency / capacity_kwh
            action = "discharge"

        soc = max(min(soc, soc_max), soc_min)

        grid_power_kw = load_kw + charge_kw - discharge_kw

        records.append(
            {
                "hour": hour,
                "load_kw": round(load_kw, 2),
                "price_type": price_type,
                "price": round(price, 4),
                "action": action,
                "charge_kw": round(charge_kw, 2),
                "discharge_kw": round(discharge_kw, 2),
                "soc": round(soc, 4),
                "soc_percent": round(soc * 100, 2),
                "grid_power_kw": round(grid_power_kw, 2),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 5. ROI 收益测算
# ============================================================

def calculate_roi(result_df: pd.DataFrame, ess: dict, finance: dict) -> dict:
    """
    作用：
    根据运行结果计算日收益、年收益、初始投资和静态回收期。
    """

    operation_days = int(finance["operation_days_per_year"])
    investment_cost_per_wh = float(finance["investment_cost_per_wh"])

    charge_cost = (result_df["charge_kw"] * result_df["price"]).sum()
    discharge_revenue = (result_df["discharge_kw"] * result_df["price"]).sum()

    daily_net_profit = discharge_revenue - charge_cost
    annual_profit = daily_net_profit * operation_days

    capacity_kwh = float(ess["capacity_kwh"])
    initial_investment = capacity_kwh * 1000 * investment_cost_per_wh

    payback_years = None
    if annual_profit > 0:
        payback_years = initial_investment / annual_profit

    charged_energy_kwh = result_df["charge_kw"].sum()
    discharged_energy_kwh = result_df["discharge_kw"].sum()

    max_original_load_kw = result_df["load_kw"].max()
    max_grid_power_kw = result_df["grid_power_kw"].max()
    peak_shaving_kw = max_original_load_kw - max_grid_power_kw

    return {
        "daily_charge_cost_yuan": round(charge_cost, 2),
        "daily_discharge_revenue_yuan": round(discharge_revenue, 2),
        "daily_net_profit_yuan": round(daily_net_profit, 2),
        "annual_profit_yuan": round(annual_profit, 2),
        "initial_investment_yuan": round(initial_investment, 2),
        "static_payback_years": round(payback_years, 2) if payback_years is not None else None,
        "charged_energy_kwh": round(charged_energy_kwh, 2),
        "discharged_energy_kwh": round(discharged_energy_kwh, 2),
        "max_original_load_kw": round(max_original_load_kw, 2),
        "max_grid_power_kw": round(max_grid_power_kw, 2),
        "peak_shaving_kw": round(peak_shaving_kw, 2),
    }


# ============================================================
# 6. 多容量方案对比
# ============================================================

def run_capacity_comparison(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    作用：
    对比不同功率/容量储能方案的收益和回收期。

    面试价值：
    说明不是拍脑袋定 500kW/1MWh，而是通过多方案比较做初步选型。
    """

    scenarios = config["capacity_scenarios"]
    base_ess = config["ess"]
    base_finance = config["finance"]
    strategy = config["strategy"]

    results = []

    for scenario in scenarios:
        ess = deepcopy(base_ess)
        finance = deepcopy(base_finance)

        ess["power_kw"] = scenario["power_kw"]
        ess["capacity_kwh"] = scenario["capacity_kwh"]
        finance["investment_cost_per_wh"] = scenario["investment_cost_per_wh"]

        result_df = simulate_strategy(data, ess, strategy)
        roi = calculate_roi(result_df, ess, finance)

        row = {
            "scenario": scenario["name"],
            "power_kw": scenario["power_kw"],
            "capacity_kwh": scenario["capacity_kwh"],
            "investment_cost_per_wh": scenario["investment_cost_per_wh"],
            **roi,
        }

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# 7. 敏感性分析
# ============================================================

def run_sensitivity_analysis(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    作用：
    分析投资成本、峰电价、谷电价变化对回收期的影响。

    面试价值：
    说明项目经济性不是固定值，而是受成本、电价结构、运行天数等条件影响。
    """

    base_ess = config["ess"]
    base_finance = config["finance"]
    strategy = config["strategy"]
    sensitivity = config["sensitivity"]

    results = []

    for investment_cost in sensitivity["investment_cost_per_wh"]:
        for peak_price in sensitivity["peak_price"]:
            for valley_price in sensitivity["valley_price"]:
                scenario_data = apply_price_scenario(data, peak_price, valley_price)

                ess = deepcopy(base_ess)
                finance = deepcopy(base_finance)
                finance["investment_cost_per_wh"] = investment_cost

                result_df = simulate_strategy(scenario_data, ess, strategy)
                roi = calculate_roi(result_df, ess, finance)

                row = {
                    "investment_cost_per_wh": investment_cost,
                    "peak_price": peak_price,
                    "valley_price": valley_price,
                    "price_spread": round(peak_price - valley_price, 2),
                    **roi,
                }

                results.append(row)

    return pd.DataFrame(results)


# ============================================================
# 8. 图表生成
# ============================================================

def plot_load_curve(result_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(result_df["hour"], result_df["load_kw"], marker="o", label="Original Load")
    plt.plot(result_df["hour"], result_df["grid_power_kw"], marker="o", label="Grid Power After ESS")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.title("Load Curve Before and After Energy Storage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "load_curve.png", dpi=200)
    plt.close()


def plot_soc_curve(result_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(result_df["hour"], result_df["soc_percent"], marker="o")
    plt.xlabel("Hour")
    plt.ylabel("SOC (%)")
    plt.title("Energy Storage SOC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "soc_curve.png", dpi=200)
    plt.close()


def plot_charge_discharge_curve(result_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.bar(result_df["hour"], result_df["charge_kw"], label="Charge Power")
    plt.bar(result_df["hour"], -result_df["discharge_kw"], label="Discharge Power")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.title("Charge and Discharge Power Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "charge_discharge_curve.png", dpi=200)
    plt.close()


def plot_capacity_comparison(comparison_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.bar(comparison_df["scenario"], comparison_df["static_payback_years"])
    plt.xlabel("Capacity Scenario")
    plt.ylabel("Static Payback Period (Years)")
    plt.title("Capacity Scenario Payback Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "capacity_payback_comparison.png", dpi=200)
    plt.close()


def plot_sensitivity_scatter(sensitivity_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.scatter(
        sensitivity_df["price_spread"],
        sensitivity_df["static_payback_years"],
        s=80
    )
    plt.xlabel("Peak-Valley Price Spread (Yuan/kWh)")
    plt.ylabel("Static Payback Period (Years)")
    plt.title("Sensitivity: Price Spread vs Payback Period")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensitivity_price_spread_payback.png", dpi=200)
    plt.close()


def generate_charts(
    base_result_df: pd.DataFrame,
    capacity_comparison_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame
):
    OUTPUT_DIR.mkdir(exist_ok=True)

    plot_load_curve(base_result_df)
    plot_soc_curve(base_result_df)
    plot_charge_discharge_curve(base_result_df)
    plot_capacity_comparison(capacity_comparison_df)
    plot_sensitivity_scatter(sensitivity_df)


# ============================================================
# 9. 结果导出
# ============================================================

def export_results(
    base_result_df: pd.DataFrame,
    base_roi: dict,
    capacity_comparison_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame
):
    OUTPUT_DIR.mkdir(exist_ok=True)

    excel_path = OUTPUT_DIR / "roi_result.xlsx"
    report_path = OUTPUT_DIR / "summary_report.md"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        base_result_df.to_excel(writer, sheet_name="base_hourly_strategy", index=False)
        pd.DataFrame([base_roi]).to_excel(writer, sheet_name="base_roi_summary", index=False)
        capacity_comparison_df.to_excel(writer, sheet_name="capacity_comparison", index=False)
        sensitivity_df.to_excel(writer, sheet_name="sensitivity_analysis", index=False)

    best_capacity = capacity_comparison_df.sort_values("static_payback_years").iloc[0]
    best_sensitivity = sensitivity_df.sort_values("static_payback_years").iloc[0]

    report = f"""# 储能收益测算摘要报告

## 1. 基准方案结果

| 指标 | 数值 |
|---|---:|
| 日充电成本 | {base_roi["daily_charge_cost_yuan"]} 元 |
| 日放电收益 | {base_roi["daily_discharge_revenue_yuan"]} 元 |
| 日净收益 | {base_roi["daily_net_profit_yuan"]} 元 |
| 年收益 | {base_roi["annual_profit_yuan"]} 元 |
| 初始投资 | {base_roi["initial_investment_yuan"]} 元 |
| 静态回收期 | {base_roi["static_payback_years"]} 年 |
| 日充电量 | {base_roi["charged_energy_kwh"]} kWh |
| 日放电量 | {base_roi["discharged_energy_kwh"]} kWh |
| 最大削峰量 | {base_roi["peak_shaving_kw"]} kW |

## 2. 多容量方案对比结论

当前容量对比中，静态回收期最短的方案为：

| 指标 | 数值 |
|---|---:|
| 方案 | {best_capacity["scenario"]} |
| 功率 | {best_capacity["power_kw"]} kW |
| 容量 | {best_capacity["capacity_kwh"]} kWh |
| 年收益 | {best_capacity["annual_profit_yuan"]} 元 |
| 初始投资 | {best_capacity["initial_investment_yuan"]} 元 |
| 静态回收期 | {best_capacity["static_payback_years"]} 年 |

说明：该结论基于当前 24 小时负荷曲线和电价条件，真实项目还需要结合全年负荷曲线、并网条件、设备报价和合同边界进一步校核。

## 3. 敏感性分析结论

当前敏感性分析中，回收期最短的组合为：

| 指标 | 数值 |
|---|---:|
| 单位投资成本 | {best_sensitivity["investment_cost_per_wh"]} 元/Wh |
| 峰电价 | {best_sensitivity["peak_price"]} 元/kWh |
| 谷电价 | {best_sensitivity["valley_price"]} 元/kWh |
| 峰谷价差 | {best_sensitivity["price_spread"]} 元/kWh |
| 年收益 | {best_sensitivity["annual_profit_yuan"]} 元 |
| 静态回收期 | {best_sensitivity["static_payback_years"]} 年 |

## 4. 方案判断

从模型结果可以看出，工商业储能项目经济性对以下因素高度敏感：

1. 峰谷价差；
2. 单位投资成本；
3. 储能容量配置；
4. 负荷曲线是否能充分吸收储能放电；
5. 是否能够叠加需量控制、光伏消纳等收益。

当前模型仍属于前期方案测算工具，不替代真实商业项目设计。

## 5. 输出文件

- roi_result.xlsx：包含基准方案、多容量方案对比、敏感性分析；
- load_curve.png：储能接入前后负荷曲线；
- soc_curve.png：SOC 曲线；
- charge_discharge_curve.png：充放电功率曲线；
- capacity_payback_comparison.png：多容量方案回收期对比图；
- sensitivity_price_spread_payback.png：峰谷价差与回收期敏感性图。
"""

    report_path.write_text(report, encoding="utf-8")


# ============================================================
# 10. 主程序入口
# ============================================================

def main():
    config = load_config()
    data = load_data()

    base_ess = config["ess"]
    base_finance = config["finance"]
    strategy = config["strategy"]

    # 基准方案
    base_result_df = simulate_strategy(data, base_ess, strategy)
    base_roi = calculate_roi(base_result_df, base_ess, base_finance)

    # 多容量方案对比
    capacity_comparison_df = run_capacity_comparison(data, config)

    # 敏感性分析
    sensitivity_df = run_sensitivity_analysis(data, config)

    # 导出结果
    export_results(
        base_result_df,
        base_roi,
        capacity_comparison_df,
        sensitivity_df
    )

    # 生成图表
    generate_charts(
        base_result_df,
        capacity_comparison_df,
        sensitivity_df
    )

    print("储能收益测算 V2.0 完成。")
    print("已生成：基准方案、多容量方案对比、敏感性分析。")
    print("结果文件已输出到 output/ 目录。")
    print()
    print("基准方案核心结果：")

    for key, value in base_roi.items():
        print(f"{key}: {value}")

    print()
    print("多容量方案对比：")
    print(capacity_comparison_df[
        [
            "scenario",
            "power_kw",
            "capacity_kwh",
            "annual_profit_yuan",
            "initial_investment_yuan",
            "static_payback_years",
        ]
    ])

    print()
    print("敏感性分析样本数量：", len(sensitivity_df))


if __name__ == "__main__":
    main()

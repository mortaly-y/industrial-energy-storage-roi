from pathlib import Path

import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "project_config.yaml"
LOAD_PATH = ROOT_DIR / "data" / "load_profile_24h.csv"
PRICE_PATH = ROOT_DIR / "data" / "electricity_price.csv"
OUTPUT_DIR = ROOT_DIR / "output"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data():
    load_df = pd.read_csv(LOAD_PATH)
    price_df = pd.read_csv(PRICE_PATH)
    data = pd.merge(load_df, price_df, on="hour", how="left")
    return data


def simulate_strategy(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    ess = config["ess"]
    strategy = config["strategy"]

    power_kw = ess["power_kw"]
    capacity_kwh = ess["capacity_kwh"]
    soc_min = ess["soc_min"]
    soc_max = ess["soc_max"]
    soc = ess["initial_soc"]
    efficiency = ess["efficiency"]

    charge_type = strategy["charge_price_type"]
    discharge_type = strategy["discharge_price_type"]
    anti_reverse_power = strategy["anti_reverse_power"]

    records = []

    for _, row in data.iterrows():
        hour = int(row["hour"])
        load_kw = float(row["load_kw"])
        price_type = row["price_type"]
        price = float(row["price"])

        charge_kw = 0.0
        discharge_kw = 0.0
        action = "idle"

        available_charge_kwh = (soc_max - soc) * capacity_kwh
        available_discharge_kwh = (soc - soc_min) * capacity_kwh

        if price_type == charge_type and available_charge_kwh > 0:
            charge_kw = min(power_kw, available_charge_kwh)
            soc += charge_kw * efficiency / capacity_kwh
            action = "charge"

        elif price_type == discharge_type and available_discharge_kwh > 0:
            discharge_limit = min(power_kw, available_discharge_kwh)

            if anti_reverse_power:
                discharge_limit = min(discharge_limit, load_kw)

            discharge_kw = discharge_limit
            soc -= discharge_kw / efficiency / capacity_kwh
            action = "discharge"

        grid_power_kw = load_kw + charge_kw - discharge_kw

        records.append(
            {
                "hour": hour,
                "load_kw": load_kw,
                "price_type": price_type,
                "price": price,
                "action": action,
                "charge_kw": round(charge_kw, 2),
                "discharge_kw": round(discharge_kw, 2),
                "soc": round(soc, 4),
                "grid_power_kw": round(grid_power_kw, 2),
            }
        )

    return pd.DataFrame(records)


def calculate_roi(result_df: pd.DataFrame, config: dict) -> dict:
    ess = config["ess"]
    finance = config["finance"]

    operation_days = finance["operation_days_per_year"]
    investment_cost_per_wh = finance["investment_cost_per_wh"]

    charge_cost = (result_df["charge_kw"] * result_df["price"]).sum()
    discharge_revenue = (result_df["discharge_kw"] * result_df["price"]).sum()

    daily_net_profit = discharge_revenue - charge_cost
    annual_profit = daily_net_profit * operation_days

    initial_investment = ess["capacity_kwh"] * 1000 * investment_cost_per_wh

    if annual_profit > 0:
        payback_years = initial_investment / annual_profit
    else:
        payback_years = None

    return {
        "daily_charge_cost_yuan": round(charge_cost, 2),
        "daily_discharge_revenue_yuan": round(discharge_revenue, 2),
        "daily_net_profit_yuan": round(daily_net_profit, 2),
        "annual_profit_yuan": round(annual_profit, 2),
        "initial_investment_yuan": round(initial_investment, 2),
        "static_payback_years": round(payback_years, 2) if payback_years else None,
    }


def export_results(result_df: pd.DataFrame, roi: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)

    excel_path = OUTPUT_DIR / "roi_result.xlsx"
    report_path = OUTPUT_DIR / "summary_report.md"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="hourly_strategy", index=False)
        pd.DataFrame([roi]).to_excel(writer, sheet_name="roi_summary", index=False)

    report = f"""# 储能收益测算摘要报告

## 核心结果

| 指标 | 数值 |
|---|---:|
| 日充电成本 | {roi["daily_charge_cost_yuan"]} 元 |
| 日放电收益 | {roi["daily_discharge_revenue_yuan"]} 元 |
| 日净收益 | {roi["daily_net_profit_yuan"]} 元 |
| 年收益 | {roi["annual_profit_yuan"]} 元 |
| 初始投资 | {roi["initial_investment_yuan"]} 元 |
| 静态回收期 | {roi["static_payback_years"]} 年 |

## 说明

本结果基于 24 小时典型负荷曲线、峰谷电价、500kW/1MWh 储能配置和基础谷充峰放策略计算，仅用于前期方案测算与作品集展示。
"""

    report_path.write_text(report, encoding="utf-8")


def main():
    config = load_config()
    data = load_data()
    result_df = simulate_strategy(data, config)
    roi = calculate_roi(result_df, config)
    export_results(result_df, roi)

    print("储能收益测算完成。")
    print("核心结果：")
    for key, value in roi.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
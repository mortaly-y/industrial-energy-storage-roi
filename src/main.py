from pathlib import Path

import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径配置
# 作用：告诉程序去哪里读取配置、负荷数据、电价数据，以及把结果输出到哪里。
# 一般不需要修改，除非你改变了仓库文件夹结构。
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT_DIR / "config" / "project_config.yaml"
LOAD_PATH = ROOT_DIR / "data" / "load_profile_24h.csv"
PRICE_PATH = ROOT_DIR / "data" / "electricity_price.csv"
OUTPUT_DIR = ROOT_DIR / "output"


# ============================================================
# 2. 读取配置文件
# 作用：读取储能功率、容量、SOC上下限、系统效率、投资成本等参数。
# 参数不要在代码里硬改，优先去 config/project_config.yaml 里修改。
# ============================================================

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ============================================================
# 3. 读取负荷曲线和电价表
# 作用：把 data/load_profile_24h.csv 和 data/electricity_price.csv 合并成一张表。
#
# 需要注意：
# - load_profile_24h.csv 必须有 hour, load_kw 两列；
# - electricity_price.csv 必须有 hour, price_type, price 三列；
# - hour 必须是 0 到 23。
# ============================================================

def load_data() -> pd.DataFrame:
    load_df = pd.read_csv(LOAD_PATH)
    price_df = pd.read_csv(PRICE_PATH)

    data = pd.merge(load_df, price_df, on="hour", how="left")

    if data["price"].isna().any():
        raise ValueError("电价数据存在缺失，请检查 electricity_price.csv 中是否覆盖 0-23 小时。")

    return data


# ============================================================
# 4. 储能运行策略模拟
# 作用：按照“谷段充电、峰段放电”的简化 EMS 策略，模拟 24 小时运行情况。
#
# 主要逻辑：
# - 谷段：如果 SOC 低于上限，则充电；
# - 峰段：如果 SOC 高于下限，则放电；
# - 防逆流：放电功率不能大于当前负荷；
# - 平段：默认不动作。
#
# 可修改位置：
# - 若想增加“平段补电/平段放电”，就在这里改策略逻辑；
# - 若想增加“需量控制”，也在这里增加 demand_limit 判断。
# ============================================================

def simulate_strategy(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    ess = config["ess"]
    strategy = config["strategy"]

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

        # 当前还能充多少电，单位 kWh
        available_charge_kwh = max((soc_max - soc) * capacity_kwh, 0)

        # 当前还能放多少电，单位 kWh
        available_discharge_kwh = max((soc - soc_min) * capacity_kwh, 0)

        # 谷段充电
        if price_type == charge_type and available_charge_kwh > 0:
            charge_kw = min(power_kw, available_charge_kwh)
            soc += charge_kw * efficiency / capacity_kwh
            action = "charge"

        # 峰段放电
        elif price_type == discharge_type and available_discharge_kwh > 0:
            discharge_limit = min(power_kw, available_discharge_kwh)

            # 防逆流：储能放电不能超过当前负荷，否则可能向电网反送电
            if anti_reverse_power:
                discharge_limit = min(discharge_limit, load_kw)

            discharge_kw = discharge_limit
            soc -= discharge_kw / efficiency / capacity_kwh
            action = "discharge"

        # 防止浮点误差导致 SOC 超界
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
# 5. 收益测算
# 作用：根据模拟出来的充放电功率和电价，计算日收益、年收益和静态回收期。
#
# 核心公式：
# - 日充电成本 = Σ 充电功率 × 当前电价
# - 日放电收益 = Σ 放电功率 × 当前电价
# - 日净收益 = 日放电收益 - 日充电成本
# - 年收益 = 日净收益 × 年运行天数
# - 初始投资 = 储能容量(kWh) × 1000 × 单位投资成本(元/Wh)
# - 静态回收期 = 初始投资 ÷ 年收益
#
# 可修改位置：
# - 投资成本在 config/project_config.yaml 里改；
# - 年运行天数在 config/project_config.yaml 里改；
# - 后续可增加运维成本、电池衰减、需量收益等。
# ============================================================

def calculate_roi(result_df: pd.DataFrame, config: dict) -> dict:
    ess = config["ess"]
    finance = config["finance"]

    operation_days = int(finance["operation_days_per_year"])
    investment_cost_per_wh = float(finance["investment_cost_per_wh"])

    charge_cost = (result_df["charge_kw"] * result_df["price"]).sum()
    discharge_revenue = (result_df["discharge_kw"] * result_df["price"]).sum()

    daily_net_profit = discharge_revenue - charge_cost
    annual_profit = daily_net_profit * operation_days

    initial_investment = float(ess["capacity_kwh"]) * 1000 * investment_cost_per_wh

    payback_years = None
    if annual_profit > 0:
        payback_years = initial_investment / annual_profit

    return {
        "daily_charge_cost_yuan": round(charge_cost, 2),
        "daily_discharge_revenue_yuan": round(discharge_revenue, 2),
        "daily_net_profit_yuan": round(daily_net_profit, 2),
        "annual_profit_yuan": round(annual_profit, 2),
        "initial_investment_yuan": round(initial_investment, 2),
        "static_payback_years": round(payback_years, 2) if payback_years is not None else None,
    }


# ============================================================
# 6. 图表生成
# 作用：生成三张图，方便放进 PPT / README / 作品集。
#
# 输出文件：
# - output/load_curve.png
# - output/soc_curve.png
# - output/charge_discharge_curve.png
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


def generate_charts(result_df: pd.DataFrame):
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_load_curve(result_df)
    plot_soc_curve(result_df)
    plot_charge_discharge_curve(result_df)


# ============================================================
# 7. 导出结果
# 作用：把小时级运行策略和收益摘要导出到 Excel，并生成 Markdown 报告。
#
# 输出文件：
# - output/roi_result.xlsx
# - output/summary_report.md
# ============================================================

def export_results(result_df: pd.DataFrame, roi: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)

    excel_path = OUTPUT_DIR / "roi_result.xlsx"
    report_path = OUTPUT_DIR / "summary_report.md"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="hourly_strategy", index=False)
        pd.DataFrame([roi]).to_excel(writer, sheet_name="roi_summary", index=False)

    report = f"""# 储能收益测算摘要报告

## 1. 核心结果

| 指标 | 数值 |
|---|---:|
| 日充电成本 | {roi["daily_charge_cost_yuan"]} 元 |
| 日放电收益 | {roi["daily_discharge_revenue_yuan"]} 元 |
| 日净收益 | {roi["daily_net_profit_yuan"]} 元 |
| 年收益 | {roi["annual_profit_yuan"]} 元 |
| 初始投资 | {roi["initial_investment_yuan"]} 元 |
| 静态回收期 | {roi["static_payback_years"]} 年 |

## 2. 输出图表

- load_curve.png：储能接入前后负荷曲线；
- soc_curve.png：储能 SOC 曲线；
- charge_discharge_curve.png：储能充放电功率曲线。

## 3. 说明

本结果基于 24 小时典型负荷曲线、峰谷电价、500kW/1MWh 储能配置和基础谷充峰放策略计算，仅用于前期方案测算与作品集展示。

真实储能项目还需要进一步考虑：

1. 当地实际分时电价政策；
2. 负荷曲线的全年波动；
3. 电池衰减；
4. 运维成本；
5. 需量收益；
6. 并网和消防要求；
7. 设备厂家实际效率和质保边界。
"""

    report_path.write_text(report, encoding="utf-8")


# ============================================================
# 8. 主程序入口
# 作用：按顺序执行完整流程。
#
# 执行顺序：
# 1. 读取配置
# 2. 读取数据
# 3. 模拟策略
# 4. 计算收益
# 5. 导出 Excel 和报告
# 6. 生成图表
# 7. 在终端打印结果
# ============================================================

def main():
    config = load_config()
    data = load_data()

    result_df = simulate_strategy(data, config)
    roi = calculate_roi(result_df, config)

    export_results(result_df, roi)
    generate_charts(result_df)

    print("储能收益测算完成。")
    print("结果文件已生成到 output/ 目录。")
    print("核心结果：")

    for key, value in roi.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

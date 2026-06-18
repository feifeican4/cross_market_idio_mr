"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import StrategyRunResult


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a small DataFrame as GitHub-style Markdown."""
    if df.empty:
        return "_空表_"
    view = df.head(max_rows).copy()
    view = view.where(pd.notna(view), "")
    headers = [str(col) for col in view.columns]
    rows = [headers]
    for row in view.itertuples(index=False, name=None):
        rows.append([str(value) for value in row])

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def render_row(values: list[str]) -> str:
        cells = [value.ljust(widths[i]) for i, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    lines = [render_row(rows[0])]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows[1:]:
        lines.append(render_row(row))
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def generate_report(result: StrategyRunResult, output_path: str | Path) -> None:
    """Write the base Chinese Markdown research report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    settings = result.config.settings
    target_count = len(result.config.target_symbols)
    benchmark_count = len(result.config.benchmark_symbols)

    lines = [
        "# 跨市场特质均值回归策略报告",
        "",
        "## 1. 策略说明",
        "本策略用滚动多因子回归剥离系统性风险，只交易资产相对因子模型的特质残差。",
        "当残差 z-score 过高时做空目标资产并做多因子对冲腿；当残差 z-score 过低时做多目标资产并做空因子对冲腿。",
        "策略目标不是预测市场方向，而是检验短期特质偏离是否存在均值回归。",
        "",
        "## 2. 作业要求对应",
        "- 数据频率：日频主流程；1-4 小时频作为可扩展接口，当前未作为主结果展示。",
        f"- 标的数量：目标资产 {target_count} 个，基准/因子 {benchmark_count} 个。",
        "- 资产类别：币股、加密 altcoins、美股科技股均在 `configs/universe.yaml` 中配置。",
        "- 因子模型：每个目标资产单独做滚动 OLS，并输出 R2 与 ADF 残差平稳性检验。",
        "- 信号规则：进入阈值使用 +/-entry_z，退出阈值使用 +/-exit_z。",
        "- 组合构建：按 pair volatility 做风险预算，并施加单资产、板块、杠杆、因子暴露约束。",
        "- 回测成本：计入手续费、滑点、借券成本和 perp funding 代理成本。",
        "- 报告分析：包含绩效、风险、参数敏感性、成本敏感性、容量与局限性。",
        "",
        "## 3. 配置",
        f"- 回测区间: {settings.start_date} ~ {settings.end_date}",
        f"- 回归窗口: {settings.regression_window}",
        f"- z-score 窗口: {settings.zscore_window}",
        f"- 进入阈值: {settings.entry_z}",
        f"- 退出阈值: {settings.exit_z}",
        f"- 单资产仓位上限: {settings.max_single_weight:.2%}",
        f"- 同板块暴露上限: {settings.max_group_weight:.2%}",
        f"- 总杠杆上限: {settings.max_gross_leverage:.1f}x",
        f"- 单因子净暴露上限: {settings.max_factor_weight:.2%}",
        f"- 股票借券成本假设: {settings.annual_borrow_rate:.2%} 年化",
        f"- perp funding 代理成本假设: {settings.annual_funding_rate:.2%} 年化",
        "",
        "## 4. 业绩指标",
    ]

    for key, value in result.summary.items():
        lines.append(f"- {key}: {_format_metric(value)}")

    lines.extend(
        [
            "",
            "## 5. 回撤期分析",
            _df_to_markdown(result.drawdown_table),
            "",
            "## 6. 因子模型诊断",
            "avg_r2 越高，说明因子模型解释力越强；ADF p-value 越低，越支持残差平稳，均值回归假设更可信。",
            _df_to_markdown(result.diagnostics.sort_values("avg_r2", ascending=False)),
            "",
            "## 7. 最近 20 天净值与成本",
            _df_to_markdown(result.backtest.daily.tail(20).reset_index().rename(columns={"index": "date"})),
            "",
            "## 8. 方法论说明",
            "1. 每个资产单独做滚动 OLS，避免把不同资产硬塞进同一个回归。",
            "2. 回归、z-score 均只使用历史数据，并在交易中 shift 一天，避免前视偏差。",
            "3. 目标资产腿和因子对冲腿同时建仓，目标是控制 BTC、ETH、SPY、QQQ、SMH 等系统风险。",
            "4. 回测按换手计入手续费和滑点，股票空头计入借券成本，perp 腿预留 funding 成本。",
            "5. demo 结果没有做参数优化；负收益同样是有效研究结果，说明当前设定下 alpha 不足或成本过高。",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def append_analysis_tables(
    report_path: str | Path,
    parameter_sensitivity: pd.DataFrame | None = None,
    cost_sensitivity: pd.DataFrame | None = None,
    capacity: pd.DataFrame | None = None,
) -> None:
    """Append sensitivity and capacity sections to an existing report."""
    path = Path(report_path)
    additions: list[str] = []

    if parameter_sensitivity is not None and not parameter_sensitivity.empty:
        additions.extend(
            [
                "",
                "## 9. 参数敏感性分析",
                "这一部分检查策略是否只在单一参数下好看。结果不用于挑选最好 Sharpe，而是用于观察稳定性。",
                _df_to_markdown(parameter_sensitivity),
            ]
        )

    if cost_sensitivity is not None and not cost_sensitivity.empty:
        additions.extend(
            [
                "",
                "## 10. 交易成本敏感性分析",
                "这一部分把手续费和滑点放大，观察策略是否会被交易成本吞噬。",
                _df_to_markdown(cost_sensitivity),
            ]
        )

    if capacity is not None and not capacity.empty:
        additions.extend(
            [
                "",
                "## 11. 容量分析",
                "这里用 ADV 参与率估算容量。若没有真实成交额数据，表中的 ADV 是代理值，不能直接作为实盘容量结论。",
                _df_to_markdown(capacity.reset_index().rename(columns={"index": "symbol"})),
            ]
        )

    additions.extend(
        [
            "",
            "## 12. 策略局限",
            "1. yfinance 和合成数据只能用于研究与代码验证，不能替代真实 Binance TradFi perp 历史价格。",
            "2. funding rate、订单簿深度、真实可成交滑点需要接入交易所历史数据后重新估计。",
            "3. ADF 通过只说明样本内残差更像平稳序列，不保证未来一定均值回归。",
            "4. 线性 beta 在 regime 切换时会失效，特别是危机行情和极端散户情绪行情。",
            "5. 交易成本和容量是这类策略的生命线；毛 alpha 很容易被手续费、滑点、funding 和 borrow 吃掉。",
        ]
    )

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + "\n" + "\n".join(additions), encoding="utf-8")


def append_bonus_report(report_path: str | Path, bonus_result: object) -> None:
    """Append bonus module results and caveats to the report."""
    path = Path(report_path)
    additions = [
        "",
        "## 13. 加分项 1：跨市场基差套利模块",
        "这里用现货价格构造合成 perp 代理，展示 spot-perp basis 的研究框架。真实提交时应替换为 Binance 美股/加密 perp 历史价格和 funding rate。",
    ]
    basis_summary = getattr(bonus_result, "basis_summary", {})
    for key, value in basis_summary.items():
        additions.append(f"- {key}: {_format_metric(value)}")

    additions.extend(
        [
            "",
            "## 14. 加分项 2：动态因子选择",
            "动态因子选择用 stepwise AIC 和 LASSO 做候选因子筛选。它展示了如何让模型随市场关系变化而调整，但也更容易过拟合，所以这里只作为研究扩展，不用来美化主策略。",
        ]
    )
    dynamic_summary = getattr(bonus_result, "dynamic_summary", {})
    for key, value in dynamic_summary.items():
        additions.append(f"- {key}: {_format_metric(value)}")

    selection = getattr(bonus_result, "dynamic_selection", pd.DataFrame())
    if isinstance(selection, pd.DataFrame) and not selection.empty:
        additions.extend(["", "动态因子选择样例：", _df_to_markdown(selection.tail(12))])

    additions.extend(
        [
            "",
            "## 15. 加分项 3：可解释 ML 残差回归过滤",
            "ML 模块使用 walk-forward logistic regression 预测未来 5 天残差绝对值是否收敛。它不直接预测价格，只作为交易过滤器，特征包括残差 z-score、残差变化、残差波动和因子波动。",
        ]
    )
    ml_summary = getattr(bonus_result, "ml_summary", {})
    for key, value in ml_summary.items():
        additions.append(f"- {key}: {_format_metric(value)}")

    ml_classification = getattr(bonus_result, "ml_classification", {})
    if ml_classification:
        additions.append("")
        additions.append("ML 分类诊断：")
        for key, value in ml_classification.items():
            additions.append(f"- {key}: {_format_metric(value)}")

    coef = getattr(bonus_result, "ml_coefficients", pd.DataFrame())
    if isinstance(coef, pd.DataFrame) and not coef.empty:
        additions.extend(["", "ML 最后一轮模型系数：", _df_to_markdown(coef)])

    additions.extend(
        [
            "",
            "## 16. 总反思",
            "1. 主策略没有追求漂亮 Sharpe。demo 结果为负，说明框架是诚实研究管线，不是参数美化器。",
            "2. 最大风险是残差不回归。因子模型只解释线性关系，市场 regime 切换时 beta 和残差分布都会变。",
            "3. 数据质量是实盘前最大缺口。yfinance、合成 perp 和默认 ADV 只能证明方法，不能证明 Binance 可成交利润。",
            "4. 动态因子和 ML 是双刃剑。它们提升适应性，但也显著增加过拟合风险，应作为稳健性和研究扩展，而不是收益宣传来源。",
        ]
    )

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + "\n" + "\n".join(additions), encoding="utf-8")

"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import StrategyRunResult


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_Empty_"
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


def generate_report(result: StrategyRunResult, output_path: str | Path) -> None:
    """Write a readable Markdown report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    settings = result.config.settings
    lines = [
        "# 跨市场特质均值回归策略报告",
        "",
        "## 1. 策略说明",
        "本策略使用滚动因子回归剥离系统性风险，用残差 z-score 触发反向交易。",
        "交易对象是资产相对因子组合的异常偏离，而不是单纯预测涨跌。",
        "",
        "## 2. 配置",
        f"- 回测区间: {settings.start_date} ~ {settings.end_date}",
        f"- 回归窗口: {settings.regression_window}",
        f"- z-score 窗口: {settings.zscore_window}",
        f"- 进场阈值: {settings.entry_z}",
        f"- 出场阈值: {settings.exit_z}",
        f"- 单票上限: {settings.max_single_weight:.2%}",
        f"- 板块上限: {settings.max_group_weight:.2%}",
        f"- 总杠杆上限: {settings.max_gross_leverage:.1f}x",
        "",
        "## 3. 绩效指标",
    ]

    for key, value in result.summary.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## 4. 回撤分析",
            _df_to_markdown(result.drawdown_table),
            "",
            "## 5. 因子模型诊断",
            _df_to_markdown(result.diagnostics.sort_values("avg_r2", ascending=False)),
            "",
            "## 6. 最近 20 天净值",
            _df_to_markdown(result.backtest.daily.tail(20).reset_index().rename(columns={"index": "date"})),
            "",
            "## 7. 研究说明",
            "1. 每个资产单独做滚动 OLS，避免把不同资产硬塞进一个回归里。",
            "2. 残差 z-score 使用历史窗口的均值和标准差，并 shift 一天避免前视偏差。",
            "3. 仓位构建时对资产腿和因子腿分别加约束，再统一控制总杠杆。",
            "4. 回测中计入手续费、滑点、借券和 funding，避免用毛收益误导判断。",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def append_analysis_tables(
    report_path: str | Path,
    parameter_sensitivity: pd.DataFrame | None = None,
    cost_sensitivity: pd.DataFrame | None = None,
    capacity: pd.DataFrame | None = None,
) -> None:
    """Append sensitivity and capacity tables to an existing report."""
    path = Path(report_path)
    additions: list[str] = []

    if parameter_sensitivity is not None and not parameter_sensitivity.empty:
        additions.extend(
            [
                "",
                "## 8. 参数敏感性分析",
                "这一部分检查策略是否只在单一参数下好看。如果多个参数组合都能保持合理表现，说明策略更不容易是过拟合。",
                _df_to_markdown(parameter_sensitivity),
            ]
        )

    if cost_sensitivity is not None and not cost_sensitivity.empty:
        additions.extend(
            [
                "",
                "## 9. 交易成本敏感性分析",
                "这一部分把手续费和滑点放大，观察策略是否会被成本吞噬。",
                _df_to_markdown(cost_sensitivity),
            ]
        )

    if capacity is not None and not capacity.empty:
        additions.extend(
            [
                "",
                "## 10. 容量分析",
                "这里用 ADV 参与率估计容量。若没有真实成交额数据，表中的 ADV 是代理值，不能直接作为实盘容量结论。",
                _df_to_markdown(capacity.reset_index().rename(columns={"index": "symbol"})),
            ]
        )

    additions.extend(
        [
            "",
            "## 11. 局限性",
            "1. yfinance 和合成数据只能用于研究与代码验证，不能替代真实 Binance TradFi perp 历史价格。",
            "2. funding rate、订单簿深度和真实可成交滑点需要接入交易所历史数据后重新估计。",
            "3. 残差平稳性会随市场状态变化，ADF 通过不代表未来一定均值回归。",
            "4. 当前版本没有做加分项中的基差套利、动态因子选择和机器学习增强。",
        ]
    )

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + "\n" + "\n".join(additions), encoding="utf-8")

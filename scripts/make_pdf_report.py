"""Generate a portrait Chinese PDF report from backtest outputs."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_market_mr.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
A4 = (8.27, 11.69)


def setup_chinese_font() -> None:
    """Use a local Chinese font when available."""
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def load_outputs(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Load available output files from a backtest directory."""
    tables = {
        "daily": _read_csv(output_dir / "backtest_daily.csv", parse_dates=["date"], index_col="date"),
        "diagnostics": _read_csv(output_dir / "diagnostics.csv"),
        "parameter": _read_csv(output_dir / "parameter_sensitivity.csv"),
        "cost": _read_csv(output_dir / "cost_sensitivity.csv"),
        "capacity": _read_csv(output_dir / "capacity_analysis.csv").rename(columns={"Unnamed: 0": "symbol"}),
        "factor_exposure": _read_csv(output_dir / "factor_exposure.csv", index_col=0, parse_dates=True),
        "bonus": _read_csv(output_dir / "bonus_summary.csv"),
        "basis_daily": _read_csv(output_dir / "basis_daily.csv", parse_dates=["date"], index_col="date"),
        "dynamic_daily": _read_csv(output_dir / "dynamic_daily.csv", parse_dates=["date"], index_col="date"),
        "ml_daily": _read_csv(output_dir / "ml_daily.csv", parse_dates=["date"], index_col="date"),
    }
    intraday_dir = output_dir.parent / "intraday_real"
    tables["intraday"] = _read_csv(
        intraday_dir / "backtest_daily.csv",
        parse_dates=["date"],
        index_col="date",
    )
    return tables


def _page_number(fig: plt.Figure, page: int) -> None:
    fig.text(0.94, 0.025, str(page), ha="right", va="bottom", fontsize=8, color="black")


def _wrap(text: str, width: int = 62) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=width))
    return "\n".join(lines)


def text_page(pdf: PdfPages, title: str, body: str, page: int) -> None:
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=20, weight="bold", color="black", va="top")
    fig.text(0.09, 0.88, _wrap(body), fontsize=11, color="black", va="top", linespacing=1.45)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _format_pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else ""


def _format_num(value: float) -> str:
    return f"{value:.3f}" if pd.notna(value) else ""


def _metric_box(ax: plt.Axes, x: float, y: float, title: str, value: str) -> None:
    ax.add_patch(plt.Rectangle((x, y), 0.38, 0.15, fill=False, edgecolor="black", linewidth=0.9))
    ax.text(x + 0.02, y + 0.10, title, fontsize=10, weight="bold", transform=ax.transAxes)
    ax.text(x + 0.02, y + 0.04, value, fontsize=16, weight="bold", transform=ax.transAxes)


def table_page(pdf: PdfPages, title: str, df: pd.DataFrame, page: int, max_rows: int = 15) -> None:
    fig, ax = plt.subplots(figsize=A4)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, fontsize=17, weight="bold", loc="left", pad=15, color="black")
    if df.empty:
        ax.text(0.05, 0.7, "无可用数据", fontsize=12, color="black", transform=ax.transAxes)
    else:
        view = df.head(max_rows).copy()
        for col in view.columns:
            if pd.api.types.is_numeric_dtype(view[col]):
                view[col] = view[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        table = ax.table(
            cellText=view.astype(str).values,
            colLabels=[str(col) for col in view.columns],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        font_size = 10 if len(view.columns) <= 5 else 8.5
        table.set_fontsize(font_size)
        table.scale(1, 1.55 if len(view.columns) <= 5 else 1.35)
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.35)
            if row == 0:
                cell.set_text_props(weight="bold", color="black")
            else:
                cell.set_text_props(color="black")
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def summary_page(pdf: PdfPages, daily: pd.DataFrame, config, page: int) -> None:
    ret = daily["net_return"].dropna()
    equity = daily["equity_curve"]
    annual_return = (1 + ret).prod() ** (252 / max(len(ret), 1)) - 1
    annual_vol = ret.std(ddof=1) * np.sqrt(252)
    sharpe = annual_return / max(annual_vol, 1e-12)
    max_dd = drawdown(equity).min()
    fig, ax = plt.subplots(figsize=A4)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    left, right = 0.07, 0.93
    ax.text(left, 0.96, "跨市场特质均值回归策略", fontsize=22, weight="bold", transform=ax.transAxes)
    ax.text(left, 0.915, "回测报告 | 真实日频主策略 + 真实4小时频加分项", fontsize=14, weight="bold", transform=ax.transAxes)
    ax.plot([left, right], [0.89, 0.89], color="black", linewidth=0.8, transform=ax.transAxes)

    # Metric panel: a clean 2x2 table with identical left/right margins.
    x0, y0, w, h = left, 0.66, right - left, 0.20
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, edgecolor="black", linewidth=0.9, transform=ax.transAxes))
    ax.plot([x0 + w / 2, x0 + w / 2], [y0, y0 + h], color="black", linewidth=0.7, transform=ax.transAxes)
    ax.plot([x0, x0 + w], [y0 + h / 2, y0 + h / 2], color="black", linewidth=0.7, transform=ax.transAxes)
    metrics = [
        ("年化收益", _format_pct(annual_return), x0 + 0.03, y0 + h * 0.63),
        ("年化波动", _format_pct(annual_vol), x0 + w / 2 + 0.03, y0 + h * 0.63),
        ("Sharpe", _format_num(sharpe), x0 + 0.03, y0 + h * 0.13),
        ("最大回撤", _format_pct(max_dd), x0 + w / 2 + 0.03, y0 + h * 0.13),
    ]
    for title, value, tx, ty in metrics:
        ax.text(tx, ty + 0.045, title, fontsize=10.5, weight="bold", transform=ax.transAxes)
        ax.text(tx, ty, value, fontsize=15.5, weight="bold", transform=ax.transAxes)

    summary_lines = [
        f"本次回测使用 {len(config.target_symbols)} 个目标资产和 {len(config.factor_symbols)} 个因子，",
        "目标资产覆盖币股、加密 altcoins、美股科技 Perp proxy 三类。",
        "",
        "策略先用 BTC、ETH、SPY、QQQ、SMH 等因子解释资产收益，",
        "再交易无法被因子解释的特质残差。",
        "",
        "组合不是单边押方向，而是目标资产腿与因子对冲腿同时建仓。",
        "回测结果显示，当前简单 residual z-score 反转在扣除成本后表现为负，",
        "主要原因是残差回归幅度不足以覆盖交易摩擦。",
    ]
    ax.text(
        left,
        0.58,
        "\n".join(summary_lines),
        fontsize=12.5,
        va="top",
        linespacing=1.55,
        transform=ax.transAxes,
    )
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def data_page(pdf: PdfPages, page: int) -> None:
    fig, ax = plt.subplots(figsize=A4)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(0.02, 0.96, "2. 数据说明", fontsize=19, weight="bold", transform=ax.transAxes)
    left = (
        "资产分类\n\n"
        "币股：\nMSTR, COIN, MARA, RIOT, CLSK, HOOD, SQ\n\n"
        "加密 Altcoins：\nSOL, BNB, ADA, XRP, DOGE, AVAX, LINK, LTC, BCH, DOT, AAVE, ARB, OP, UNI\n\n"
        "美股科技 Perp Proxy：\nNVDA, TSLA, AMD, META, MSFT, AAPL, GOOGL, AMZN"
    )
    right = (
        "因子与频率\n\n"
        "因子：BTC, ETH, SPY, QQQ, SMH\n\n"
        "主流程：日频 close-to-close\n\n"
        "加分项：Binance 官方历史归档真实 4 小时 K 线实验\n\n"
        "代码：\nconfigs/universe.yaml\nsrc/cross_market_mr/data.py\nscripts/run_intraday_real.py"
    )
    ax.text(0.05, 0.86, _wrap(left, 36), fontsize=10.8, va="top", linespacing=1.35, transform=ax.transAxes)
    ax.text(0.55, 0.86, _wrap(right, 34), fontsize=10.8, va="top", linespacing=1.35, transform=ax.transAxes)
    ax.plot([0.50, 0.50], [0.16, 0.88], color="black", linewidth=0.6, transform=ax.transAxes)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def method_page(pdf: PdfPages, page: int) -> None:
    fig, ax = plt.subplots(figsize=A4)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(0.02, 0.96, "3. 方法与代码对应", fontsize=19, weight="bold", transform=ax.transAxes)
    steps = [
        ("1", "滚动因子模型", "factor_model.py::fit_rolling_factor_model"),
        ("2", "残差 z-score", "signals.py::rolling_zscore"),
        ("3", "进出场信号", "signals.py::build_hysteresis_signal"),
        ("4", "多腿对冲组合", "portfolio.py::build_pair_weight_frame"),
        ("5", "风控约束", "portfolio.py::apply_risk_caps"),
        ("6", "成本后回测", "backtest.py::run_backtest"),
    ]
    y = 0.83
    for num, name, code in steps:
        ax.add_patch(plt.Rectangle((0.06, y - 0.055), 0.10, 0.08, fill=False, edgecolor="black", linewidth=0.8))
        ax.text(0.105, y - 0.015, num, fontsize=14, weight="bold", ha="center", transform=ax.transAxes)
        ax.text(0.20, y + 0.005, name, fontsize=12, weight="bold", transform=ax.transAxes)
        ax.text(0.20, y - 0.035, code, fontsize=10, transform=ax.transAxes)
        y -= 0.12
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_equity(pdf: PdfPages, daily: pd.DataFrame, page: int) -> None:
    equity = daily["equity_curve"]
    dd = drawdown(equity)
    fig, axes = plt.subplots(2, 1, figsize=A4, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("4. 净值与回撤", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    axes[0].plot(equity.index, equity, color="black", linewidth=1.5)
    axes[0].set_ylabel("净值")
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(dd.index, dd.values, 0, color="black", alpha=0.25)
    axes[1].set_ylabel("回撤")
    axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    axes[1].grid(True, alpha=0.25)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_returns_costs(pdf: PdfPages, daily: pd.DataFrame, page: int) -> None:
    ret = daily["net_return"].dropna()
    costs = daily[["transaction_cost", "borrow_cost", "funding_cost"]].cumsum()
    fig, axes = plt.subplots(2, 1, figsize=A4)
    fig.suptitle("5. 收益分布与成本", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    axes[0].hist(ret, bins=45, color="black", alpha=0.8)
    axes[0].axvline(ret.mean(), color="black", linestyle="--", linewidth=1)
    axes[0].set_title("日度净收益分布")
    axes[0].grid(True, alpha=0.25)
    costs.plot(ax=axes[1], linewidth=1.4)
    axes[1].legend(["交易成本", "借券成本", "Funding 成本"], fontsize=9)
    axes[1].set_title("累计成本")
    axes[1].grid(True, alpha=0.25)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(pdf: PdfPages, diagnostics: pd.DataFrame, page: int) -> None:
    view = diagnostics.sort_values("avg_r2", ascending=False).head(15)
    fig, axes = plt.subplots(2, 1, figsize=A4)
    fig.suptitle("6. 因子模型诊断", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    axes[0].barh(view["asset"], view["avg_r2"], color="black", alpha=0.8)
    axes[0].invert_yaxis()
    axes[0].set_title("平均 R2 Top 15")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[1].scatter(diagnostics["avg_r2"], diagnostics["adf_p_value"], color="black", alpha=0.8)
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("avg_r2")
    axes[1].set_ylabel("ADF p-value")
    axes[1].set_title("残差平稳性检查")
    axes[1].grid(True, alpha=0.25)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_exposure_activity(pdf: PdfPages, tables: dict[str, pd.DataFrame], page: int) -> None:
    diagnostics = tables["diagnostics"].sort_values("active_days", ascending=False).head(12)
    exposure = tables["factor_exposure"]
    fig, axes = plt.subplots(2, 1, figsize=A4)
    fig.suptitle("7. 交易活跃度与因子暴露", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    axes[0].barh(diagnostics["asset"], diagnostics["active_days"], color="black", alpha=0.8)
    axes[0].invert_yaxis()
    axes[0].set_title("持仓天数 Top 12")
    axes[0].grid(True, axis="x", alpha=0.25)
    cols = [col for col in ["net_BTC", "net_ETH", "net_SPY", "net_QQQ", "net_SMH"] if col in exposure.columns]
    if cols:
        exposure[cols].rolling(20).mean().plot(ax=axes[1], linewidth=1.1)
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(-0.05, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("20期滚动平均净因子暴露")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_cost(pdf: PdfPages, tables: dict[str, pd.DataFrame], page: int) -> None:
    parameter = tables["parameter"]
    cost = tables["cost"]
    fig, axes = plt.subplots(2, 1, figsize=A4, constrained_layout=True)
    fig.suptitle("8. 参数与成本敏感性", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    if not parameter.empty:
        labels = parameter.apply(lambda r: f"z={r['entry_z']},w={int(r['regression_window'])}", axis=1)
        axes[0].bar(labels, parameter["sharpe"], color="black", alpha=0.8)
        axes[0].tick_params(axis="x", rotation=28, labelsize=7)
    axes[0].set_title("参数敏感性：不同参数下 Sharpe")
    axes[0].grid(True, axis="y", alpha=0.25)
    if not cost.empty:
        axes[1].plot(cost["cost_multiplier"], cost["annual_return"], color="black", marker="o", label="年化收益")
        axes[1].plot(cost["cost_multiplier"], cost["sharpe"], color="black", marker="x", linestyle="--", label="Sharpe")
        axes[1].legend(fontsize=8)
    axes[1].set_title("成本敏感性")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlabel("cost multiplier")
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_capacity(pdf: PdfPages, tables: dict[str, pd.DataFrame], page: int) -> None:
    capacity = tables["capacity"].dropna(subset=["capacity_usd_proxy"]).sort_values("capacity_usd_proxy").head(18)
    fig, ax = plt.subplots(figsize=A4, constrained_layout=True)
    fig.suptitle("9. 容量压力测试", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    if not capacity.empty:
        ax.barh(capacity["symbol"], capacity["capacity_usd_proxy"] / 1_000_000, color="black", alpha=0.8)
        ax.invert_yaxis()
    ax.set_title("容量代理值，百万美元；基于 ADV 参与率假设")
    ax.set_xlabel("USD million")
    ax.grid(True, axis="x", alpha=0.25)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_bonus_page_one(pdf: PdfPages, tables: dict[str, pd.DataFrame], page: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=A4)
    fig.suptitle("9. 加分项曲线（一）", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    items = [
        ("basis_daily", "Basis 套利模块"),
        ("dynamic_daily", "动态因子模块"),
    ]
    for ax, (key, title) in zip(axes, items):
        df = tables.get(key, pd.DataFrame())
        if not df.empty and "equity_curve" in df:
            ax.plot(df.index, df["equity_curve"], color="black", linewidth=1.2)
        ax.set_title(title, pad=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20)
    fig.subplots_adjust(top=0.90, hspace=0.35)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_bonus_page_two(pdf: PdfPages, tables: dict[str, pd.DataFrame], page: int) -> None:
    fig, ax = plt.subplots(1, 1, figsize=A4)
    fig.suptitle("10. 加分项曲线（二）", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    ml_df = tables.get("ml_daily", pd.DataFrame())
    if not ml_df.empty and "equity_curve" in ml_df:
        ax.plot(ml_df.index, ml_df["equity_curve"], color="black", linewidth=1.2, label="ML gating")
    basis_df = tables.get("basis_daily", pd.DataFrame())
    if not basis_df.empty and "equity_curve" in basis_df:
        ax.plot(basis_df.index, basis_df["equity_curve"], color="black", linewidth=1.0, linestyle="--", label="真实 basis")
    ax.set_title("ML gating 与真实 basis 对比", pad=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(fontsize=8)
    fig.subplots_adjust(top=0.90)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_intraday_real(pdf: PdfPages, tables: dict[str, pd.DataFrame], page: int) -> None:
    intraday = tables.get("intraday", pd.DataFrame())
    fig, axes = plt.subplots(2, 1, figsize=A4, sharex=True, constrained_layout=True)
    fig.suptitle("10. 真实4小时频加分项", fontsize=17, weight="bold", x=0.08, ha="left", color="black")
    if not intraday.empty and "equity_curve" in intraday:
        equity = intraday["equity_curve"]
        dd = drawdown(equity)
        axes[0].plot(equity.index, equity, color="black", linewidth=1.2)
        axes[0].set_title("Binance 真实4小时K线 residual MR 净值")
        axes[1].fill_between(dd.index, dd.values, 0, color="black", alpha=0.25)
        axes[1].set_title("真实4小时频回撤")
        axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    else:
        axes[0].text(0.05, 0.5, "无真实4小时频结果", transform=axes[0].transAxes)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20)
    _page_number(fig, page)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def bonus_strategy_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bonus = tables["bonus"].copy()
    if bonus.empty:
        return bonus
    bonus = bonus[bonus["module"].isin(["basis", "dynamic", "ml"])].copy()
    cols = [
        "module",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "active_days",
        "avg_selected_factors",
    ]
    cols = [col for col in cols if col in bonus.columns]
    result = bonus[cols].copy()
    rename = {
        "module": "模块",
        "annual_return": "年化收益",
        "annual_volatility": "年化波动",
        "sharpe": "Sharpe",
        "max_drawdown": "最大回撤",
        "active_days": "活跃天数",
        "avg_selected_factors": "平均因子数",
    }
    return result.rename(columns=rename)


def ml_classifier_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bonus = tables["bonus"].copy()
    if bonus.empty or "module" not in bonus.columns:
        return pd.DataFrame()
    row = bonus[bonus["module"] == "ml_classifier"].copy()
    if row.empty:
        return pd.DataFrame()
    cols = ["accuracy", "precision", "recall", "positive_rate", "predicted_positive_rate"]
    cols = [col for col in cols if col in row.columns]
    result = row[cols].copy()
    return result.rename(
        columns={
            "accuracy": "准确率",
            "precision": "精确率",
            "recall": "召回率",
            "positive_rate": "正样本比例",
            "predicted_positive_rate": "预测正样本比例",
        }
    )


def analysis_conclusion_page(pdf: PdfPages, page: int) -> None:
    body = (
        "分析结果如下：\n"
        "1. 主策略扣成本后亏损，说明简单残差 z-score 反转不足以覆盖手续费、滑点、借券和 funding 成本。\n"
        "2. 因子模型对部分币股解释力较强，但解释力高不等于残差一定会回归。\n"
        "3. ADF 检验在样本内支持部分残差平稳，但样本内平稳不保证未来仍然平稳。\n"
        "4. 成本敏感性显示策略对交易成本很敏感，这类短周期均值回归策略容易被成本吃掉。\n"
        "5. 加分项中，basis 模块使用 Binance 真实 spot/perp 历史价格；实盘结论还需要接入逐期 funding rate 和成交深度。\n"
        "6. 动态因子和 ML gating 没有扭转主策略亏损，说明更复杂模型不能自动创造 alpha，反而需要更严格防止过拟合。\n\n"
        "结论：研究框架完成；当前简单版本不能证明可实盘盈利。"
    )
    text_page(pdf, "12. 分析结果与反思", body, page)


def write_pdf_report(output_dir: str | Path) -> Path:
    """Write a portrait PDF backtest report into the project report directory."""
    setup_chinese_font()
    output_path = Path(output_dir)
    tables = load_outputs(output_path)
    daily = tables["daily"]
    if daily.empty:
        raise FileNotFoundError(f"No backtest_daily.csv found in {output_path}")

    config = load_config(PROJECT_ROOT / "configs" / "universe.yaml")
    pdf_path = output_path / "strategy_report.pdf"

    page = 1
    with PdfPages(pdf_path) as pdf:
        summary_page(pdf, daily, config, page)
        page += 1
        data_page(pdf, page)
        page += 1
        method_page(pdf, page)
        page += 1
        plot_equity(pdf, daily, page)
        page += 1
        plot_returns_costs(pdf, daily, page)
        page += 1
        if not tables["diagnostics"].empty:
            plot_diagnostics(pdf, tables["diagnostics"], page)
            page += 1
        if not tables["factor_exposure"].empty and not tables["diagnostics"].empty:
            plot_exposure_activity(pdf, tables, page)
            page += 1
        if not tables["parameter"].empty or not tables["cost"].empty:
            plot_sensitivity_cost(pdf, tables, page)
            page += 1
        if not tables["capacity"].empty:
            plot_capacity(pdf, tables, page)
            page += 1
        plot_bonus_page_one(pdf, tables, page)
        page += 1
        plot_bonus_page_two(pdf, tables, page)
        page += 1
        plot_intraday_real(pdf, tables, page)
        page += 1
        table_page(pdf, "11.1 加分项策略结果", bonus_strategy_table(tables), page, max_rows=5)
        page += 1
        table_page(pdf, "11.2 ML 分类指标", ml_classifier_table(tables), page, max_rows=3)
        page += 1
        analysis_conclusion_page(pdf, page)

    return pdf_path


def main() -> None:
    pdf_path = write_pdf_report(PROJECT_ROOT / "reports" / "live")
    print(f"PDF report written to {pdf_path}")


if __name__ == "__main__":
    main()

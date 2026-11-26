"""Write RQ2 markdown summary with key checks."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from common import OUT, TABLES, LOGS, RAW, ensure_dirs, log

def run():
    ensure_dirs()
    lines = ["# 4.2 RQ2 实验数据核对汇总稿\n"]
    lines.append("本文件由 `RQ2/` 流水线自动生成，数值均来自真实预测记录。\n")
    # coverage
    cov = pd.read_csv(LOGS / "panel_coverage_check.csv")
    lines.append("## 面板覆盖\n")
    lines.append("```\n"+cov.to_string(index=False)+"\n```")
    lines.append("\n")
    chk = pd.read_csv(LOGS / "rq1_reopen_mase_check.csv")
    lines.append("## 与 RQ1 重新开放期 MASE 核对\n")
    lines.append(f"最大绝对差 = {chk.abs_diff.max():.6g}\n\n")
    lines.append("```\n"+chk.to_string(index=False)+"\n```")
    lines.append("\n")
    for name in sorted(TABLES.glob("table_S*.csv")):
        df = pd.read_csv(name)
        lines.append(f"## {name.name}\n")
        lines.append(f"行数={len(df)} 列数={df.shape[1]}\n\n")
        lines.append("```\n"+df.head(20).to_string(index=False)+"\n```")
        lines.append("\n")
    # contribution sum checks
    if (RAW / "s4_market_decomposition_detail.csv").exists():
        dec = pd.read_csv(RAW / "s4_market_decomposition_detail.csv")
        lines.append("## 市场贡献加总核对（应≈100）\n")
        for (split, a, b), part in dec.groupby(["split_name", "model_a", "model_b"]):
            lines.append(f"- {split} {a}-{b}: MASE share sum={part.share_MASE_pct.sum():.4f}, MAE share sum={part.share_MAE_pct.sum():.4f}\n")
    md = OUT / "4.2_RQ2_实验数据核对汇总稿.md"
    md.write_text("".join(lines), encoding="utf-8")
    log(f"wrote {md}")

if __name__ == "__main__":
    run()

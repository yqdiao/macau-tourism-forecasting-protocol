"""Statutory holiday calendar for the registered lightweight-model plan.

Mainland China rows drop make-up rest days (调休, 补假, 休息日) and the 2020
extended Spring Festival. Other countries keep the library's public holidays.
The Hong Kong destination series are not included; HK is only the Macau source market.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import holidays
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "shared/data/holiday_calendar.parquet"
START = date(2016, 12, 1)
END = date(2026, 12, 31)
CN_DROP = ("调休", "补假", "休息日", "延长")
MARKETS = {
    "AU_Australia": "AU", "BR_Brazil": "BR", "CA_Canada": "CA", "CH_Switzerland": "CH",
    "CN_Chinese_mainland": "CN", "DE_Germany": "DE", "ES_Spain": "ES", "FR_France": "FR",
    "GB_United_Kingdom": "GB", "HK_Hong_Kong_SAR": "HK", "ID_Indonesia": "ID",
    "IN_India": "IN", "IT_Italy": "IT", "JP_Japan": "JP", "KR_Republic_of_Korea": "KR",
    "MY_Malaysia": "MY", "NL_Netherlands": "NL", "NZ_New_Zealand": "NZ",
    "PH_Philippines": "PH", "PT_Portugal": "PT", "RU_Russian_Federation": "RU",
    "SG_Singapore": "SG", "TH_Thailand": "TH", "TW_Taiwan_region": "TW",
    "US_United_States_of_America": "US", "VN_Viet_Nam": "VN", "ZA_South_Africa": "ZA",
}


def _dates(country: str) -> list[tuple[date, str]]:
    found = holidays.country_holidays(country, years=range(2016, 2027))
    rows = []
    for day, name in found.items():
        if day < START or day > END:
            continue
        if country == "CN" and any(token in name for token in CN_DROP):
            continue
        rows.append((day, name))
    return rows


def build() -> pd.DataFrame:
    rows = []
    for day, name in _dates("MO"):
        rows.append({"date": day, "country": "MO", "name": name, "role": "macau", "market": None})
    for day, name in _dates("CN"):
        rows.append({"date": day, "country": "CN", "name": name, "role": "mainland", "market": None})
    for market, country in MARKETS.items():
        for day, name in _dates(country):
            rows.append({"date": day, "country": country, "name": name, "role": "source", "market": market})
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["role", "market", "date", "name"], kind="stable").reset_index(drop=True)


def main() -> None:
    frame = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    print(f"{len(frame)} rows -> {OUTPUT}")


if __name__ == "__main__":
    main()

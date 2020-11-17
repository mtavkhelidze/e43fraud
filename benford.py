#!/usr/bin/env python3

# Data come from for x in $(seq 1 30); do echo $x; http "https://results.cec.gov.ge/assets/data/election_43/prop/districts/$x.json" > prop-$x.json; done
# Change 'prop' to 'major' for majoritarian results.

import json
import re
from functools import reduce
from math import log10
from typing import Iterable, List, TypedDict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from scipy.stats import chisquare


class VoteResult(TypedDict):
    id: int
    votes: int
    station: str
    name: str


def cec_item_get(*, station: str, item: object) -> VoteResult:
    return {
        "id": item["number"],
        "votes": item["vote"],
        "station": station,
        "name": cec_name_clean(item["name"])
    }


def remove_marks(s: str) -> str:
    return re.sub("[\"„“”()]", "", s)


def remove_extra_ws(s: str) -> str:
    return re.sub("[ \t]+", " ", s)


def strip(s: str) -> str:
    return s.strip()


def english_names(name: str) -> str:
    return name.split("|")[1]


def cec_name_clean(cec: str) -> str:
    return reduce(
        lambda x, f: f(x),
        [
            english_names,
            remove_marks,
            remove_extra_ws,
            strip,
        ],
        cec,
    )


def read_vote_results(path: str) -> Iterable[VoteResult]:
    with open(path, "r") as fp:
        json_data = json.load(fp=fp)
        return (
            cec_item_get(station=x["number"], item=item)
            for x in json_data["items"]
            for item in x["subjects"]
        )


def get_data_for(*, system: str, count: int) -> DataFrame:
    result: List[Iterable[VoteResult]] = []
    for i in range(1, count):
        result.append(read_vote_results(f"./data.2020-11-17/{system}-{i}.json"))
    return pd.DataFrame(x for block in result for x in block)


def first_digit(n: int) -> int:
    return int(str(n)[0])


def chi_squared_stat(observed, expected) -> float:
    return sum((o - e) ** 2 / e for (o, e) in zip(observed, expected))


def bendord_counts_for(*, total: int) -> List[int]:
    return [round(total * log10(1 + 1 / n)) for n in range(1, 10)]


def process_data(*, system: str, count=30) -> DataFrame:
    df = get_data_for(system=system, count=count)

    first_digits = [first_digit(x) for x in df["votes"] if x > 10]
    observed_counts = [first_digits.count(digit) for digit in range(1, 10)]
    expected_counts = bendord_counts_for(total=sum(observed_counts))

    chi_stat = chi_squared_stat(observed_counts, expected_counts)
    chi = chisquare(observed_counts, expected_counts, ddof=1)

    print(f"System: {system}")
    print(f"Total votes in {df['votes'].unique().size} stations: {df['votes'].sum()}")
    print(f"Observed counts: {observed_counts}")
    print(f"Expected counts: {expected_counts}")
    print(f"Chi-Square stat (mine): {chi_stat}")
    print(f"Chi_Square stat (from scypi.stats) {chi}")
    print(df['votes'].size)

    return DataFrame({
        "observed": observed_counts,
        "expected": expected_counts,
    }, index=range(1, 10))


df = process_data(system='prop')

df["obs_pct"] = (df["observed"] / df["observed"].sum()).round(4) * 100
df["exp_pct"] = (df["expected"] / df["expected"].sum()).round(4) * 100

df.loc[df['obs_pct'] >= df['exp_pct'], 'more'] = df['obs_pct']
df.loc[df['obs_pct'] < df['exp_pct'], 'less'] = df['obs_pct']
df['more'] = df['more'].fillna(0)
df['less'] = df['less'].fillna(0)

df.loc[df['obs_pct'] < df['exp_pct'], 'more_exp'] = df['exp_pct']
df['more_exp'] = df['more_exp'].fillna(0)

df['diff'] = df['obs_pct'] - df['exp_pct']
print(df)

figsize = (15, 8)
digits = range(1, 10)

fig, ax = plt.subplots(figsize=figsize)

more_bars = ax.bar(
    digits, df['more'], width=0.3, color='red', alpha=1, zorder=1, label='Observed %'
)

exp_bars = ax.bar(
    digits, df['exp_pct'], width=0.3, color='grey', alpha=1, zorder=2, label='Expected %'
)

more_exp_bars = ax.bar(
    digits, df['less'], width=0.3, color='grey', alpha=1, zorder=4
)

less_bars = ax.bar(
    digits, df['less'], width=0.3, color='red', alpha=1, zorder=4
)

for d, bar in zip(df['diff'], more_bars):
    height = bar.get_height()
    if height == 0:
        continue
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{d:.2f}',
        ha='center',
        va='bottom',
        fontsize=12,
        color='blue'
    )

for d, bar in zip(df['diff'], exp_bars):
    height = bar.get_height()
    if d > 0:
        continue
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{d:.2f}',
        ha='center',
        va='bottom',
        fontsize=12,
        color='blue'
    )

ax.scatter(digits, df['exp_pct'], s=30, c='black', zorder=10)

ax.plot(digits, df["exp_pct"], color='grey', linewidth=2)

ax.set_ylabel('Frequency (%)', fontsize=12)
ax.set_xlabel('Digits', fontsize=12)
ax.set_xticks(digits)
ax.set_xticklabels(digits, fontsize=12)

blue_patch = mpatches.Patch(color='blue', label='Difference %')
ax.legend(handles=[more_bars, exp_bars, blue_patch])
ax.set_title('არჩევნებზე მიცემული ხმების რიცხვებში პირველი ციფრების განაწილება (პროპორციული)')
plt.show()

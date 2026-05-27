from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "산불발생위치도_지형특성계산.csv"
OUT_DIR = ROOT / "outputs" / "terrain_eda"


ASPECT_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ASPECT_LABELS_KO = {
    "N": "북향",
    "NE": "북동향",
    "E": "동향",
    "SE": "남동향",
    "S": "남향",
    "SW": "남서향",
    "W": "서향",
    "NW": "북서향",
}


def configure_plot() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def add_aspect_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["사면방향(도)"] = (
        np.degrees(np.arctan2(result["사면방향_sin"], result["사면방향_cos"])) + 360
    ) % 360

    bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    result["사면방향"] = pd.cut(
        result["사면방향(도)"], bins=bins, labels=labels, include_lowest=True, right=False, ordered=False
    )
    result["사면방향_한글"] = result["사면방향"].map(ASPECT_LABELS_KO)
    return result


def clean_twi(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["TWI_시각화용"] = result["TWI(지형다습지수)"].where(
        result["TWI(지형다습지수)"].between(0, 30)
    )
    return result


def save_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [
        ("경사도(도)", "경사도 분포", "경사도(도)"),
        ("고도(m)", "고도 분포", "고도(m)"),
        ("TPI(지형위치지수)", "TPI 분포", "TPI"),
        ("TWI_시각화용", "TWI 분포(0~30만 표시)", "TWI"),
    ]

    for ax, (col, title, xlabel) in zip(axes.ravel(), metrics):
        sns.histplot(df[col].dropna(), bins=35, kde=True, ax=ax, color="#3b82f6")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("산불 발생 건수")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "terrain_histograms.png", dpi=160)
    plt.close(fig)


def save_aspect_frequency(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["사면방향"]
        .value_counts()
        .reindex(ASPECT_ORDER)
        .rename_axis("aspect")
        .reset_index(name="count")
    )
    counts["percent"] = counts["count"] / counts["count"].sum() * 100
    counts["label_ko"] = counts["aspect"].map(ASPECT_LABELS_KO)
    counts.to_csv(OUT_DIR / "aspect_frequency.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=counts, x="label_ko", y="count", ax=ax, color="#10b981")
    ax.set_title("사면방향별 산불 발생 빈도")
    ax.set_xlabel("사면방향")
    ax.set_ylabel("산불 발생 건수")
    for i, row in counts.iterrows():
        ax.text(i, row["count"], f"{row['percent']:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "aspect_frequency.png", dpi=160)
    plt.close(fig)
    return counts


def save_boxplots(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["사면방향"]).copy()
    plot_df["사면방향"] = pd.Categorical(plot_df["사면방향"], ASPECT_ORDER, ordered=True)
    plot_df["사면방향_한글"] = plot_df["사면방향"].map(ASPECT_LABELS_KO)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.boxplot(data=plot_df, x="사면방향_한글", y="경사도(도)", ax=axes[0], color="#93c5fd")
    axes[0].set_title("사면방향별 경사도 Boxplot")
    axes[0].set_xlabel("사면방향")
    axes[0].set_ylabel("경사도(도)")

    sns.boxplot(data=plot_df, x="사면방향_한글", y="고도(m)", ax=axes[1], color="#86efac")
    axes[1].set_title("사면방향별 고도 Boxplot")
    axes[1].set_xlabel("사면방향")
    axes[1].set_ylabel("고도(m)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "aspect_boxplots.png", dpi=160)
    plt.close(fig)


def save_slope_elevation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    slope_bins = [0, 5, 10, 15, 20, 25, 30, 35, 45]
    elev_bins = [0, 50, 100, 200, 300, 500, 800, 1200, 1800]
    heat_df = df.copy()
    heat_df["경사도구간"] = pd.cut(heat_df["경사도(도)"], slope_bins, right=False)
    heat_df["고도구간"] = pd.cut(heat_df["고도(m)"], elev_bins, right=False)
    pivot = pd.crosstab(heat_df["고도구간"], heat_df["경사도구간"])
    pivot.to_csv(OUT_DIR / "slope_elevation_crosstab.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt="d", linewidths=0.4, ax=ax)
    ax.set_title("고도-경사도 구간별 산불 발생 건수")
    ax.set_xlabel("경사도 구간(도)")
    ax.set_ylabel("고도 구간(m)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slope_elevation_heatmap.png", dpi=160)
    plt.close(fig)
    return pivot


def format_interval(interval: pd.Interval) -> str:
    return f"{interval.left:g}~{interval.right:g}"


def make_summary(df: pd.DataFrame, aspect_counts: pd.DataFrame, heatmap: pd.DataFrame) -> None:
    n = len(df)
    slope_mean = df["경사도(도)"].mean()
    slope_median = df["경사도(도)"].median()
    slope_25_plus = (df["경사도(도)"] >= 25).sum()
    slope_15_plus = (df["경사도(도)"] >= 15).sum()

    elev_mean = df["고도(m)"].mean()
    elev_median = df["고도(m)"].median()
    elev_under_300 = (df["고도(m)"] < 300).sum()
    elev_mode_bin = pd.cut(
        df["고도(m)"], [0, 50, 100, 200, 300, 500, 800, 1200, 1800], right=False
    ).value_counts().idxmax()

    top_aspect = aspect_counts.sort_values("count", ascending=False).iloc[0]
    top_cell = heatmap.stack().idxmax()
    twi_invalid = df["TWI_시각화용"].isna().sum()

    summary = f"""# 지형 특성별 산불 발생 EDA

## 데이터
- 분석 대상: 산불 발생 지점 {n:,}건
- 사용 변수: 고도(m), 경사도(도), 사면방향, TPI, TWI
- 주의: 이 데이터는 산불 발생 지점만 포함하므로, 전체 지형 면적 대비 '발생 확률'이 아니라 '발생 지점의 분포'를 해석합니다.
- TWI는 0~30 범위를 벗어난 값 {twi_invalid:,}건을 시각화에서 제외했습니다.

## 핵심 결과
- 산불 발생 지점의 평균 경사도는 {slope_mean:.1f}도, 중앙값은 {slope_median:.1f}도입니다.
- 경사도 15도 이상 지점은 {slope_15_plus:,}건({slope_15_plus / n * 100:.1f}%), 25도 이상 지점은 {slope_25_plus:,}건({slope_25_plus / n * 100:.1f}%)입니다.
- 산불 발생 지점의 평균 고도는 {elev_mean:.1f}m, 중앙값은 {elev_median:.1f}m입니다.
- 고도 300m 미만 지점은 {elev_under_300:,}건({elev_under_300 / n * 100:.1f}%)으로 저고도 발생 지점이 많습니다.
- 가장 많이 나타난 고도 구간은 {format_interval(elev_mode_bin)}m입니다.
- 사면방향 중 발생 건수가 가장 많은 방향은 {ASPECT_LABELS_KO[top_aspect['aspect']]}({top_aspect['count']:,}건, {top_aspect['percent']:.1f}%)입니다.
- 고도-경사도 조합에서는 고도 {format_interval(top_cell[0])}m, 경사도 {format_interval(top_cell[1])}도 구간이 가장 많습니다.

## 산출물
- `terrain_histograms.png`: 경사도, 고도, TPI, TWI 히스토그램
- `aspect_frequency.png`: 사면방향별 산불 발생 빈도
- `aspect_boxplots.png`: 사면방향별 경사도/고도 boxplot
- `slope_elevation_heatmap.png`: 고도-경사도 구간별 빈도 heatmap
- `aspect_frequency.csv`, `slope_elevation_crosstab.csv`: 표 형태 요약

## 보고서에 쓸 수 있는 문장
- "산불 발생 지점은 평균 경사도 {slope_mean:.1f}도, 중앙값 {slope_median:.1f}도로 완만한~중간 경사지에 많이 분포했다."
- "고도는 300m 미만 지점이 {elev_under_300 / n * 100:.1f}%로, 분석 데이터에서는 저고도 산불 발생 지점이 우세했다."
- "사면방향은 {ASPECT_LABELS_KO[top_aspect['aspect']]}에서 가장 많은 발생 건수를 보였다."
- "다만 본 분석은 발생 지점만 대상으로 하므로, 특정 지형이 실제로 더 위험한지 판단하려면 비발생 지점 또는 전체 산림 지형 면적을 함께 비교해야 한다."
"""
    (OUT_DIR / "terrain_eda_summary.md").write_text(summary, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot()
    df = pd.read_csv(DATA_PATH)
    df = add_aspect_columns(df)
    df = clean_twi(df)

    save_histograms(df)
    aspect_counts = save_aspect_frequency(df)
    save_boxplots(df)
    heatmap = save_slope_elevation_heatmap(df)
    make_summary(df, aspect_counts, heatmap)


if __name__ == "__main__":
    main()

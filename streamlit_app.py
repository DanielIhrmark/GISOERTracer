import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots


# ----------------------------
# Paths (relative to project root)
# ----------------------------
BASE_PARQUET = Path("pdf_corpus_enriched_step2.parquet")
WIDE_PARQUET = Path("bertopic_output_Wide/docs_with_topics.parquet")
WIDE_TOPIC_INFO = Path("bertopic_output_Wide/topic_info.csv")  # optional


# ----------------------------
# Phrase categories (editable)
# ----------------------------
PHRASE_CATEGORIES: Dict[str, List[str]] = {
    "Generic GIS markers": [
        "gis", "geographic information system", "geographic information systems",
        "geoinformation system", "geoinformation systems",
        "open source gis", "free and open source", "foss", "foss4g",
        "web mapping", "digital mapping", "geospatial",
    ],
    "ESRI stack": [
        "arcgis pro", "arcgis", "arcmap", "esri", "storymaps", "arcgis online",
    ],
    "FOSS GIS": [
        "qgis", "grass gis", "grass", "saga gis", "saga", "geoda", "mapinfo",
    ],
    "Databases / spatial SQL": [
        "postgis", "postgresql", "spatialite",
    ],
    "Geo libraries / tooling": [
        "gdal", "ogr", "proj", "pyproj", "geopandas", "shapely", "rasterio",
        "sf", "sp", "terra", "raster", "tmap", "ggplot2",
    ],
    "Web mapping / viz": [
        "leaflet", "mapbox", "openlayers", "kepler.gl", "folium",
    ],
    "DH mapping tools": [
        "neatline", "recogito", "pelagios",
    ],
    "Methods / operations": [
        "georeferencing", "georeference", "geocoding", "gazetteer",
        "digitizing", "digitisation", "digitization",
        "kernel density", "kde", "interpolation", "buffer", "spatial join", "overlay",
        "remote sensing", "satellite imagery", "aerial photography", "lidar",
        "classification", "supervised classification", "unsupervised classification",
        "geoparsing",
    ],
    "Programming environments": [
        "python", "r ", "rstudio", "r studio",
    ],
}

# ----------------------------
# Regex for exact word counts without findall() lists  (CHANGE #2)
# ----------------------------
WORD_RE = re.compile(r"\b[\w']+\b")


# ----------------------------
# Utilities
# ----------------------------
def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return pd.read_parquet(path)


def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_int64_na(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def normalize_year(df: pd.DataFrame) -> pd.DataFrame:
    if "year" in df.columns:
        df["year"] = to_int64_na(df["year"])
        return df
    for c in ["bib_year", "published_year"]:
        if c in df.columns:
            df["year"] = to_int64_na(df[c])
            return df
    df["year"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df


def parse_openalex_concepts(raw: Any) -> List[Dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            return []
    return []


def concepts_to_names(concepts: List[Dict[str, Any]]) -> List[str]:
    out = []
    for c in concepts:
        name = c.get("display_name") or c.get("name")
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def load_topic_labels(topic_info_csv: Path) -> Dict[int, str]:
    if not topic_info_csv.exists():
        return {}
    try:
        ti = pd.read_csv(topic_info_csv)
        if "Topic" in ti.columns and "Name" in ti.columns:
            m = {}
            for _, r in ti.iterrows():
                try:
                    m[int(r["Topic"])] = str(r["Name"])
                except Exception:
                    pass
            return m
    except Exception:
        return {}
    return {}


def build_topic_label(topic_id: int, label_map: Dict[int, str]) -> str:
    if topic_id in label_map:
        return f"{topic_id}: {label_map[topic_id]}"
    return str(topic_id)


def compile_phrase_regex(phrase: str) -> re.Pattern:
    """
    Robust regex for a phrase:
    - case-insensitive
    - tolerates hyphen/space variations between tokens
    - word boundaries where appropriate
    Special cases:
      - "r " -> standalone R
      - punctuation tokens like kepler.gl -> no strict boundaries
    """
    p = phrase.strip()
    if not p:
        return re.compile(r"a^")  # matches nothing

    if p == "r ":
        return re.compile(r"(?i)\bR\b")

    tokens = [re.escape(t) for t in p.split()]
    if len(tokens) == 1:
        if re.search(r"[.\-_/]", p):
            return re.compile(r"(?i)" + re.escape(p))
        return re.compile(r"(?i)\b" + re.escape(p) + r"\b")

    joined = r"[\s\-]+".join(tokens)
    if "." in p:
        return re.compile(r"(?i)" + joined)

    return re.compile(r"(?i)\b" + joined + r"\b")


@dataclass
class PhraseSpec:
    category: str
    phrase: str
    regex: re.Pattern


# ----------------------------
# Load + merge (cached)
# ----------------------------
@st.cache_data(show_spinner=True)
def load_master() -> Tuple[pd.DataFrame, Dict[int, str]]:
    base = safe_read_parquet(BASE_PARQUET)
    wide = safe_read_parquet(WIDE_PARQUET)

    base = normalize_year(base)

    # OA concepts
    if "openalex_concepts_json" in base.columns:
        concepts_parsed = base["openalex_concepts_json"].apply(parse_openalex_concepts)
        base["concept_names"] = concepts_parsed.apply(concepts_to_names)
        base["concept_names_str"] = base["concept_names"].apply(lambda xs: "; ".join(xs[:12]) if isinstance(xs, list) else "")
        base["has_concepts"] = base["concept_names"].apply(lambda xs: bool(xs) if isinstance(xs, list) else False)
    else:
        base["concept_names"] = [[] for _ in range(len(base))]
        base["concept_names_str"] = ""
        base["has_concepts"] = False

    # Merge wide topics by pdf_path (preferred)
    key = pick_first_existing_col(base, ["pdf_path", "file_path", "path"])
    if not key:
        raise ValueError("Base parquet missing 'pdf_path' (or similar).")

    if key not in wide.columns:
        wide_key = pick_first_existing_col(wide, ["pdf_path", "file_path", "path"])
        if wide_key:
            wide = wide.rename(columns={wide_key: key})
        else:
            raise ValueError("Wide topics parquet missing 'pdf_path' (or similar).")

    wide_topic_col = pick_first_existing_col(wide, ["topic", "Topic", "wide_topic"])
    wide_prob_col = pick_first_existing_col(wide, ["topic_prob_max", "prob_max", "wide_prob"])

    if not wide_topic_col:
        raise ValueError("Wide topics parquet missing a 'topic' column.")

    wide_sel = wide[[key, wide_topic_col] + ([wide_prob_col] if wide_prob_col else [])].copy()
    wide_sel = wide_sel.rename(columns={wide_topic_col: "wide_topic"})
    if wide_prob_col:
        wide_sel = wide_sel.rename(columns={wide_prob_col: "wide_prob"})
    else:
        wide_sel["wide_prob"] = pd.NA

    master = base.merge(wide_sel, on=key, how="left")

    # display title
    if "bib_title" in master.columns:
        master["title_display"] = master["bib_title"].fillna(master.get("file_name", ""))
    else:
        master["title_display"] = master.get("file_name", "")

    # ensure text is str
    master["text"] = master.get("text", "").astype(str)

    # exact word counts WITHOUT findall() (CHANGE #2)
    master["word_count"] = master["text"].str.count(WORD_RE).astype("int64")

    # types
    master["wide_topic"] = pd.to_numeric(master["wide_topic"], errors="coerce").astype("Int64")
    master["wide_prob"] = pd.to_numeric(master["wide_prob"], errors="coerce")

    topic_labels = load_topic_labels(WIDE_TOPIC_INFO)
    return master, topic_labels


def filter_by_concepts(df: pd.DataFrame, selected: List[str], mode_any: bool) -> pd.DataFrame:
    if not selected:
        return df
    selected_set = set(selected)

    def match(xs: Any) -> bool:
        if not isinstance(xs, list) or not xs:
            return False
        xset = set(xs)
        return bool(xset & selected_set) if mode_any else selected_set.issubset(xset)

    return df[df["concept_names"].apply(match)]


def compute_phrase_counts_by_year(df: pd.DataFrame, specs: List[PhraseSpec]) -> pd.DataFrame:
    """
    Returns:
      year, phrase, category, hits, total_words, per_1000

    CHANGE #2: use vectorized Series.str.count() instead of regex.findall()
    to avoid allocating match lists (large memory spikes).
    """
    work = df.dropna(subset=["year"]).copy()
    work = work[work["year"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["year", "phrase", "category", "hits", "total_words", "per_1000"])

    work["year"] = work["year"].astype(int)

    totals = work.groupby("year")["word_count"].sum().reset_index(name="total_words")

    frames = []
    text_series = work["text"].astype(str)

    for spec in specs:
        hits_by_year = (
            text_series.str.count(spec.regex)
            .groupby(work["year"])
            .sum()
            .reset_index(name="hits")
        )
        hits_by_year["phrase"] = spec.phrase
        hits_by_year["category"] = spec.category
        frames.append(hits_by_year)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["year", "hits", "phrase", "category"]
    )
    out = out.merge(totals, on="year", how="left")
    out["per_1000"] = (out["hits"] / out["total_words"].replace(0, np.nan)) * 1000.0
    out["per_1000"] = out["per_1000"].fillna(0.0)
    return out[["year", "phrase", "category", "hits", "total_words", "per_1000"]]


def compute_topic_share_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    year, topic, count, share
    share is topic share of docs per year.
    """
    tmp = df.dropna(subset=["year"]).copy()
    tmp = tmp[tmp["year"].notna()].copy()
    tmp["year"] = tmp["year"].astype(int)

    tmp = tmp[tmp["wide_topic"].notna()].copy()
    tmp["wide_topic"] = tmp["wide_topic"].astype(int)

    counts = tmp.groupby(["year", "wide_topic"]).size().reset_index(name="count")
    denom = counts.groupby("year")["count"].transform("sum")
    counts["share"] = counts["count"] / denom
    counts = counts.rename(columns={"wide_topic": "topic"})
    return counts


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="GeoHum Topic Tracer", layout="wide")
st.title("GeoHum Topic Tracer")

master, topic_labels = load_master()

with st.sidebar:
    st.header("Filters")
    years = master["year"].dropna().astype(int)
    year_min, year_max = (int(years.min()), int(years.max())) if len(years) else (1900, 2026)
    year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))

    journal_opts = sorted([j for j in master.get("journal_name", pd.Series()).dropna().unique().tolist() if str(j).strip()])
    selected_journals = st.multiselect("Journal(s)", options=journal_opts, default=journal_opts)

    # OA concepts filter (top 300 by frequency)
    all_concepts = master["concept_names"].explode().dropna()
    all_concepts = all_concepts[all_concepts.astype(str).str.len() > 0]
    concept_freq = all_concepts.value_counts().head(300)
    concept_opts = concept_freq.index.tolist()

    st.caption("OA concept filter uses the top 300 most frequent concepts.")
    selected_concepts = st.multiselect("OpenAlex concepts", options=concept_opts, default=[])
    concept_mode = st.radio("Concept matching", options=["ANY (OR)", "ALL (AND)"], index=0, horizontal=True)
    concept_any = concept_mode.startswith("ANY")

    st.divider()
    st.header("Topic background")
    # NEW: choose how to select topics
    topic_select_mode = st.radio("Topic selection", options=["Top N topics", "Select topics manually"], index=0)

    include_outlier_topic = st.checkbox("Include outlier topic (-1)", value=False)

    st.divider()
    st.header("Phrases to plot")

    # NEW: custom category inputs
    st.subheader("Custom category")
    custom_cat_name = st.text_input("Custom category name", value="Custom")
    custom_cat_phrases_raw = st.text_area(
        "Comma-separated phrases",
        value="",
        help="Example: arcgis, qgis, postgis, geopandas"
    )

    st.divider()
    st.subheader("Built-in categories")
    cats = list(PHRASE_CATEGORIES.keys())
    selected_cats = st.multiselect(
        "Category",
        options=cats,
        default=["ESRI stack", "FOSS GIS", "Geo libraries / tooling", "Methods / operations"]
    )

    available_phrases = sorted({p for c in selected_cats for p in PHRASE_CATEGORIES.get(c, [])})

    selected_phrases = st.multiselect(
        "Built-in phrases (lines)",
        options=available_phrases,
        default=available_phrases[:6]
    )

    # include custom phrases in plotted set?
    include_custom = st.checkbox("Include custom category phrases", value=True)

    aggregate_mode = st.radio(
        "Aggregation",
        options=["Plot each phrase", "Aggregate by category", "Aggregate all selected"],
        index=0
    )

    st.caption("Phrase series are occurrences per 1,000 words (year-normalized).")
    show_doc_stats = st.checkbox("Show document stats table", value=False)

# Apply filters
df = master  # CHANGE #1: avoid duplicating full text dataframe in memory

df = df[(df["year"].isna()) | ((df["year"] >= year_range[0]) & (df["year"] <= year_range[1]))]

if selected_journals and "journal_name" in df.columns:
    df = df[df["journal_name"].isin(selected_journals)]

df = filter_by_concepts(df, selected_concepts, mode_any=concept_any)

# Keep this copy (smallish) because downstream may add columns / subset;
# it does NOT duplicate 'master' anymore (we removed master.copy()).
df_for_freq = df[df["word_count"] > 0].copy()

# Prepare phrase specs (built-in + custom)
phrase_to_cat: Dict[str, str] = {}
for cat in selected_cats:
    for p in PHRASE_CATEGORIES.get(cat, []):
        phrase_to_cat[p] = cat

custom_phrases: List[str] = []
if include_custom and custom_cat_phrases_raw.strip():
    custom_phrases = [p.strip() for p in custom_cat_phrases_raw.split(",") if p.strip()]
    for p in custom_phrases:
        phrase_to_cat[p] = custom_cat_name.strip() or "Custom"

# The full list of phrases to compute
all_selected_phrases = list(dict.fromkeys(selected_phrases + (custom_phrases if include_custom else [])))

specs: List[PhraseSpec] = []
for p in all_selected_phrases:
    cat = phrase_to_cat.get(p, "Uncategorized")
    specs.append(PhraseSpec(category=cat, phrase=p, regex=compile_phrase_regex(p)))

# Compute frequency series
freq_df = compute_phrase_counts_by_year(df_for_freq, specs)

# Compute topic shares
topic_df = compute_topic_share_by_year(df)

# Topic selection UI depends on filtered corpus — build available topics from the filtered df
available_topic_ids = sorted([int(x) for x in df["wide_topic"].dropna().unique().tolist() if pd.notna(x)])

# Build background topic df based on selection
if not topic_df.empty:
    bg = topic_df.copy()
    if not include_outlier_topic:
        bg = bg[bg["topic"] != -1].copy()

    if topic_select_mode == "Top N topics":
        # default N
        top_n = st.sidebar.slider("Top N topics in background", 5, 30, 12)
        top_topic_ids = (
            bg.groupby("topic")["count"].sum().sort_values(ascending=False).head(top_n).index.tolist()
        )
        selected_topic_ids = top_topic_ids
    else:
        # manual selection
        # show labels for selection
        topic_label_opts = [build_topic_label(t, topic_labels) for t in available_topic_ids]
        label_to_id = {build_topic_label(t, topic_labels): t for t in available_topic_ids}
        default_labels = topic_label_opts[: min(8, len(topic_label_opts))]
        chosen_labels = st.sidebar.multiselect(
            "Select topics to show in background",
            options=topic_label_opts,
            default=default_labels
        )
        selected_topic_ids = [label_to_id[l] for l in chosen_labels] if chosen_labels else []

    topic_df_bg = bg[bg["topic"].isin(selected_topic_ids)].copy()
    topic_df_bg["topic_label"] = topic_df_bg["topic"].apply(lambda t: build_topic_label(int(t), topic_labels))
else:
    topic_df_bg = pd.DataFrame(columns=["year", "topic", "count", "share", "topic_label"])
    selected_topic_ids = []

# Aggregate frequency series if requested
plot_df = freq_df.copy()
if aggregate_mode == "Aggregate by category":
    plot_df = (
        plot_df.groupby(["year", "category"], as_index=False)
        .agg(hits=("hits", "sum"), total_words=("total_words", "first"))
    )
    plot_df["per_1000"] = (plot_df["hits"] / plot_df["total_words"].replace(0, np.nan)) * 1000.0
    plot_df["per_1000"] = plot_df["per_1000"].fillna(0.0)
    plot_df["series"] = plot_df["category"]

elif aggregate_mode == "Aggregate all selected":
    plot_df = (
        plot_df.groupby(["year"], as_index=False)
        .agg(hits=("hits", "sum"), total_words=("total_words", "first"))
    )
    plot_df["per_1000"] = (plot_df["hits"] / plot_df["total_words"].replace(0, np.nan)) * 1000.0
    plot_df["per_1000"] = plot_df["per_1000"].fillna(0.0)
    plot_df["series"] = "ALL_SELECTED"

else:
    plot_df["series"] = plot_df["phrase"]

# Layout
left, right = st.columns([2.2, 1], gap="large")

with left:
    st.subheader("How to use:")
    st.info("Keyword frequencies in foreground, BERTopic topics in background. Double-click in legend to isolate, single-click to toggle.")

    if plot_df.empty:
        st.info(
            "No phrase frequency data after filters.\n\n"
            "Try widening year range, removing concept filters, selecting more phrases, or adding a custom category."
        )
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Background topics (stacked share)
        if not topic_df_bg.empty:
            for lbl, g in topic_df_bg.groupby("topic_label"):
                g = g.sort_values("year")
                fig.add_trace(
                    go.Scatter(
                        x=g["year"],
                        y=g["share"],
                        mode="lines",
                        stackgroup="topics",
                        name=f"Topic share: {lbl}",
                        hovertemplate="Year=%{x}<br>Share=%{y:.1%}<extra></extra>",
                        opacity=0.25,
                    ),
                    secondary_y=True,
                )
        else:
            st.caption("No background topics selected (or no topics available after filtering).")

        # Foreground frequency lines
        for series, g in plot_df.groupby("series"):
            g = g.sort_values("year")
            fig.add_trace(
                go.Scatter(
                    x=g["year"],
                    y=g["per_1000"],
                    mode="lines+markers",
                    name=f"{series} (per 1k)",
                    hovertemplate="Year=%{x}<br>Per 1k=%{y:.3f}<extra></extra>",
                ),
                secondary_y=False,
            )

        fig.update_layout(
            height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Occurrences per 1,000 words", secondary_y=False)
        fig.update_yaxes(title_text="Topic share of documents", tickformat=".0%", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Quick stats (filtered view)")
    st.metric("Documents", len(df))
    st.metric("Total words", int(df_for_freq["word_count"].sum()))
    st.metric("Docs with OA concepts", int(df.get("has_concepts", pd.Series([False] * len(df))).sum()))
    st.metric("Docs with topic", int(df["wide_topic"].notna().sum()))

    st.divider()
    st.subheader("Top phrases (overall, filtered)")
    if not freq_df.empty:
        top_phrase = (
            freq_df.groupby(["phrase", "category"], as_index=False)
            .agg(hits=("hits", "sum"), total_words=("total_words", "sum"))
        )
        top_phrase["per_1000"] = (top_phrase["hits"] / top_phrase["total_words"].replace(0, np.nan)) * 1000.0
        top_phrase["per_1000"] = top_phrase["per_1000"].fillna(0.0)
        top_phrase = top_phrase.sort_values("per_1000", ascending=False).head(20)

        fig2 = px.bar(
            top_phrase,
            x="per_1000",
            y="phrase",
            color="category",
            orientation="h",
            title="Occurrences per 1,000 words (top phrases)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No phrase data computed.")

st.divider()

if show_doc_stats:
    st.subheader("Document-level table (filtered)")
    # Compute snippet ONLY when needed (avoids keeping a second big string column resident)
    df_view = df.copy()
    df_view["text_snip"] = df_view["text"].str.slice(0, 350)

    cols = [
        "year", "journal_name", "journal_issue", "title_display", "file_name",
        "wide_topic", "wide_prob", "concept_names_str", "word_count", "text_snip", "pdf_path"
    ]
    cols = [c for c in cols if c in df_view.columns]
    st.dataframe(df_view[cols], use_container_width=True, hide_index=True)

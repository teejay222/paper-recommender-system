"""
app/app.py — Streamlit frontend for the Academic Paper Recommendation System.

Tabs: Browse (corpus explorer), Recommendations (semantic similarity + category
boost, with year/category filters), Analytics (corpus + embedding-space views),
and Evaluation (known-item retrieval + proxy experiments). Project Info is the
remaining placeholder.

Pure data logic lives in app/analytics.py and app/evaluation.py; this file only
caches and renders.

Run from the PROJECT ROOT (so the relative data/model paths resolve):
    streamlit run app/app.py
"""

import math
import re
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()

# Make `src` importable when launched as `streamlit run app/app.py` from root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import PaperRecommender, DATA_FILE
from analytics import (
    compute_analytics, read_projections, read_separation, read_similarity,
    parse_tags,
)
from evaluation import (
    read_known_item, read_known_item_queries, read_proxy, read_proxy_sweep,
    read_hybrid_sweep, overall_metrics, recall_curve, model_delta,
    failure_picks, hybrid_best,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Academic Paper Recommender", layout="wide")

MODEL_CHOICES = {
    "SPECTER2 (primary)":   "specter2",
    "SciBERT (comparison)": "scibert",
}
MODE_CHOICES = {
    "Title — find papers similar to a known paper": "title",
    "Free query — search by keywords or a question": "keywords",
}


# ---------------------------------------------------------------------------
# Cached loaders (load each model once per session, not on every interaction)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_recommender(model_value: str) -> PaperRecommender:
    """Loads and caches a PaperRecommender for the given model."""
    return PaperRecommender(model=model_value)


@st.cache_data(show_spinner="Loading corpus…")
def load_corpus() -> pd.DataFrame:
    """Loads the columns the browse tab needs (cached once per session).

    Does NOT load embeddings — browsing is a plain metadata view, independent
    of the recommender. arxiv_id is read as str to avoid float truncation.
    """
    cols = [
        "arxiv_id", "title", "authors", "category",
        "publication_year", "citation_count", "abstract", "keybert_tags_v2",
    ]
    return pd.read_csv(DATA_FILE, usecols=cols, dtype={"arxiv_id": str})


FIGURES_DIR = Path("results/figures")

# Fixed, high-contrast palette so the four categories are easy to tell apart
# everywhere (especially the PCA / t-SNE maps). Explicit domain->range keeps each
# category the same colour across every chart, regardless of row order.
CATEGORY_DOMAIN = ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]
CATEGORY_COLORS = ["#E63946", "#1D7DF2", "#2EB82E", "#FF8C00"]  # red, blue, green, orange
CAT_SCALE = alt.Scale(domain=CATEGORY_DOMAIN, range=CATEGORY_COLORS)

# Two-class scale (vivid, clearly distinct).
COMPARISON_SCALE = alt.Scale(domain=["Same category", "Different category"],
                             range=["#2EB82E", "#E63946"])          # green vs red

# Vibrant single-series bar colours (anything but the dull default blue).
COLOR_PAPERS  = "#1FB6C1"   # teal
COLOR_TAGS    = "#8E5BD6"   # purple
COLOR_CIT     = "#EE6C4D"   # coral

EVAL_DIR = Path("results/evaluation")

# Model + query-type display labels and a two-model colour scale for the
# Evaluation tab.
MODEL_LABELS = {"specter2": "SPECTER2", "scibert": "SciBERT"}
QTYPE_LABELS = {
    "keyword_query": "Keyword", "task_query": "Task",
    "problem_query": "Problem", "natural_query": "Natural",
}
MODEL_SCALE = alt.Scale(domain=["SPECTER2", "SciBERT"],
                        range=["#1D7DF2", "#FF8C00"])   # blue vs orange


@st.cache_data(show_spinner=False)
def get_filter_options() -> tuple[list, list]:
    """Distinct categories and years for the filter dropdowns.

    Reads only the two needed columns from the CSV (cheap, cached once) so the
    filters can populate without loading the full recommender/embeddings.
    Returns (categories sorted A->Z, years sorted newest first as strings).
    """
    import pandas as pd
    df = pd.read_csv(DATA_FILE, usecols=["category", "publication_year"])
    cats = sorted(df["category"].dropna().astype(str).unique().tolist())
    years = sorted(
        df["publication_year"].dropna().astype(int).unique().tolist(), reverse=True
    )
    return cats, [str(y) for y in years]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def matching_keywords(query: str, tags: list) -> list:
    """Tags that share a word with the query (context only, not the ranker)."""
    qwords = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    out = []
    for t in tags:
        if any(w in qwords for w in re.findall(r"[a-z0-9]+", t.lower())):
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Analytics data (thin cached wrappers over src/analytics.py)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Computing analytics…")
def analytics_data() -> dict:
    return compute_analytics(load_corpus())


@st.cache_data(show_spinner=False)
def load_projections(model: str):
    return read_projections(FIGURES_DIR, model)


@st.cache_data(show_spinner=False)
def load_separation():
    return read_separation(FIGURES_DIR)


@st.cache_data(show_spinner=False)
def load_similarity():
    return read_similarity(FIGURES_DIR)


@st.cache_data(show_spinner=False)
def load_known_item():
    return read_known_item(EVAL_DIR)


@st.cache_data(show_spinner=False)
def load_known_item_queries(model, qtype):
    return read_known_item_queries(EVAL_DIR, model, qtype)


PROXY_DIR = Path("results/wandb_exports")


@st.cache_data(show_spinner=False)
def load_proxy(name: str):
    return read_proxy(PROXY_DIR, name)


@st.cache_data(show_spinner=False)
def load_proxy_sweep():
    return read_proxy_sweep(PROXY_DIR)


@st.cache_data(show_spinner=False)
def load_hybrid_sweep():
    return read_hybrid_sweep(EVAL_DIR)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_result(position: int, paper: dict, query: str, mode: str) -> None:
    """Renders one recommendation as a bordered card with an explanation."""
    with st.container(border=True):
        st.markdown(f"**{position}. {paper['title']}**")
        st.caption(paper["authors"])
        st.caption(
            f"{paper['category']}  ·  {paper['publication_year']}  ·  "
            f"{paper['citation_count']} citations"
        )

        st.markdown(
            f"Cosine **{paper['similarity_score']:.3f}**  ·  "
            f"Final **{paper['final_score']:.3f}**  ·  "
            f"Category boost: **{'Applied' if paper['boosted'] else 'None'}**"
        )

        st.write(paper["abstract_preview"] + "…")

        with st.expander("Why was this recommended?"):
            st.markdown(
                f"- **Ranked by semantic similarity** (cosine = {paper['similarity_score']:.3f})."
            )
            if paper["boosted"]:
                st.markdown(
                    f"- **Category boost applied** — shares the query paper's "
                    f"category (*{paper['category']}*)."
                )
            else:
                st.markdown("- **No category boost** — free-query mode has no category to match.")

            tags = parse_tags(paper.get("tags", ""))
            matched = matching_keywords(query, tags)
            if matched:
                st.markdown("- **Keywords overlapping your query:** " + ", ".join(matched))
            if tags:
                st.markdown("- **Key concepts in this paper:** " + ", ".join(tags))
            st.caption(
                "Final score = cosine × (1 + 0.03 × same-category). Ranking is "
                "driven by embedding similarity and the category boost; keywords "
                "are shown for context only."
            )

        st.markdown(f"[View on arXiv](https://arxiv.org/abs/{paper['arxiv_id']})")


def render_matched_box(matched: dict) -> None:
    """Compact 'you searched for this' box, shown in the right column in title
    mode so the user can confirm the fuzzy match alongside the recommendations."""
    with st.container(border=True):
        st.markdown("**Your search matched**")
        st.markdown(f"**{matched['title']}**")
        st.caption(matched["authors"])
        st.caption(f"{matched['category']}  ·  {matched['publication_year']}")
        st.caption(f"Title match: {matched['match_score']:.0f}/100")
        st.markdown(f"[View on arXiv](https://arxiv.org/abs/{matched['arxiv_id']})")
        st.caption(
            "Recommendations are other papers similar to this one. If this isn't "
            "what you meant, refine your title."
        )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def recommendations_tab() -> None:
    st.subheader("Recommendations")
    st.write(
        "Find related papers — either from a known paper title, or from a "
        "free-text query. SPECTER2 is the primary model; SciBERT is included "
        "for comparison."
    )

    c1, c2, c3 = st.columns([3, 2, 1])
    mode_label  = c1.selectbox("Search mode", list(MODE_CHOICES.keys()))
    model_label = c2.selectbox("Embedding model", list(MODEL_CHOICES.keys()))
    top_k       = c3.slider("Results", min_value=5, max_value=20, value=10, step=5)

    cats, years = get_filter_options()
    f1, f2 = st.columns(2)
    category_label = f1.selectbox("Filter by category", ["Any category"] + cats)
    year_label     = f2.selectbox("Filter by year", ["Any year"] + years)
    category = None if category_label == "Any category" else category_label
    year     = None if year_label == "Any year" else year_label

    mode  = MODE_CHOICES[mode_label]
    model = MODEL_CHOICES[model_label]

    if mode == "title":
        placeholder = "e.g. paste a paper title from the dataset"
        helptext = "Enter a paper title; the closest match is used to find similar work."
    else:
        placeholder = "e.g. transformer models for time series forecasting"
        helptext = "Enter keywords or a question describing what you're looking for."

    query = st.text_input("Your query", placeholder=placeholder, help=helptext)
    go = st.button("Get recommendations", type="primary")

    if go:
        if not query.strip():
            st.warning("Enter a title or query first.")
        else:
            try:
                with st.spinner("Loading model and finding papers…"):
                    recommender = get_recommender(model)
                    fallback = False
                    if mode == "title":
                        matched = recommender.match_title(query.strip())
                        if matched is None:
                            # No title clears the bar — don't fake a match.
                            # Fall back to a free-text search on the title text.
                            fallback = True
                    else:
                        matched = None

                    search_mode = "keywords" if (mode == "keywords" or fallback) else "title"
                    t0 = time.perf_counter()
                    results = recommender.recommend(
                        query.strip(), mode=search_mode, top_k=top_k,
                        category=category, year=year,
                    )
                    elapsed = time.perf_counter() - t0
                    corpus_size = len(recommender.arxiv_ids)
                st.session_state["rec_results"] = results
                st.session_state["rec_meta"] = {
                    "model_label": model_label, "mode": mode,
                    "query": query.strip(), "corpus_size": corpus_size,
                    "elapsed": elapsed, "matched": matched, "fallback": fallback,
                    "category": category, "year": year,
                }
            except ValueError as e:
                st.session_state.pop("rec_results", None)
                st.error(str(e))
            except Exception as e:                          # noqa: BLE001
                st.session_state.pop("rec_results", None)
                st.error(f"Something went wrong while searching: {e}")

    # Render whatever the last successful search produced (persists across
    # slider/widget reruns, so results don't vanish when you adjust a control).
    results = st.session_state.get("rec_results")
    meta = st.session_state.get("rec_meta")

    if meta and not results:
        # A search ran but nothing came back — almost always over-strict filters.
        active = []
        if meta.get("category"):
            active.append(f"category = {meta['category']}")
        if meta.get("year"):
            active.append(f"year = {meta['year']}")
        if active:
            st.info(
                "No papers match those filters (" + ", ".join(active) +
                "). Try widening the category or year."
            )
        else:
            st.info("No results for that query.")

    if results and meta:
        filter_bits = []
        if meta.get("category"):
            filter_bits.append(f"category {meta['category']}")
        if meta.get("year"):
            filter_bits.append(f"year {meta['year']}")
        filter_txt = ("  ·  Filtered to " + " & ".join(filter_bits)) if filter_bits else ""
        caption = (
            f"Retrieved from {meta['corpus_size']:,} papers  ·  "
            f"Model: {meta['model_label']}  ·  "
            f"Search time: {meta['elapsed']:.2f} s" + filter_txt
        )

        if meta.get("matched"):
            # Title mode with a confirmed match: recommendations on the left,
            # the matched paper in a box on the right.
            st.success(f"Top {len(results)} similar papers — {meta['model_label']}")
            st.caption(caption)
            main_col, side_col = st.columns([3, 1.3])
            with side_col:
                render_matched_box(meta["matched"])
            with main_col:
                for i, paper in enumerate(results, start=1):
                    render_result(i, paper, meta["query"], meta["mode"])
        else:
            if meta.get("fallback"):
                st.warning(
                    "We don't have that exact paper in the corpus — showing papers "
                    "related to your search instead."
                )
                st.success(f"Top {len(results)} related papers — {meta['model_label']}")
            else:
                st.success(f"Top {len(results)} results — {meta['model_label']}")
            st.caption(caption)
            for i, paper in enumerate(results, start=1):
                render_result(i, paper, meta["query"], meta["mode"])


def render_browse_row(row: pd.Series) -> None:
    """Compact metadata card for one paper in the browse tab (no scores)."""
    try:
        cites = int(row["citation_count"]) if pd.notna(row["citation_count"]) else 0
    except (ValueError, TypeError):
        cites = 0
    with st.container(border=True):
        st.markdown(f"**{row['title']}**")
        st.caption(str(row["authors"]))
        st.caption(
            f"{row['category']}  ·  {row['publication_year']}  ·  {cites} citations"
        )
        with st.expander("Abstract"):
            st.write(str(row["abstract"]))
        st.markdown(f"[View on arXiv](https://arxiv.org/abs/{row['arxiv_id']})")


def browse_tab() -> None:
    st.subheader("Browse the corpus")
    df = load_corpus()
    st.write(
        f"Explore the {len(df):,} papers in the dataset — filter and sort, no "
        "query needed. This is a plain metadata view; for similarity-based "
        "results use the Recommendations tab."
    )

    cats, years = get_filter_options()
    c1, c2, c3 = st.columns(3)
    category = c1.selectbox("Category", ["All"] + cats, key="browse_cat")
    year     = c2.selectbox("Year", ["All"] + years, key="browse_year")
    sort_label = c3.selectbox(
        "Sort by",
        ["Most cited", "Newest first", "Oldest first", "Title A–Z"],
        key="browse_sort",
    )

    # Reset to page 1 whenever the filter/sort selection changes (otherwise you
    # could be stranded on a page that no longer exists for the new result set).
    sig = (category, year, sort_label)
    if st.session_state.get("browse_sig") != sig:
        st.session_state["browse_sig"] = sig
        st.session_state["browse_page"] = 1

    view = df
    if category != "All":
        view = view[view["category"] == category]
    if year != "All":
        view = view[view["publication_year"].astype(str) == year]

    if sort_label == "Most cited":
        view = view.sort_values("citation_count", ascending=False)
    elif sort_label == "Newest first":
        view = view.sort_values("publication_year", ascending=False)
    elif sort_label == "Oldest first":
        view = view.sort_values("publication_year", ascending=True)
    else:  # Title A–Z
        view = view.sort_values("title", key=lambda s: s.astype(str).str.lower())

    total = len(view)
    if total == 0:
        st.info("No papers match these filters.")
        return

    PAGE = 20
    n_pages = max(1, math.ceil(total / PAGE))
    page = st.number_input(
        "Page", min_value=1, max_value=n_pages, step=1, key="browse_page"
    )
    start = (int(page) - 1) * PAGE
    chunk = view.iloc[start:start + PAGE]
    st.caption(
        f"Showing {start + 1:,}–{min(start + PAGE, total):,} of {total:,}  ·  "
        f"page {int(page)} of {n_pages}"
    )

    for _, row in chunk.iterrows():
        render_browse_row(row)


def render_embedding_section() -> None:
    sep = load_separation()
    model_label = st.radio(
        "Model", ["SPECTER2 (primary)", "SciBERT (comparison)"],
        horizontal=True, key="emb_model",
    )
    model = "specter2" if model_label.startswith("SPECTER2") else "scibert"
    proj = load_projections(model)

    if proj is None:
        st.info(
            "Embedding projections haven't been generated yet. On the machine "
            "that holds the embeddings (the cluster), run from the project root:\n\n"
            "```\npython src/compute_projections.py\n```\n\n"
            "That writes `results/figures/projections_*.csv` and "
            "`embedding_separation.json`; this section then renders the PCA and "
            "t-SNE maps automatically."
        )
        return

    proj_label = st.radio("Projection", ["t-SNE", "PCA"], horizontal=True, key="emb_proj")
    xcol, ycol = ("tsne_x", "tsne_y") if proj_label == "t-SNE" else ("pca_x", "pca_y")

    st.caption(
        f"{len(proj):,}-paper stratified sample, coloured by category. "
        "Hover for titles; drag to zoom."
    )
    chart = (
        alt.Chart(proj).mark_circle(size=18, opacity=0.5).encode(
            x=alt.X(f"{xcol}:Q", title=f"{proj_label} 1",
                    axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y(f"{ycol}:Q", title=f"{proj_label} 2",
                    axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("category:N", title="Category", scale=CAT_SCALE),
            tooltip=["title:N", "category:N"],
        ).interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    if sep:
        c = st.columns(2)
        s_spec, s_sci = sep.get("specter2"), sep.get("scibert")
        c[0].metric("SPECTER2 category separation",
                    f"{s_spec:.3f}" if pd.notna(s_spec) else "—")
        c[1].metric("SciBERT category separation",
                    f"{s_sci:.3f}" if pd.notna(s_sci) else "—")
        st.caption(
            "Silhouette score of the four categories in embedding space (range "
            "−1 to 1). Both models score near zero — the categories do **not** form "
            "cleanly separated clusters, which is expected given how much these "
            "fields overlap. SPECTER2 edges ahead of SciBERT, but neither separates "
            "by primary category; that isn't what the recommender relies on."
        )

    sim = load_similarity()
    if sim is not None:
        st.markdown("**Same- vs different-category similarity**")
        st.caption(
            "Cosine similarity for random same- vs different-category pairs. The "
            "two distributions nearly overlap — SPECTER2 shows a slight gap "
            "(same-category pairs marginally higher), SciBERT essentially none — "
            "consistent with the near-zero silhouette: primary category is not a "
            "strong axis of separation in this corpus."
        )
        st.altair_chart(
            alt.Chart(sim).mark_area(opacity=0.45, interpolate="step").encode(
                x=alt.X("cosine:Q", bin=alt.Bin(maxbins=40),
                        title="Cosine similarity"),
                y=alt.Y("count()", stack=None, title="Pairs"),
                color=alt.Color("comparison:N", title=None, scale=COMPARISON_SCALE),
            ).properties(height=220).facet(column=alt.Column("model:N", title=None)),
            use_container_width=True)
    else:
        st.caption(
            "Tip: the latest `compute_projections.py` also writes a "
            "`similarity_pairs.csv`; re-run it to add the same-vs-different "
            "category similarity histogram here."
        )


def analytics_tab() -> None:
    st.subheader("Analytics")
    st.write(
        "What the curated corpus looks like and how the embedding models organise "
        "it. Every figure here is computed from the dataset itself."
    )
    d = analytics_data()
    h = d["headline"]

    m = st.columns(4)
    m[0].metric("Papers", f"{h['papers']:,}")
    m[1].metric("Categories", h["categories"])
    m[2].metric("Years", f"{h['year_min']}–{h['year_max']}")
    m[3].metric("% cited ≥ once", f"{h['pct_cited']}%")
    m2 = st.columns(4)
    m2[0].metric("Unique keywords", f"{d['n_unique_tags']:,}")
    m2[1].metric("Mean citations", f"{h['mean_citations']:.1f}")
    m2[2].metric("Most citations", f"{h['max_citations']:,}")
    m2[3].metric("Embedding dim", "768")

    st.divider()
    st.markdown("#### Corpus composition")
    col = st.columns(2)
    with col[0]:
        st.caption("Papers per year — balanced by design.")
        st.altair_chart(
            alt.Chart(d["per_year"]).mark_bar(color=COLOR_PAPERS).encode(
                x=alt.X("publication_year:O", title="Year"),
                y=alt.Y("papers:Q", title="Papers"),
                tooltip=["publication_year", "papers"],
            ), use_container_width=True)
    with col[1]:
        st.caption("Category mix — cs.AI dominates (a known imbalance).")
        st.altair_chart(
            alt.Chart(d["per_cat"]).mark_bar().encode(
                x=alt.X("papers:Q", title="Papers"),
                y=alt.Y("category:N", sort="-x", title=None),
                color=alt.Color("category:N", scale=CAT_SCALE, legend=None),
                tooltip=["category", "papers"],
            ), use_container_width=True)

    st.divider()
    st.markdown("#### Citation landscape")
    col = st.columns(2)
    with col[0]:
        st.caption("Citation distribution — long-tailed; many recent papers uncited.")
        st.altair_chart(
            alt.Chart(d["cit_buckets"]).mark_bar(color=COLOR_CIT).encode(
                x=alt.X("bucket:N", sort=["0", "1-5", "6-20", "21-100", "100+"],
                        title="Citations"),
                y=alt.Y("papers:Q", title="Papers"),
                tooltip=["bucket", "papers"],
            ), use_container_width=True)
    with col[1]:
        st.caption("Most-cited papers in the corpus.")
        st.dataframe(
            d["top_cited"][["title", "category", "publication_year",
                            "citation_count", "arxiv_url"]],
            hide_index=True, use_container_width=True,
            column_config={
                "title": "Title", "category": "Category",
                "publication_year": st.column_config.NumberColumn("Year", format="%d"),
                "citation_count": st.column_config.NumberColumn("Citations", format="%d"),
                "arxiv_url": st.column_config.LinkColumn("arXiv", display_text="open"),
            })

    col = st.columns(2)
    with col[0]:
        st.caption("Mean citations by category — pulled up by a few mega-cited "
                   "papers, so read it as a rough signal, not a typical value.")
        st.altair_chart(
            alt.Chart(d["cit_by_cat"]).mark_bar().encode(
                x=alt.X("category:N", sort="-y", title=None),
                y=alt.Y("mean:Q", title="Mean citations"),
                color=alt.Color("category:N", scale=CAT_SCALE, legend=None),
                tooltip=["category:N", "mean:Q"],
            ), use_container_width=True)
    with col[1]:
        st.caption("Mean citations by year — recent papers haven't had time to "
                   "accumulate citations (a citation-lag effect, not lower quality).")
        st.altair_chart(
            alt.Chart(d["cit_by_year"]).mark_bar(color=COLOR_CIT).encode(
                x=alt.X("publication_year:O", title="Year"),
                y=alt.Y("mean:Q", title="Mean citations"),
                tooltip=["publication_year:O", "mean:Q"],
            ), use_container_width=True)

    st.divider()
    st.markdown("#### Topics & trends")
    st.caption(f"Most common KeyBERT keywords across {h['papers']:,} papers.")
    st.altair_chart(
        alt.Chart(d["top_tags"]).mark_bar(color=COLOR_TAGS).encode(
            x=alt.X("count:Q", title="Papers"),
            y=alt.Y("tag:N", sort="-x", title=None),
            tooltip=["tag", "count"],
        ), use_container_width=True)

    trend = st.radio("Show keywords that are…", ["Rising", "Declining"],
                     horizontal=True, key="topic_trend")
    frame = d["rising"] if trend == "Rising" else d["falling"]
    st.caption(
        f"{'Fastest-rising' if trend == 'Rising' else 'Fastest-declining'} "
        f"keywords by yearly count ({d['years'][0]}→{d['years'][-1]})."
    )
    st.altair_chart(
        alt.Chart(frame).mark_line(point=True).encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("count:Q", title="Papers tagged"),
            color=alt.Color("tag:N", title="Keyword",
                            scale=alt.Scale(scheme="tableau10")),
            tooltip=["tag", "year", "count"],
        ), use_container_width=True)

    st.divider()
    st.markdown("#### Embedding space")
    st.write(
        "How the two models lay the corpus out in 2-D. A caveat up front: the four "
        "arXiv categories overlap heavily and papers are often cross-listed, so "
        "expect the colours to mix rather than form clean clusters. The embeddings "
        "organise papers by fine-grained topic, not cleanly by primary category — "
        "the recommender's real strength is instance-level retrieval (see the "
        "Evaluation tab), not category clustering."
    )
    render_embedding_section()


def evaluation_tab() -> None:
    st.subheader("Evaluation")
    st.write(
        "Two evaluations of the recommender, reading the files the eval scripts "
        "write: a **known-item** retrieval test, and the earlier **proxy** "
        "experiments used to pick the configuration. Nothing here is hardcoded — "
        "missing files show how to generate them."
    )

    # ---- Known-item retrieval (both models) ----
    st.markdown("#### Known-item retrieval — SPECTER2 vs SciBERT")
    ki = load_known_item()
    if ki is None:
        st.info(
            "Not generated yet. Run from the project root (where the embeddings "
            "are available):\n\n```\npython src/evaluation.py\n```\n\n"
            "It writes `results/evaluation/known_item_summary.csv`."
        )
    else:
        st.caption(
            "Each query's source paper is the one known-relevant document "
            "(provenance-based ground truth, independent of the embeddings). With "
            "one relevant item per query, Recall@K is simply whether that paper "
            "lands in the top K — deliberately conservative."
        )

        overall = overall_metrics(ki)
        delta = model_delta(overall)

        # Headline — how far the primary model leads the baseline.
        if delta:
            names = {"mrr": "MRR", "recall@10": "Recall@10"}
            cards = st.columns(len(delta))
            for c, (key, dv) in zip(cards, delta.items()):
                lead = f"{dv['factor']:.1f}× vs SciBERT" if dv["factor"] else None
                c.metric(f"SPECTER2 {names.get(key, key)}",
                         f"{dv['primary']:.3f}", lead)

        # Section 1 — overall metrics, averaged across all query formulations.
        st.markdown("**Overall metrics** — averaged across all four query formulations")
        ov_disp = overall.rename(columns={
            "model": "Model", "mrr": "MRR", "recall@5": "Recall@5",
            "recall@10": "Recall@10", "recall@20": "Recall@20", "ndcg@10": "NDCG@10",
        })
        ov_disp["Model"] = ov_disp["Model"].map(MODEL_LABELS).fillna(ov_disp["Model"])
        st.dataframe(ov_disp.round(3), hide_index=True, use_container_width=True)

        # Section 2 — Recall@K curve (only K = 5/10/20 were computed).
        rc = recall_curve(ki)
        if not rc.empty:
            rc = rc.copy()
            rc["Model"] = rc["model"].map(MODEL_LABELS).fillna(rc["model"])
            st.caption("**Recall@K** — share of queries whose source paper is in "
                       "the top K (cutoffs 5/10/20; K = 1 and 50 were not computed).")
            st.altair_chart(
                alt.Chart(rc).mark_line(point=True).encode(
                    x=alt.X("k:O", title="K"),
                    y=alt.Y("recall:Q", title="Recall@K"),
                    color=alt.Color("Model:N", title=None, scale=MODEL_SCALE),
                    tooltip=["Model:N", "k:O", "recall:Q"],
                ), use_container_width=True)

        # By query type — the per-formulation breakdown.
        opts = {"Recall@10": "hit@10", "Recall@5": "hit@5",
                "NDCG@10": "ndcg@10", "MRR": "mrr"}
        opts = {k: v for k, v in opts.items() if v in ki.columns}
        if opts:
            st.markdown("**By query type**")
            metric_label = st.radio("Metric", list(opts.keys()),
                                    horizontal=True, key="ki_metric")
            col = opts[metric_label]
            cf = pd.DataFrame({
                "Query type": ki["query_type"].map(QTYPE_LABELS).fillna(ki["query_type"]),
                "Model": ki["model"].map(MODEL_LABELS).fillna(ki["model"]),
                "value": ki[col],
            })
            st.altair_chart(
                alt.Chart(cf).mark_bar().encode(
                    x=alt.X("Query type:N", title=None),
                    y=alt.Y("value:Q", title=metric_label),
                    color=alt.Color("Model:N", title=None, scale=MODEL_SCALE),
                    xOffset="Model:N",
                    tooltip=["Query type:N", "Model:N", "value:Q"],
                ), use_container_width=True)

        # Section 5 — failure / quality analysis from the per-query ranks.
        st.markdown("**Where it does best and worst** — by the source paper's rank")
        sel = st.columns(2)
        with sel[0]:
            fm = st.selectbox("Model", list(MODEL_LABELS.keys()),
                              format_func=lambda m: MODEL_LABELS.get(m, m),
                              key="fail_model")
        qkeys = list(QTYPE_LABELS.keys())
        with sel[1]:
            fq = st.selectbox(
                "Query type", qkeys,
                index=qkeys.index("natural_query") if "natural_query" in qkeys else 0,
                format_func=lambda q: QTYPE_LABELS.get(q, q), key="fail_qtype")
        picks = failure_picks(load_known_item_queries(fm, fq))
        if picks is None:
            st.info("Per-query ranks not found for that combination.")
        else:
            pcols = st.columns(3)
            for c, (_, row) in zip(pcols, picks.iterrows()):
                with c:
                    st.markdown(f"**{row['label']}** · rank **{int(row['rank'])}**")
                    st.caption(row["query"])
            st.caption(
                "Each rank is where that query's own source paper landed in the "
                "full ranking — concrete best/median/worst cases for the selected "
                "model and query type. Across formulations, natural-language "
                "queries resolve best and the more abstractive task/problem "
                "phrasings worst, matching the chart above."
            )

        with st.expander("Full results table (all metrics, per query type)"):
            full = ki.rename(columns={"hit@5": "recall@5", "hit@10": "recall@10",
                                      "hit@20": "recall@20"})
            st.dataframe(full, hide_index=True, use_container_width=True)

    # ---- Hybrid late-fusion experiment (optional file) ----
    hyb = load_hybrid_sweep()
    if hyb is not None and not hyb.empty:
        st.divider()
        st.markdown("#### Hybrid — SPECTER2 + SciBERT (late fusion)")
        best = hybrid_best(hyb)
        if best:
            txt = (f"Best blend at **w = {best['weight']:.2f}** "
                   f"(~{best['weight'] * 100:.0f}% SPECTER2): overall MRR "
                   f"{best['mrr']:.3f}")
            if best["baseline"] is not None:
                txt += (f" vs {best['baseline']:.3f} at SPECTER2-only "
                        f"(+{best['mrr'] - best['baseline']:.3f})")
            st.caption(txt + ".")
        hmetric = st.radio("Metric", ["MRR", "Recall@10"], horizontal=True,
                           key="hybrid_metric")
        hcol = "mrr" if hmetric == "MRR" else "hit@10"
        hf = pd.DataFrame({
            "Weight on SPECTER2": hyb["weight_specter2"],
            "Query type": hyb["query_type"].map(QTYPE_LABELS).fillna(hyb["query_type"]),
            "value": hyb[hcol],
        })
        best_w = best["weight"] if best else 0.9
        line_chart = alt.Chart(hf).mark_line(point=True).encode(
            x=alt.X("Weight on SPECTER2:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("value:Q", title=hmetric),
            color=alt.Color("Query type:N", title=None),
            tooltip=["Query type:N", "Weight on SPECTER2:Q", "value:Q"],
        )
        rule = alt.Chart(pd.DataFrame({"Weight on SPECTER2": [best_w]})).mark_rule(
            strokeDash=[4, 4], color="#888", size=1).encode(x="Weight on SPECTER2:Q")
        st.altair_chart(line_chart + rule, use_container_width=True)
        st.caption(
            "Late fusion: each paper scored w·cos(SPECTER2) + (1−w)·cos(SciBERT); "
            "w = 1.0 is SPECTER2 only, 0.0 is SciBERT only (dashed line = best "
            "overall blend). A 90/10 blend slightly helps the natural and problem "
            "queries and is roughly flat on the others; below ~80% SPECTER2 the "
            "weak model drags results down sharply."
        )

    # ---- Proxy experiments (configuration selection) ----
    st.divider()
    st.markdown("#### Proxy experiments — configuration selection")
    st.caption(
        "Earlier experiments on an automatic proxy ground truth: a 2026 temporal "
        "hold-out where a retrieved paper counts as relevant if it shares the "
        "query paper's category AND at least one keyword. These scores are on that "
        "proxy definition and are **not** comparable to the known-item numbers "
        "above — they were used to choose the recommender's configuration. "
        "Exported from Weights & Biases."
    )

    first = load_proxy("first")
    sweep = load_proxy_sweep()
    cat_exp = load_proxy("category")
    restr = load_proxy("restricted")

    if all(x is None for x in (first, sweep, cat_exp, restr)):
        st.info("No proxy-experiment exports found in `results/wandb_exports/`.")
    else:
        # Embedding model & distance metric.
        if first is not None and {"model", "metric", "k", "ndcg_at_k"} <= set(first.columns):
            f10 = first[first["k"] == 10].copy()
            f10["Model"] = f10["model"].map(MODEL_LABELS).fillna(f10["model"])
            f10["Distance"] = f10["metric"].str.capitalize()
            st.caption("**Embedding model & distance** — SPECTER2 clearly beats "
                       "SciBERT; cosine and Euclidean are practically tied (NDCG@10).")
            st.altair_chart(
                alt.Chart(f10).mark_bar().encode(
                    x=alt.X("Model:N", title=None),
                    y=alt.Y("ndcg_at_k:Q", title="NDCG@10"),
                    color=alt.Color("Model:N", title=None, scale=MODEL_SCALE),
                    xOffset="Distance:N",
                    tooltip=["Model:N", "Distance:N", "ndcg_at_k:Q"],
                ), use_container_width=True)

        # Category-boost α sweep.
        if sweep is not None and {"category", "k", "alpha", "ndcg_at_k"} <= set(sweep.columns):
            g = (sweep[(sweep["category"] == "global") & (sweep["k"] == 10)]
                 [["alpha", "ndcg_at_k"]].sort_values("alpha"))
            if not g.empty:
                st.caption("**Category-boost strength (α)** — NDCG@10 climbs then "
                           "plateaus; the knee is around α = 0.03 (dashed line), the "
                           "value the recommender uses.")
                line = alt.Chart(g).mark_line(point=True, color=COLOR_PAPERS).encode(
                    x=alt.X("alpha:Q", title="Category-boost α"),
                    y=alt.Y("ndcg_at_k:Q", title="NDCG@10"),
                    tooltip=["alpha:Q", "ndcg_at_k:Q"])
                rule = (alt.Chart(pd.DataFrame({"alpha": [0.03]}))
                        .mark_rule(color="#E63946", strokeDash=[4, 4]).encode(x="alpha:Q"))
                st.altair_chart(line + rule, use_container_width=True)

        cc = st.columns(2)
        # Per-category difficulty.
        if cat_exp is not None and {"category", "k", "ndcg_at_k"} <= set(cat_exp.columns):
            with cc[0]:
                c10 = cat_exp[cat_exp["k"] == 10]
                st.caption("**By category** — some fields are easier than others.")
                st.altair_chart(
                    alt.Chart(c10).mark_bar().encode(
                        x=alt.X("category:N", sort="-y", title=None),
                        y=alt.Y("ndcg_at_k:Q", title="NDCG@10"),
                        color=alt.Color("category:N", scale=CAT_SCALE, legend=None),
                        tooltip=["category:N", "ndcg_at_k:Q"],
                    ), use_container_width=True)
        # Boost vs restrict vs neither.
        if sweep is not None and restr is not None:
            try:
                g2 = sweep[(sweep["category"] == "global") & (sweep["k"] == 10)]
                nob = float(g2[g2["alpha"] == 0.0]["ndcg_at_k"].iloc[0])
                boost = float(g2[g2["alpha"] == 0.03]["ndcg_at_k"].iloc[0])
                r10 = float(restr[restr["k"] == 10]["ndcg_at_k"].iloc[0])
                comp = pd.DataFrame({
                    "Setting": ["No boost", "Boost α=0.03", "Category-restricted"],
                    "ndcg": [nob, boost, r10],
                })
                with cc[1]:
                    st.caption("**Boost vs restrict** — a soft α-boost ≈ hard "
                               "category-restriction, both well above no boost.")
                    st.altair_chart(
                        alt.Chart(comp).mark_bar(color="#8E5BD6").encode(
                            x=alt.X("Setting:N", sort=None, title=None),
                            y=alt.Y("ndcg:Q", title="NDCG@10"),
                            tooltip=["Setting:N", "ndcg:Q"],
                        ), use_container_width=True)
            except (IndexError, KeyError):
                pass

    st.divider()
    with st.expander("How these evaluations work (methodology & caveats)"):
        st.markdown(
            "- **Known-item retrieval.** Every query was generated from a specific "
            "source paper, so that paper is treated as the one relevant document. "
            "Objective and independent of the embeddings, but conservative: a "
            "genuinely relevant *other* paper ranked above the source counts "
            "against the score.\n"
            "- **Proxy experiments.** A 2026 temporal hold-out with an automatic "
            "relevance rule (same category AND a shared keyword). Cheap and fully "
            "reproducible, but only an approximation of true relevance — hence used "
            "for *configuration choices* (model, distance, boost), with the "
            "known-item test as the cleaner check. Proxy and known-item NDCG are "
            "not directly comparable.\n"
            "- **Hyperparameter selection.** The title-mode category boost "
            "(α = 0.03) is the knee of the sweep above; α is the value set in "
            "`recommender.py`."
        )


REPO_URL = "https://github.com/teejay222/paper-recommender-system"


def project_info_tab() -> None:
    st.subheader("Project Info")
    h = analytics_data()["headline"]
    n_tags = analytics_data()["n_unique_tags"]

    # 1 — Overview / objective.
    st.markdown("#### Overview")
    st.write(
        "An end-to-end data-science workflow that builds a content-based "
        "recommender for computer-science research papers: the user supplies a "
        "paper they are interested in (or a free-text query) and the system "
        "returns the most similar papers in the corpus. Metadata comes from the "
        "arXiv and Semantic Scholar APIs. The project is organised around the "
        "course work packages, summarised in the table further down."
    )

    # 2 — Dataset (live numbers).
    st.markdown("#### Dataset")
    d = st.columns(4)
    d[0].metric("Papers", f"{h['papers']:,}")
    d[1].metric("Categories", h["categories"])
    d[2].metric("Years", f"{h['year_min']}–{h['year_max']}")
    d[3].metric("Unique keywords", f"{n_tags:,}")
    st.caption(
        "Computer-science papers (cs.AI, cs.CL, cs.CV, cs.LG) scraped from the "
        "arXiv API and enriched with Semantic Scholar citation counts. The raw "
        "pull is balanced by year (≈8,333 papers per year) and de-duplicated; "
        f"{h['pct_cited']}% of papers have at least one citation, and the mean is "
        f"{h['mean_citations']} (heavily skewed by a few very highly-cited papers)."
    )

    # 3 — Pipeline.
    st.markdown("#### Pipeline")
    st.markdown(
        "1. **Collection** — paper metadata scraped from the arXiv API "
        "(`data_collection.py`).\n"
        "2. **Enrichment** — citation and reference counts from Semantic Scholar "
        "(`semantic_scholar_enrichment.py`).\n"
        "3. **Quality & cleaning** — missing-value, duplicate and citation/"
        "reference checks, then text cleaning (`data_validation.py`, "
        "`preprocessing.py`).\n"
        "4. **Annotation** — automated domain labels and dictionary tags "
        "(`annotation.py`).\n"
        "5. **Keywords** — KeyBERT key-phrases per paper "
        "(`keyword_extraction_v2.py`).\n"
        "6. **Embeddings** — SPECTER2 and SciBERT vectors, generated on an A100 "
        "GPU pod on the Kubernetes cluster (`embeddings_specter2.py`, "
        "`embeddings_scibert.py`).\n"
        "7. **Recommend** — rank by cosine similarity with a category boost "
        "(`recommender.py`).\n"
        "8. **Evaluate & track** — known-item and proxy experiments, logged to "
        "Weights & Biases (`evaluation.py`, `experiments*.py`)."
    )

    # 4 — Models.
    st.markdown("#### Models")
    st.write(
        "Two frozen, off-the-shelf scientific text encoders (no fine-tuning), "
        "each turning a paper's title + abstract into a 768-dimensional vector:"
    )
    st.markdown(
        "- **SPECTER2** — the primary model; document-level embeddings trained "
        "with citation context, suited to paper-level similarity.\n"
        "- **SciBERT** — a scientific-domain BERT, used as the baseline."
    )

    # 5 — Recommender.
    st.markdown("#### Recommender")
    st.write(
        "Recommendations are the nearest neighbours by cosine similarity in the "
        "SPECTER2 space, with a small category boost (α = 0.03) that nudges "
        "same-category papers upward. It works from either an existing paper "
        "(look it up, get similar ones) or a free-text query. The value of α was "
        "chosen from the sweep shown in the Evaluation tab."
    )

    # 6 — Evaluation summary (pointer, not a duplicate).
    st.markdown("#### Evaluation")
    st.write(
        "Two complementary methods — see the **Evaluation** tab for the numbers: "
        "a **known-item** retrieval test (each query's source paper is the one "
        "relevant document, scored with Precision / Recall / NDCG@K) and a "
        "**proxy** temporal hold-out used for configuration choices. The brief "
        "noted SciDocs for this package; that was discussed with the supervisor "
        "and waived in favour of these two task-level methods."
    )

    # 7 — Work-package coverage + deviations.
    st.markdown("#### Work packages & deviations")
    wp = pd.DataFrame(
        [
            ["Data Scraping", "", "Done", "arXiv API + Semantic Scholar enrichment"],
            ["Data Annotation", "", "Done", "Automated domain labels + dictionary tags (not Label Studio)"],
            ["Data Quality", "Yes", "Done", "Missing-value, duplicate & citation/reference checks"],
            ["Kubernetes Cluster", "", "Done", "A100 GPU pod (PVC-backed) for embedding generation"],
            ["Experiments Logging", "Yes", "Done", "Weights & Biases (model, distance, α sweep, per-category)"],
            ["Vector Embeddings", "Yes", "Done", "SPECTER2 (primary) + SciBERT (baseline), 768-dim"],
            ["Hyperparameter Tuning", "", "Done", "Category-boost α sweep (knee at 0.03)"],
            ["Recommender System", "Yes", "Done", "Cosine similarity + category boost"],
            ["Performance Evaluation", "Yes", "Done", "Known-item + proxy (Precision / Recall / NDCG@K)"],
            ["Perturbation Analysis", "", "Not attempted", "—"],
            ["Frontend Application", "", "Done", "Streamlit app (five tabs)"],
        ],
        columns=["Work package", "Mandatory", "Status", "How it was done"],
    )
    st.dataframe(wp, hide_index=True, use_container_width=True)
    st.markdown(
        "**Deviations from the original brief**\n"
        "- **SciDocs evaluation** — discussed and waived; replaced by two "
        "task-level methods (known-item + proxy).\n"
        "- **Data annotation** — performed automatically (domain labels + "
        "dictionary tags), not via Label Studio.\n"
        "- **OpenReview** — explored (`openreview_processing.py`) then dropped; "
        "the corpus is arXiv + Semantic Scholar only.\n"
        "- **Citation-based ground truth** — explored "
        "(`data/evaluation/citation_ground_truth.json`) but not used for the "
        "headline evaluation.\n"
        "- **Search tab → Browse** — the planned search view became a full "
        "corpus browser that needs no query.\n"
        "- **Perturbation analysis** — not attempted (optional package)."
    )

    # 8 — Tech stack & repository.
    st.markdown("#### Tech stack & repository")
    st.write(
        "Python · pandas · sentence-transformers / adapters (SPECTER2) · "
        "Hugging Face Transformers (SciBERT) · KeyBERT · scikit-learn · Altair · "
        "Streamlit · Weights & Biases · Docker / Kubernetes (A100 GPU pod)."
    )
    st.markdown(f"Repository: [{REPO_URL}]({REPO_URL})")


def main() -> None:
    st.title("Academic Paper Recommendation System")

    tabs = st.tabs(["Browse", "Recommendations", "Analytics", "Evaluation", "Project Info"])

    with tabs[0]:
        browse_tab()
    with tabs[1]:
        recommendations_tab()
    with tabs[2]:
        analytics_tab()
    with tabs[3]:
        evaluation_tab()
    with tabs[4]:
        project_info_tab()


if __name__ == "__main__":
    main()
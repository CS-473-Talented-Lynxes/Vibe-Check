import streamlit as st
import pandas as pd
import pydeck as pdk

from src.embeddings import ComplaintSearcher
from src.clustering import LocationClusterer


st.set_page_config(page_title="Vibe Check", layout="wide")


@st.cache_resource(show_spinner=False)
def get_searcher():
    return ComplaintSearcher()


@st.cache_resource(show_spinner=False)
def get_clusterer():
    return LocationClusterer()


def category_label(problem, detail):
    return f"{problem} - {detail}"


def add_selection(label, payload, similarity_store):
    current = st.session_state.selected_labels
    if label in current:
        similarity_store[label] = max(similarity_store.get(label, 0.0), payload["similarity"])
        return
    if len(current) >= 50:
        st.warning("You can select up to 50 concerns at a time.")
        return
    current.append(label)
    similarity_store[label] = payload["similarity"]


def build_match_payload(selected_labels, category_lookup, similarity_store):
    payload = []
    for label in selected_labels:
        problem, detail = category_lookup[label]
        payload.append({
            "problem": problem,
            "detail": detail,
            "similarity": float(similarity_store.get(label, 1.0)),
        })
    return payload


def build_map(points_df, recommendations):
    base_view = pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=10, pitch=35)
    layers = []

    best_clusters = recommendations.get("best", [])
    worst_clusters = recommendations.get("worst", [])

    if best_clusters:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=best_clusters,
                get_position="[center_lon, center_lat]",
                get_radius=900,
                get_fill_color="[44, 160, 44, 180]",
                get_line_color="[0, 80, 0, 220]",
                line_width_min_pixels=2,
                pickable=True,
            )
        )

    if worst_clusters:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=worst_clusters,
                get_position="[center_lon, center_lat]",
                get_radius=900,
                get_fill_color="[220, 38, 38, 180]",
                get_line_color="[127, 29, 29, 220]",
                line_width_min_pixels=2,
                pickable=True,
            )
        )

    tooltip = {
        "html": (
            "<b>{primary_borough}</b> {primary_zip}<br/>"
            "Normalized severity: {normalized_severity}<br/>"
            "Severity: {severity_score}<br/>"
            "Matched complaints: {complaint_count}"
        ),
        "style": {"backgroundColor": "#111827", "color": "white"},
    }

    return pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=base_view,
        layers=layers,
        tooltip=tooltip,
    )


def run_analysis(clusterer, matched_categories):
    cluster_results = clusterer.cluster_extremes(
        matched_categories,
        k_clusters=300,
        top_n=50,
    )

    return {
        "matched_categories": matched_categories,
        "best_clusters": cluster_results["best"],
        "worst_clusters": cluster_results["worst"],
    }


def render_selected_concerns():
    selected_labels = st.session_state.selected_labels
    if not selected_labels:
        st.info("Pick at least one concern to generate recommendations.")
        return

    st.caption("Current concern set")
    for label in selected_labels:
        st.write(f"- {label}")


def render_results_page():
    st.title("Results")
    render_selected_concerns()

    results = st.session_state.analysis_results
    best_clusters = results["best_clusters"]
    worst_clusters = results["worst_clusters"]

    if not best_clusters and not worst_clusters:
        st.warning("No ranked areas were available for this selection.")
        return

    st.caption(
        "These map markers and ranked lists come from the KMeans clustering pipeline. Clusters are now ranked by normalized severity, which means total concern weight divided by matched complaint count."
    )

    map_col, list_col = st.columns([1.35, 1])

    with map_col:
        st.subheader("City Map")
        st.pydeck_chart(
            build_map(
                pd.DataFrame(),
                {"best": best_clusters, "worst": worst_clusters},
            ),
            use_container_width=True,
        )
        legend_cols = st.columns(2)
        with legend_cols[0]:
            st.success("Green = lower normalized-severity clusters")
        with legend_cols[1]:
            st.error("Red = strongest normalized-severity hotspots")

    with list_col:
        if best_clusters:
            st.subheader("Lower-Concern Clusters")
            for idx, row in enumerate(best_clusters):
                area_label = f"{row['primary_borough']} {row['primary_zip']}"
                st.markdown(
                    f"""
                    **{idx + 1}. {area_label}**  
                    Normalized severity: `{row['normalized_severity']:.4f}`  
                    Severity: `{row['severity_score']:.2f}`  
                    Matched complaints: `{int(row['complaint_count'])}`  
                    Center: `{row['center_lat']:.4f}, {row['center_lon']:.4f}`
                    """
                )
        else:
            st.info("No lower-severity clusters were available.")

        st.divider()

        if worst_clusters:
            st.subheader("Strongest Concern Hotspots")
            for idx, row in enumerate(worst_clusters):
                area_label = f"{row['primary_borough']} {row['primary_zip']}"
                st.markdown(
                    f"""
                    **{idx + 1}. {area_label}**  
                    Normalized severity: `{row['normalized_severity']:.4f}`  
                    Severity: `{row['severity_score']:.2f}`  
                    Matched complaints: `{int(row['complaint_count'])}`  
                    Center: `{row['center_lat']:.4f}, {row['center_lon']:.4f}`
                    """
                )
        else:
            st.info("No high-severity clusters were available for this selection.")


def render_home_page(searcher, clusterer, category_lookup, all_category_labels):
    st.title("Vibe Check")
    st.markdown(
        "Build a concern profile for NYC apartment hunting. Start with semantic search,"
        " refine the complaint categories you care about, and then generate a map plus ranked lists."
    )

    control_col, selection_col = st.columns([1.4, 1])

    common_categories_df = (
        clusterer.df.groupby(["Problem", "Problem Detail"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(12)
    )
    common_categories = [
        {
            "label": category_label(row["Problem"], row["Problem Detail"]),
            "problem": row["Problem"],
            "detail": row["Problem Detail"],
        }
        for _, row in common_categories_df.iterrows()
    ]

    with control_col:
        st.subheader("1. Search by description")
        semantic_query = st.text_input(
            "Describe what you want to avoid",
            placeholder="Examples: loud music at night, rats, broken heating, potholes",
        )
        if st.button("Find matching categories", use_container_width=True):
            if semantic_query.strip():
                st.session_state.search_results = searcher.search(semantic_query.strip(), top_k=50)
            else:
                st.session_state.search_results = []

        if st.session_state.search_results:
            st.caption("Semantic matches from your text")
            suggestion_cols = st.columns(2)
            for idx, result in enumerate(st.session_state.search_results):
                label = category_label(result["problem"], result["detail"])
                with suggestion_cols[idx % 2]:
                    st.write(f"**{label}**")
                    st.caption(f"Similarity: {result['similarity']:.2f}")
                    if st.button(f"Add suggestion {idx + 1}", key=f"suggestion-{idx}", use_container_width=True):
                        add_selection(label, result, st.session_state.similarity_store)

        st.subheader("2. Or pick a common category")
        common_cols = st.columns(3)
        for idx, item in enumerate(common_categories):
            payload = {"problem": item["problem"], "detail": item["detail"], "similarity": 1.0}
            with common_cols[idx % 3]:
                if st.button(item["label"], key=f"common-{idx}", use_container_width=True):
                    add_selection(item["label"], payload, st.session_state.similarity_store)

    with selection_col:
        st.subheader("3. Build your concern list")
        selected_labels = st.multiselect(
            "Choose up to 50 concerns",
            options=all_category_labels,
            default=st.session_state.selected_labels,
            max_selections=50,
            placeholder="Pick categories for this run",
        )
        st.session_state.selected_labels = selected_labels

        render_selected_concerns()

        if st.button(
            "Run Vibe Check",
            disabled=not selected_labels,
            type="primary",
            use_container_width=True,
        ):
            matched_categories = build_match_payload(
                selected_labels,
                category_lookup,
                st.session_state.similarity_store,
            )
            with st.spinner("Ranking areas..."):
                st.session_state.analysis_results = run_analysis(clusterer, matched_categories)
            st.session_state.current_page = "Results"
            st.rerun()

        if st.session_state.analysis_results:
            st.divider()
            st.caption("You already have a saved analysis in this session.")
            if st.button("Open Results", use_container_width=True):
                st.session_state.current_page = "Results"
                st.rerun()


searcher = get_searcher()
clusterer = get_clusterer()

if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = []
if "similarity_store" not in st.session_state:
    st.session_state.similarity_store = {}
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

category_lookup = {
    category_label(row["Problem"], row["Problem Detail"]): (row["Problem"], row["Problem Detail"])
    for row in searcher.categories
}
all_category_labels = sorted(category_lookup.keys())

with st.sidebar:
    st.header("Navigate")
    page_options = ["Home"]
    if st.session_state.analysis_results:
        page_options.append("Results")

    current_page = st.radio(
        "Page",
        options=page_options,
        index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0,
    )
    st.session_state.current_page = current_page

    if st.session_state.analysis_results:
        if st.button("Clear Current Analysis", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.current_page = "Home"
            st.rerun()

if st.session_state.current_page == "Home":
    render_home_page(searcher, clusterer, category_lookup, all_category_labels)
elif st.session_state.current_page == "Results" and st.session_state.analysis_results:
    render_results_page()
else:
    st.session_state.current_page = "Home"
    render_home_page(searcher, clusterer, category_lookup, all_category_labels)

st.divider()
st.caption("Semantic search uses sentence-transformer matching over 311 problem categories. KMeans hotspots are ranked by normalized severity: total matched concern weight divided by matched complaint count.")

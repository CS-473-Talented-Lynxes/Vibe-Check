import streamlit as st
import pandas as pd
import pydeck as pdk

from src.embeddings import ComplaintSearcher
from src.clustering import LocationClusterer


st.set_page_config(page_title="Vibe Check", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: #16a34a !important;
        box-shadow: 0 0 0 1px #16a34a !important;
    }

    div[data-testid="stTextInput"] input:focus {
        outline: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


CLUSTERING_METHOD_LABELS = {
    "seed": "Seed-based nearest-center",
    "kmeans": "K-means",
}
MAP_COLUMN_RADIUS = 300
MAP_ELEVATION_SCALE = 5000


@st.cache_resource(show_spinner=False)
def get_searcher():
    return ComplaintSearcher()


@st.cache_resource(show_spinner=False)
def get_clusterer():
    return LocationClusterer()


def category_label(problem, detail):
    return f"{problem} - {detail}"


def clustering_method_label(method):
    return CLUSTERING_METHOD_LABELS.get(method, method)


def reset_current_analysis():
    st.session_state.selected_labels = []
    st.session_state.similarity_store = {}
    st.session_state.search_results = []
    st.session_state.analysis_results = None
    st.session_state.current_page = "Home"
    st.session_state.semantic_query = ""
    st.session_state.selected_clustering_method = "seed"


def add_selection(label, payload, similarity_store):
    current = list(st.session_state.selected_labels)
    if label in current:
        similarity_store[label] = max(similarity_store.get(label, 0.0), payload["similarity"])
        st.session_state.selected_labels = current
        return
    if len(current) >= 50:
        st.warning("You can select up to 50 concerns at a time.")
        return
    current.append(label)
    st.session_state.selected_labels = current
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
    base_view = pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=10, pitch=50)
    layers = []

    best_clusters = recommendations.get("best", [])
    worst_clusters = recommendations.get("worst", [])

    def with_severity_height(clusters):
        elevated_clusters = []
        for cluster in clusters:
            normalized_severity = float(cluster.get("normalized_severity", 0.0))
            elevated_clusters.append({
                **cluster,
                "display_elevation": normalized_severity,
            })
        return elevated_clusters

    if best_clusters:
        best_clusters = with_severity_height(best_clusters)
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=best_clusters,
                get_position="[center_lon, center_lat]",
                radius=MAP_COLUMN_RADIUS,
                get_elevation="display_elevation",
                get_fill_color="[44, 160, 44, 180]",
                get_line_color="[0, 80, 0, 220]",
                elevation_scale=MAP_ELEVATION_SCALE,
                extruded=True,
                line_width_min_pixels=2,
                pickable=True,
            )
        )

    if worst_clusters:
        worst_clusters = with_severity_height(worst_clusters)
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=worst_clusters,
                get_position="[center_lon, center_lat]",
                radius=MAP_COLUMN_RADIUS,
                get_elevation="display_elevation",
                get_fill_color="[220, 38, 38, 180]",
                get_line_color="[127, 29, 29, 220]",
                elevation_scale=MAP_ELEVATION_SCALE,
                extruded=True,
                line_width_min_pixels=2,
                pickable=True,
            )
        )

    tooltip = {
        "html": (
            "<b>{primary_borough}</b> {primary_zip}<br/>"
            "Normalized severity: {normalized_severity}<br/>"
            "Concern share: {concern_share}<br/>"
            "Reliability factor: {reliability_factor}<br/>"
            "Concern score: {severity_score}<br/>"
            "Baseline score: {baseline_score}<br/>"
            "Matched complaints: {complaint_count}<br/>"
            "All complaints in cluster: {baseline_complaint_count}"
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


def run_analysis(clusterer, matched_categories, clustering_method):
    cluster_results = clusterer.cluster_extremes(
        matched_categories,
        k_clusters=300,
        top_n=50,
        method=clustering_method,
    )

    return {
        "matched_categories": matched_categories,
        "best_clusters": cluster_results["best"],
        "worst_clusters": cluster_results["worst"],
        "clustering_method": clustering_method,
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
    clustering_method = results.get("clustering_method", "seed")

    if not best_clusters and not worst_clusters:
        st.warning("No ranked areas were available for this selection.")
        return

    method_summary = (
        "the seed-based nearest-center pipeline"
        if clustering_method == "seed"
        else "the native K-means pipeline"
    )
    st.caption(
        f"Clustering method: {clustering_method_label(clustering_method)}. "
        f"These map markers and ranked lists come from {method_summary}. "
        "Clusters are ranked by concern share: matched concern weight divided by total weighted 311 complaint volume in the same cluster."
    )

    lower_col, higher_col = st.columns(2)

    with lower_col:
        st.subheader("Lower-Concern")
        with st.container(height=560):
            if best_clusters:
                for idx, row in enumerate(best_clusters):
                    area_label = f"{row['primary_borough']} {row['primary_zip']}"
                    st.markdown(
                        f"""
                        **{idx + 1}. {area_label}**  
                        Normalized severity: `{row['normalized_severity']:.4f}`  
                        Concern share: `{row['concern_share']:.4f}`  
                        Reliability factor: `{row['reliability_factor']:.2f}`  
                        Concern score: `{row['severity_score']:.2f}`  
                        Baseline score: `{row['baseline_score']:.2f}`  
                        Matched complaints: `{int(row['complaint_count'])}`  
                        All complaints: `{int(row['baseline_complaint_count'])}`  
                        Center: `{row['center_lat']:.4f}, {row['center_lon']:.4f}`
                        """
                    )
                    st.divider()
            else:
                st.info("No lower-severity clusters were available.")

        st.caption("Green markers show lower concern-share clusters.")
        st.pydeck_chart(
            build_map(
                pd.DataFrame(),
                {"best": best_clusters, "worst": []},
            ),
            use_container_width=True,
        )

    with higher_col:
        st.subheader("Hotspots")
        with st.container(height=560):
            if worst_clusters:
                for idx, row in enumerate(worst_clusters):
                    area_label = f"{row['primary_borough']} {row['primary_zip']}"
                    st.markdown(
                        f"""
                        **{idx + 1}. {area_label}**  
                        Normalized severity: `{row['normalized_severity']:.4f}`  
                        Concern share: `{row['concern_share']:.4f}`  
                        Reliability factor: `{row['reliability_factor']:.2f}`  
                        Concern score: `{row['severity_score']:.2f}`  
                        Baseline score: `{row['baseline_score']:.2f}`  
                        Matched complaints: `{int(row['complaint_count'])}`  
                        All complaints: `{int(row['baseline_complaint_count'])}`  
                        Center: `{row['center_lat']:.4f}, {row['center_lon']:.4f}`
                        """
                    )
                    st.divider()
            else:
                st.info("No high-severity clusters were available for this selection.")

        st.caption("Red markers show strongest concern-share hotspots.")
        st.pydeck_chart(
            build_map(
                pd.DataFrame(),
                {"best": [], "worst": worst_clusters},
            ),
            use_container_width=True,
        )


def render_home_page(searcher, clusterer, category_lookup):
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
        .head(30)
    )
    common_categories = [
        {
            "label": category_label(row["Problem"], row["Problem Detail"]),
            "problem": row["Problem"],
            "detail": row["Problem Detail"],
            "similarity": 1.0,
        }
        for _, row in common_categories_df.iterrows()
    ]

    with control_col:
        st.subheader("1. Search by description")
        with st.form("semantic-search-form"):
            semantic_query = st.text_input(
                "Describe what you want to avoid",
                placeholder="Examples: loud music at night, rats, broken heating, potholes",
                key="semantic_query",
            )
            submitted_search = st.form_submit_button(
                "Find matching categories",
                use_container_width=True,
            )

        if submitted_search:
            if semantic_query.strip():
                st.session_state.search_results = searcher.search(semantic_query.strip(), top_k=50)
            else:
                st.session_state.search_results = []

        if st.session_state.search_results:
            with st.expander(
                f"Browse semantic matches ({len(st.session_state.search_results)})",
                expanded=True,
            ):
                with st.container(height=360):
                    for idx, result in enumerate(st.session_state.search_results):
                        label = category_label(result["problem"], result["detail"])
                        match_cols = st.columns([5, 1.2, 0.8])
                        with match_cols[0]:
                            st.write(f"**{label}**")
                        with match_cols[1]:
                            st.caption(f"Similarity: {result['similarity']:.2f}")
                        with match_cols[2]:
                            if st.button(
                                "Add",
                                key=f"suggestion-{idx}",
                            ):
                                add_selection(label, result, st.session_state.similarity_store)

        st.subheader("2. Or pick a common category")
        with st.container(height=360):
            common_cols = st.columns(2)
            for idx, item in enumerate(common_categories):
                payload = {
                    "problem": item["problem"],
                    "detail": item["detail"],
                    "similarity": item["similarity"],
                }
                with common_cols[idx % 2]:
                    if st.button(item["label"], key=f"common-{idx}", use_container_width=True):
                        add_selection(item["label"], payload, st.session_state.similarity_store)

    with selection_col:
        st.subheader("3. Build your concern list")
        render_selected_concerns()

        st.subheader("4. Choose clustering method")
        st.selectbox(
            "Clustering algorithm",
            options=list(CLUSTERING_METHOD_LABELS.keys()),
            format_func=clustering_method_label,
            key="selected_clustering_method",
        )
        st.caption(
            "Seed-based uses sampled complaint locations as fixed centers. K-means updates the centers iteratively until clusters stabilize."
        )

        if st.button(
            "Run Vibe Check",
            disabled=not st.session_state.selected_labels,
            type="primary",
            use_container_width=True,
        ):
            matched_categories = build_match_payload(
                st.session_state.selected_labels,
                category_lookup,
                st.session_state.similarity_store,
            )
            with st.spinner("Ranking areas..."):
                st.session_state.analysis_results = run_analysis(
                    clusterer,
                    matched_categories,
                    st.session_state.selected_clustering_method,
                )
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
if "semantic_query" not in st.session_state:
    st.session_state.semantic_query = ""
if "selected_clustering_method" not in st.session_state:
    st.session_state.selected_clustering_method = "seed"

category_lookup = {
    category_label(row["Problem"], row["Problem Detail"]): (row["Problem"], row["Problem Detail"])
    for row in searcher.categories
}

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
            reset_current_analysis()
            st.rerun()

if st.session_state.current_page == "Home":
    render_home_page(searcher, clusterer, category_lookup)
elif st.session_state.current_page == "Results" and st.session_state.analysis_results:
    render_results_page()
else:
    st.session_state.current_page = "Home"
    render_home_page(searcher, clusterer, category_lookup)

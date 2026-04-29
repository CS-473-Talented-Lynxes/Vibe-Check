import streamlit as st
import streamlit.components.v1 as components
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

BOROUGH_ZILLOW_SLUGS = {
    "manhattan": "manhattan",
    "bronx": "bronx",
    "brooklyn": "brooklyn",
    "queens": "queens",
    "staten island": "staten-island",
}


@st.cache_resource(show_spinner=False)
def get_searcher():
    return ComplaintSearcher()


@st.cache_resource(show_spinner=False)
def get_clusterer():
    return LocationClusterer()


def category_label(problem, detail):
    return f"{problem} - {detail}"


def reset_current_analysis():
    st.session_state.selected_labels = []
    st.session_state.similarity_store = {}
    st.session_state.search_results = []
    st.session_state.analysis_results = None
    st.session_state.current_page = "Home"
    st.session_state.semantic_query = ""
    st.session_state.highlighted_cluster_id = None
    st.session_state.scroll_to_cluster_id = None


def set_highlighted_cluster(cluster_id, scroll=False):
    st.session_state.highlighted_cluster_id = int(cluster_id)
    st.session_state.scroll_to_cluster_id = int(cluster_id) if scroll else None


def zillow_area_url(borough, zip_code):
    zip_text = "".join(ch for ch in str(zip_code) if ch.isdigit())[:5]
    if not zip_text:
        return "https://www.zillow.com/homes/"

    borough_slug = BOROUGH_ZILLOW_SLUGS.get(str(borough).strip().lower())
    if borough_slug:
        return f"https://www.zillow.com/{borough_slug}-new-york-ny-{zip_text}/"

    return f"https://www.zillow.com/homes/{zip_text}_rb/"


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


def render_cluster_details(row):
    st.write(f"Normalized severity: `{row['normalized_severity']:.4f}`")
    st.write(f"Concern share: `{row['concern_share']:.4f}`")
    st.write(f"Reliability factor: `{row['reliability_factor']:.2f}`")
    st.write(f"Concern score: `{row['severity_score']:.2f}`")
    st.write(f"Baseline score: `{row['baseline_score']:.2f}`")
    st.write(f"Matched complaints: `{int(row['complaint_count'])}`")
    st.write(f"All complaints: `{int(row['baseline_complaint_count'])}`")
    st.write(f"Center: `{row['center_lat']:.4f}, {row['center_lon']:.4f}`")


def extract_selected_cluster_id(selection_state):
    if selection_state is None:
        return None

    if hasattr(selection_state, "to_dict"):
        selection_state = selection_state.to_dict()

    if not selection_state:
        return None

    selection = selection_state.get("selection", selection_state)
    objects = selection.get("objects", [])

    if isinstance(objects, dict):
        candidates = []
        for layer_objects in objects.values():
            if isinstance(layer_objects, list):
                candidates.extend(layer_objects)
            elif layer_objects:
                candidates.append(layer_objects)
    elif isinstance(objects, list):
        candidates = objects
    else:
        candidates = []

    for candidate in candidates:
        if isinstance(candidate, dict) and "cluster_id" in candidate:
            return int(candidate["cluster_id"])

    return None


def scroll_selected_cluster_into_view(cluster_id):
    if cluster_id is None:
        return

    components.html(
        f"""
        <script>
        const target = window.parent.document.getElementById("cluster-{int(cluster_id)}");
        if (target) {{
            target.scrollIntoView({{ behavior: "smooth", block: "center" }});
        }}
        </script>
        """,
        height=0,
    )


def render_cluster_map(deck, highlighted_cluster_id):
    map_key = f"cluster-map-{highlighted_cluster_id if highlighted_cluster_id is not None else 'none'}"
    try:
        return st.pydeck_chart(
            deck,
            use_container_width=True,
            key=map_key,
            on_select="rerun",
            selection_mode="single-object",
        )
    except TypeError:
        st.pydeck_chart(deck, use_container_width=True)
        return None


def build_map(points_df, recommendations, highlighted_cluster_id=None):
    base_view = pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=10, pitch=35)
    layers = []

    best_clusters = recommendations.get("best", [])
    worst_clusters = recommendations.get("worst", [])

    if best_clusters:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="lower-concern-clusters",
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
                id="hotspot-clusters",
                data=worst_clusters,
                get_position="[center_lon, center_lat]",
                get_radius=900,
                get_fill_color="[220, 38, 38, 180]",
                get_line_color="[127, 29, 29, 220]",
                line_width_min_pixels=2,
                pickable=True,
            )
        )

    highlighted_cluster = next(
        (
            cluster
            for cluster in best_clusters + worst_clusters
            if int(cluster["cluster_id"]) == highlighted_cluster_id
        ),
        None,
    )
    if highlighted_cluster:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="highlighted-cluster",
                data=[highlighted_cluster],
                get_position="[center_lon, center_lat]",
                get_radius=1400,
                stroked=True,
                filled=False,
                get_line_color="[37, 99, 235, 255]",
                line_width_min_pixels=5,
                pickable=False,
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

    highlighted_cluster_id = st.session_state.highlighted_cluster_id
    map_col, list_col = st.columns([1.15, 1])

    with map_col:
        st.subheader("City Map")
        map_selection = render_cluster_map(
            build_map(
                pd.DataFrame(),
                {"best": best_clusters, "worst": worst_clusters},
                highlighted_cluster_id=highlighted_cluster_id,
            ),
            highlighted_cluster_id,
        )
        selected_cluster_id = extract_selected_cluster_id(map_selection)
        if (
            selected_cluster_id is not None
            and selected_cluster_id != st.session_state.highlighted_cluster_id
        ):
            set_highlighted_cluster(selected_cluster_id, scroll=True)
            st.rerun()

        legend_cols = st.columns(2)
        with legend_cols[0]:
            st.success("Green = lower concern-share clusters")
        with legend_cols[1]:
            st.error("Red = strongest concern-share hotspots")

    with list_col:
        lower_col, higher_col = st.columns(2)

        with lower_col:
            st.subheader("Lower-Concern")
            with st.container(height=560):
                if best_clusters:
                    for idx, row in enumerate(best_clusters):
                        cluster_id = int(row["cluster_id"])
                        st.markdown(f'<div id="cluster-{cluster_id}"></div>', unsafe_allow_html=True)
                        area_label = f"{row['primary_borough']} {row['primary_zip']}"
                        area_url = zillow_area_url(row["primary_borough"], row["primary_zip"])
                        st.markdown(f"**{idx + 1}. [{area_label}]({area_url})**")
                        if cluster_id == st.session_state.highlighted_cluster_id:
                            st.info("Highlighted on map")
                        render_cluster_details(row)
                        st.button(
                            "Highlight on map",
                            key=f"highlight-best-{cluster_id}",
                            on_click=set_highlighted_cluster,
                            args=(cluster_id,),
                            use_container_width=True,
                        )
                        st.divider()
                else:
                    st.info("No lower-severity clusters were available.")

        with higher_col:
            st.subheader("Hotspots")
            with st.container(height=560):
                if worst_clusters:
                    for idx, row in enumerate(worst_clusters):
                        cluster_id = int(row["cluster_id"])
                        st.markdown(f'<div id="cluster-{cluster_id}"></div>', unsafe_allow_html=True)
                        area_label = f"{row['primary_borough']} {row['primary_zip']}"
                        st.markdown(f"**{idx + 1}. {area_label}**")
                        if cluster_id == st.session_state.highlighted_cluster_id:
                            st.info("Highlighted on map")
                        render_cluster_details(row)
                        st.button(
                            "Highlight on map",
                            key=f"highlight-worst-{cluster_id}",
                            on_click=set_highlighted_cluster,
                            args=(cluster_id,),
                            use_container_width=True,
                        )
                        st.divider()
                else:
                    st.info("No high-severity clusters were available for this selection.")

    if st.session_state.scroll_to_cluster_id is not None:
        scroll_selected_cluster_into_view(st.session_state.scroll_to_cluster_id)
        st.session_state.scroll_to_cluster_id = None


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
if "semantic_query" not in st.session_state:
    st.session_state.semantic_query = ""
if "highlighted_cluster_id" not in st.session_state:
    st.session_state.highlighted_cluster_id = None
if "scroll_to_cluster_id" not in st.session_state:
    st.session_state.scroll_to_cluster_id = None

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

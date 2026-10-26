import html

import streamlit as st


def apply_global_theme():
    st.markdown(
        """
        <style>
        :root {
          --meridian-bg:#f6f5f1; --meridian-surface:rgba(255,255,255,.82);
          --meridian-ink:#20231f; --meridian-muted:#74776f; --meridian-line:#e5e4dd;
          --meridian-forest:#31584d; --meridian-sage:#dfe9e2; --meridian-warm:#b8654c;
        }
        html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        .stApp { background:var(--meridian-bg); color:var(--meridian-ink); }
        [data-testid="stHeader"] { background:rgba(246,245,241,.9); }
        [data-testid="stAppViewContainer"] > .main { background:var(--meridian-bg); }
        .block-container { max-width:1500px; padding-top:2rem; padding-bottom:3rem; }
        [data-testid="stSidebar"] { background:#f1f0eb; border-right:1px solid var(--meridian-line); }
        [data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--meridian-muted); }
        [data-testid="stMetric"] { border:1px solid var(--meridian-line); border-radius:14px; padding:14px 16px; background:linear-gradient(145deg,rgba(255,255,255,.88),rgba(255,255,255,.65)); box-shadow:0 5px 18px rgba(32,35,31,.035); backdrop-filter:blur(8px); }
        [data-testid="stMetricLabel"] { color:var(--meridian-muted); font-size:.78rem; }
        [data-testid="stMetricValue"] { color:var(--meridian-ink); font-weight:650; letter-spacing:-.035em; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--meridian-line)!important; border-radius:16px!important; background:linear-gradient(145deg,rgba(255,255,255,.86),rgba(255,255,255,.7)); box-shadow:0 7px 22px rgba(32,35,31,.035); backdrop-filter:blur(9px); }
        .stButton > button, .stDownloadButton > button { border-radius:9px; border:1px solid var(--meridian-line); background:#fff; color:var(--meridian-ink); font-weight:560; transition:background .16s ease,border-color .16s ease,transform .16s ease; }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color:#b5c8bc; background:#f3f7f3; transform:translateY(-1px); }
        .stButton > button[kind="primary"] { background:var(--meridian-forest); color:#fff; border-color:var(--meridian-forest); }
        .stButton > button[kind="primary"]:hover { background:#28493f; border-color:#28493f; }
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { border-radius:9px; }
        [data-testid="stDataFrame"] { border:1px solid var(--meridian-line); border-radius:12px; overflow:hidden; }
        .meridian-eyebrow { color:var(--meridian-forest); font-size:.69rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin:0 0 .7rem; }
        .meridian-title { color:var(--meridian-ink); font-size:clamp(2rem,3.4vw,3rem); line-height:1.08; font-weight:680; letter-spacing:-.055em; margin:0 0 .65rem; }
        .meridian-description { color:var(--meridian-muted); font-size:1rem; max-width:760px; margin:0 0 1.4rem; }
        .meridian-brand { color:var(--meridian-ink); font-size:1rem; font-weight:700; letter-spacing:-.035em; }
        .meridian-brand span { display:block; color:var(--meridian-muted); font-size:.7rem; font-weight:500; letter-spacing:.04em; margin-top:.1rem; }
        .meridian-caption { color:var(--meridian-muted); font-size:.78rem; }
        .meridian-status { display:inline-flex; align-items:center; gap:7px; color:var(--meridian-muted); font-size:.76rem; }
        .meridian-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#809b87; }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition:none!important; scroll-behavior:auto!important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, description, eyebrow="PRODUCT INTELLIGENCE"):
    st.markdown(
        f'<p class="meridian-eyebrow">{html.escape(eyebrow)}</p>'
        f'<h1 class="meridian-title">{html.escape(title)}</h1>'
        f'<p class="meridian-description">{html.escape(description)}</p>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, delta=None, help_text=None):
    st.metric(label=label, value=value, delta=delta, help=help_text)


def section_title(title, description=None):
    st.markdown(f"#### {title}")
    if description:
        st.caption(description)


def chart_panel(title, description, status=None):
    container = st.container(border=True)
    with container:
        heading, context = st.columns([4, 1])
        with heading:
            st.markdown(f"**{title}**")
            if description:
                st.caption(description)
        if status:
            with context:
                st.caption(status)
    return container


def empty_state(title, description, action_label=None, key=None):
    st.markdown(f"**{title}**")
    st.caption(description)
    if action_label:
        return st.button(action_label, key=key)
    return False


def render_sidebar(identity, rows, categories, model_ready):
    st.sidebar.markdown(
        '<div class="meridian-brand">MERIDIAN<span>PRODUCT INTELLIGENCE</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{rows:,} product offers")
    st.sidebar.caption(f"{categories} product categories")
    status = "Classifier ready" if model_ready else "Classifier not trained"
    st.sidebar.markdown(
        f'<span class="meridian-status"><i class="meridian-dot"></i>{status}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"DATA SOURCE  ·  {identity}")

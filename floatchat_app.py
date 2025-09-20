# app.py — FloatChat (Explore + Chat) with improved prompting
import os, json, re, time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import plotly.express as px
import pydeck as pdk
from argopy import DataFetcher as ArgoDataFetcher

# ================= Azure OpenAI setup =================
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")      # e.g., https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # your deployment name
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

client = None
client_mode = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        client_mode = "responses" if "preview" in AZURE_OPENAI_API_VERSION else "chat"
    except Exception:
        client = None
        client_mode = None

# ================= Streamlit page setup =================
st.set_page_config(page_title="FloatChat — Ask the Ocean", layout="wide")
st.title("FloatChat — Ask the Ocean")
st.caption("SIH25040 · Data source: IFREMER ERDDAP (ArgoFloats). We prefer adjusted data when QC is good.")

# ================= Prompt pack (improved) =================
# 1) Parameter extraction (few-shot + schema)
PARAMS_SYSTEM = (
    "You convert short ocean questions into JSON parameters for an ARGO query.\n"
    "Always return ONLY valid JSON (no extra text). If something is missing, use defaults.\n"
    "Defaults: region_name='bay_of_bengal', bbox=[78,100,5,22], last 60 days, depth=0–500 m, variable='temperature'.\n"
    "Allowed region_name: 'bay_of_bengal', 'arabian_sea', 'custom'. "
    "If 'custom', user may give bbox as lon/lat ranges.\n"
    "Map common words:\n"
    "- 'Bengal', 'Bay' => bay_of_bengal\n"
    "- 'Arabian' => arabian_sea\n"
    "- 'salt', 'salinity', 'psal' => variable=salinity\n"
    "- 'temp', 'temperature', '°C' => variable=temperature\n"
    "Dates like 'June to August 2024', 'last 45 days' or 'past month' must be converted to YYYY-MM-DD.\n"
)

# Few-shot examples make extraction more reliable
PARAMS_FEWSHOTS = [
    {
        "role": "user",
        "content": "Show Arabian Sea last 45 days salinity up to 300 m"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "region_name": "arabian_sea",
            "bbox": [55, 78, 5, 23],
            "start_date": (datetime.utcnow().date() - timedelta(days=45)).isoformat(),
            "end_date": datetime.utcnow().date().isoformat(),
            "depth_min": 0, "depth_max": 300,
            "variable": "salinity"
        })
    },
    {
        "role": "user",
        "content": "Bengal me monsoon 2024 May se Aug, depth 0-200m, temp"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "region_name": "bay_of_bengal",
            "bbox": [78, 100, 5, 22],
            "start_date": "2024-05-01",
            "end_date": "2024-08-31",
            "depth_min": 0, "depth_max": 200,
            "variable": "temperature"
        })
    },
    {
        "role": "user",
        "content": "custom 60–80E, 10–20N, past 30 days, temperature"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "region_name": "custom",
            "bbox": [60, 80, 10, 20],
            "start_date": (datetime.utcnow().date() - timedelta(days=30)).isoformat(),
            "end_date": datetime.utcnow().date().isoformat(),
            "depth_min": 0, "depth_max": 500,
            "variable": "temperature"
        })
    }
]

PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "region_name": {"type":"string","enum":["bay_of_bengal","arabian_sea","custom"]},
        "bbox": {"type":"array","items":{"type":"number"}, "minItems":4, "maxItems":4},
        "start_date": {"type":"string","description":"YYYY-MM-DD"},
        "end_date": {"type":"string","description":"YYYY-MM-DD"},
        "depth_min": {"type":"number"},
        "depth_max": {"type":"number"},
        "variable": {"type":"string","enum":["temperature","salinity"]}
    },
    "required": ["region_name","start_date","end_date","depth_min","depth_max","variable"]
}

# 2) Style prompt to rewrite summaries if you enable the toggle
STYLE_SYSTEM = (
    "Rewrite the given technical summary so a non-expert can understand it.\n"
    "Rules:\n"
    "- Use short sentences and simple words.\n"
    "- Keep it factual; do not invent numbers.\n"
    "- If temperature: add a quick feel-word (warm, cool, mild) based on the given mean.\n"
    "- Keep it under 120 words.\n"
    "- Maintain any dates/units exactly."
)

# ================= Helper functions =================
def normalize_variable(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ["temperature", "temp", "t", "temp_c", "°c", "degc"]:
        return "temperature"
    if v in ["salinity", "salt", "psal", "s"]:
        return "salinity"
    return "temperature"

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_region(bbox, start_date, end_date, zmin, zmax, params=None) -> xr.Dataset:
    lonW, lonE, latS, latN = map(float, bbox)
    t0, t1 = str(start_date), str(end_date)
    fetcher = ArgoDataFetcher(src="erddap", params=params) if params else ArgoDataFetcher(src="erddap")
    ds = fetcher.region([lonW, lonE, latS, latN, float(zmin), float(zmax), t0, t1]).to_xarray()
    return ds

def prefer_adjusted(ds: xr.Dataset, var_choice: str) -> pd.DataFrame:
    if var_choice == "temperature":
        raw, adj, qc_raw, qc_adj = "TEMP", "TEMP_ADJUSTED", "TEMP_QC", "TEMP_ADJUSTED_QC"
    else:
        raw, adj, qc_raw, qc_adj = "PSAL", "PSAL_ADJUSTED", "PSAL_QC", "PSAL_ADJUSTED_QC"

    needed = ["PRES", "LATITUDE", "LONGITUDE", "TIME"]
    for k in needed:
        if k not in ds:
            return pd.DataFrame(columns=["time","lat","lon","depth_m","value","mode_used","qc_used"])

    if raw not in ds:
        return pd.DataFrame(columns=["time","lat","lon","depth_m","value","mode_used","qc_used"])

    df_raw = ds[[raw, "PRES", "LATITUDE", "LONGITUDE", "TIME"]].to_dataframe().reset_index(drop=True)

    if qc_raw in ds:
        df_qcr = ds[[qc_raw]].to_dataframe().reset_index(drop=True)
        df_raw[qc_raw] = df_qcr[qc_raw].astype(str) if not df_qcr.empty else "9"
    else:
        df_raw[qc_raw] = "9"

    if adj in ds:
        adj_cols = [c for c in [adj, qc_adj, "PRES", "LATITUDE", "LONGITUDE", "TIME"] if c in ds]
        df_adj = ds[adj_cols].to_dataframe().reset_index(drop=True)
    else:
        df_adj = pd.DataFrame(columns=[adj, qc_adj, "PRES", "LATITUDE", "LONGITUDE", "TIME"])

    join_keys = [k for k in ["PRES", "TIME", "LATITUDE", "LONGITUDE"] if k in df_raw.columns and k in df_adj.columns]
    if len(join_keys) == 0:
        df = df_raw.copy()
        df[adj] = np.nan
        df[qc_adj] = "9"
    else:
        df = df_raw.merge(
            df_adj[[c for c in [adj, qc_adj] + join_keys if c in df_adj.columns]],
            how="left", on=join_keys
        )

    def choose(row):
        v_adj = row.get(adj)
        q_adj = str(row.get(qc_adj, "9"))
        v_raw = row.get(raw)
        q_raw = str(row.get(qc_raw, "9"))
        if pd.notna(v_adj) and q_adj in {"1","2"}:
            return v_adj, "adjusted", q_adj
        if pd.notna(v_raw) and q_raw in {"1","2"}:
            return v_raw, "raw", q_raw
        return np.nan, "bad", q_adj if pd.notna(v_adj) else q_raw

    vals, modes, qcs = [], [], []
    for _, r in df.iterrows():
        v, m, q = choose(r)
        vals.append(v); modes.append(m); qcs.append(q)

    out = pd.DataFrame({
        "time": pd.to_datetime(df["TIME"], errors="coerce"),
        "lat": df.get("LATITUDE"),
        "lon": df.get("LONGITUDE"),
        "depth_m": pd.to_numeric(df.get("PRES"), errors="coerce"),
        "value": vals,
        "mode_used": modes,
        "qc_used": qcs
    }).dropna(subset=["time","lat","lon","depth_m","value"])
    return out

def make_ts_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    for k in ["TEMP","PSAL","PRES"]:
        if k not in ds:
            return pd.DataFrame(columns=["TEMP","PSAL","depth_m"])
    df = ds[["TEMP","PSAL","PRES"]].to_dataframe().dropna().reset_index(drop=True)
    df = df.rename(columns={"PRES":"depth_m"})
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df = df.dropna(subset=["TEMP","PSAL","depth_m"])
    return df

# ---------- Humanizer helpers ----------
def _fmt_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

def _fmt_float(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _fmt_date(d):
    try:
        return pd.to_datetime(d).date().isoformat()
    except Exception:
        return str(d)

def _temp_words(mean_c):
    if mean_c is None: return ""
    m = float(mean_c)
    if m >= 28:  return "very warm"
    if m >= 24:  return "warm"
    if m >= 18:  return "mild"
    if m >= 10:  return "cool"
    return "cold"

def humanize_summary(region_name, bbox, start_date, end_date,
                     depth_min, depth_max, variable,
                     n_points, vmin, vmean, vmax,
                     lang="english", emojis=True):
    box_str = f"[{bbox[0]}, {bbox[1]}] lon × [{bbox[2]}, {bbox[3]}] lat"
    sd, ed = _fmt_date(start_date), _fmt_date(end_date)
    dp = f"{int(depth_min)}–{int(depth_max)} m"
    pts = _fmt_int(n_points)
    vmin1, vmean1, vmax1 = _fmt_float(vmin), _fmt_float(vmean), _fmt_float(vmax)
    var_label = "Temperature (°C)" if variable == "temperature" else "Salinity (PSU)"
    temp_feel = _temp_words(vmean) if variable == "temperature" else ""
    star = "⭐ " if emojis else ""
    pin  = "📍 " if emojis else ""
    cal  = "🗓️ " if emojis else ""
    dep  = "🌊 " if emojis else ""
    csv  = "⬇️ " if emojis else ""

    if lang.lower() == "hinglish":
        lines = [
            f"{star}**Simple Summary (Hinglish):**",
            f"{pin} *Jagah*: {region_name.replace('_',' ').title()} ({box_str})",
            f"{cal} *Dates*: {sd} se {ed} tak",
            f"{dep} *Gehrai*: {dp}",
            f"• *Data points*: {pts}",
            "",
            ("*Matlab*: Paani average mein **"
             f"{_fmt_float(vmean,1)}°C** tha ({temp_feel}), "
             f"upar zyada garam (~{vmax1}°C) aur neeche thanda (~{vmin1}°C).")
             if variable=="temperature" else
             ("*Matlab*: Namak (salinity) average **"
              f"{vmean1} PSU** tha, range {vmin1}–{vmax1} PSU."),
            "Heatmap time aur depth ke hisaab se badlav dikhata hai,",
            "T–S graph temperature–salinity ka relation dikhata hai.",
            "",
            f"{csv} Neeche diye gaye CSV se exactly yahi data download karein."
        ]
        return "\n".join(lines)

    if lang.lower() == "hindi":
        lines = [
            f"{star}**सरल सारांश (Hindi):**",
            f"{pin} *क्षेत्र*: {region_name.replace('_',' ').title()} ({box_str})",
            f"{cal} *तिथि*: {sd} से {ed} तक",
            f"{dep} *गहराई*: {dp}",
            f"• *डेटा बिंदु*: {pts}",
            "",
            (f"*मतलब*: औसत तापमान **{_fmt_float(vmean,1)}°C** ({temp_feel}) रहा, "
             f"ऊपर अधिक गर्म (~{vmax1}°C) और नीचे ठंडा (~{vmin1}°C).")
             if variable=="temperature" else
             (f"*मतलब*: औसत लवणता **{vmean1} PSU** रही, दायरा {vmin1}–{vmax1} PSU."),
            "हीटमैप समय और गहराई के साथ बदलाव दिखाता है,",
            "T–S ग्राफ तापमान–लवणता का संबंध दिखाता है।",
            "",
            f"{csv} नीचे दिए CSV से यही डेटा डाउनलोड करें।"
        ]
        return "\n".join(lines)

    # default English
    lines = [
        f"{star}**Simple summary:**",
        f"{pin} *Area*: {region_name.replace('_',' ').title()} ({box_str})",
        f"{cal} *Dates*: {sd} to {ed}",
        f"{dep} *Depth*: {dp}",
        f"• *Data points*: {pts}",
        "",
        (f"*What it means*: The water was **{temp_feel}** on average "
         f"(**{vmean1}°C**), from **{vmin1}°C** (deeper) to **{vmax1}°C** (near surface).")
         if variable=="temperature" else
        (f"*What it means*: Saltiness averaged **{vmean1} PSU**, "
         f"ranging **{vmin1}–{vmax1} PSU**."),
        "The heatmap shows how values change with time and depth,",
        "and the T–S chart shows the link between temperature and salinity.",
        "",
        f"{csv} You can download the exact slice below (CSV)."
    ]
    return "\n".join(lines)

# ---------- LLM callers ----------
def extract_params_with_llm(question: str):
    """Use Azure OpenAI to extract JSON params. Robust to both Responses and Chat APIs."""
    if client is None:
        return None

    # Build messages
    messages = [{"role":"system","content":PARAMS_SYSTEM}]
    messages += PARAMS_FEWSHOTS
    messages += [{"role":"user","content":question}]

    try:
        if client_mode == "responses":
            resp = client.responses.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                input=messages,
                temperature=0.2,
                response_format={"type":"json_schema","json_schema":{"name":"params","schema":PARAMS_SCHEMA}}
            )
            text = getattr(resp, "output_text", None)
            if not text:
                text = ""
                try:
                    for item in resp.output:
                        if hasattr(item, "content"):
                            for c in item.content:
                                if getattr(c, "type", "") == "output_text":
                                    text += c.text
                except Exception:
                    pass
        else:
            # Chat Completions path
            comp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.2
            )
            text = comp.choices[0].message.content

        # Strip fences & parse JSON
        text = text.strip()
        text = re.sub(r"^```json|```$", "", text, flags=re.MULTILINE).strip()
        return json.loads(text)
    except Exception:
        return None

def rewrite_human_style(text: str):
    """Optional: use LLM to polish the human summary (short, clear)."""
    if client is None:
        return text
    try:
        if client_mode == "responses":
            resp = client.responses.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                input=[{"role":"system","content":STYLE_SYSTEM},
                       {"role":"user","content":text}],
                temperature=0.3
            )
            polished = getattr(resp, "output_text", None)
            if polished:
                return polished.strip()
            return text
        else:
            comp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role":"system","content":STYLE_SYSTEM},
                          {"role":"user","content":text}],
                temperature=0.3
            )
            return comp.choices[0].message.content.strip()
    except Exception:
        return text

# ---------- Viz helpers ----------
def render_map(df, bbox):
    st.subheader("Float locations")
    map_df = df.groupby(["lat","lon"], as_index=False).size()
    if len(map_df) == 0:
        st.info("No float locations to display for this query.")
    else:
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(map_df["lat"].mean()) if len(map_df) else (bbox[2]+bbox[3])/2,
                longitude=float(map_df["lon"].mean()) if len(map_df) else (bbox[0]+bbox[1])/2,
                zoom=3.5
            ),
            layers=[pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_radius=20000,
                pickable=True
            )]
        ))

def render_heatmap(df, variable):
    st.subheader("Time–Depth (heatmap)")
    df2 = df.copy()
    df2["time_bin"] = pd.to_datetime(df2["time"]).dt.floor("D")
    df2["depth_bin"] = (df2["depth_m"]/25).round()*25
    pivot = df2.pivot_table(index="depth_bin", columns="time_bin", values="value", aggfunc="mean")
    if pivot.size == 0:
        st.info("Not enough data to render a heatmap. Try a larger time window.")
    else:
        fig, ax = plt.subplots(figsize=(7,4))
        im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, shading="auto")
        ax.invert_yaxis()
        ax.set_xlabel("Time"); ax.set_ylabel("Depth (m)")
        ax.set_title("Temperature (°C)" if variable=="temperature" else "Salinity (PSU)")
        fig.colorbar(im, ax=ax, label=("°C" if variable=="temperature" else "PSU"))
        st.pyplot(fig, clear_figure=True)

def render_ts(ds):
    st.subheader("T–S Diagram")
    for k in ["TEMP","PSAL","PRES"]:
        if k not in ds:
            st.info("T–S diagram needs both temperature (TEMP) and salinity (PSAL). Try changing dates/region.")
            return
    tdf = ds[["TEMP","PSAL","PRES"]].to_dataframe().dropna().reset_index(drop=True)
    tdf = tdf.rename(columns={"PRES":"depth_m"})
    fig = px.scatter(tdf, x="PSAL", y="TEMP", hover_data=["depth_m"], opacity=0.6)
    fig.update_layout(xaxis_title="Salinity (PSU)", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)

def render_summary_and_download(df, start_date, end_date, depth_min, depth_max, bbox):
    st.subheader("Summary & Download")
    st.write({
        "n_points": int(len(df)),
        "time_range": [str(df['time'].min()), str(df['time'].max())] if len(df) else None,
        "depth_range_m": [float(df['depth_m'].min()), float(df['depth_m'].max())] if len(df) else None,
        "value_min": float(np.nanmin(df['value'])) if len(df) else None,
        "value_mean": float(np.nanmean(df['value'])) if len(df) else None,
        "value_max": float(np.nanmax(df['value'])) if len(df) else None
    })
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       "floatchat_subset.csv", "text/csv")
    st.caption("QC: prefer *_ADJUSTED with QC∈{1,2}; else raw with QC∈{1,2}. Pressure (dbar) ≈ depth (m) near surface.")
    st.write(
        f"**Server**: IFREMER ERDDAP · **Window**: {start_date} → {end_date} · "
        f"**Depth**: {depth_min}–{depth_max} m · **Region**: "
        f"[{bbox[0]}, {bbox[1]}] lon × [{bbox[2]}, {bbox[3]}] lat"
    )

# ================= UI: Tabs =================
tab1, tab2 = st.tabs(["🔎 Explore", "💬 Chat"])

# ---------- Explore ----------
with tab1:
    st.subheader("Manual exploration")
    region_sel = st.selectbox("Region", ["Bay of Bengal", "Arabian Sea", "Custom"], key="expl_region")
    if region_sel == "Bay of Bengal":
        bbox = [78, 100, 5, 22]
    elif region_sel == "Arabian Sea":
        bbox = [55, 78, 5, 23]
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: lonW = st.number_input("lonW", value=78.0, format="%.3f", key="lonW_e")
        with c2: lonE = st.number_input("lonE", value=100.0, format="%.3f", key="lonE_e")
        with c3: latS = st.number_input("latS", value=5.0, format="%.3f", key="latS_e")
        with c4: latN = st.number_input("latN", value=22.0, format="%.3f", key="latN_e")
        bbox = [float(lonW), float(lonE), float(latS), float(latN)]

    colA, colB = st.columns(2)
    with colA:
        end_date = st.date_input("End date (UTC)", value=datetime.utcnow().date(), key="end_e")
    with colB:
        start_date = st.date_input("Start date (UTC)", value=(datetime.utcnow()-timedelta(days=90)).date(), key="start_e")

    variable = normalize_variable(st.selectbox("Variable", ["temperature", "salinity"], key="var_e"))
    depth_min, depth_max = st.slider("Depth range (m)", 0, 2000, (0, 500), step=50, key="depth_e")

    lang_choice = st.selectbox("Summary language", ["English","Hinglish","Hindi"], key="lang_e")
    use_emojis  = st.checkbox("Add emojis", value=True, key="emo_e")
    llm_polish  = st.checkbox("Polish summary with Azure OpenAI", value=False, key="polish_e")

    if st.button("Fetch & Plot", key="go_e"):
        try:
            with st.spinner("Fetching from IFREMER ERDDAP (via argopy)…"):
                params = ["TEMP","PSAL"]  # keep both so T–S can render when possible
                ds = fetch_region(bbox, start_date, end_date, depth_min, depth_max, params=params)
            df = prefer_adjusted(ds, variable)
            if df.empty:
                st.warning("No points returned. Try a shorter date window or a different depth range.")
            else:
                df = df[(df["depth_m"] >= depth_min) & (df["depth_m"] <= depth_max)]
                render_map(df, bbox)
                c1, c2 = st.columns(2)
                with c1: render_heatmap(df, variable)
                with c2: render_ts(ds)

                # Human summary
                vmin = np.nanmin(df["value"]); vmean = np.nanmean(df["value"]); vmax = np.nanmax(df["value"])
                region_name = "bay_of_bengal" if region_sel=="Bay of Bengal" else "arabian_sea" if region_sel=="Arabian Sea" else "custom"
                human = humanize_summary(region_name, bbox, start_date, end_date, depth_min, depth_max,
                                         variable, len(df), vmin, vmean, vmax,
                                         lang=lang_choice.lower(), emojis=use_emojis)
                if llm_polish:
                    human = rewrite_human_style(human)
                st.markdown(human)

                render_summary_and_download(df, start_date, end_date, depth_min, depth_max, bbox)
        except Exception as e:
            st.error(f"Could not fetch/plot data: {e}")
            st.info("Tips: shrink the time window (e.g., last 30–60 days), reduce depth, or change region. Check internet access.")

# ---------- Chat ----------
with tab2:
    st.subheader("Ask in plain English")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Display past turns
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    lang_chat = st.selectbox("Answer language", ["English","Hinglish","Hindi"], key="lang_c")
    emo_chat  = st.checkbox("Add emojis", True, key="emo_c")
    polish_chat = st.checkbox("Polish chat answers with Azure OpenAI", value=False, key="polish_c")

    user_q = st.chat_input("Example: 'Show Bay of Bengal from June to August, temperature up to 500 m'")
    if user_q:
        st.session_state.chat_history.append(("user", user_q))
        with st.chat_message("assistant"):
            placeholder = st.empty()

            # 1) Extract params via LLM (or default guess)
            def default_guess():
                return {
                    "region_name": "bay_of_bengal",
                    "bbox": [78, 100, 5, 22],
                    "start_date": (datetime.utcnow().date() - timedelta(days=60)).isoformat(),
                    "end_date": datetime.utcnow().date().isoformat(),
                    "depth_min": 0, "depth_max": 500,
                    "variable": "temperature"
                }

            parsed = extract_params_with_llm(user_q) or default_guess()

            # Normalize & fill
            region_name = parsed.get("region_name","bay_of_bengal")
            if region_name == "bay_of_bengal":
                bbox = [78, 100, 5, 22]
            elif region_name == "arabian_sea":
                bbox = [55, 78, 5, 23]
            else:
                bbox = parsed.get("bbox") or [78, 100, 5, 22]

            start_date = parsed.get("start_date", (datetime.utcnow().date()-timedelta(days=60)).isoformat())
            end_date   = parsed.get("end_date", datetime.utcnow().date().isoformat())
            depth_min  = float(parsed.get("depth_min", 0))
            depth_max  = float(parsed.get("depth_max", 500))
            variable   = normalize_variable(parsed.get("variable", "temperature"))

            st.markdown("**Interpreted query:**")
            st.json({
                "region_name": region_name, "bbox": bbox,
                "start_date": start_date, "end_date": end_date,
                "depth_min": depth_min, "depth_max": depth_max,
                "variable": variable
            })

            # 2) Fetch + analyze + respond
            try:
                with st.spinner("Fetching from IFREMER ERDDAP (via argopy)…"):
                    ds = fetch_region(bbox, start_date, end_date, depth_min, depth_max, params=["TEMP","PSAL"])
                df = prefer_adjusted(ds, variable)
                if df.empty:
                    answer = "I didn’t find data for that selection. Try a shorter date window or a smaller depth range."
                    if polish_chat:
                        answer = rewrite_human_style(answer)
                    placeholder.markdown(answer)
                    st.session_state.chat_history.append(("assistant", answer))
                else:
                    df = df[(df["depth_m"] >= depth_min) & (df["depth_m"] <= depth_max)]
                    vmin = float(np.nanmin(df["value"]))
                    vmean = float(np.nanmean(df["value"]))
                    vmax = float(np.nanmax(df["value"]))
                    # Deterministic human summary
                    human = humanize_summary(region_name, bbox, start_date, end_date,
                                             depth_min, depth_max, variable,
                                             len(df), vmin, vmean, vmax,
                                             lang=lang_chat.lower(), emojis=emo_chat)
                    if polish_chat:
                        human = rewrite_human_style(human)
                    placeholder.markdown(human)

                    # Plots + download
                    render_map(df, bbox)
                    c1, c2 = st.columns(2)
                    with c1: render_heatmap(df, variable)
                    with c2: render_ts(ds)
                    render_summary_and_download(df, start_date, end_date, depth_min, depth_max, bbox)

                    st.session_state.chat_history.append(("assistant", human))
            except Exception as e:
                msg = f"Could not fetch/plot data: {e}"
                if polish_chat:
                    msg = rewrite_human_style(msg)
                placeholder.markdown(msg)
                st.session_state.chat_history.append(("assistant", msg))

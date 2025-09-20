# backend/server.py
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
import os
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import tempfile
import json
import netCDF4 as nc
import numpy as np
import re
import aiosqlite

# Azure OpenAI (Cognitive Services) async SDK
from openai import AsyncAzureOpenAI

# ---- Load env ----
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env", override=True)

# ---- DB functions (your module) ----
from database import (
    init_database,
    save_chat_message,
    get_chat_history,
    save_netcdf_metadata,
    get_netcdf_files,
    DATABASE_PATH,
)

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# =============================================================================
# App & CORS
# =============================================================================
app = FastAPI(
    title="FloatChat AI - ARGO Ocean Data Explorer",
    description="AI-powered conversational interface for ARGO ocean data discovery and visualization",
    version="1.0.0",
)

cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
allow_credentials = not (len(cors_origins) == 1 and cors_origins[0] == "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")

# =============================================================================
# Schemas
# =============================================================================
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    provider: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = None
    context: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    status: str = "success"
    provider: Optional[str] = None


class NetCDFResponse(BaseModel):
    file_id: str
    filename: str
    dimensions: Dict[str, Any]
    variables: Dict[str, Any]
    global_attributes: Dict[str, Any]
    total_variables: int
    total_dimensions: int


# =============================================================================
# JSON helpers (NumPy/bytes -> JSON-safe)
# =============================================================================
def _json_safe(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return x.hex()
    return x


# =============================================================================
# Azure OpenAI (Cognitive Services) client
# =============================================================================
def _azure_client() -> AsyncAzureOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not api_key or not endpoint:
        raise RuntimeError(
            "Azure OpenAI not fully configured. "
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env"
        )
    return AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


@api_router.get("/azure/ping")
async def azure_ping():
    try:
        client = _azure_client()
        dep = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not dep:
            return {"ok": False, "error": "AZURE_OPENAI_DEPLOYMENT is missing"}
        resp = await client.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Say hi in one word."},
            ],
            max_tokens=5,
            temperature=0,
        )
        return {"ok": True, "text": resp.choices[0].message.content}
    except Exception as e:
        logger.error(f"/azure/ping failed: {e}")
        return {"ok": False, "error": str(e)}


# =============================================================================
# Helpers to pull full metadata (dimensions/variables/attrs) from DB
# =============================================================================
async def get_all_netcdf_metadata() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, filename, dimensions, variables, global_attributes "
            "FROM netcdf_files ORDER BY upload_timestamp DESC"
        ) as cur:
            rows = await cur.fetchall()
            for r in rows:
                try:
                    out.append({
                        "id": r["id"],
                        "filename": r["filename"],
                        "dimensions": json.loads(r["dimensions"] or "{}"),
                        "variables": json.loads(r["variables"] or "{}"),
                        "global_attributes": json.loads(r["global_attributes"] or "{}"),
                    })
                except Exception:
                    out.append({
                        "id": r["id"],
                        "filename": r["filename"],
                        "dimensions": {},
                        "variables": {},
                        "global_attributes": {},
                    })
    return out


# =============================================================================
# ARGO ingestion + simple data Q&A
# =============================================================================
ARGO_NAME_MAP = {
    "temperature": ["TEMP_ADJUSTED", "TEMP", "TEMP_K", "TEMP_C"],
    "salinity": ["PSAL_ADJUSTED", "PSAL", "SALINITY"],
    "pressure": ["PRES_ADJUSTED", "PRES", "PRESSURE"],
    "depth": ["DEPTH", "DEPH"],
    "latitude": ["LATITUDE", "LAT"],
    "longitude": ["LONGITUDE", "LON"],
    "time": ["JULD", "TIME", "DATE_TIME"],
    "float_id": ["PLATFORM_NUMBER", "PLATFORM_CODE", "FLOAT_SERIAL_NO"],
    "oxygen": ["DOXY_ADJUSTED", "DOXY", "OXYGEN"],
}


def _find_first_var(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    return None


def _to_array(ds, varname):
    if not varname:
        return None
    arr = ds.variables[varname][:]
    try:
        return np.array(arr.filled(np.nan))
    except Exception:
        return np.array(arr)


def _flatten(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None:
        return None
    a = np.squeeze(a)
    if a.ndim == 2:
        return a.reshape(-1)
    return a


def _expand(arr: Optional[np.ndarray], length: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.full(length, arr.item())
    if arr.size == length:
        return arr
    if length % arr.size == 0:
        rep = length // arr.size
        return np.repeat(arr, rep)
    out = np.full(length, np.nan)
    out[: min(length, arr.size)] = arr[: min(length, arr.size)]
    return out


async def ingest_argo_to_db(file_id: str, nc_path: str):
    """
    Read NetCDF and insert rows into argo_data:
    (id, file_id, float_id, latitude, longitude, timestamp, depth, temperature, salinity, pressure, oxygen, quality_flag)
    - Accepts 1D and 2D profile layouts.
    - Ingests even if lat/lon are missing (useful for profiles/TS).
    - If depth is missing but pressure exists, uses pressure as depth.
    - If time is missing, keeps timestamp as NULL so profiles/scatter still work.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        with nc.Dataset(nc_path, "r") as ds:
            v_lat = _find_first_var(ds, ARGO_NAME_MAP["latitude"])
            v_lon = _find_first_var(ds, ARGO_NAME_MAP["longitude"])
            v_time = _find_first_var(ds, ARGO_NAME_MAP["time"])
            v_temp = _find_first_var(ds, ARGO_NAME_MAP["temperature"])
            v_psal = _find_first_var(ds, ARGO_NAME_MAP["salinity"])
            v_pres = _find_first_var(ds, ARGO_NAME_MAP["pressure"])
            v_depth = _find_first_var(ds, ARGO_NAME_MAP["depth"])
            v_fid = _find_first_var(ds, ARGO_NAME_MAP["float_id"])
            v_oxy = _find_first_var(ds, ARGO_NAME_MAP["oxygen"])

            # Must have at least one physical var to ingest anything
            if not any([v_temp, v_psal, v_pres, v_oxy]):
                return

            A_lat = _to_array(ds, v_lat)
            A_lon = _to_array(ds, v_lon)
            A_time = _to_array(ds, v_time)
            A_temp = _to_array(ds, v_temp)
            A_psal = _to_array(ds, v_psal)
            A_pres = _to_array(ds, v_pres)
            A_depth = _to_array(ds, v_depth)
            A_oxy = _to_array(ds, v_oxy)

            # Flatten (profiles -> 1D)
            A_temp_f = _flatten(A_temp)
            A_psal_f = _flatten(A_psal)
            A_pres_f = _flatten(A_pres)
            A_depth_f = _flatten(A_depth)
            A_oxy_f = _flatten(A_oxy)

            # Determine output length from available series
            candidates = [x for x in [A_temp_f, A_psal_f, A_pres_f, A_depth_f, A_oxy_f] if isinstance(x, np.ndarray)]
            if not candidates:
                return
            N = max(x.size for x in candidates)

            lat = _expand(A_lat, N)
            lon = _expand(A_lon, N)
            tim = _expand(A_time, N)  # may be None (OK for profile/scatter)
            tem = _expand(A_temp_f, N)
            sal = _expand(A_psal_f, N)
            pres = _expand(A_pres_f, N)
            dep = _expand(A_depth_f, N)
            oxy = _expand(A_oxy_f, N)

            if dep is None and pres is not None:
                dep = pres.copy()

            # float id (stringify)
            fid = np.array([""] * N)
            if v_fid:
                raw = _to_array(ds, v_fid)
                if raw is not None:
                    try:
                        if raw.dtype.kind in ("U", "S"):
                            raw = raw.astype(str)
                        else:
                            raw = raw.astype(str)
                        fid = _expand(raw, N)
                        fid = np.array([str(x) if x is not None else "" for x in fid])
                    except Exception:
                        pass

            rows = []
            for i in range(N):
                # keep row if at least one numeric observation exists
                has_value = False
                for arr in [tem, sal, pres, dep, oxy]:
                    if arr is not None:
                        val = arr[i]
                        if isinstance(val, (int, float, np.floating)) and not np.isnan(val):
                            has_value = True
                            break
                if not has_value:
                    continue

                rows.append(
                    (
                        str(uuid.uuid4()),
                        file_id,
                        (fid[i] if isinstance(fid, np.ndarray) else "") or "",
                        float(lat[i]) if lat is not None and isinstance(lat[i], (int, float, np.floating)) and not np.isnan(lat[i]) else None,
                        float(lon[i]) if lon is not None and isinstance(lon[i], (int, float, np.floating)) and not np.isnan(lon[i]) else None,
                        float(tim[i]) if tim is not None and isinstance(tim[i], (int, float, np.floating)) and not np.isnan(tim[i]) else None,
                        float(dep[i]) if dep is not None and isinstance(dep[i], (int, float, np.floating)) and not np.isnan(dep[i]) else None,
                        float(tem[i]) if tem is not None and isinstance(tem[i], (int, float, np.floating)) and not np.isnan(tem[i]) else None,
                        float(sal[i]) if sal is not None and isinstance(sal[i], (int, float, np.floating)) and not np.isnan(sal[i]) else None,
                        float(pres[i]) if pres is not None and isinstance(pres[i], (int, float, np.floating)) and not np.isnan(pres[i]) else None,
                        float(oxy[i]) if oxy is not None and isinstance(oxy[i], (int, float, np.floating)) and not np.isnan(oxy[i]) else None,
                        None,
                    )
                )

            if rows:
                await db.executemany(
                    """
                    INSERT INTO argo_data
                    (id, file_id, float_id, latitude, longitude, timestamp, depth, temperature, salinity, pressure, oxygen, quality_flag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                await db.commit()


# import re
# import aiosqlite
# from typing import Optional

# Variable synonyms → DB columns
VAR_MAP = {
    "temperature": ["temp", "temperature"],
    "salinity": ["sal", "salinity", "psal"],
    "pressure": ["pressure", "pres"],
    "oxygen": ["oxygen", "o2", "doxy"],
    "depth": ["depth"],
    "latitude": ["lat", "latitude"],
    "longitude": ["lon", "longitude"],
}

# Supported stats
STAT_FUNCS = {
    "avg": "AVG",
    "average": "AVG",
    "mean": "AVG",
    "min": "MIN",
    "minimum": "MIN",
    "max": "MAX",
    "maximum": "MAX",
    "count": "COUNT",
    "std": "STDDEV",  # SQLite needs extension for STDDEV
}

def _detect_variable(user_text: str) -> Optional[str]:
    for col, aliases in VAR_MAP.items():
        for alias in aliases:
            if re.search(rf"\b{alias}\b", user_text):
                print(f"[DEBUG] Matched variable alias '{alias}' → column '{col}'")
                return col
    print("[DEBUG] No variable match found.")
    return None

def _detect_stat(user_text: str) -> Optional[str]:
    for word, sqlfunc in STAT_FUNCS.items():
        if re.search(rf"\b{word}\b", user_text):
            print(f"[DEBUG] Matched stat word '{word}' → SQL '{sqlfunc}'")
            return sqlfunc
    print("[DEBUG] No stat match found.")
    return None

async def try_answer_from_data(user_text: str) -> Optional[str]:
    t = user_text.lower().strip()
    print("\n==============================")
    print("try_answer_from_data received:", t)
    print("==============================")

    # -------------------
    # Special: Latest position
    # -------------------
    if "latest position" in t:
        print("[DEBUG] Detected 'latest position' query")
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            row = await (await db.execute(
                "SELECT latitude, longitude FROM argo_data "
                "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
                "ORDER BY timestamp DESC NULLS LAST LIMIT 1"
            )).fetchone()
            print("[DEBUG] SQL executed for latest position, row:", row)
            if row:
                lat, lon = row
                return f"Latest known position: lat={lat:.4f}, lon={lon:.4f}."
            return "No positions found in the data."

    # -------------------
    # Handle direct latitude/longitude queries (no stat)
    # -------------------
    if "latitude" in t or "longitude" in t:
        print("[DEBUG] Detected latitude/longitude query without stat")
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            rows = await (await db.execute(
                "SELECT latitude, longitude FROM argo_data "
                "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
                "LIMIT 5"
            )).fetchall()
            print("[DEBUG] SQL executed for lat/lon sample, rows:", rows)
            if not rows:
                return "No latitude/longitude data found."
            
            formatted = [f"({r[0]:.4f}, {r[1]:.4f})" for r in rows]
            return "Here are some latitude/longitude pairs: " + ", ".join(formatted)

    # -------------------
    # Detect variable/stat
    # -------------------
    var = _detect_variable(t)
    stat = _detect_stat(t)

    # If not recognized → give suggestions
    if not var or not stat:
        print(f"[DEBUG] Either variable ({var}) or stat ({stat}) not detected.")
        valid_vars = ", ".join(VAR_MAP.keys())
        valid_stats = ", ".join(set(STAT_FUNCS.keys()))
        return (
            "I couldn’t understand your query.\n\n"
            "👉 Try asking in formats like:\n"
            f"- average temperature at 100m\n"
            f"- mean salinity between 50 and 200m\n"
            f"- max oxygen at 20m\n"
            f"- count pressure values between 0 and 1000m\n\n"
            f"Supported variables: {valid_vars}\n"
            f"Supported stats: {valid_stats}"
        )

    # -------------------
    # Depth filters
    # -------------------
    depth_filter = ""
    params = []

    # "at 100m"
    m = re.search(r"(?:at|@)\s*(\d+)\s*m", t)
    if m:
        depth = float(m.group(1))
        tol = max(2.0, depth * 0.05)
        depth_filter = "AND depth BETWEEN ? AND ?"
        params.extend([depth - tol, depth + tol])
        print(f"[DEBUG] Depth filter detected: {depth}m ±{tol:.1f}m")

    # "between 50 and 200m"
    m2 = re.search(r"(?:between|from)\s*(\d+)\s*(?:to|and|-)\s*(\d+)\s*m", t)
    if m2:
        d1, d2 = float(m2.group(1)), float(m2.group(2))
        lo, hi = min(d1, d2), max(d1, d2)
        depth_filter = "AND depth BETWEEN ? AND ?"
        params = [lo, hi]  # overwrite if both filters exist
        print(f"[DEBUG] Depth range detected: {lo}–{hi} m")

    # -------------------
    # Build SQL
    # -------------------
    sql = f"SELECT {stat}({var}) FROM argo_data WHERE {var} IS NOT NULL {depth_filter}"
    print("[DEBUG] Final SQL:", sql)
    print("[DEBUG] Params:", params)

    # -------------------
    # Run SQL
    # -------------------
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        row = await (await db.execute(sql, params)).fetchone()
        print("[DEBUG] SQL executed, row:", row)

    if row and row[0] is not None:
        val = row[0]
        print(f"[DEBUG] Got result: {val}")
        if stat == "COUNT":
            return f"There are {int(val)} {var} records {('at the requested depth range' if depth_filter else '')}."
        return f"{stat.capitalize()} {var} {('at the requested depth range' if depth_filter else '')}: {val:.3f}"
    else:
        print(f"[DEBUG] No matching data found for {var}.")
        return f"I couldn't find {var} data matching your query."


# =============================================================================
# Chat
# =============================================================================
@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # -----------------------
        # Step 1: Run your direct logic
        # -----------------------
        direct = await try_answer_from_data(request.message)
        print("direct:", direct)

        # Save user message
        await save_chat_message(session_id, "user", request.message)

        # -----------------------
        # Step 2: Build system + history
        # -----------------------
        history = await get_chat_history(session_id, limit=10)

        files_md = await get_all_netcdf_metadata()
        if files_md:
            lines = []
            for f in files_md:
                lines.append(
                    f"File: {f['filename']} — Dimensions: {len(f['dimensions'])}, Variables: {len(f['variables'])}"
                )
            history.append({
                "role": "system",
                "content": "User has uploaded NetCDF files:\n" + "\n".join(lines)
            })

        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is missing.")
        client = _azure_client()

        system_message = (
             "You are FloatChat, an expert AI assistant for ARGO ocean data analysis.\n\n"
                "You have two sources of truth:\n"
                "1. Direct SQL results provided by the backend (labeled as 'Direct data query result').\n"
                "2. Your own reasoning about NetCDF metadata and user intent.\n\n"
                "Rules:\n"
                "- If a direct result is provided, always incorporate it naturally in your answer.\n"
                "- If no direct result is available because the query lacked a statistic (avg, min, max, count, std), "
                "do not just say you cannot answer. Instead, guide the user with **clear example queries** "
                "that they can run, based on the variable and context they asked about.\n"
                "- Example:\n"
                "  User: 'oxygen level'\n"
                "  Response: 'I need to know whether you want the **average**, **minimum**, or **maximum** oxygen level. "
                "You can ask: \"average oxygen at 100m\", \"max oxygen between 50 and 200m\", or \"count oxygen values.\"'\n"
                "- Example:\n"
                "  User: 'oxygen at 100m'\n"
                "  Response: 'Do you want the **average oxygen at 100m across all profiles**, "
                "or the **oxygen at 100m for a specific profile (e.g., D2900766_001.nc)?' Suggest both options clearly.\n\n"
                "Always keep your answers practical and user-friendly, and when possible, propose the exact follow-up "
                "queries the user should try, using the supported variables and statistics.\n\n"
                "Supported variables: temperature, salinity, pressure, oxygen, depth, latitude, longitude.\n"
                "Supported stats: avg/average/mean, min/minimum, max/maximum, count, std.\n"
            )

        # -----------------------
        # Step 3: Build AI messages
        # -----------------------
        messages = [{"role": "system", "content": system_message}]
        for m in history[-10:]:
            r = m.get("role") or "user"
            if r not in ("user", "assistant", "system"):
                r = "user"
            messages.append({"role": r, "content": m.get("content", "")})

        # Inject direct results into the context
        if direct:
            messages.append({
                "role": "system",
                "content": f"Direct data query result:\n{direct}"
            })

        messages.append({"role": "user", "content": request.message})

        # -----------------------
        # Step 4: Ask AI
        # -----------------------
        resp = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=0.4,
            top_p=1.0,
            max_tokens=800,
        )

        text = (resp.choices[0].message.content or "").strip()
        print("AI response:", text)

        # Save AI message
        msg_id = await save_chat_message(session_id, "assistant", text, provider=f"azure-cs:{deployment}")
        print("Saved message ID:", msg_id)

        return ChatResponse(
            response=text,
            session_id=session_id,
            message_id=msg_id,
            status="success",
            provider=f"azure-cs:{deployment}",
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            response="I encountered an error processing your request. Please try again.",
            session_id=request.session_id or "error",
            message_id=str(uuid.uuid4()),
            status="error",
        )


@api_router.get("/chat/history/{session_id}")
async def get_session_history(session_id: str):
    try:
        history = await get_chat_history(session_id)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@api_router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            await db.commit()
        return {"message": f"Session {session_id} cleared", "status": "success"}
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session")


@api_router.delete("/chat/all")
async def clear_all_sessions():
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute("DELETE FROM chat_messages")
            await db.execute("DELETE FROM chat_sessions")
            await db.commit()
        return {"message": "All chat history cleared", "status": "success"}
    except Exception as e:
        logger.error(f"Clear-all error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear chats")


# =============================================================================
# NetCDF upload/inspect
# =============================================================================
@api_router.post("/data/upload", response_model=NetCDFResponse)
async def upload_netcdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".nc", ".netcdf", ".cdf")):
        raise HTTPException(status_code=400, detail="File must be a NetCDF file (.nc, .netcdf, .cdf)")

    tmp_path = None
    try:
        content = await file.read()
        file_size = len(content)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        with nc.Dataset(tmp_path, "r") as ds:
            dimensions = {
                name: {"size": len(dim), "unlimited": dim.isunlimited()}
                for name, dim in ds.dimensions.items()
            }

            variables = {}
            for name, var in ds.variables.items():
                info = {
                    "type": str(var.dtype),
                    "dimensions": list(var.dimensions),
                    "shape": list(var.shape),
                }
                attrs = {}
                for attr_name in var.ncattrs():
                    val = getattr(var, attr_name)
                    if isinstance(val, bytes):
                        try:
                            val = val.decode("utf-8", errors="ignore")
                        except Exception:
                            val = val.hex()
                    elif isinstance(val, np.ndarray):
                        val = val.tolist()
                    elif isinstance(val, (np.integer, np.floating)):
                        val = val.item()
                    attrs[attr_name] = val
                if attrs:
                    info["attributes"] = attrs
                variables[name] = info

            global_attributes = {}
            for attr_name in ds.ncattrs():
                val = getattr(ds, attr_name)
                if isinstance(val, bytes):
                    try:
                        val = val.decode("utf-8", errors="ignore")
                    except Exception:
                        val = val.hex()
                elif isinstance(val, np.ndarray):
                    val = val.tolist()
                elif isinstance(val, (np.integer, np.floating)):
                    val = val.item()
                global_attributes[attr_name] = val

            metadata = {
                "dimensions": dimensions,
                "variables": variables,
                "global_attributes": global_attributes,
            }

            file_id = await save_netcdf_metadata(file.filename, file_size, metadata)

        # Ingest into argo_data
        try:
            await ingest_argo_to_db(file_id, tmp_path)
        except Exception as ie:
            logger.warning(f"ARGO ingest warning for {file.filename}: {ie}")

        return NetCDFResponse(
            file_id=file_id,
            filename=file.filename,
            dimensions=dimensions,
            variables=variables,
            global_attributes=global_attributes,
            total_variables=len(variables),
            total_dimensions=len(dimensions),
        )

    except Exception as e:
        logger.error(f"NetCDF processing error: {e}")
        raise HTTPException(status_code=422, detail=f"Error processing NetCDF file: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@api_router.get("/data/files")
async def list_netcdf_files():
    try:
        return {"files": await get_netcdf_files()}
    except Exception as e:
        logger.error(f"File listing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve file list")


@api_router.delete("/data/all")
async def clear_all_data():
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute("DELETE FROM argo_data")
            await db.execute("DELETE FROM netcdf_files")
            await db.commit()
        return {"message": "All data cleared", "status": "success"}
    except Exception as e:
        logger.error(f"Data clear error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear data")


# =============================================================================
# VARIABLES & PREVIEW
# =============================================================================
@api_router.get("/data/variables")
async def list_variables(file_id: str):
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            row = await (await db.execute(
                "SELECT variables, filename FROM netcdf_files WHERE id = ?", (file_id,)
            )).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found")

        vars_json, fname = row
        variables = json.loads(vars_json or "{}")

        out = []
        for name, info in variables.items():
            attrs = info.get("attributes", {})
            out.append({
                "name": name,
                "dtype": info.get("type"),
                "shape": info.get("shape"),
                "dimensions": info.get("dimensions"),
                "units": attrs.get("units") or attrs.get("unit") or attrs.get("Units"),
                "standard_name": attrs.get("standard_name") or attrs.get("long_name"),
            })
        return {"file_id": file_id, "filename": fname, "variables": out}
    except Exception as e:
        logger.error(f"/data/variables error: {e}")
        raise HTTPException(status_code=500, detail="Failed to read variables")


@api_router.get("/data/preview")
async def preview_variable(file_id: str, var: str, limit: int = 200):
    var_lc = var.lower()
    numeric_cols = {"temperature": "temperature", "salinity": "salinity", "pressure": "pressure", "depth": "depth",
                    "oxygen": "oxygen"}

    for key, col in numeric_cols.items():
        if key in var_lc:
            async with aiosqlite.connect(DATABASE_PATH) as db:
                await db.execute("PRAGMA foreign_keys = ON;")
                rows = await (await db.execute(
                    f"SELECT {col} FROM argo_data WHERE {col} IS NOT NULL AND file_id = ? LIMIT ?",
                    (file_id, limit)
                )).fetchall()
            vals = [r[0] for r in rows]
            if not vals:
                return {"file_id": file_id, "var": var, "values": [], "count": 0}
            arr = np.array(vals, dtype=float)
            return {
                "file_id": file_id,
                "var": var,
                "count": int(arr.size),
                "values": [_json_safe(v) for v in arr.tolist()],
                "stats": {
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                }
            }

    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        row = await (await db.execute(
            "SELECT variables FROM netcdf_files WHERE id = ?", (file_id,)
        )).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    variables = json.loads(row[0] or "{}")
    info = variables.get(var)
    if not info:
        raise HTTPException(status_code=404, detail="Variable not found in metadata")
    return {"file_id": file_id, "var": var, "info": info}


# =============================================================================
# PLOT ENDPOINTS (return Plotly JSON) — always return a usable figure
# =============================================================================
def _placeholder(title: str, note: str):
    return {
        "layout": {
            "title": title,
            "annotations": [{
                "text": note,
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5, "showarrow": False
            }],
            "margin": {"l": 60, "r": 20, "t": 40, "b": 50},
        },
        "data": [{
            "type": "scatter",
            "mode": "markers",
            "x": [0], "y": [0],
            "marker": {"opacity": 0}  # invisible point to keep axes alive
        }]
    }


@api_router.get("/plot/profile")
async def plot_profile(file_id: str = None, temp_var: str = "TEMP", depth_var: str = "DEPTH", profile_index: int = 0):
    sql = "SELECT depth, temperature FROM argo_data WHERE depth IS NOT NULL AND temperature IS NOT NULL "
    params = []
    if file_id:
        sql += "AND file_id = ? "
        params.append(file_id)
    sql += "ORDER BY depth ASC LIMIT 5000"

    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        rows = await (await db.execute(sql, params)).fetchall()

    if rows:
        depths = np.array([r[0] for r in rows], dtype=float)
        temps = np.array([r[1] for r in rows], dtype=float)

        order = np.argsort(depths)
        depths = depths[order]
        temps = temps[order]

        fig = {
            "layout": {
                "title": f"Vertical Profile {f'(file: {file_id})' if file_id else ''}",
                "xaxis": {"title": f"{temp_var} (°C)"},
                "yaxis": {"title": f"{depth_var} (m)", "autorange": "reversed"},
                "margin": {"l": 60, "r": 20, "t": 40, "b": 50},
            },
            "data": [{
                "type": "scatter",
                "mode": "lines+markers",
                "x": [_json_safe(x) for x in temps.tolist()],
                "y": [_json_safe(y) for y in depths.tolist()],
                "name": "Profile",
            }]
        }
        return fig

    return _placeholder("Profile unavailable", "No (depth, temperature) pairs found for this file.")


@api_router.get("/plot/timeseries")
async def plot_timeseries(file_id: str = None, var: str = "temperature", depth: float = 10.0):
    var = var.lower().strip()
    col = {"temperature": "temperature", "salinity": "salinity", "pressure": "pressure", "oxygen": "oxygen"}.get(var)
    if not col:
        raise HTTPException(status_code=400, detail="var must be temperature, salinity, pressure, or oxygen")

    tol = max(2.0, 0.05 * depth)
    sql = f"SELECT timestamp, {col} FROM argo_data WHERE {col} IS NOT NULL AND depth BETWEEN ? AND ? "
    params = [depth - tol, depth + tol]
    if file_id:
        sql += "AND file_id = ? "
        params.append(file_id)
    sql += "ORDER BY timestamp ASC"

    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        rows = await (await db.execute(sql, params)).fetchall()

    if not rows:
        return _placeholder("No data at requested depth", "Try a different depth or variable.")

    t = [r[0] for r in rows]
    v = [r[1] for r in rows]

    # If timestamps are None, build an index
    if all(x is None for x in t):
        x_vals = list(range(len(v)))
        x_title = "Sample Index"
    else:
        # Replace None with previous or 0
        x_vals = []
        last = 0.0
        for x in t:
            if x is None:
                x_vals.append(last)
            else:
                xv = float(x)
                x_vals.append(xv)
                last = xv
        x_title = "Time (dataset units)"

    return {
        "layout": {
            "title": f"{var.capitalize()} time series @ ~{depth} m {f'(file: {file_id})' if file_id else ''}",
            "xaxis": {"title": x_title},
            "yaxis": {"title": var.capitalize()},
            "margin": {"l": 60, "r": 20, "t": 40, "b": 50},
        },
        "data": [{
            "type": "scatter",
            "mode": "lines+markers",
            "x": [_json_safe(x) for x in x_vals],
            "y": [_json_safe(y) for y in v],
            "name": var,
        }]
    }


@api_router.get("/plot/map")
async def plot_map(file_id: Optional[str] = None, limit: int = 1000):
    sql = (
        "SELECT latitude, longitude FROM argo_data "
        "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
    )
    params = []
    if file_id:
        sql += "AND file_id = ? "
        params.append(file_id)
    sql += "ORDER BY timestamp DESC LIMIT ?"

    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        rows = await (await db.execute(sql, (*params, limit))).fetchall()

    if not rows:
        # Return empty map frame
        return {
            "layout": {
                "title": "No positions available",
                "geo": {"projection": {"type": "natural earth"}},
                "annotations": [{
                    "text": "Upload a file with LATITUDE/LONGITUDE to view positions.",
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": 0.5, "showarrow": False
                }],
            },
            "data": [{
                "type": "scattergeo",
                "lat": [], "lon": [],
                "mode": "markers",
                "marker": {"size": 6},
                "name": "Positions",
            }]
        }

    lats = [r[0] for r in rows]
    lons = [r[1] for r in rows]

    return {
        "layout": {
            "title": f"ARGO Float Positions {f'(file: {file_id})' if file_id else ''}",
            "geo": {"projection": {"type": "natural earth"}}
        },
        "data": [{
            "type": "scattergeo",
            "lat": [_json_safe(v) for v in lats],
            "lon": [_json_safe(v) for v in lons],
            "mode": "markers",
            "marker": {"size": 6},
            "name": "Positions",
        }]
    }


@api_router.get("/plot/scatter")
async def plot_scatter(file_id: str = None, x_var: str = "temperature", y_var: str = "salinity"):
    lut = {"temperature": "temperature", "salinity": "salinity", "pressure": "pressure", "oxygen": "oxygen"}
    x_col = lut.get(x_var.lower())
    y_col = lut.get(y_var.lower())

    if not x_col or not y_col:
        raise HTTPException(status_code=400, detail="Variables must be temperature, salinity, pressure, or oxygen")

    sql = f"SELECT {x_col}, {y_col} FROM argo_data WHERE {x_col} IS NOT NULL AND {y_col} IS NOT NULL "
    params = []
    if file_id:
        sql += "AND file_id = ? "
        params.append(file_id)
    sql += "ORDER BY timestamp ASC LIMIT 5000"

    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON;")
        rows = await (await db.execute(sql, params)).fetchall()

    if not rows:
        return _placeholder("No data for scatter plot", f"Need {x_var.upper()} and {y_var.upper()} values.")

    x_vals = [r[0] for r in rows]
    y_vals = [r[1] for r in rows]

    return {
        "layout": {
            "title": f"{x_var.capitalize()} vs {y_var.capitalize()} {f'(file: {file_id})' if file_id else ''}",
            "xaxis": {"title": x_var.capitalize()},
            "yaxis": {"title": y_var.capitalize()},
            "margin": {"l": 60, "r": 20, "t": 40, "b": 50},
        },
        "data": [{
            "type": "scatter",
            "mode": "markers",
            "x": [_json_safe(x) for x in x_vals],
            "y": [_json_safe(y) for y in y_vals],
            "name": f"{x_var} vs {y_var}",
        }]
    }


# =============================================================================
# Health
# =============================================================================
@api_router.get("/health")
async def health_check():
    try:
        async with aiosqlite.connect(DATABASE_PATH) as db:
            await db.execute("SELECT 1")

        ai_configured = all(
            [
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_OPENAI_ENDPOINT"),
                os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            ]
        )

        return {
            "status": "healthy",
            "service": "FloatChat AI Backend",
            "database": "connected",
            "ai_service": "configured" if ai_configured else "not_configured",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---- Mount API ----
app.include_router(api_router)

# ---- Serve frontend (../frontend) ----
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/api")
async def api_root():
    return {
        "message": "FloatChat AI Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# ---- Startup ----
@app.on_event("startup")
async def startup_event():
    await init_database()
    logger.info("Database initialized successfully")


# ---- Dev entrypoint ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)

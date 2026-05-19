
"""
This is the stable API for CastaneaGPT.
It uses FastAPI to provide endpoints for querying the RAG system.
"""

import uuid
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.api.schemas import ChatRequest
from app.api.memory import InMemorySessionStore, format_memory
from app.api.geo import router as geo_router
from app.rag.rag import query_rag
from app.utils.stand_context import (
    build_ndvi_context_block,
    build_stand_context_block,
    should_include_ndvi_context,
)
from app.utils.quiz import quiz

from app.api.ndvi import router as ndvi_router

load_dotenv()



# -----------------------------
# App + Memory
# -----------------------------
app = FastAPI(title="CastaneaGPT API", version="0.2")

session_store = InMemorySessionStore(max_turns=8, ttl_minutes=120)
app.include_router(ndvi_router, prefix="/geo")
app.include_router(geo_router)


# -----------------------------
# Helpers
# -----------------------------
def is_uuid(val: str) -> bool:
    try:
        uuid.UUID(val)
        return True
    except Exception:
        return False


def _enum_to_str(x) -> str:
    """Supports Enum values from Pydantic (e.g., AnswerMode.professional)."""
    return x.value if hasattr(x, "value") else str(x)


# -----------------------------
# Schemas for Swagger
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceItem(BaseModel):
    id: int
    score: float
    source_type: Optional[str] = None
    file_name: Optional[str] = None
    source_path: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None
    clean_file: Optional[str] = None
    snippet: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class QuizGenerateRequest(BaseModel):
    topic: str = "general chestnut knowledge"
    difficulty: str = "intermediate"
    n_questions: int = 5
    format_type: str = "multiple_choice"
    language: str = "auto"


class QuizGradeRequest(BaseModel):
    quiz_id: str
    user_answers: Dict[str, str]


class MemoryResetRequest(BaseModel):
    session_id: str


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):

    print("DEBUG stand_id:", req.stand_id)

    # Convert Enums -> strings
    mode = _enum_to_str(req.mode)
    verbosity = _enum_to_str(req.verbosity)
    cite_style = _enum_to_str(req.cite_style)

    # -----------------------------
    # LANGUAGE HIERARCHY
    # 1. If user selects language in UI -> force it
    # 2. If UI = auto -> mirror user message language
    # -----------------------------
    if req.language and req.language != "auto":
        language = req.language
    else:
        language = "auto"

    # -----------------------------
    # Session handling
    # -----------------------------
    session_id = req.session_id

    if req.use_session_memory:
        if not session_id or not is_uuid(session_id):
            session_id = str(uuid.uuid4())
    else:
        session_id = None

    # -----------------------------
    # Memory block
    # -----------------------------
    memory_block = ""
    recent_source_keys = []

    if req.use_session_memory and session_id:
        mem = session_store.get(session_id)
        memory_block = format_memory(mem.turns)
        recent_source_keys = list(mem.recent_source_keys)

    # -----------------------------
    # Stand context injection
    # -----------------------------
    stand_context_block = ""

    if getattr(req, "stand_id", None):
        stand_context_block = build_stand_context_block(req.stand_id)

    # -----------------------------
    # NDVI context injection
    # -----------------------------
    ndvi_context_block = ""

    if getattr(req, "stand_id", None) and should_include_ndvi_context(req.query):
        ndvi_context_block = build_ndvi_context_block(req.stand_id)

    # -----------------------------
    # Build retrieval question
    # -----------------------------
    question_block = req.query

    if memory_block:
        question_block = f"{memory_block}\n\nNew question:\n{req.query}"

    context_block = ""

    if stand_context_block:
        context_block += stand_context_block.strip() + "\n\n"

    if ndvi_context_block:
        context_block += ndvi_context_block.strip() + "\n\n"

    if context_block:
        retrieval_question = f"{context_block}User question:\n{question_block}"
    else:
        retrieval_question = question_block

    print("PROMPT TO RAG:\n", retrieval_question)

    # -----------------------------
    # Step 1: RAG retrieval
    # -----------------------------
    rag_out = query_rag(
        retrieval_question,
        mode=mode,
        verbosity=verbosity,
        language=language,
        cite_style=cite_style,
        language_hint_text=req.query,
        recent_source_keys=recent_source_keys,
    )

    final_answer = rag_out.get("answer", "")
    sources = rag_out.get("sources", []) or []

    # -----------------------------
    # Normalize sources
    # -----------------------------
    normalized_sources = []
    seen = set()

    for s in sources:
        label = s.get("label") or s.get("source") or "document"
        snippet = (s.get("snippet") or "").strip()
        page = s.get("page")
        score = s.get("score")

        key = (label, page)
        if key in seen:
            continue
        seen.add(key)

        normalized_sources.append({
            "id": s.get("id"),
            "label": label,
            "page": page,
            "score": score,
            "snippet": snippet,
        })

    # -----------------------------
    # Citation formatting
    # -----------------------------
    if cite_style != "none" and normalized_sources:
        citation_text = "\n\nSource reference list:\n"

        for s in normalized_sources[:8]:
            source_id = s.get("id") or "?"
            citation_text += f"[Source {source_id}] {s['label']}"
            if s["page"]:
                citation_text += f", p.{s['page']}"
            citation_text += "\n"

        final_answer = final_answer + citation_text

    # -----------------------------
    # Update memory
    # -----------------------------
    if req.use_session_memory and session_id:
        session_store.append(session_id, "user", req.query)
        session_store.append(session_id, "assistant", final_answer)
        session_store.set_recent_sources(
            session_id,
            [s.get("source_key") or s.get("label") or "" for s in sources[:8]],
        )

    # -----------------------------
    # Compact sources for frontend
    # -----------------------------
    compact_sources = [
        {
            "id": s["id"],
            "label": s["label"],
            "page": s["page"],
            "score": s["score"],
            "snippet": s["snippet"],
        }
        for s in normalized_sources[:8]
    ]

    return {
        "answer": final_answer,
        "sources": compact_sources,
        "session_id": session_id,
    }


@app.post("/memory/reset")
def reset_memory(req: MemoryResetRequest):
    session_store.clear(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


@app.get("/map", response_class=HTMLResponse, tags=["demo"])
def map_view():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CastaneaGPT Map Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet"
        href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html, body { height: 100%; margin: 0; }
    #map { height: 100%; }

    .panel {
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 999;
      background: white;
      padding: 14px;
      border-radius: 14px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.15);
      font-family: system-ui, sans-serif;
      font-size: 14px;
      width: 380px;
      max-height: 92%;
      overflow-y: auto;
    }

    .section { margin-top: 12px; }
    .mono { font-family: ui-monospace, monospace; font-size: 12px; }

    label {
      display: block;
      margin-top: 10px;
      margin-bottom: 4px;
    }

    select {
      display: block;
      width: 100%;
      margin-bottom: 6px;
    }

    textarea {
      width: 100%;
      min-height: 60px;
      resize: vertical;
      padding: 6px;
      margin-top: 10px;
    }

    button {
      padding: 6px 10px;
      cursor: pointer;
      margin-top: 10px;
    }

    .stand-row {
      display: flex;
      gap: 6px;
      align-items: flex-start;
      margin-bottom: 6px;
    }

    .stand-row button {
      margin-top: 0;
    }

    .stand-row .stand-main-btn {
      flex: 1;
    }

    .stand-row .stand-toggle-btn {
      white-space: nowrap;
      font-size: 12px;
    }

    .button-row {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }

    .button-row button {
      margin-top: 0;
    }

    .answer {
      margin-top: 8px;
      font-size: 13px;
      line-height: 1.4;
    }
  </style>
</head>
<body>

<div id="map"></div>



<!--THIS IS THE PANEL SECTION-------------------------------------------------------!-->


<div class="panel">
  <div><b>CastaneaGPT</b></div>

  <div class="section">
    <div id="status" class="mono">Loading stands…</div>
    <div id="picked" class="mono"></div>
    <!-- ✅ NEW: NDVI display (insert HERE) -->
    <div id="ndvi" class="mono"></div>
  </div>

  <!-- ✅ NEW: Stand list section -->
  <div class="section">
    <b>Particelle disponibili / Available stands</b>
    <div id="standList"></div>
  </div>

  <div class="section">
    <b>Chiedi informazioni in generale e su questa particella / Ask general questions and questions about this stand</b>
    <label>Modalita / Mode:</label>
    <select id="modeSelect">
      <option value="professional">Professionale / Professional</option>
      <option value="decision" selected>Supporto decisionale / Decision Support</option>
      <option value="playful">Giocoso / Playful</option>
    </select>

    <label>Livello di dettaglio / Verbosity:</label>
    <select id="verbositySelect">
      <option value="short">Basso / Low</option>
      <option value="normal" selected>Medio / Medium</option>
      <option value="long">Alto / High</option>
    </select>

    <label>Lingua / Language:</label>
    <select id="languageSelect">
      <option value="auto" selected>Auto</option>
      <option value="en">English</option>
      <option value="it">Italiano</option>
      <option value="de">Deutsch</option>
    </select>

    <textarea id="chatInput"
              placeholder="e.g. What thinning intervention is recommended this year?"></textarea>
    <div class="button-row">
      <button id="askBtn">Chiedi / Ask</button>
      <button id="toggleAnswerBtn">Nascondi / Hide</button>
    </div>
    <div id="answerBox" class="answer"></div>
  </div>

  <div class="section">
    <b>Quiz sulla conoscenza del castagno / Chestnut Knowledge Quiz</b>

    <label>Tema / Topic:</label>
    <select id="quiz-topic">
      <option value="general chestnut knowledge">Generale / General</option>
      <option value="chestnut silviculture">Selvicoltura / Silviculture</option>
      <option value="chestnut forest operations">Utilizzazioni forestali / Forest Operations</option>
      <option value="chestnut pests and diseases">Parassiti e malattie / Pests and diseases</option>
      <option value="chestnut ecology">Ecologia / Ecology</option>
      <option value="chestnut economics">Economia / Economics</option>
      <option value="chestnut cultural role">Ruolo culturale / Cultural role</option>
    </select>

    <label>Difficolta / Difficulty:</label>
    <select id="quiz-difficulty">
      <option value="beginner">Principiante / Beginner</option>
      <option value="intermediate" selected>Intermedio / Intermediate</option>
      <option value="advanced">Avanzato / Advanced</option>
    </select>

    <label>Lingua / Language:</label>
    <select id="quiz-language">
      <option value="auto" selected>Auto</option>
      <option value="en">English</option>
      <option value="it">Italiano</option>
      <option value="de">Deutsch</option>
    </select>

    <button id="quiz-start" onclick="startQuiz()">Avvia quiz / Start Quiz</button>
    <div id="quiz-container"></div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
const UI_TRANSLATIONS = {
  auto: {
    loadingStands: "Loading stands…",
    noStandsInView: "No stands found in the current map view.",
    loadingStandSummary: (standId) => `Loading stand ${standId} summary…`,
    failedStandSummary: "Failed to load stand summary.",
    ndviTitle: "NDVI",
    noNdvi: "No NDVI data available.",
    latest: "Latest",
    mean: "Mean",
    min: "Min",
    max: "Max",
    timeSeries: "Time series",
    noTimeSeries: "No time series available.",
    species: "Species",
    management: "Management",
    ageClass: "Age class",
    areaHa: "Area (ha)",
    altitudeM: "Altitude (m)",
    notes: "Notes",
    standSelected: (standId) => `Stand ${standId} selected.`,
    loadedStands: (count) => `Loaded ${count} stand(s) in view.`,
    loadedNoStands: "Loaded 0 stands. The database may be empty, or the stands may be outside the current map extent.",
  },
  en: {
    loadingStands: "Loading stands…",
    noStandsInView: "No stands found in the current map view.",
    loadingStandSummary: (standId) => `Loading stand ${standId} summary…`,
    failedStandSummary: "Failed to load stand summary.",
    ndviTitle: "NDVI",
    noNdvi: "No NDVI data available.",
    latest: "Latest",
    mean: "Mean",
    min: "Min",
    max: "Max",
    timeSeries: "Time series",
    noTimeSeries: "No time series available.",
    species: "Species",
    management: "Management",
    ageClass: "Age class",
    areaHa: "Area (ha)",
    altitudeM: "Altitude (m)",
    notes: "Notes",
    standSelected: (standId) => `Stand ${standId} selected.`,
    loadedStands: (count) => `Loaded ${count} stand(s) in view.`,
    loadedNoStands: "Loaded 0 stands. The database may be empty, or the stands may be outside the current map extent.",
  },
  it: {
    loadingStands: "Caricamento particelle…",
    noStandsInView: "Nessuna particella trovata nell'area visibile della mappa.",
    loadingStandSummary: (standId) => `Caricamento riepilogo della particella ${standId}…`,
    failedStandSummary: "Impossibile caricare il riepilogo della particella.",
    ndviTitle: "NDVI",
    noNdvi: "Nessun dato NDVI disponibile.",
    latest: "Ultimo",
    mean: "Media",
    min: "Min",
    max: "Max",
    timeSeries: "Serie temporale",
    noTimeSeries: "Nessuna serie temporale disponibile.",
    species: "Specie",
    management: "Gestione",
    ageClass: "Classe di eta",
    areaHa: "Superficie (ha)",
    altitudeM: "Altitudine (m)",
    notes: "Note",
    standSelected: (standId) => `Particella ${standId} selezionata.`,
    loadedStands: (count) => `${count} particella/e caricate nella vista.`,
    loadedNoStands: "Caricate 0 particelle. Il database potrebbe essere vuoto oppure le particelle potrebbero trovarsi fuori dall'area visibile.",
  },
  de: {
    loadingStands: "Bestaende werden geladen…",
    noStandsInView: "Keine Bestaende im aktuellen Kartenausschnitt gefunden.",
    loadingStandSummary: (standId) => `Zusammenfassung fuer Bestand ${standId} wird geladen…`,
    failedStandSummary: "Zusammenfassung des Bestands konnte nicht geladen werden.",
    ndviTitle: "NDVI",
    noNdvi: "Keine NDVI-Daten verfuegbar.",
    latest: "Neuester Wert",
    mean: "Mittelwert",
    min: "Min",
    max: "Max",
    timeSeries: "Zeitreihe",
    noTimeSeries: "Keine Zeitreihe verfuegbar.",
    species: "Art",
    management: "Bewirtschaftung",
    ageClass: "Altersklasse",
    areaHa: "Flaeche (ha)",
    altitudeM: "Hoehe (m)",
    notes: "Hinweise",
    standSelected: (standId) => `Bestand ${standId} ausgewaehlt.`,
    loadedStands: (count) => `${count} Bestand/Bestaende im Kartenausschnitt geladen.`,
    loadedNoStands: "0 Bestaende geladen. Die Datenbank koennte leer sein oder die Bestaende liegen ausserhalb des aktuellen Kartenausschnitts.",
  },
};

const map = L.map('map').setView([43.0, 12.5], 6);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

let standsLayer = null;
let didAutoFit = false;
let selectedStandId = null;
let selectedStandSummary = null;
let selectedStandNdvi = null;
let selectedStandCollapsed = false;
let lastStandFeatureCollection = null;
let answerCollapsed = false;

function getUiLanguage() {
  const value = document.getElementById("languageSelect")?.value || "auto";
  return UI_TRANSLATIONS[value] ? value : "auto";
}

function t() {
  return UI_TRANSLATIONS[getUiLanguage()];
}

function formatMaybeEmpty(value) {
  return value == null || value === "" ? "-" : value;
}

function updateAnswerVisibility() {
  const answerBox = document.getElementById("answerBox");
  const toggleAnswerBtn = document.getElementById("toggleAnswerBtn");
  if (!answerBox || !toggleAnswerBtn) return;

  answerBox.style.display = answerCollapsed ? "none" : "block";
  toggleAnswerBtn.textContent = answerCollapsed ? "Mostra / Show" : "Nascondi / Hide";
}

function bboxParam() {
  const b = map.getBounds();
  return `${b.getWest()},${b.getSouth()},${b.getEast()},${b.getNorth()}`;
}

function standToggleLabel(standId) {
  if (selectedStandId === standId && selectedStandSummary && !selectedStandCollapsed) {
    return "Nascondi / Hide";
  }
  return "Mostra / Show";
}

function setHighlight(feature) {
  if (!feature) return;

  const layer = L.geoJSON(feature).addTo(map);
  map.fitBounds(layer.getBounds().pad(0.3));

  if (window._highlight) map.removeLayer(window._highlight);
  window._highlight = layer;
}

async function loadSelectedStand(standId, feature = null) {
  const status = document.getElementById('status');
  if (!standId) return;

  selectedStandId = standId;
  selectedStandCollapsed = false;

  if (feature) {
    setHighlight(feature);
  }

  status.textContent = t().loadingStandSummary(standId);

  const resSummary = await fetch(`/geo/stands/${standId}/summary`);
  if (!resSummary.ok) {
    status.textContent = t().failedStandSummary;
    return;
  }

  selectedStandSummary = await resSummary.json();
  selectedStandNdvi = null;

  try {
    const resNdvi = await fetch(`/geo/stands/${standId}/ndvi`);
    if (resNdvi.ok) {
      selectedStandNdvi = await resNdvi.json();
    }
  } catch (e) {
    console.error("NDVI fetch error:", e);
  }

  renderSelectedStand();
  if (lastStandFeatureCollection) {
    renderStandList(lastStandFeatureCollection);
  }
  status.textContent = t().standSelected(standId);
}


function renderStandList(fc) {
  const container = document.getElementById("standList");
  container.innerHTML = "";

  const features = Array.isArray(fc.features) ? fc.features : [];

  if (!features.length) {
    container.innerHTML = `<div style='color:#666;'>${t().noStandsInView}</div>`;
    return;
  }

  features.forEach(f => {
    const p = f.properties || {};
    const standId = p.id;

    const row = document.createElement("div");
    row.className = "stand-row";

    const btn = document.createElement("button");
    btn.className = "stand-main-btn";
    btn.textContent = "Particella / Stand " + standId;

    btn.onclick = async () => {
      await loadSelectedStand(standId, f);
    };

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "stand-toggle-btn";
    toggleBtn.textContent = standToggleLabel(standId);
    toggleBtn.onclick = async () => {
      if (selectedStandId !== standId || !selectedStandSummary) {
        await loadSelectedStand(standId, f);
      }

      selectedStandCollapsed = !selectedStandCollapsed;
      renderSelectedStand();
      if (lastStandFeatureCollection) {
        renderStandList(lastStandFeatureCollection);
      }
    };

    row.appendChild(btn);
    row.appendChild(toggleBtn);
    container.appendChild(row);
  });
}


function formatNdviValue(value) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}


function buildNdviHtml(ndvi) {
  const labels = t();
  const emptyHtml = `<br><b>${labels.ndviTitle}</b><br><span style='color:#666;'>${labels.noNdvi}</span>`;

  if (!ndvi) {
    return emptyHtml;
  }

  const latest = ndvi.latest ? ndvi.latest : null;
  const series = Array.isArray(ndvi.series) ? [...ndvi.series] : [];
  series.sort((a, b) => {
    const monthA = typeof a.month === "number" ? a.month : 99;
    const monthB = typeof b.month === "number" ? b.month : 99;
    if (monthA !== monthB) return monthA - monthB;

    const yearA = typeof a.year === "number" ? a.year : 9999;
    const yearB = typeof b.year === "number" ? b.year : 9999;
    return yearA - yearB;
  });

  let html = `<br><b>${labels.ndviTitle}</b><br>`;
  html += labels.latest + " (" + (latest && latest.label ? latest.label : "-") + ")<br>";
  html += labels.mean + ": " + formatNdviValue(latest ? latest.mean : null) + "<br>";
  html += labels.min + ": " + formatNdviValue(latest ? latest.min : null) + "<br>";
  html += labels.max + ": " + formatNdviValue(latest ? latest.max : null) + "<br>";
  html += `<br><b>${labels.timeSeries}</b><br>`;

  if (!series.length) {
    html += `<div>${labels.noTimeSeries}</div>`;
    return html;
  }

  series.forEach(function(entry) {
    html += "<div>"
      + entry.label
      + ": mean=" + formatNdviValue(entry.mean)
      + ", min=" + formatNdviValue(entry.min)
      + ", max=" + formatNdviValue(entry.max)
      + "</div>";
  });

  return html;
}

function renderSelectedStand() {
  const picked = document.getElementById('picked');
  if (!selectedStandSummary) {
    picked.innerHTML = "";
    return;
  }

  if (selectedStandCollapsed) {
    picked.innerHTML = "<i>Informazioni sulla particella nascoste / Stand information hidden</i>";
    return;
  }

  const labels = t();
  const s = selectedStandSummary;
  const ndviHtml = buildNdviHtml(selectedStandNdvi);

  picked.innerHTML = `
    <b>${formatMaybeEmpty(s.name)}</b><br>
    ${labels.species}: ${formatMaybeEmpty(s.species)}<br>
    ${labels.management}: ${formatMaybeEmpty(s.management_type)}<br>
    ${labels.ageClass}: ${formatMaybeEmpty(s.age_class)}<br>
    ${labels.areaHa}: ${formatMaybeEmpty(s.area_ha)}<br>
    ${labels.altitudeM}: ${formatMaybeEmpty(s.altitude_m)}<br>
    ${labels.notes}: ${formatMaybeEmpty(s.notes)}
    ${ndviHtml}
  `;
}



async function loadStands() {
  const status = document.getElementById('status');
  status.textContent = t().loadingStands;

  const url = `/geo/stands?bbox=${encodeURIComponent(bboxParam())}`;
  const res = await fetch(url);
  const fc = await res.json();
  lastStandFeatureCollection = fc;

  renderStandList(fc);

  if (standsLayer) standsLayer.remove();

  standsLayer = L.geoJSON(fc, {
    onEachFeature: (feature, layer) => {
      layer.on('click', async () => {
        const p = feature.properties || {};
        const standId = p.id;
        if (!standId) return;
        await loadSelectedStand(standId, feature);
      });
    }
  }).addTo(map);

  const n = (fc.features || []).length;
  status.textContent = n
    ? t().loadedStands(n)
    : t().loadedNoStands;
}


document.getElementById("askBtn").addEventListener("click", async () => {
  const question = document.getElementById("chatInput").value.trim();
  const askBtn = document.getElementById("askBtn");
  const chatInput = document.getElementById("chatInput");
  const answerBox = document.getElementById("answerBox");

  if (!question) return;
  if (!selectedStandId) {
    answerBox.innerHTML = "<i>Please select a stand first.</i>";
    return;
  }

  askBtn.disabled = true;
  chatInput.disabled = true;
  answerCollapsed = false;
  updateAnswerVisibility();
  answerBox.innerHTML = "<i>Thinking…</i>";

  const mode = document.getElementById("modeSelect").value;
  const verbosity = document.getElementById("verbositySelect").value;
  const language = document.getElementById("languageSelect").value;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: question,
        stand_id: selectedStandId,
        use_session_memory: true,
        mode: mode,
        verbosity: verbosity,
        language: language,
        cite_style: "inline"
      })
    });

    if (!res.ok) {
      throw new Error("Chat request failed");
    }

    const data = await res.json();
    answerBox.innerHTML = data.answer;
  } catch (err) {
    console.error("Chat request failed:", err);
    answerBox.innerHTML = "<i>Request failed. Please try again.</i>";
  } finally {
    askBtn.disabled = false;
    chatInput.disabled = false;
  }
});

document.getElementById("toggleAnswerBtn").addEventListener("click", () => {
  answerCollapsed = !answerCollapsed;
  updateAnswerVisibility();
});

map.on('moveend', loadStands);
document.getElementById("languageSelect").addEventListener("change", () => {
  renderSelectedStand();
  loadStands();
});
loadStands();
updateAnswerVisibility();

let currentQuizId = null;
let currentQuestions = [];


// =======================
// START QUIZ
// =======================
async function startQuiz() {
  const startBtn = document.getElementById("quiz-start");
  const container = document.getElementById("quiz-container");
  startBtn.disabled = true;
  container.innerHTML = "<i>Generating quiz…</i>";

  const topic = document.getElementById("quiz-topic").value;
  const difficulty = document.getElementById("quiz-difficulty").value;
  const language = document.getElementById("quiz-language").value;

  try {
    const res = await fetch("/quiz/generate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        topic: topic,
        difficulty: difficulty,
        language: language,
        n_questions: 5,
        format_type: "multiple_choice"
      })
    });

    if (!res.ok) {
      throw new Error("Quiz generation failed");
    }

    const data = await res.json();
    console.log("QUIZ DATA:", data);

    if (!data.questions || data.questions.length === 0) {
      container.innerHTML = "<i>No quiz questions returned.</i>";
      startBtn.disabled = false;
      return;
    }

    currentQuizId = data.quiz_id;
    currentQuestions = data.questions;

    renderQuiz(currentQuestions);
  } catch (err) {
    console.error("Quiz generation failed:", err);
    container.innerHTML = "<i>Quiz generation failed.</i>";
    startBtn.disabled = false;
  }
}

// =======================
// RENDER QUIZ
// =======================
function renderQuiz(questions) {
  const container = document.getElementById("quiz-container");
  container.innerHTML = "";

  questions.forEach((q, i) => {
    let html = `
      <div style="margin-bottom:20px;">
        <b>Question ${i+1}</b><br>
        ${q.question}<br><br>
    `;

    if (!q.options || q.options.length === 0) {
      html += "<i>No options available</i>";
    } else {
      q.options.forEach(opt => {
        html += `
          <label style="display:block;">
            <input type="radio" name="${q.id}" value="${opt}">
            ${opt}
          </label>
        `;
      });
    }

    html += "</div>";
    container.innerHTML += html;
  });

  container.innerHTML += `
    <button id="quiz-submit" onclick="submitQuiz()" style="margin-top:10px;">
      Invia risposte / Submit Answers
    </button>
  `;
}

// =======================
// SUBMIT QUIZ
// =======================
async function submitQuiz() {
  const container = document.getElementById("quiz-container");
  const submitBtn = document.getElementById("quiz-submit");
  const answers = {};

  currentQuestions.forEach(q => {
    const selected = document.querySelector(`input[name="${q.id}"]:checked`);
    if (selected) {
      answers[String(q.id)] = selected.value;
    }
  });

  console.log("USER ANSWERS:", answers);

  if (Object.keys(answers).length === 0) {
    alert("Please select at least one answer.");
    return;
  }

  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = "Correzione in corso... / Grading...";
  }

  try {
    const res = await fetch("/quiz/grade", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        quiz_id: currentQuizId,
        user_answers: answers
      })
    });

    if (!res.ok) {
      throw new Error("Quiz grading failed");
    }

    const data = await res.json();
    showResults(data);
  } catch (err) {
    console.error("Quiz grading failed:", err);
    container.innerHTML += "<div style='margin-top:10px; color:#666;'><i>Quiz grading failed.</i></div>";
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = "Invia risposte / Submit Answers";
    }
  }
}



// =======================
// SHOW RESULTS
// =======================
function showResults(data) {
  console.log("GRADE RESPONSE:", data);

  const container = document.getElementById("quiz-container");
  const score = data.score != null ? data.score : 0;
  const total = data.total != null ? data.total : (data.results ? data.results.length : 0);

  let html = `
    <div style="margin-bottom:15px;">
      <b>Score:</b> ${score} / ${total}
    </div>
  `;

  if (data.results) {
    data.results.forEach((r, i) => {
      const color = r.is_correct ? "green" : "red";

      html += `
        <div style="margin-bottom:12px; padding:8px; border:1px solid #ddd;">
          <b>Question ${i+1}</b><br>
          ${r.question}<br><br>

          <b>Your answer:</b>
          <span style="color:${color}; font-weight:bold;">
            ${r.your_answer != null ? r.your_answer : "No answer"}
          </span><br>

          <b>Correct answer:</b>
          <span style="color:green; font-weight:bold;">
            ${r.correct_answer}
          </span><br>

          <b>Explanation:</b> ${r.explanation}
        </div>
      `;
    });
  }

  html += `
    <button onclick="restartQuiz()" style="margin-top:10px;">
      Riavvia quiz / Restart Quiz
    </button>
  `;

  container.innerHTML = html;
}

// =======================
// RESTART QUIZ
// =======================
function restartQuiz() {
  currentQuizId = null;
  currentQuestions = [];
  document.getElementById("quiz-container").innerHTML = "";
  document.getElementById("quiz-start").disabled = false;
}
</script>

</body>
</html>
"""
    return HTMLResponse(content=html)


@app.post("/quiz/generate")
def generate_quiz(req: QuizGenerateRequest):
    return quiz.generate(
        topic=req.topic,
        difficulty=req.difficulty,
        n_questions=req.n_questions,
        format_type=req.format_type,
        language=req.language,
    )


@app.post("/quiz/grade")
def grade_quiz(req: QuizGradeRequest):
    return quiz.grade(
        quiz_id=req.quiz_id,
        user_answers=req.user_answers,
    )

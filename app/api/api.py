
"""
This is the stable API for CastaneaGPT.
It uses FastAPI to provide endpoints for querying the RAG system.
"""

import os
import uuid
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from app.api.schemas import ChatRequest
from app.api.prompting import build_response_policy
from app.api.memory import InMemorySessionStore, format_memory
from app.api.geo import router as geo_router
from app.rag.rag import query_rag
from app.utils.stand_context import build_ndvi_context_block, build_stand_context_block
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


def rewrite_answer_with_policy(
    grounded_answer: str,
    context_digest: str,
    mode: str,
    verbosity: str,
    language: str,
    language_instruction: str,
    cite_style: str,
) -> str:
    """
    Rewrite layer:
    - Style + verbosity + citation style enforcement happens here
    - Must NOT introduce new facts beyond the retrieved context + grounded answer
    """

    # Practical guidance:
    # short  -> ~200 to 300
    # normal -> ~600 to 800
    # long   -> ~1000 to 1500
    max_tokens = 300 if verbosity == "short" else 1200 if verbosity == "long" else 700

    llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        max_tokens=max_tokens,
    )

    policy = build_response_policy(
        mode=mode,
        verbosity=verbosity,
        language=language,   # language code: en / it / de / auto
        cite_style=cite_style,
    )

    prompt = (
        f"{policy}\n\n"
        "REWRITE TASK:\n"
        "1) Write the final answer in the requested MODE and VERBOSITY.\n"
        "2) Use ONLY the information contained in:\n"
        "   - CONTEXT (retrieved snippets)\n"
        "   - GROUNDED ANSWER (baseline)\n"
        "3) Do NOT introduce any new facts not supported by CONTEXT.\n"
        "4) If CONTEXT is insufficient for a requested detail, explicitly say so.\n"
        "5) Apply the requested citation style when referring to sources.\n\n"
        "LANGUAGE RULE:\n"
        f"{language_instruction}\n\n"
        "CONTEXT (retrieved snippets):\n"
        f"{context_digest}\n\n"
        "GROUNDED ANSWER (baseline):\n"
        f"{grounded_answer}\n"
    )

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="You rewrite answers using provided context only, following the given policy.",
        ),
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]

    try:
        resp = llm.chat(messages)
        out = (resp.message.content or "").strip()
        return out or grounded_answer
    except Exception:
        return grounded_answer


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
    lang_name_map = {
        "en": "English",
        "it": "Italian",
        "de": "German",
    }

    if req.language and req.language != "auto":
        language = req.language
        language_name = lang_name_map.get(language, "English")
        language_instruction = (
            f"The answer MUST be written entirely in {language_name}. "
            f"Do not switch language. Do not mix languages."
        )
    else:
        language = "auto"
        language_instruction = (
            "Write the answer in the SAME language as the user's question. "
            "Do not switch languages. Do not mix languages."
        )

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

    if req.use_session_memory and session_id:
        mem = session_store.get(session_id)
        memory_block = format_memory(mem.turns)

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

    if getattr(req, "stand_id", None):
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
    rag_out = query_rag(retrieval_question)

    grounded_answer = rag_out.get("answer", "")
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
            "label": label,
            "page": page,
            "score": score,
            "snippet": snippet,
        })

    # -----------------------------
    # Context digest
    # -----------------------------
    digest_lines = []
    for s in normalized_sources[:5]:
        if s["snippet"]:
            digest_lines.append(f"- {s['label']}: {s['snippet']}")

    context_digest = "\n".join(digest_lines) if digest_lines else "(no retrieved snippets)"

    # -----------------------------
    # Step 2: Rewrite answer
    # -----------------------------
    final_answer = rewrite_answer_with_policy(
        grounded_answer=grounded_answer,
        context_digest=context_digest + "\n\n" + ndvi_context_block,
        mode=mode,
        verbosity=verbosity,
        language=language,
        language_instruction=language_instruction,
        cite_style=cite_style,
    )

    # -----------------------------
    # Citation formatting
    # -----------------------------
    if cite_style != "none" and normalized_sources:
        citation_text = "\n\nSources:\n"

        for i, s in enumerate(normalized_sources[:5]):
            citation_text += f"[{i+1}] {s['label']}"
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

    # -----------------------------
    # Compact sources for frontend
    # -----------------------------
    compact_sources = [
        {
            "label": s["label"],
            "page": s["page"],
            "score": s["score"],
            "snippet": s["snippet"],
        }
        for s in normalized_sources[:5]
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

    textarea {
      width: 100%;
      min-height: 60px;
      resize: vertical;
      padding: 6px;
    }

    button {
      padding: 6px 10px;
      cursor: pointer;
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
  <div><b>CastaneaGPT</b> – Geo Decision Demo</div>

  <div class="section">
    <div id="status" class="mono">Loading stands…</div>
    <div id="picked" class="mono"></div>
    <!-- ✅ NEW: NDVI display (insert HERE) -->
    <div id="ndvi" class="mono"></div>
  </div>

  <!-- ✅ NEW: Stand list section -->
  <div class="section">
    <b>Available stands</b>
    <div id="standList"></div>
  </div>

  <div class="section">
    <b>Ask about this stand</b>
    <label>Mode:</label>
    <select id="modeSelect">
      <option value="professional">Professional</option>
      <option value="decision" selected>Decision Support</option>
      <option value="playful">Playful</option>
    </select>

    <label>Verbosity:</label>
    <select id="verbositySelect">
      <option value="short">Low</option>
      <option value="normal" selected>Medium</option>
      <option value="long">High</option>
    </select>

    <label>Language:</label>
    <select id="languageSelect">
      <option value="en" selected>English</option>
      <option value="it">Italian</option>
      <option value="auto">Auto</option>
    </select>

    <textarea id="chatInput"
              placeholder="e.g. What thinning intervention is recommended this year?"></textarea>
    <button id="askBtn">Ask</button>
    <div id="answerBox" class="answer"></div>
  </div>

  <div class="section">
    <b>Chestnut Knowledge Quiz</b>

    <label>Topic:</label>
    <select id="quiz-topic">
      <option value="general chestnut knowledge">General</option>
      <option value="Castanea sativa management">Management</option>
      <option value="coppice systems">Coppice systems</option>
      <option value="chestnut pests and diseases">Pests and diseases</option>
    </select>

    <label>Difficulty:</label>
    <select id="quiz-difficulty">
      <option value="beginner">Beginner</option>
      <option value="intermediate" selected>Intermediate</option>
      <option value="advanced">Advanced</option>
    </select>

    <label>Language:</label>
    <select id="quiz-language">
      <option value="auto" selected>Auto</option>
      <option value="en">English</option>
      <option value="it">Italiano</option>
      <option value="de">Deutsch</option>
    </select>

    <button id="quiz-start" onclick="startQuiz()">Start Quiz</button>
    <div id="quiz-container"></div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
const map = L.map('map').setView([43.0, 12.5], 6);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

let standsLayer = null;
let didAutoFit = false;
let selectedStandId = null;

function bboxParam() {
  const b = map.getBounds();
  return `${b.getWest()},${b.getSouth()},${b.getEast()},${b.getNorth()}`;
}


function renderStandList(fc) {
  const container = document.getElementById("standList");
  container.innerHTML = "";

  const features = Array.isArray(fc.features) ? fc.features : [];

  if (!features.length) {
    container.innerHTML = "<div style='color:#666;'>No stands found in the current map view.</div>";
    return;
  }

  features.forEach(f => {
    const p = f.properties || {};
    const standId = p.id;

    const btn = document.createElement("button");
    btn.textContent = "Stand " + standId;
    btn.style.display = "block";
    btn.style.marginBottom = "6px";

    btn.onclick = () => {
      selectedStandId = standId;

      const layer = L.geoJSON(f).addTo(map);
      map.fitBounds(layer.getBounds().pad(0.3));

      if (window._highlight) map.removeLayer(window._highlight);
      window._highlight = layer;
    };

    container.appendChild(btn);
  });
}


function formatNdviValue(value) {
  return typeof value === "number" ? value.toFixed(3) : "-";
}


function buildNdviHtml(ndvi) {
  const emptyHtml = "<br><b>NDVI</b><br><span style='color:#666;'>No NDVI data available.</span>";

  if (!ndvi) {
    return emptyHtml;
  }

  const latest = ndvi.latest ? ndvi.latest : null;
  const series = Array.isArray(ndvi.series) ? ndvi.series : [];

  let html = "<br><b>NDVI</b><br>";
  html += "Latest (" + (latest && latest.label ? latest.label : "-") + ")<br>";
  html += "Mean: " + formatNdviValue(latest ? latest.mean : null) + "<br>";
  html += "Min: " + formatNdviValue(latest ? latest.min : null) + "<br>";
  html += "Max: " + formatNdviValue(latest ? latest.max : null) + "<br>";
  html += "<br><b>Time series</b><br>";

  if (!series.length) {
    html += "<div>No time series available.</div>";
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



async function loadStands() {
  const status = document.getElementById('status');
  status.textContent = 'Loading stands…';

  const url = `/geo/stands?bbox=${encodeURIComponent(bboxParam())}`;
  const res = await fetch(url);
  const fc = await res.json();

  renderStandList(fc);

  if (standsLayer) standsLayer.remove();

  standsLayer = L.geoJSON(fc, {
    onEachFeature: (feature, layer) => {
      layer.on('click', async () => {
        const p = feature.properties || {};
        const standId = p.id;
        if (!standId) return;

        selectedStandId = standId;
        status.textContent = `Loading stand ${standId} summary…`;

        // -------------------------
        // 1) Load summary
        // -------------------------
        const resSummary = await fetch(`/geo/stands/${standId}/summary`);
        if (!resSummary.ok) {
          status.textContent = "Failed to load stand summary.";
          return;
        }

        const s = await resSummary.json();

        // -------------------------
        // 2) Load NDVI
        // -------------------------
        let ndviHtml = "<br><b>NDVI</b><br><span style='color:#666;'>No NDVI data available.</span>";

        try {
          const resNdvi = await fetch(`/geo/stands/${standId}/ndvi`);
          if (resNdvi.ok) {
            const ndvi = await resNdvi.json();
            ndviHtml = buildNdviHtml(ndvi);
          }
        } catch (e) {
          console.error("NDVI fetch error:", e);
        }

        // -------------------------
        // 3) Render panel
        // -------------------------
        document.getElementById('picked').innerHTML = `
          <b>${s.name}</b><br>
          Species: ${s.species}<br>
          Management: ${s.management_type}<br>
          Age class: ${s.age_class}<br>
          Area (ha): ${s.area_ha}<br>
          Altitude (m): ${s.altitude_m}<br>
          Notes: ${s.notes}
          ${ndviHtml}
        `;

        status.textContent = `Stand ${standId} selected.`;
      });
    }
  }).addTo(map);

  const n = (fc.features || []).length;
  status.textContent = n
    ? `Loaded ${n} stand(s) in view.`
    : 'Loaded 0 stands. The database may be empty, or the stands may be outside the current map extent.';
}


document.getElementById("askBtn").addEventListener("click", async () => {
  const question = document.getElementById("chatInput").value.trim();
  const answerBox = document.getElementById("answerBox");

  if (!question) return;
  if (!selectedStandId) {
    answerBox.innerHTML = "<i>Please select a stand first.</i>";
    return;
  }

  answerBox.innerHTML = "<i>Thinking…</i>";

  const mode = document.getElementById("modeSelect").value;
  const verbosity = document.getElementById("verbositySelect").value;
  const language = document.getElementById("languageSelect").value;

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: question,
      stand_id: selectedStandId,
      use_session_memory: false,
      mode: mode,
      verbosity: verbosity,
      language: language,
      cite_style: "inline"
    })
  });

  const data = await res.json();
  answerBox.innerHTML = data.answer;
});

map.on('moveend', loadStands);
loadStands();

let currentQuizId = null;
let currentQuestions = [];


// =======================
// START QUIZ
// =======================
async function startQuiz() {
  const startBtn = document.getElementById("quiz-start");
  startBtn.disabled = true;

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

    const data = await res.json();
    console.log("QUIZ DATA:", data);

    if (!data.questions || data.questions.length === 0) {
      alert("No quiz questions returned.");
      startBtn.disabled = false;
      return;
    }

    currentQuizId = data.quiz_id;
    currentQuestions = data.questions;

    renderQuiz(currentQuestions);
  } catch (err) {
    console.error("Quiz generation failed:", err);
    alert("Quiz generation failed.");
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
    <button onclick="submitQuiz()" style="margin-top:10px;">
      Submit Answers
    </button>
  `;
}

// =======================
// SUBMIT QUIZ
// =======================
async function submitQuiz() {
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

  try {
    const res = await fetch("/quiz/grade", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        quiz_id: currentQuizId,
        user_answers: answers
      })
    });

    const data = await res.json();
    showResults(data);
  } catch (err) {
    console.error("Quiz grading failed:", err);
    alert("Quiz grading failed.");
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
      Restart Quiz
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

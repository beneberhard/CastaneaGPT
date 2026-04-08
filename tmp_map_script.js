
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

  (fc.features || []).forEach(f => {
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
        let ndviHtml = "<br><b>NDVI</b><br><span style=\"color:#666;\">No NDVI data available.</span>";

        try {
          const resNdvi = await fetch(`/geo/stands/${standId}/ndvi`);
          if (resNdvi.ok) {
            const ndvi = await resNdvi.json();
            const latest = ndvi && ndvi.latest ? ndvi.latest : null;
            const series = Array.isArray(ndvi.series) ? ndvi.series : [];
            const seriesHtml = series.length
              ? series.map(entry => `
                  <div>${entry.label}: mean=${typeof entry.mean === "number" ? entry.mean.toFixed(3) : "-"}, min=${typeof entry.min === "number" ? entry.min.toFixed(3) : "-"}, max=${typeof entry.max === "number" ? entry.max.toFixed(3) : "-"}</div>
                `).join("")
              : "<div>No time series available.</div>";

            ndviHtml = `
              <br><b>NDVI</b><br>
              Latest (${latest && latest.label ? latest.label : "-"})<br>
              Mean: ${latest && typeof latest.mean === "number" ? latest.mean.toFixed(3) : "-"}<br>
              Min: ${latest && typeof latest.min === "number" ? latest.min.toFixed(3) : "-"}<br>
              Max: ${latest && typeof latest.max === "number" ? latest.max.toFixed(3) : "-"}<br>
              <br><b>Time series</b><br>
              ${seriesHtml}
            `;
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
  status.textContent = `Loaded ${n} stand(s) in view.`;
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
  const score = data.score ?? 0;
  const total = data.total ?? (data.results ? data.results.length : 0);

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
            ${r.your_answer ?? "No answer"}
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

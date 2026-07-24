const CATEGORIES = [
  "Marriage & family", "Labour dispute", "Traffic accident", "Debt dispute",
  "Criminal defence", "Property dispute", "Consumer complaint", "Cybercrime",
  "Medical negligence", "Housing / tenancy", "Education dispute"
];

const categoryList = document.getElementById("category-list");
CATEGORIES.forEach(c => {
  const li = document.createElement("li");
  li.textContent = c;
  categoryList.appendChild(li);
});

const log = document.getElementById("case-log");
const form = document.getElementById("composer");
const textarea = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");

let docketNumber = 1;

function nextDocket() {
  return "DOCKET " + String(docketNumber++).padStart(3, "0");
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function addUserEntry(text) {
  const entry = document.createElement("div");
  entry.className = "entry user";
  entry.innerHTML = `
    <div class="entry-head">
      <span class="docket">${nextDocket()}</span>
      <span class="tag">You</span>
    </div>
    <p></p>
  `;
  entry.querySelector("p").textContent = text;
  log.appendChild(entry);
  scrollToBottom();
}

function addTypingEntry() {
  const entry = document.createElement("div");
  entry.className = "entry system";
  entry.id = "typing-entry";
  entry.innerHTML = `
    <div class="entry-head">
      <span class="docket">${nextDocket()}</span>
      <span class="tag">Assistant</span>
    </div>
    <div class="typing"><span></span><span></span><span></span></div>
  `;
  log.appendChild(entry);
  scrollToBottom();
  return entry;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderAnswerEntry(entry, payload) {
  const head = entry.querySelector(".entry-head");
  const docket = head.querySelector(".docket").outerHTML;

  const typeLabel = payload.type ? escapeHtml(payload.type) : "Unclassified";
  const tagClass = payload.confident ? "tag stamp" : "tag low-confidence";

  head.innerHTML = `${docket}<span class="${tagClass}">${typeLabel}</span>`;

  let body = "";

  if (payload.results && payload.results.length) {
    payload.results.forEach((item, i) => {
      body += `
        <div class="candidate">
          <p class="matched-q">Closest on file: “${escapeHtml(item.question)}” <span class="score-bar">match ${(item.score * 100).toFixed(0)}%</span></p>
          <ul class="answers">
            ${item.answers.map(a => `<li>${escapeHtml(a)}</li>`).join("")}
          </ul>
        </div>
      `;
    });
  } else {
    body += `<p>${escapeHtml(payload.note || "No matching entry found.")}</p>`;
    if (payload.generic_advice && payload.generic_advice.length) {
      body += `<ul class="answers">${payload.generic_advice.map(a => `<li>${escapeHtml(a)}</li>`).join("")}</ul>`;
    }
  }

  entry.insertAdjacentHTML("beforeend", body);
  entry.removeAttribute("id");
  scrollToBottom();
}

async function askQuestion(question) {
  const typingEntry = addTypingEntry();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || "Something went wrong.");
    }

    const payload = await res.json();
    renderAnswerEntry(typingEntry, payload);
  } catch (e) {
    typingEntry.querySelector(".typing")?.remove();
    typingEntry.insertAdjacentHTML("beforeend", `<p>Sorry — ${escapeHtml(e.message)}</p>`);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = textarea.value.trim();
  if (!question) return;

  addUserEntry(question);
  textarea.value = "";
  textarea.style.height = "auto";
  sendBtn.disabled = true;

  askQuestion(question).finally(() => {
    sendBtn.disabled = false;
    textarea.focus();
  });
});

textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + "px";
});

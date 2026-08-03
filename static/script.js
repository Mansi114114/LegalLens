/* LegalLens assistant UI.
 *
 * Conversations live in localStorage, not on the server. The knowledge base is
 * static and every answer is derived from the question alone, so there is
 * nothing a backend session would add except a database to host and a privacy
 * question to answer — and the free tier this deploys to has an ephemeral
 * disk, so server-side history would silently vanish on restart anyway.
 */

const STORAGE_KEY = "legallens.chats.v1";
const TITLE_MAX = 44;

const CATEGORIES = [
  "Marriage & family", "Labour dispute", "Traffic accident", "Debt dispute",
  "Criminal defence", "Property dispute", "Consumer complaint", "Cybercrime",
  "Medical negligence", "Housing / tenancy", "Education dispute", "Insurance claims",
  "Inheritance & succession"
];

const el = {
  log: document.getElementById("case-log"),
  thread: document.getElementById("thread"),
  intro: document.getElementById("intro"),
  form: document.getElementById("composer"),
  textarea: document.getElementById("question"),
  sendBtn: document.getElementById("send-btn"),
  chatList: document.getElementById("chat-list"),
  listLabel: document.getElementById("list-label"),
  listEmpty: document.getElementById("list-empty"),
  search: document.getElementById("chat-search"),
  newChat: document.getElementById("new-chat-btn"),
  title: document.getElementById("chat-title"),
  sidebar: document.getElementById("sidebar"),
  scrim: document.getElementById("scrim"),
  drawerOpen: document.getElementById("drawer-open"),
  drawerClose: document.getElementById("drawer-close"),
  categoryList: document.getElementById("category-list"),
};

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/* ---------------------------------------------------------------- store -- */

let state = { chats: [], activeId: null };

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed?.chats)) {
      state.chats = parsed.chats;
      state.activeId = parsed.activeId ?? null;
    }
  } catch (e) {
    // Corrupt or unavailable storage shouldn't take the whole page down —
    // start empty and carry on in memory.
    console.warn("Could not restore chats:", e);
  }
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    // Private browsing / quota exceeded. The session still works, it just
    // won't survive a reload.
    console.warn("Could not save chats:", e);
  }
}

function activeChat() {
  return state.chats.find(c => c.id === state.activeId) || null;
}

function newId() {
  return `c_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function createChat() {
  const chat = { id: newId(), title: "New chat", updatedAt: Date.now(), messages: [] };
  state.chats.unshift(chat);
  state.activeId = chat.id;
  saveState();
  return chat;
}

/** Ensure there's somewhere to append to, without creating empty chats on load. */
function ensureActiveChat() {
  return activeChat() || createChat();
}

function titleFrom(text) {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > TITLE_MAX ? clean.slice(0, TITLE_MAX - 1) + "…" : clean;
}

function touchChat(chat) {
  chat.updatedAt = Date.now();
  // Most-recent-first, matching where a new chat is inserted.
  state.chats.sort((a, b) => b.updatedAt - a.updatedAt);
  saveState();
}

/* -------------------------------------------------------------- sidebar -- */

function chatMatches(chat, query) {
  if (!query) return true;
  const q = query.toLowerCase();
  if (chat.title.toLowerCase().includes(q)) return true;
  return chat.messages.some(m => {
    if (m.role === "user") return m.text.toLowerCase().includes(q);
    const p = m.payload || {};
    return (p.results || []).some(r =>
      r.question.toLowerCase().includes(q) ||
      r.answers.some(a => a.toLowerCase().includes(q))
    );
  });
}

const ICONS = {
  dots: `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><circle cx="5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="19" cy="12" r="1.7"/></svg>`,
  pencil: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z"/></svg>`,
  trash: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>`,
};

/* -- chat row context menu ------------------------------------------------
 * The menu is appended to <body> and positioned with fixed coordinates rather
 * than nested inside the row. The chat list is a scroll container, so a menu
 * living inside it would be clipped at the container's edge.
 */
let openMenu = null;

function closeChatMenu() {
  if (!openMenu) return;
  openMenu.el.remove();
  openMenu.anchor?.setAttribute("aria-expanded", "false");
  openMenu = null;
}

function openChatMenu(chat, anchor) {
  const wasOpenForThis = openMenu?.chatId === chat.id;
  closeChatMenu();
  if (wasOpenForThis) return;

  const menu = document.createElement("div");
  menu.className = "row-menu";
  menu.setAttribute("role", "menu");
  menu.innerHTML = `
    <button type="button" role="menuitem" data-act="rename">${ICONS.pencil}<span>Rename</span></button>
    <button type="button" role="menuitem" data-act="delete" class="danger">${ICONS.trash}<span>Delete</span></button>
  `;
  document.body.appendChild(menu);

  // Position under the trigger, flipping up or clamping left when the menu
  // would otherwise run off-screen.
  const r = anchor.getBoundingClientRect();
  const m = menu.getBoundingClientRect();
  const gap = 6;
  const top = r.bottom + gap + m.height > window.innerHeight
    ? Math.max(gap, r.top - gap - m.height)
    : r.bottom + gap;
  const left = Math.min(
    Math.max(gap, r.right - m.width),
    window.innerWidth - m.width - gap
  );
  menu.style.top = `${top}px`;
  menu.style.left = `${left}px`;

  anchor.setAttribute("aria-expanded", "true");
  openMenu = { el: menu, chatId: chat.id, anchor };

  menu.querySelector('[data-act="rename"]').addEventListener("click", () => {
    closeChatMenu();
    startRename(chat);
  });
  menu.querySelector('[data-act="delete"]').addEventListener("click", () => {
    closeChatMenu();
    deleteChat(chat);
  });
  menu.querySelector("button").focus();
}

function startRename(chat) {
  // Re-render first so the row is guaranteed to be in the list even if a
  // search filter was just changed underneath the menu.
  renderSidebar();
  const li = el.chatList.querySelector(`.chat-item[data-id="${chat.id}"]`);
  if (!li) return;

  const input = document.createElement("input");
  input.type = "text";
  input.className = "chat-rename-input";
  input.value = chat.title;
  input.maxLength = 80;
  input.setAttribute("aria-label", "Chat name");

  li.innerHTML = "";
  li.appendChild(input);
  input.focus();
  input.select();

  let settled = false;
  const finish = (save) => {
    if (settled) return;
    settled = true;
    const next = input.value.trim();
    if (save && next && next !== chat.title) {
      chat.title = next;
      // Remember that the user named this one, so sending the first message
      // doesn't overwrite it with an auto-generated title.
      chat.renamed = true;
      saveState();
      if (chat.id === state.activeId && chat.messages.length) setTitle(chat.title, true);
    }
    renderSidebar();
  };

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    e.stopPropagation();
  });
  input.addEventListener("blur", () => finish(true));
}

function deleteChat(chat) {
  if (!confirm(`Delete “${chat.title}”? This can't be undone.`)) return;

  const wasActive = chat.id === state.activeId;
  state.chats = state.chats.filter(c => c.id !== chat.id);
  if (wasActive) state.activeId = state.chats[0]?.id ?? null;
  saveState();

  renderSidebar();
  // Only the active thread needs redrawing; deleting a background chat
  // shouldn't wipe what's on screen.
  if (wasActive) renderChat();
}

function renderSidebar() {
  const query = el.search.value.trim();
  const visible = state.chats.filter(c => chatMatches(c, query));

  el.listLabel.textContent = query ? `Results (${visible.length})` : "Chats";
  el.chatList.innerHTML = "";

  visible.forEach((chat) => {
    const li = document.createElement("li");
    li.className = "chat-item" + (chat.id === state.activeId ? " active" : "");
    li.dataset.id = chat.id;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chat-item-btn";
    btn.textContent = chat.title;
    btn.title = chat.title;
    btn.addEventListener("click", () => selectChat(chat.id));

    const menuBtn = document.createElement("button");
    menuBtn.type = "button";
    menuBtn.className = "chat-menu-btn";
    menuBtn.innerHTML = ICONS.dots;
    menuBtn.setAttribute("aria-label", `Options for ${chat.title}`);
    menuBtn.setAttribute("aria-haspopup", "menu");
    menuBtn.setAttribute("aria-expanded", "false");
    menuBtn.addEventListener("click", (e) => {
      // Without this the click falls through to the row and switches chats.
      e.stopPropagation();
      openChatMenu(chat, menuBtn);
    });

    li.append(btn, menuBtn);
    el.chatList.appendChild(li);
  });

  const nothing = visible.length === 0;
  el.listEmpty.hidden = !nothing;
  el.listEmpty.textContent = query ? "No chats match that search." : "No chats yet.";
}

function selectChat(id) {
  state.activeId = id;
  saveState();
  renderSidebar();
  renderChat();
  closeDrawer();
  el.textarea.focus();
}

/* ------------------------------------------------------ answer rendering -- */

function answerListHtml(answers) {
  return `<ul class="answers">${answers.map(a => `<li>${escapeHtml(a)}</li>`).join("")}</ul>`;
}

function citationsHtml(item) {
  const refs = [...(item.acts || []), ...(item.sections || [])];
  if (!refs.length) return "";
  return `<p class="citations">${refs.map(r => `<span class="cite">${escapeHtml(r)}</span>`).join("")}</p>`;
}

function docketTag(n) {
  return `<span class="docket">DOCKET&nbsp;${String(n).padStart(3, "0")}</span>`;
}

/**
 * Two candidates the retriever couldn't separate, shown as peers.
 *
 * They are deliberately NOT presented as ranked. The backend only sends a
 * choice when the scores are within AMBIGUITY_MARGIN of each other, so
 * calling one of them "the answer" would imply a confidence the scores
 * don't support — the whole reason this path exists.
 */
function renderChoice(entry, payload, initialChoice, onPick) {
  const [a, b] = payload.results;

  const card = (item, index) => `
    <div class="choice-card" data-index="${index}">
      <div class="choice-card-head">
        <span class="choice-label">Option ${index + 1}</span>
        <span class="tag low-confidence">${escapeHtml(item.type || "Unclassified")}</span>
      </div>
      <p class="matched-q">Closest question: “${escapeHtml(item.question)}”
        <span class="score-bar">Match: ${Math.min(item.score, 0.99).toFixed(2)}</span>
      </p>
      ${answerListHtml(item.answers)}
      ${citationsHtml(item)}
      <button type="button" class="choice-select">Select this answer</button>
    </div>
  `;

  const prompt = payload.note ||
    "Two entries match about equally well. Pick whichever fits your situation.";

  entry.insertAdjacentHTML("beforeend", `
    <p class="choice-prompt">${escapeHtml(prompt)}</p>
    <div class="choice-grid">${card(a, 0)}${card(b, 1)}</div>
  `);

  const grid = entry.querySelector(".choice-grid");

  function resolve(index, notify) {
    if (grid.classList.contains("resolved")) return;
    grid.classList.add("resolved");

    grid.querySelectorAll(".choice-card").forEach((cardEl) => {
      const isPick = Number(cardEl.dataset.index) === index;
      cardEl.classList.add(isPick ? "chosen" : "dismissed");
      const btn = cardEl.querySelector(".choice-select");
      // Keep the rejected option readable rather than removing it — the user
      // may want to compare again after picking.
      if (btn) {
        btn.outerHTML = isPick
          ? `<p class="chosen-flag">✓ Selected</p>`
          : `<p class="dismissed-flag">Not selected</p>`;
      }
    });

    if (notify) onPick(index, payload.results[index]);
  }

  grid.querySelectorAll(".choice-card").forEach((cardEl) => {
    cardEl.querySelector(".choice-select").addEventListener(
      "click", () => resolve(Number(cardEl.dataset.index), true)
    );
  });

  // Restoring a chat: re-apply an earlier pick without re-firing side effects.
  if (initialChoice === 0 || initialChoice === 1) resolve(initialChoice, false);
}

function renderAnswerEntry(entry, payload, initialChoice, onPick) {
  // The "..." placeholder lives on until the answer replaces it. Only the
  // error path used to clear it, so every completed answer kept a stray set
  // of pulsing dots above it.
  entry.querySelector(".typing")?.remove();

  const head = entry.querySelector(".entry-head");
  const docket = head.querySelector(".docket").outerHTML;

  // Small talk and off-topic input aren't cases, so don't stamp them with a
  // legal category — that framing is what made every stray message look like
  // a filed matter.
  const KIND_LABELS = {
    greeting: "LegalLens",
    out_of_scope: "Not a legal query",
    empty: "LegalLens",
  };
  const setTag = (label, cls) => {
    head.innerHTML = `${docket}<span class="${cls}">${escapeHtml(label)}</span>`;
  };

  if (payload.kind === "choice") {
    // Both options are on the table, so don't stamp a single category yet
    // unless the two happen to agree on one.
    const picked = payload.results[initialChoice];
    setTag(picked ? (picked.type || "Selected") : (payload.type || "Which applies?"),
           picked ? "tag stamp" : "tag low-confidence");
    entry.removeAttribute("id");
    renderChoice(entry, payload, initialChoice, (index, item) => {
      setTag(item.type || "Selected", "tag stamp");
      onPick?.(index);
      scrollToBottom();
    });
    return;
  }

  const typeLabel = KIND_LABELS[payload.kind] || payload.type || "No close match";
  setTag(typeLabel, payload.confident ? "tag stamp" : "tag low-confidence");

  let body = "";

  if (payload.results && payload.results.length) {
    const topResult = payload.results[0];
    const topScore = Math.min(topResult.score, 0.99).toFixed(2);

    // A hedge from the backend ("no close match — treat as background")
    // belongs above the answer, not buried under it.
    if (payload.note) {
      body += `<p class="hedge">${escapeHtml(payload.note)}</p>`;
    }

    body += `
      <div class="candidate primary-candidate">
        <p class="matched-q">Most similar question: “${escapeHtml(topResult.question)}” <span class="score-bar">Similarity Score: ${topScore}</span></p>
        ${answerListHtml(topResult.answers)}
        ${citationsHtml(topResult)}
      </div>
    `;

    if (payload.results.length > 1) {
      const others = payload.results.slice(1);
      body += `
        <div class="alternatives-container">
          <button type="button" class="alternatives-toggle" onclick="
            const list = this.nextElementSibling;
            list.classList.toggle('open');
            const isOpen = list.classList.contains('open');
            this.textContent = isOpen ? '▲ Hide other relevant results' : '▼ View other relevant results (${others.length})';
          ">
            ▼ View other relevant results (${others.length})
          </button>
          <div class="alternatives-list">
      `;

      others.forEach((item) => {
        body += `
          <div class="candidate alt-candidate">
            <p class="matched-q">Most similar question: “${escapeHtml(item.question)}” <span class="score-bar">Similarity Score: ${Math.min(item.score, 0.99).toFixed(2)}</span></p>
            ${answerListHtml(item.answers)}
            ${citationsHtml(item)}
          </div>
        `;
      });

      body += `</div></div>`;
    }
  } else {
    body += `<p>${escapeHtml(payload.note || "No matching entry found.")}</p>`;
    if (payload.generic_advice && payload.generic_advice.length) {
      body += answerListHtml(payload.generic_advice);
    }
  }

  entry.insertAdjacentHTML("beforeend", body);
  entry.removeAttribute("id");
}

/* -------------------------------------------------------- thread rendering */

function scrollToBottom() {
  el.thread.scrollTo({ top: el.thread.scrollHeight, behavior: "smooth" });
}

function userEntryEl(text, docketNo) {
  const entry = document.createElement("div");
  entry.className = "entry user";
  entry.innerHTML = `
    <div class="entry-head">${docketTag(docketNo)}<span class="tag">You</span></div>
    <p></p>
  `;
  entry.querySelector("p").textContent = text;
  return entry;
}

function assistantShellEl(docketNo, typing) {
  const entry = document.createElement("div");
  entry.className = "entry system";
  entry.innerHTML = `
    <div class="entry-head">${docketTag(docketNo)}<span class="tag">Assistant</span></div>
    ${typing ? `<div class="typing"><span></span><span></span><span></span></div>` : ""}
  `;
  return entry;
}

/** Full re-render of the active chat, used on load and on chat switch. */
function setTitle(text, isChat) {
  el.title.textContent = text;
  el.title.classList.toggle("is-chat", isChat);
}

function renderChat() {
  const chat = activeChat();
  el.log.innerHTML = "";

  const messages = chat?.messages ?? [];
  el.intro.hidden = messages.length > 0;
  setTitle(messages.length ? chat.title : "Your own AI-Lawyer", messages.length > 0);

  messages.forEach((msg, i) => {
    if (msg.role === "user") {
      el.log.appendChild(userEntryEl(msg.text, i + 1));
      return;
    }
    const entry = assistantShellEl(i + 1, false);
    el.log.appendChild(entry);
    renderAnswerEntry(entry, msg.payload, msg.chosen, (index) => {
      msg.chosen = index;
      saveState();
    });
  });

  el.thread.scrollTop = el.thread.scrollHeight;
}

/* ------------------------------------------------------------------ ask -- */

async function askQuestion(chat, question) {
  const entry = assistantShellEl(chat.messages.length + 1, true);
  el.log.appendChild(entry);
  scrollToBottom();

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
    const msg = { role: "assistant", payload, chosen: null };
    chat.messages.push(msg);
    touchChat(chat);

    renderAnswerEntry(entry, payload, null, (index) => {
      msg.chosen = index;
      saveState();
    });
    scrollToBottom();
  } catch (e) {
    // Transport failures aren't persisted — a reload should retry cleanly
    // rather than replay a stale error.
    entry.querySelector(".typing")?.remove();
    entry.insertAdjacentHTML("beforeend", `<p class="hedge">Sorry — ${escapeHtml(e.message)}</p>`);
    entry.removeAttribute("id");
    scrollToBottom();
  }
}

/* --------------------------------------------------------------- drawer -- */

function openDrawer() {
  el.sidebar.classList.add("open");
  el.scrim.hidden = false;
  el.drawerOpen.setAttribute("aria-expanded", "true");
}

function closeDrawer() {
  el.sidebar.classList.remove("open");
  el.scrim.hidden = true;
  el.drawerOpen.setAttribute("aria-expanded", "false");
}

/* ----------------------------------------------------------------- init -- */

CATEGORIES.forEach((c) => {
  const li = document.createElement("li");
  li.textContent = c;
  el.categoryList.appendChild(li);
});

el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = el.textarea.value.trim();
  if (!question) return;

  const chat = ensureActiveChat();
  const isFirst = chat.messages.length === 0;

  chat.messages.push({ role: "user", text: question });
  // A title the user typed themselves outranks the auto-generated one.
  if (isFirst && !chat.renamed) chat.title = titleFrom(question);
  touchChat(chat);

  el.intro.hidden = true;
  setTitle(chat.title, true);
  el.log.appendChild(userEntryEl(question, chat.messages.length));
  renderSidebar();

  el.textarea.value = "";
  el.textarea.style.height = "auto";
  el.sendBtn.disabled = true;
  scrollToBottom();

  askQuestion(chat, question).finally(() => {
    el.sendBtn.disabled = false;
    el.textarea.focus();
  });
});

el.textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    el.form.requestSubmit();
  }
});

el.textarea.addEventListener("input", () => {
  el.textarea.style.height = "auto";
  el.textarea.style.height = Math.min(el.textarea.scrollHeight, 160) + "px";
});

el.newChat.addEventListener("click", () => {
  const current = activeChat();
  // An untouched "New chat" is already a blank slate — reuse it instead of
  // stacking up empty entries every time the button is pressed.
  if (!current || current.messages.length > 0) createChat();
  el.search.value = "";
  renderSidebar();
  renderChat();
  closeDrawer();
  el.textarea.focus();
});

el.search.addEventListener("input", renderSidebar);

el.drawerOpen.addEventListener("click", openDrawer);
el.drawerClose.addEventListener("click", closeDrawer);
el.scrim.addEventListener("click", closeDrawer);

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  // Escape should peel off one layer at a time: the menu before the drawer.
  if (openMenu) closeChatMenu();
  else closeDrawer();
});

// Any click outside an open menu dismisses it. Capture phase so it still
// fires when a handler further down stops propagation.
document.addEventListener("click", (e) => {
  if (openMenu && !openMenu.el.contains(e.target)) closeChatMenu();
}, true);

// A fixed-position menu doesn't travel with its anchor, so close it rather
// than let it drift away from the row it belongs to.
el.chatList.parentElement.addEventListener("scroll", closeChatMenu, { passive: true });
window.addEventListener("resize", closeChatMenu);

loadState();
renderSidebar();
renderChat();

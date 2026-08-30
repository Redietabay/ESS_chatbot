// ESS Popup Widget — standalone script for the /widget iframe page.
// Guest Q&A through /ask_stream, PLUS full desktop feature parity:
// mic (speech-to-text), per-answer Copy / Listen / Stop / timing,
// and a mic-language selector. Guest-only originally; login was added later
// (in-popup auth panel below) but chat history was never wired up — every
// message was sent with session_id: null, so the server never persisted
// logged-in widget conversations at all. Session tracking + a History panel
// were added to fix that (search "SESSION HISTORY" below).
// Same-origin with the Flask app, so cookies/CSRF work without CORS.

const STREAM_META_SEP = '\x00__ESS_META__\x00';
const LANG_TO_TTS_LOCALE = { en: 'en-US', am: 'am-ET', om: 'om-ET' };
const MAX_UPLOAD_MB = 15; // matches app.py's UPLOAD_MAX_BYTES exactly

// UI (interface chrome) strings — separate from the mic-language selector,
// which only controls what language speech-to-text listens for.
const UI_STRINGS = {
  en: {
    welcome: '👋 Hi! Ask me about Ethiopian census data, economic surveys, livestock metrics, or inflation rates.',
    placeholder: 'Type your question...',
    listening: 'Listening...',
    attachTitle: 'Attach a PDF to ask questions about it',
    micTitle: 'Speak your question',
    closeTitle: 'Close',
    signedInAs: 'Signed in as',
    history: 'History',
    logout: 'Logout',
    newChat: '+ New Chat',
    noHistory: 'No past chats yet.',
    loading: 'Loading...',
    historyLoadError: 'Could not load history.',
    chatLoadError: 'Could not load that chat.',
    freeQuestionsLeft: 'free question', freeQuestionsLeftPlural: 'free questions', left: 'left',
    signIn: 'Sign in',
    signingIn: 'Signing in...',
    creatingAccount: 'Creating account...',
    username: 'Username',
    email: 'Email',
    password: 'Password',
    passwordMin: 'Password (min 8 characters)',
    createAccount: 'Create account',
    noAccount: "Don't have an account?",
    haveAccount: 'Already have an account?',
    register: 'Register',
    connectionError: 'Connection error — please try again.',
    somethingWrong: 'Something went wrong.',
    couldNotReach: 'Sorry, I could not reach the assistant. Please try again.',
    cancel: 'Cancel',
    freeQuestionsUsed: 'Free questions used.',
    createFreeAccount: 'Create a free account',
    or: 'or',
    guestModeUnlimited: "You're chatting as a guest — answers won't be saved."
  },
  am: {
    welcome: '👋 ሰላም! ስለ ኢትዮጵያ የህዝብ ቆጠራ፣ የኢኮኖሚ ጥናቶች፣ የከብት እርባታ መረጃ ወይም የዋጋ ግሽበት ይጠይቁኝ።',
    placeholder: 'ጥያቄዎን እዚህ ይጻፉ...',
    listening: 'በማዳመጥ ላይ...',
    attachTitle: 'ጥያቄ ለመጠየቅ PDF ያያይዙ',
    micTitle: 'ጥያቄዎን ይናገሩ',
    closeTitle: 'ዝጋ',
    signedInAs: 'የገቡት እንደ',
    history: 'ታሪክ',
    logout: 'ውጣ',
    newChat: '+ አዲስ ውይይት',
    noHistory: 'እስካሁን ያለፈ ውይይት የለም።',
    loading: 'በመጫን ላይ...',
    historyLoadError: 'ታሪኩን መጫን አልተቻለም።',
    chatLoadError: 'ያንን ውይይት መጫን አልተቻለም።',
    freeQuestionsLeft: 'ነጻ ጥያቄ', freeQuestionsLeftPlural: 'ነጻ ጥያቄዎች', left: 'ቀርተዋል',
    signIn: 'ግባ',
    signingIn: 'በመግባት ላይ...',
    creatingAccount: 'መለያ በመፍጠር ላይ...',
    username: 'የተጠቃሚ ስም',
    email: 'ኢሜይል',
    password: 'የይለፍ ቃል',
    passwordMin: 'የይለፍ ቃል (ቢያንስ 8 ቁምፊ)',
    createAccount: 'መለያ ፍጠር',
    noAccount: 'መለያ የለዎትም?',
    haveAccount: 'መለያ አለዎት?',
    register: 'ተመዝገብ',
    connectionError: 'የግንኙነት ስህተት — እባክዎ እንደገና ይሞክሩ።',
    somethingWrong: 'የሆነ ችግር ተፈጥሯል።',
    couldNotReach: 'ይቅርታ፣ ረዳቱን ማግኘት አልተቻለም። እባክዎ እንደገና ይሞክሩ።',
    cancel: 'ይቅር',
    freeQuestionsUsed: 'ነጻ ጥያቄዎች አልቀዋል።',
    createFreeAccount: 'ነጻ መለያ ይፍጠሩ',
    or: 'ወይም',
    guestModeUnlimited: 'እንደ እንግዳ እየተወያዩ ነው — መልሶች አይቀመጡም።'
  }
};
let uiLang = 'en';

const wMessages    = document.getElementById('wMessages');
const wInput       = document.getElementById('wInput');
const wSendBtn     = document.getElementById('wSendBtn');
const wCloseBtn    = document.getElementById('wCloseBtn');
const wSuggestions = document.getElementById('wSuggestions');
const wGuestBanner = document.getElementById('wGuestBanner');

let recognition;
let isListening = false;

// ═══════════════════════════════════════
// SESSION HISTORY
// ═══════════════════════════════════════
let wIsLoggedIn  = false;
const WSESSION_STORAGE_KEY = 'ess_widget_session_id';
// In-memory wSessionId used to just be `let wSessionId = null`, reset on
// every iframe reload (e.g. the host page gets refreshed) even though the
// iframe itself stays open the rest of the time — so a page refresh looked
// like "history is broken" even after login/cookie issues are fixed.
// sessionStorage is scoped to the iframe's own origin+tab, so it survives
// reloads but still clears when the tab actually closes, matching how a
// normal browser tab's in-memory state would behave.
let wSessionId = (() => {
  try {
    const saved = sessionStorage.getItem(WSESSION_STORAGE_KEY);
    return saved ? Number(saved) : null;
  } catch (e) { return null; } // storage can throw in locked-down/private contexts — degrade to in-memory only
})();

function setWSessionId(id) {
  wSessionId = id;
  try {
    if (id === null) sessionStorage.removeItem(WSESSION_STORAGE_KEY);
    else sessionStorage.setItem(WSESSION_STORAGE_KEY, String(id));
  } catch (e) { /* non-fatal — falls back to in-memory-only for this tab */ }
}

const wHistoryPanel = document.getElementById('wHistoryPanel');
let currentLanguage = 'am-ET'; // default mic language, matches desktop

document.addEventListener('DOMContentLoaded', () => {
  injectUiLangToggle();
  loadSuggestions();
  loadGuestStatus();
  setupInput();
  setupViewportFix();
  injectFileUpload();
  restoreUploadedFileChip();
  injectLangSelector();
  initSpeechRecognition();
});

// ═══════════════════════════════════════
// UI LANGUAGE TOGGLE (EN / አማርኛ)
// ═══════════════════════════════════════
function injectUiLangToggle() {
  const toggle = document.getElementById('wUiLangToggle');
  if (!toggle) return;
  toggle.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => setUiLang(btn.dataset.lang));
  });
}

function setUiLang(lang) {
  if (!UI_STRINGS[lang]) return;
  uiLang = lang;
  const t = UI_STRINGS[lang];

  const toggle = document.getElementById('wUiLangToggle');
  if (toggle) {
    toggle.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.lang === lang));
  }

  const welcomeEl = document.getElementById('wWelcomeText');
  if (welcomeEl) welcomeEl.textContent = t.welcome;

  if (!isListening) wInput.placeholder = t.placeholder;

  const attachBtn = document.getElementById('wAttachBtn');
  if (attachBtn) attachBtn.title = t.attachTitle;

  const micBtn = document.getElementById('wMicBtn');
  if (micBtn) micBtn.title = t.micTitle;

  wCloseBtn.title = t.closeTitle;

  // Re-render whichever banner/panel is currently on screen so it picks up
  // the new language too — not just the welcome text and placeholder.
  loadGuestStatus();
  if (wHistoryPanel && wHistoryPanel.style.display === 'block') renderHistoryPanel();
  if (wAuthPanel && wAuthPanel.style.display === 'block') renderAuthPanel(wAuthPanel.dataset.mode);
  if (document.getElementById('wSuggestions')) loadSuggestions();
}

// ═══════════════════════════════════════
// MOBILE VIEWPORT / KEYBOARD HANDLING
// Fixes input + send button getting hidden behind the on-screen keyboard:
// the visualViewport API reports the space actually visible above the
// keyboard, so we size the widget to that instead of the full window.
// ═══════════════════════════════════════
function setupViewportFix() {
  if (!window.visualViewport) return;
  const vv = window.visualViewport;
  const root = document.getElementById('widgetRoot');

  function apply() {
    if (root) root.style.height = vv.height + 'px';
    scrollToBottom();
  }

  vv.addEventListener('resize', apply);
  vv.addEventListener('scroll', apply);
  apply();
}

// ═══════════════════════════════════════
// FILE UPLOAD
// ═══════════════════════════════════════
function injectFileUpload() {
  const inputArea = document.querySelector('.w-input-area');
  if (!inputArea || document.getElementById('wFileInput')) return;

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.id = 'wFileInput';
  fileInput.accept = '.pdf';
  fileInput.style.display = 'none';
  fileInput.addEventListener('change', handleFileSelected);
  document.body.appendChild(fileInput);

  const attachBtn = document.createElement('button');
  attachBtn.id = 'wAttachBtn';
  attachBtn.type = 'button';
  attachBtn.className = 'w-attach-btn';
  attachBtn.title = UI_STRINGS[uiLang].attachTitle;
  attachBtn.innerHTML = '📎';
  attachBtn.addEventListener('click', () => fileInput.click());
  inputArea.insertBefore(attachBtn, wInput);
}

async function restoreUploadedFileChip() {
  try {
    const res = await fetch('/upload_status');
    const data = await res.json();
    if (data.filename) showAttachedFileChip(data.filename);
  } catch (e) { /* non-fatal */ }
}

async function handleFileSelected(e) {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;

  // Client-side validation first — instant feedback, no round trip.
  if (!/\.pdf$/i.test(file.name)) {
    showUploadToast('Only PDF files are supported.', 'error');
    return;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    showUploadToast(`"${file.name}" is over ${MAX_UPLOAD_MB}MB — please upload a smaller file.`, 'error');
    return;
  }

  const attachBtn = document.getElementById('wAttachBtn');
  if (attachBtn) { attachBtn.innerHTML = '⏳'; attachBtn.disabled = true; }
  showUploadingChip(file.name);

  try {
    const token = await getCsrfToken();
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/upload_document', {
      method: 'POST',
      headers: { 'X-CSRFToken': token },
      body: formData
    });

    if (res.status === 429) {
      const chip = document.getElementById('wUploadChip');
      if (chip) chip.remove();
      showUploadToast("You're uploading too quickly — wait a moment and try again.", 'error');
      return;
    }
    if (res.status === 413) {
      const chip = document.getElementById('wUploadChip');
      if (chip) chip.remove();
      showUploadToast(`File is too large — max ${MAX_UPLOAD_MB}MB.`, 'error');
      return;
    }
    const data = await res.json();

    if (!res.ok) {
      const chip = document.getElementById('wUploadChip');
      if (chip) chip.remove();
      showUploadToast(data.error || 'Could not process that file.', 'error');
      return;
    }
    showAttachedFileChip(data.filename, data);
    showUploadToast(`"${data.filename}" is ready — ask a question about it.`, 'success');
  } catch (err) {
    console.error('Upload failed:', err);
    const chip = document.getElementById('wUploadChip');
    if (chip) chip.remove();
    showUploadToast('Upload failed — check your connection and try again.', 'error');
  } finally {
    if (attachBtn) { attachBtn.innerHTML = '📎'; attachBtn.disabled = false; }
  }
}

function showUploadingChip(filename) {
  const inputArea = document.querySelector('.w-input-area');
  let chip = document.getElementById('wUploadChip');
  if (!chip && inputArea) {
    chip = document.createElement('div');
    chip.id = 'wUploadChip';
    inputArea.parentNode.insertBefore(chip, inputArea);
  }
  if (!chip) return;
  chip.className = 'w-upload-chip uploading';
  chip.innerHTML = `<span class="w-upload-chip-spinner"></span><span>Uploading ${escapeWidgetHtml(filename)}...</span>`;
}

function showAttachedFileChip(filename, stats) {
  let chip = document.getElementById('wUploadChip');
  const inputArea = document.querySelector('.w-input-area');
  if (!chip && inputArea) {
    chip = document.createElement('div');
    chip.id = 'wUploadChip';
    inputArea.parentNode.insertBefore(chip, inputArea);
  }
  if (!chip) return;
  chip.className = 'w-upload-chip';

  let warning = '';
  if (stats && stats.pages_empty > 0) {
    warning = ` <span class="w-upload-chip-warn">(${stats.pages_empty} page(s) unreadable)</span>`;
  }

  const safeName = escapeWidgetHtml(filename);
  chip.innerHTML = `
    <span>📄 ${safeName}</span>${warning}
    <button type="button" class="w-upload-chip-remove" title="Remove attached file">&times;</button>
  `;
  chip.querySelector('.w-upload-chip-remove').addEventListener('click', () => removeAttachedFile());
}

async function removeAttachedFile() {
  try {
    const token = await getCsrfToken();
    await fetch('/clear_upload', { method: 'POST', headers: { 'X-CSRFToken': token } });
  } catch (e) { /* non-fatal */ }
  const chip = document.getElementById('wUploadChip');
  if (chip) chip.remove();
}

// Replaces alert() — alerts inside an embedded iframe on mobile are
// jarring and sometimes get silently blocked by the host page.
function showUploadToast(msg, type) {
  const inputArea = document.querySelector('.w-input-area');
  if (!inputArea) return;
  const old = document.getElementById('wUploadToast');
  if (old) old.remove();

  const toast = document.createElement('div');
  toast.id = 'wUploadToast';
  toast.className = `w-toast ${type === 'error' ? 'w-toast-error' : 'w-toast-success'}`;
  toast.textContent = msg;
  inputArea.parentNode.insertBefore(toast, inputArea);

  setTimeout(() => { if (toast.parentNode) toast.remove(); }, type === 'error' ? 5000 : 3500);
}

// ── Close button tells the parent page (loader script) to hide the iframe ──
wCloseBtn.addEventListener('click', () => {
  stopSpeaking();
  if (window.parent) {
    window.parent.postMessage({ type: 'ess-widget-close' }, '*');
  }
});

async function getCsrfToken() {
  try {
    const res = await fetch('/csrf-token');
    const data = await res.json();
    return data.csrf_token;
  } catch (e) {
    console.error('Failed to fetch CSRF token:', e);
    return '';
  }
}

async function loadSuggestions() {
  try {
    const res = await fetch(`/suggestions?lang=${uiLang}`);
    const data = await res.json();
    if (data.suggestions && data.suggestions.length > 0) {
      wSuggestions.innerHTML = '';
      data.suggestions.slice(0, 3).forEach(q => {
        const btn = document.createElement('button');
        btn.className = 'w-suggestion-btn';
        btn.textContent = q;
        btn.addEventListener('click', () => {
          wInput.value = q;
          handleSend();
        });
        wSuggestions.appendChild(btn);
      });
    }
  } catch (e) {
    console.error('Error loading suggestions:', e);
  }
}

async function loadGuestStatus() {
  try {
    const res = await fetch('/guest_status');
    const data = await res.json();
    if (data.is_guest) {
      // The server no longer recognizes this browser as logged in (expired
      // cookie, or the cross-site cookie never made it through) — drop any
      // stale saved session id instead of sending it as an "owned" session
      // to a user the server now sees as a guest.
      if (wIsLoggedIn) setWSessionId(null);
      wIsLoggedIn = false;
      updateGuestBanner(data.remaining);
    } else if (!data.is_guest) {
      wIsLoggedIn = true;
      updateSignedInBanner(data.username);
    }
  } catch (e) { /* non-fatal */ }
}

// Shown when logged in. Logout is now a button that calls /api/logout via
// fetch and stays in the popup — previously this linked to /logout (a full
// page) which is what took people out of the widget in the first place.
function updateSignedInBanner(username) {
  hideAuthPanel();
  const t = UI_STRINGS[uiLang];
  const name = username ? username : 'account';
  wGuestBanner.innerHTML = `${t.signedInAs} ${escapeWidgetHtml(name)} · <button type="button" class="w-banner-link" id="wHistoryBtn">${t.history}</button> · <button type="button" class="w-banner-link" id="wLogoutBtn">${t.logout}</button>`;
  document.getElementById('wLogoutBtn').addEventListener('click', doLogout);
  document.getElementById('wHistoryBtn').addEventListener('click', toggleHistoryPanel);
}

async function toggleHistoryPanel() {
  if (wHistoryPanel.style.display === 'block') {
    wHistoryPanel.style.display = 'none';
    return;
  }
  wHistoryPanel.style.display = 'block';
  await renderHistoryPanel();
}

async function renderHistoryPanel() {
  const t = UI_STRINGS[uiLang];
  wHistoryPanel.innerHTML = `<div class="w-history-loading">${t.loading}</div>`;
  try {
    const res = await fetch('/get_sessions');
    const data = await res.json();
    const sessions = data.sessions || [];
    wHistoryPanel.innerHTML = '';

    const newBtn = document.createElement('button');
    newBtn.className = 'w-history-new-btn';
    newBtn.textContent = t.newChat;
    newBtn.addEventListener('click', startNewSession);
    wHistoryPanel.appendChild(newBtn);

    if (sessions.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'w-history-empty';
      empty.textContent = t.noHistory;
      wHistoryPanel.appendChild(empty);
    }
    sessions.forEach(s => {
      const item = document.createElement('button');
      item.className = 'w-history-item';
      item.innerHTML = `<span class="w-history-item-title">${escapeWidgetHtml(s.title || 'New Chat')}</span><span class="w-history-item-date">${escapeWidgetHtml(s.created_at || '')}</span>`;
      item.addEventListener('click', () => loadSessionIntoChat(s.id));
      wHistoryPanel.appendChild(item);
    });
  } catch (e) {
    wHistoryPanel.innerHTML = `<div class="w-history-empty">${UI_STRINGS[uiLang].historyLoadError}</div>`;
  }
}

function startNewSession() {
  setWSessionId(null);
  wHistoryPanel.style.display = 'none';
  wMessages.innerHTML = `<div class="w-welcome"><p>${UI_STRINGS[uiLang].welcome}</p></div><div id="wSuggestions" class="w-suggestions"></div>`;
  loadSuggestions();
}

async function loadSessionIntoChat(sessionId) {
  try {
    const res = await fetch(`/get_history/${sessionId}`);
    if (!res.ok) throw new Error('not found');
    const data = await res.json();
    setWSessionId(sessionId);
    wHistoryPanel.style.display = 'none';
    wMessages.innerHTML = '';
    (data.history || []).forEach(m => {
      const bubble = appendMessage(m.sender === 'user' ? 'user' : 'bot', m.message);
      if (m.sender !== 'user' && m.source_doc) {
        appendSourceTag(bubble, m.source_doc, m.source_page);
      }
    });
    scrollToBottom();
  } catch (e) {
    wHistoryPanel.innerHTML = `<div class="w-history-empty">${UI_STRINGS[uiLang].chatLoadError}</div>`;
    wHistoryPanel.style.display = 'block';
  }
}

function escapeWidgetHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

async function doLogout() {
  try {
    const token = await getCsrfToken();
    await fetch('/api/logout', { method: 'POST', headers: { 'X-CSRFToken': token } });
  } catch (e) { /* non-fatal — UI resets either way */ }
  setWSessionId(null);
  wIsLoggedIn = false;
  if (wHistoryPanel) wHistoryPanel.style.display = 'none';
  await loadGuestStatus();
}

// ═══════════════════════════════════════
// IN-POPUP LOGIN / REGISTER — replaces navigating to /login or /register.
// Sign in / Create account buttons in the guest banner open this panel
// instead of a link; on success we just refresh the banner, no navigation,
// so the popup (and the host page it's embedded on) never closes.
// ═══════════════════════════════════════
const wAuthPanel = document.getElementById('wAuthPanel');

function showAuthPanel(mode) {
  wAuthPanel.style.display = 'block';
  wAuthPanel.dataset.mode = mode;
  renderAuthPanel(mode);
}

function hideAuthPanel() {
  wAuthPanel.style.display = 'none';
  wAuthPanel.innerHTML = '';
}

function renderAuthPanel(mode, error) {
  const isLogin = mode === 'login';
  const t = UI_STRINGS[uiLang];
  wAuthPanel.innerHTML = `
    ${error ? `<div class="w-auth-error">${escapeWidgetHtml(error)}</div>` : ''}
    <input type="text" id="wAuthUsername" placeholder="${t.username}" autocomplete="off">
    ${isLogin ? '' : `<input type="email" id="wAuthEmail" placeholder="${t.email}" autocomplete="off">`}
    <input type="password" id="wAuthPassword" placeholder="${isLogin ? t.password : t.passwordMin}">
    <div class="w-auth-row">
      <button type="button" class="w-auth-submit" id="wAuthSubmit">${isLogin ? t.signIn : t.createAccount}</button>
      <button type="button" class="w-auth-cancel" id="wAuthCancel">${t.cancel}</button>
    </div>
    <div class="w-auth-switch">
      ${isLogin
        ? `${t.noAccount} <button type="button" id="wAuthSwitch">${t.register}</button>`
        : `${t.haveAccount} <button type="button" id="wAuthSwitch">${t.signIn}</button>`}
    </div>
  `;

  document.getElementById('wAuthCancel').addEventListener('click', hideAuthPanel);
  document.getElementById('wAuthSwitch').addEventListener('click', () => showAuthPanel(isLogin ? 'register' : 'login'));
  document.getElementById('wAuthSubmit').addEventListener('click', () => submitAuth(mode));

  const pwField = document.getElementById('wAuthPassword');
  pwField.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitAuth(mode);
  });
}

async function submitAuth(mode) {
  const username = document.getElementById('wAuthUsername').value.trim();
  const password = document.getElementById('wAuthPassword').value.trim();
  const emailField = document.getElementById('wAuthEmail');
  const email = emailField ? emailField.value.trim() : undefined;

  const t = UI_STRINGS[uiLang];
  const submitBtn = document.getElementById('wAuthSubmit');
  submitBtn.disabled = true;
  submitBtn.textContent = mode === 'login' ? t.signingIn : t.creatingAccount;

  try {
    const token = await getCsrfToken();
    const payload = mode === 'login' ? { username, password } : { username, email, password };
    const res = await fetch(mode === 'login' ? '/api/login' : '/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': token },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) {
      renderAuthPanel(mode, data.error || t.somethingWrong);
      return;
    }

    hideAuthPanel();
    updateSignedInBanner(data.username);
  } catch (e) {
    console.error('Auth failed:', e);
    renderAuthPanel(mode, t.connectionError);
  }
}

// Sign-in/register in the guest banner now open the in-popup auth panel
// (see showAuthPanel below) instead of linking anywhere — the whole flow
// stays inside the iframe, so the host page and popup never close.
function updateGuestBanner(remaining) {
  hideAuthPanel();
  const t = UI_STRINGS[uiLang];
  if (remaining === null || remaining === undefined) {
    wGuestBanner.innerHTML = `${t.guestModeUnlimited} <button type="button" class="w-banner-link" id="wSignInBtn">${t.signIn}</button> ${t.or} <button type="button" class="w-banner-link" id="wRegisterBtn">${t.createFreeAccount}</button>`;
  } else if (remaining <= 0) {
    wGuestBanner.innerHTML = `${t.freeQuestionsUsed} <button type="button" class="w-banner-link" id="wRegisterBtn">${t.createFreeAccount}</button> ${t.or} <button type="button" class="w-banner-link" id="wSignInBtn">${t.signIn}</button>`;
  } else {
    wGuestBanner.innerHTML = `${remaining} ${remaining === 1 ? t.freeQuestionsLeft : t.freeQuestionsLeftPlural} ${t.left} · <button type="button" class="w-banner-link" id="wSignInBtn">${t.signIn}</button>`;
  }
  const signInBtn = document.getElementById('wSignInBtn');
  if (signInBtn) signInBtn.addEventListener('click', () => showAuthPanel('login'));
  const registerBtn = document.getElementById('wRegisterBtn');
  if (registerBtn) registerBtn.addEventListener('click', () => showAuthPanel('register'));
}

function setupInput() {
  wInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
  wInput.addEventListener('input', () => {
    wInput.style.height = 'auto';
    wInput.style.height = Math.min(wInput.scrollHeight, 90) + 'px';
  });
  // On mobile, the keyboard opening can leave the textarea scrolled out of
  // view before visualViewport catches up — nudge it into view explicitly.
  wInput.addEventListener('focus', () => {
    setTimeout(() => wInput.scrollIntoView({ block: 'end', behavior: 'smooth' }), 300);
  });
  wSendBtn.addEventListener('click', handleSend);
}

function scrollToBottom() {
  wMessages.scrollTop = wMessages.scrollHeight;
}

// Markdown table (rows of "| a | b |" with a "|---|---|" separator row)
// -> a real HTML <table>. Returns null if `start` isn't the start of a
// table; otherwise the table's HTML plus how many lines it consumed, so
// the caller can skip past them.
function _consumeMarkdownTable(lines, start) {
  const isRow = i => i < lines.length && lines[i].trim().startsWith('|');
  const isSep = i => isRow(i) && /^\|?\s*:?-{2,}/.test(lines[i].trim().replace(/^\|/, ''));

  if (!isRow(start) || !isSep(start + 1)) return null;

  const toCells = line => line.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());

  const header = toCells(lines[start]);
  let i = start + 2; // skip header + separator rows
  const bodyRows = [];
  while (isRow(i) && !isSep(i)) {
    bodyRows.push(toCells(lines[i]));
    i++;
  }

  let html = '<div class="md-table-wrap"><table class="md-table"><thead><tr>';
  header.forEach(c => { html += `<th>${c}</th>`; });
  html += '</tr></thead><tbody>';
  bodyRows.forEach(row => {
    html += '<tr>';
    header.forEach((_, ci) => { html += `<td>${row[ci] !== undefined ? row[ci] : ''}</td>`; });
    html += '</tr>';
  });
  html += '</tbody></table></div>';

  return { html, linesConsumed: i - start };
}

function formatMarkdown(text) {
  if (!text) return '';
  let clean = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Pull fenced code blocks out into placeholders BEFORE the final
  // line-by-line split below. Previously the ```block``` -> <pre><code>
  // replacement ran first but the newlines *inside* the block were still
  // there, so the later `.split('\n').map(...<p>)` step chopped a single
  // multi-line code block into one <p> per line, destroying its structure
  // (and putting <pre><code> tags around only the first line). Stashing
  // each block as a one-line placeholder keeps it intact through the split,
  // then we swap the real HTML back in afterwards.
  const codeBlocks = [];
  clean = clean.replace(/```([\s\S]*?)```/g, (match, code) => {
    codeBlocks.push(`<pre><code>${code}</code></pre>`);
    return `\u0000CODEBLOCK${codeBlocks.length - 1}\u0000`;
  });

  clean = clean.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  clean = clean.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Tables get the same placeholder treatment as code blocks, for the same
  // reason: a multi-line table would otherwise get one <p> per row.
  const srcLines = clean.split('\n');
  const outLines = [];
  for (let i = 0; i < srcLines.length; i++) {
    const table = _consumeMarkdownTable(srcLines, i);
    if (table) {
      codeBlocks.push(table.html);
      outLines.push(`\u0000CODEBLOCK${codeBlocks.length - 1}\u0000`);
      i += table.linesConsumed - 1;
      continue;
    }
    outLines.push(srcLines[i]);
  }

  let html = outLines.map(line => line.trim() ? `<p>${line}</p>` : '').join('');

  // Swap placeholders back in, unwrapping the <p> tag the split loop added
  // around the placeholder line — <pre>/<table> are block elements and
  // shouldn't nest inside <p>.
  html = html.replace(/<p>\u0000CODEBLOCK(\d+)\u0000<\/p>/g, (match, i) => codeBlocks[Number(i)]);

  return html;
}

// ═══════════════════════════════════════
// MESSAGE BUBBLES (with Copy / Listen / Stop on bot answers)
// ═══════════════════════════════════════
function appendMessage(sender, text) {
  const welcome = wMessages.querySelector('.w-welcome');
  if (welcome) welcome.remove();
  if (wSuggestions) wSuggestions.remove();

  const div = document.createElement('div');
  div.className = `w-msg ${sender}`;

  const textSpan = document.createElement('span');
  textSpan.className = 'w-bubble-text';
  textSpan.innerHTML = formatMarkdown(text);
  div.appendChild(textSpan);

  if (sender === 'bot') {
    const actions = document.createElement('div');
    actions.className = 'w-msg-actions';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'w-action-btn w-copy-btn';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => {
      const target = textSpan.innerText || textSpan.textContent;
      fallbackCopyText(target, copyBtn);
    });

    const speakBtn = document.createElement('button');
    speakBtn.className = 'w-action-btn w-speak-btn';
    speakBtn.innerHTML = '🔊';
    speakBtn.title = 'Listen';
    speakBtn.addEventListener('click', () => {
      const target = textSpan.innerText || textSpan.textContent;
      speakText(target, div.dataset.answerLang, speakBtn);
    });

    const stopBtn = document.createElement('button');
    stopBtn.className = 'w-action-btn w-stop-btn';
    stopBtn.innerHTML = '🔇';
    stopBtn.title = 'Stop';
    stopBtn.addEventListener('click', stopSpeaking);

    actions.appendChild(copyBtn);
    actions.appendChild(speakBtn);
    actions.appendChild(stopBtn);
    div.appendChild(actions);
  }

  wMessages.appendChild(div);
  scrollToBottom();
  return div;
}

function appendSourceTag(bubble, source, page) {
  if (!source || source === 'Unknown') return;
  if (bubble.querySelector('.w-source-tag')) return;
  const clean = source.substring(source.lastIndexOf('/') + 1);
  const tag = document.createElement('div');
  tag.className = 'w-source-tag';
  tag.textContent = page ? `Source: ${clean} (p. ${page})` : `Source: ${clean}`;
  bubble.appendChild(tag);
}

// Mirrors desktop's appendTimingTag — was missing here, which is why the
// popup never showed "Answered in Xs" the way the desktop chat does. The
// backend already sends meta.elapsed on both endpoints; this just renders it.
function appendTimingTag(bubble, elapsedSeconds) {
  if (bubble.querySelector('.w-timing-tag')) return;
  const tag = document.createElement('div');
  tag.className = 'w-timing-tag';
  tag.textContent = `⏱ Answered in ${elapsedSeconds}s`;
  bubble.appendChild(tag);
}

function showTyping() {
  const el = document.createElement('div');
  el.className = 'w-msg bot w-typing';
  el.innerHTML = '<span></span><span></span><span></span>';
  wMessages.appendChild(el);
  scrollToBottom();
  return el;
}

async function handleSend() {
  const question = wInput.value.trim();
  if (!question) return;

  stopSpeaking();

  wInput.value = '';
  wInput.style.height = 'auto';
  wSendBtn.disabled = true;

  appendMessage('user', question);
  const typing = showTyping();

  try {
    const token = await getCsrfToken();

    // Logged-in users get a real session so the conversation is persisted
    // and shows up in History. Guests keep session_id: null (unchanged) —
    // the server doesn't save unauthenticated chats anyway.
    if (wIsLoggedIn && !wSessionId) {
      try {
        const sRes = await fetch('/new_session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-CSRFToken': token },
          body: JSON.stringify({ title: question.slice(0, 50) })
        });
        const sData = await sRes.json();
        if (sData.success) setWSessionId(sData.session_id);
      } catch (e) { /* if this fails, fall through and send session_id: null below */ }
    }

    const response = await fetch('/ask_stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': token
      },
      body: JSON.stringify({ question: question, session_id: wIsLoggedIn ? wSessionId : null, ui_lang: uiLang })
    });

    typing.remove();

    if (!response.ok) {
      if (response.status === 403) {
        try {
          const errData = await response.json();
          if (errData.code === 'guest_limit_reached') {
            appendMessage('bot', errData.error);
            updateGuestBanner(0);
            wSendBtn.disabled = false;
            return;
          }
        } catch (e) { /* fall through */ }
      }
      appendMessage('bot', UI_STRINGS[uiLang].couldNotReach);
      wSendBtn.disabled = false;
      return;
    }

    const bubble = appendMessage('bot', '');
    const bubbleText = bubble.querySelector('.w-bubble-text');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let metaReceived = false;

    // The read loop itself used to have no try/catch, so a network drop
    // mid-stream (reader.read() throwing) skipped straight to the outer
    // catch with whatever partial buffer was already rendered — leaving a
    // half-written answer with no source tag, no timing tag, and no
    // indication anything went wrong. Wrapping the loop lets us close out
    // the bubble cleanly (finish formatting what we got, note the drop)
    // instead of leaving it looking like a normal completed answer.
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        if (buffer.includes(STREAM_META_SEP)) {
          const parts = buffer.split(STREAM_META_SEP);
          bubbleText.innerHTML = formatMarkdown(parts[0]);
          try {
            const meta = JSON.parse(parts[1]);
            metaReceived = true;
            appendSourceTag(bubble, meta.source, meta.page);
            if (typeof meta.elapsed === 'number') {
              appendTimingTag(bubble, meta.elapsed);
            }
            if (meta.language) bubble.dataset.answerLang = meta.language; // en | am | om
            if (!wIsLoggedIn && meta.guest_remaining !== undefined) {
              updateGuestBanner(meta.guest_remaining);
            }
          } catch (e) {
            console.error('Meta parse error:', e);
          }
        } else if (!metaReceived) {
          bubbleText.innerHTML = formatMarkdown(buffer);
        }
        scrollToBottom();
      }
    } catch (streamErr) {
      console.error('Stream reader dropped mid-response:', streamErr);
      if (!bubbleText.textContent.trim()) {
        bubbleText.innerHTML = formatMarkdown(UI_STRINGS[uiLang].connectionError);
      } else {
        const notice = document.createElement('div');
        notice.className = 'w-timing-tag';
        notice.textContent = uiLang === 'am' ? '⚠ ግንኙነት ተቋርጧል' : '⚠ Connection dropped — answer may be incomplete';
        bubble.appendChild(notice);
      }
    }
  } catch (e) {
    console.error(e);
    typing.remove();
    appendMessage('bot', 'An unexpected connection issue occurred.');
  } finally {
    wSendBtn.disabled = false;
  }
}

// ═══════════════════════════════════════
// CLIPBOARD COPY
// ═══════════════════════════════════════
function fallbackCopyText(text, buttonElement) {
  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.style.position = 'absolute';
  textArea.style.left = '-9999px';
  textArea.style.top = '0';
  document.body.appendChild(textArea);
  textArea.select();
  textArea.setSelectionRange(0, 99999);

  try {
    const ok = document.execCommand('copy');
    buttonElement.textContent = ok ? 'Copied!' : 'Failed!';
  } catch (err) {
    console.error('Fallback copy failed', err);
    buttonElement.textContent = 'Error!';
  }

  document.body.removeChild(textArea);
  setTimeout(() => { buttonElement.textContent = 'Copy'; }, 1800);
}

// ═══════════════════════════════════════
// TEXT-TO-SPEECH
// Hybrid strategy — mirrors chat.js:
//  - English: free, instant, zero-network Web Speech API (works everywhere).
//  - Amharic / Afaan Oromoo: almost no browser/OS ships an am-ET or om-ET
//    voice. Setting utterance.lang to one that isn't installed does NOT
//    fail loudly — it silently substitutes a default (usually English)
//    voice, which is the "wrong sound" bug. So for am/om we call the
//    backend's /tts endpoint (server-side gTTS, free) for real audio, and
//    only fall back to the broken browser voice if that request fails.
// ═══════════════════════════════════════
const TTS_ENDPOINT = '/tts';
let _activeTtsAudio = null;

function stopSpeaking() {
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  if (_activeTtsAudio) {
    _activeTtsAudio.pause();
    _activeTtsAudio.src = '';
    _activeTtsAudio = null;
  }
}

// Chrome (desktop and Android especially) returns [] from getVoices() until
// the async 'voiceschanged' event fires, sometimes well after page load.
// Returning true on an empty list let English through immediately (fine,
// since a browser voice is very likely there) but the SAME empty-list check
// also silently OK'd non-English locales the first time this ran — before
// voices had actually loaded, so hasBrowserVoiceFor('am') could return true
// with no am-ET voice installed at all, skipping the server TTS fallback
// and playing wrong-language audio. Now we only take the "assume yes"
// shortcut for English specifically; every other locale waits for a real
// (possibly empty) voice list before answering.
function hasBrowserVoiceFor(localePrefix) {
  if (!('speechSynthesis' in window)) return false;
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return localePrefix === 'en'; // English is near-universal; other locales must be confirmed
  return voices.some(v => v.lang && v.lang.toLowerCase().startsWith(localePrefix));
}

async function speakText(text, answerLang, btnEl) {
  stopSpeaking();

  const cleanText = text.replace(/[*_`#]/g, '').replace(/<\/?[^>]+(>|$)/g, '').trim();
  if (!cleanText) return;

  const lang = (answerLang && LANG_TO_TTS_LOCALE[answerLang])
    ? answerLang
    : (/[\u1200-\u137F]/.test(cleanText) ? 'am' : 'en');

  if (lang === 'en' && hasBrowserVoiceFor('en')) {
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = 'en-US';
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
    return;
  }

  const setBtnLoading = (loading) => {
    if (!btnEl) return;
    if (loading) {
      btnEl.dataset.origLabel = btnEl.innerHTML;
      btnEl.innerHTML = '⏳';
      btnEl.disabled = true;
    } else {
      btnEl.innerHTML = btnEl.dataset.origLabel || '🔊';
      btnEl.disabled = false;
    }
  };

  // Small helper so a single dropped connection (the observed failure mode —
  // "Failed to connect" — is very often transient) doesn't immediately fall
  // back to a wrong-language browser voice or a bare console error. One
  // quick retry before giving up.
  const fetchTts = () => fetch(`${TTS_ENDPOINT}?lang=${lang}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: cleanText })
  });

  setBtnLoading(true);
  try {
    let res;
    try {
      res = await fetchTts();
    } catch (networkErr) {
      // Connection-level failure (DNS/refused/etc, not an HTTP error status)
      // — retry once after a short pause before giving up on the server.
      await new Promise(r => setTimeout(r, 600));
      res = await fetchTts();
    }

    if (!res.ok) {
      let detail = null;
      try { detail = await res.json(); } catch (e) { /* not JSON */ }
      if (detail && detail.unsupported) {
        showMicStatus(detail.error || 'Voice for this language is not available yet.');
        return; // don't fall back to a garbled browser voice for om
      }
      throw new Error((detail && detail.error) || `TTS request failed (${res.status})`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    _activeTtsAudio = new Audio(url);
    _activeTtsAudio.play();
    _activeTtsAudio.onended = () => URL.revokeObjectURL(url);
  } catch (e) {
    console.error('Server TTS failed after retry:', e);
    // Only fall back to the browser's built-in voice if one is actually
    // installed for this language — for am/om that's almost never true, and
    // silently substituting an English voice for Amharic text is the
    // "wrong sound" bug this whole server-TTS path exists to avoid. When no
    // matching voice exists, tell the person plainly instead of guessing.
    const browserLocale = LANG_TO_TTS_LOCALE[lang] || 'en-US';
    if ('speechSynthesis' in window && hasBrowserVoiceFor(browserLocale.split('-')[0])) {
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.lang = browserLocale;
      window.speechSynthesis.speak(utterance);
    } else {
      showMicStatus('Voice playback is temporarily unavailable. Please try again in a moment.');
    }
  } finally {
    setBtnLoading(false);
  }
}

// ═══════════════════════════════════════
// MIC LANGUAGE SELECTOR + SPEECH-TO-TEXT
// ═══════════════════════════════════════
function injectLangSelector() {
  const inputArea = document.querySelector('.w-input-area');
  if (!inputArea || document.getElementById('wLangSelector')) return;

  const wrap = document.createElement('div');
  wrap.id = 'wLangSelectorWrap';
  wrap.innerHTML = `
    <select id="wLangSelector" class="w-lang-select" title="Mic language">
      <option value="en-US">EN</option>
      <option value="am-ET" selected>አማ</option>
    </select>
  `;
  inputArea.parentNode.insertBefore(wrap, inputArea);

  const sel = document.getElementById('wLangSelector');
  sel.addEventListener('change', (e) => {
    currentLanguage = e.target.value;
    if (recognition) recognition.lang = currentLanguage;
  });
  currentLanguage = sel.value;
}

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  // iOS Safari (and some in-app browsers) have no Web Speech API at all.
  // Don't inject a mic button that can never work — that's a dead control
  // with no feedback, which is worse than no button.
  if (!SpeechRecognition) {
    console.warn('Speech recognition is not supported in this browser.');
    return;
  }

  const inputArea = document.querySelector('.w-input-area');
  if (inputArea && !document.getElementById('wMicBtn')) {
    const micBtn = document.createElement('button');
    micBtn.type = 'button';
    micBtn.id = 'wMicBtn';
    micBtn.className = 'w-mic-btn';
    micBtn.innerHTML = '🎙️';
    micBtn.title = UI_STRINGS[uiLang].micTitle;
    inputArea.insertBefore(micBtn, wInput);
    micBtn.addEventListener('click', toggleMic);
  }

  recognition = new SpeechRecognition();
  // continuous=true: keep listening until the user clicks the mic button
  // again, instead of the browser auto-stopping after a short pause of
  // silence. Start/stop is fully user-controlled via toggleMic().
  recognition.continuous = true;
  recognition.lang = currentLanguage;
  recognition.interimResults = false;

  recognition.onstart = () => {
    isListening = true;
    const micBtn = document.getElementById('wMicBtn');
    if (micBtn) { micBtn.classList.add('recording'); micBtn.innerHTML = '🔴'; }
    wInput.placeholder = UI_STRINGS[uiLang].listening;
    showMicStatus('');
  };

  recognition.onend = () => {
    isListening = false;
    const micBtn = document.getElementById('wMicBtn');
    if (micBtn) { micBtn.classList.remove('recording'); micBtn.innerHTML = '🎙️'; }
    wInput.placeholder = UI_STRINGS[uiLang].placeholder;
  };

  // Just fill the input box — never auto-send. The user reviews the
  // transcribed text and sends it themselves (Send button or Enter).
  // With continuous=true, results accumulate across the whole session,
  // so we re-join every recognized chunk each time instead of only
  // keeping the first one.
  recognition.onresult = (event) => {
    let transcript = '';
    for (let i = 0; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    wInput.value = transcript;
    wInput.focus();
  };

  // Mobile browsers hit these far more than desktop (permission prompts,
  // flaky data connections) — surface them inline instead of just logging.
  recognition.onerror = (event) => {
    console.error('Speech Recognition Error:', event.error);
    const messages = {
      'not-allowed': 'Microphone access denied — allow it in your browser settings to use voice input.',
      'service-not-allowed': 'Microphone access denied — allow it in your browser settings to use voice input.',
      'audio-capture': 'No microphone found on this device.',
      'no-speech': "Didn't catch that — tap the mic and try again.",
      'network': 'Voice input needs an internet connection.'
    };
    showMicStatus(messages[event.error] || 'Voice input failed — please type your question instead.');
    recognition.stop();
  };
}

function showMicStatus(msg) {
  const el = document.getElementById('wMicStatus');
  if (!el) return;
  el.textContent = msg;
  clearTimeout(showMicStatus._t);
  if (msg) {
    showMicStatus._t = setTimeout(() => { el.textContent = ''; }, 4000);
  }
}

function toggleMic() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return; // the button wouldn't exist in this case anyway

  if (isListening) {
    recognition.stop();
    return;
  }

  // Mobile browsers require the mic permission prompt to originate directly
  // from this click handler — no awaited calls before .start().
  recognition.lang = currentLanguage;
  try {
    recognition.start();
  } catch (err) {
    console.error('Mic start failed:', err);
    showMicStatus('Could not start voice input — please try again.');
  }
}
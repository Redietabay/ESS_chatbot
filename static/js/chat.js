let currentSessionId = null;

// Marker the backend appends after the streamed answer text
const STREAM_META_SEP = '\x00__ESS_META__\x00';
const EMPTY_SESSIONS_HTML = '<div class="empty-sessions">No active sessions found.<br>Start a new chat below.</div>';

let recognition;
let isListening = false;
let currentLanguage = 'en-US'; // Default Language

// UI (interface chrome) strings — separate from currentLanguage above,
// which only controls what language speech-to-text listens for.
const UI_STRINGS = {
    en: {
        welcomeTitle: 'እንኳን ደህና መጡ! 👋',
        welcomeSubtitle: 'Ask me anything about Ethiopian census data, economic surveys, livestock metrics, or inflation rates.',
        placeholder: 'Type your data question here... (Press Enter to send)',
        listening: 'Listening... Speak now!',
        attachTitle: 'Attach a PDF to ask questions about it',
        micTitle: 'Speak your question'
    },
    am: {
        welcomeTitle: 'እንኳን ደህና መጡ! 👋',
        welcomeSubtitle: 'ስለ ኢትዮጵያ የህዝብ ቆጠራ፣ የኢኮኖሚ ጥናቶች፣ የከብት እርባታ መረጃ ወይም የዋጋ ግሽበት ይጠይቁኝ።',
        placeholder: 'ጥያቄዎን እዚህ ይጻፉ... (ለመላክ Enter ይጫኑ)',
        listening: 'በማዳመጥ ላይ... አሁን ይናገሩ!',
        attachTitle: 'ጥያቄ ለመጠየቅ PDF ያያይዙ',
        micTitle: 'ጥያቄዎን ይናገሩ'
    }
};
let uiLang = 'en';

function injectUiLangToggle() {
    const toggle = document.getElementById('uiLangToggle');
    if (!toggle) return;
    toggle.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => setUiLang(btn.dataset.lang));
    });
}

function setUiLang(lang) {
    if (!UI_STRINGS[lang]) return;
    uiLang = lang;
    const t = UI_STRINGS[lang];

    const toggle = document.getElementById('uiLangToggle');
    if (toggle) {
        toggle.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.lang === lang));
    }

    const titleEl = document.getElementById('welcomeTitle');
    if (titleEl) titleEl.textContent = t.welcomeTitle;
    const subtitleEl = document.getElementById('welcomeSubtitle');
    if (subtitleEl) subtitleEl.textContent = t.welcomeSubtitle;

    if (!isListening) questionInput.placeholder = t.placeholder;

    const attachBtn = document.getElementById('attachBtn');
    if (attachBtn) attachBtn.title = t.attachTitle;

    const micBtn = document.getElementById('micBtn');
    if (micBtn) micBtn.title = t.micTitle;

    // Refresh the suggestion chips in the new language too (only relevant
    // on the still-empty welcome screen; a no-op once a chat has started).
    if (document.getElementById('suggestionsGrid')) loadSuggestions();
}

// Replaces alert() for upload/mic feedback — alert() blocks the whole tab
// and looks broken on mobile; an inline toast doesn't interrupt typing.
function showToast(msg, type) {
    const inputArea = document.querySelector('.input-area');
    if (!inputArea) return;
    const old = document.getElementById('essToast');
    if (old) old.remove();

    const toast = document.createElement('div');
    toast.id = 'essToast';
    toast.className = `toast ${type === 'error' ? 'toast-error' : 'toast-success'}`;
    toast.textContent = msg;
    inputArea.insertBefore(toast, inputArea.firstChild);

    setTimeout(() => { if (toast.parentNode) toast.remove(); }, type === 'error' ? 5000 : 3500);
}

function showMicStatus(msg) {
    const el = document.getElementById('micStatus');
    if (!el) return;
    el.textContent = msg;
    clearTimeout(showMicStatus._t);
    if (msg) {
        showMicStatus._t = setTimeout(() => { el.textContent = ''; }, 4000);
    }
}

// Dynamic CSRF fetcher
async function getCsrfToken() {
    try {
        const response = await fetch('/csrf-token');
        const data = await response.json();
        return data.csrf_token;
    } catch (e) {
        console.error("Failed to fetch CSRF token:", e);
        return '';
    }
}

const messagesArea = document.getElementById('messagesArea');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const sessionsList = document.getElementById('sessionsList');
const chatTitle = document.getElementById('chatTitle');
const suggestionsGrid = document.getElementById('suggestionsGrid');

document.addEventListener('DOMContentLoaded', () => {
    injectUiLangToggle();
    loadSuggestions();
    setupSessionClickHandlers();
    setupInputListeners();
    injectLanguageSelector(); // Inject Amharic / Afaan Oromoo / English selector
    initSpeechRecognition();
    loadChartJS(); // Load Chart.js dynamically to save loading time
    injectFileUpload(); // Inject the 📎 attach-document button + chip
    restoreUploadedFileChip(); // Show the chip again if a doc survived a page reload
    if (window.IS_GUEST) initGuestBanner();
});

// ═══════════════════════════════════════
// FILE UPLOAD — attach a PDF and ask questions about it directly
// (server-side text/table/OCR extraction happens in /upload_document;
// see pdf_extract.py). While a file is attached, /ask_stream answers from
// it instead of the ESS corpus — see the chip's remove (×) button to
// go back to normal ESS Q&A.
// ═══════════════════════════════════════
let attachedFilename = null;

function injectFileUpload() {
    const inputRow = document.querySelector('.input-row');
    if (!inputRow || document.getElementById('fileUploadInput')) return;

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'fileUploadInput';
    fileInput.accept = '.pdf';
    fileInput.style.display = 'none';
    fileInput.addEventListener('change', handleFileSelected);
    document.body.appendChild(fileInput);

    const attachBtn = document.createElement('button');
    attachBtn.id = 'attachBtn';
    attachBtn.type = 'button';
    attachBtn.title = 'Attach a PDF to ask questions about it';
    attachBtn.className = 'attach-btn';
    attachBtn.innerHTML = '📎';
    attachBtn.addEventListener('click', () => fileInput.click());
    inputRow.insertBefore(attachBtn, questionInput);
}

async function restoreUploadedFileChip() {
    try {
        const res = await fetch('/upload_status');
        const data = await res.json();
        if (data.filename) {
            attachedFilename = data.filename;
            showAttachedFileChip(data.filename);
        }
    } catch (e) { /* non-fatal */ }
}

const MAX_UPLOAD_MB = 15; // matches app.py's UPLOAD_MAX_BYTES exactly

async function handleFileSelected(e) {
    const file = e.target.files[0];
    e.target.value = ''; // allow re-selecting the same file later
    if (!file) return;

    // Client-side validation first — instant feedback, no round trip.
    if (!/\.pdf$/i.test(file.name)) {
        showToast('Only PDF files are supported.', 'error');
        return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        showToast(`"${file.name}" is over ${MAX_UPLOAD_MB}MB — please upload a smaller file.`, 'error');
        return;
    }

    const attachBtn = document.getElementById('attachBtn');
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
            const chip = document.getElementById('uploadChip');
            if (chip) chip.remove();
            showToast("You're uploading too quickly — wait a moment and try again.", 'error');
            return;
        }
        if (res.status === 413) {
            const chip = document.getElementById('uploadChip');
            if (chip) chip.remove();
            showToast(`File is too large — max ${MAX_UPLOAD_MB}MB.`, 'error');
            return;
        }
        const data = await res.json();

        if (!res.ok) {
            const chip = document.getElementById('uploadChip');
            if (chip) chip.remove();
            showToast(data.error || 'Could not process that file.', 'error');
            return;
        }

        attachedFilename = data.filename;
        showAttachedFileChip(data.filename, data);
        showToast(`"${data.filename}" is ready — ask a question about it.`, 'success');
    } catch (err) {
        console.error('Upload failed:', err);
        const chip = document.getElementById('uploadChip');
        if (chip) chip.remove();
        showToast('Upload failed — check your connection and try again.', 'error');
    } finally {
        if (attachBtn) { attachBtn.innerHTML = '📎'; attachBtn.disabled = false; }
    }
}

function showUploadingChip(filename) {
    let chip = document.getElementById('uploadChip');
    const inputArea = document.querySelector('.input-area');
    if (!chip && inputArea) {
        chip = document.createElement('div');
        chip.id = 'uploadChip';
        inputArea.insertBefore(chip, inputArea.firstChild);
    }
    if (!chip) return;
    chip.className = 'upload-chip uploading';
    chip.innerHTML = `<span class="upload-chip-spinner"></span><span>Uploading ${escapeHtml(filename)}...</span>`;
}

function showAttachedFileChip(filename, stats) {
    let chip = document.getElementById('uploadChip');
    const inputArea = document.querySelector('.input-area');
    if (!chip && inputArea) {
        chip = document.createElement('div');
        chip.id = 'uploadChip';
        chip.className = 'upload-chip';
        inputArea.insertBefore(chip, inputArea.firstChild);
    }
    if (!chip) return;
    chip.className = 'upload-chip';

    let warning = '';
    if (stats && stats.pages_empty > 0) {
        warning = ` <span class="upload-chip-warn">(${stats.pages_empty} page(s) unreadable${stats.ocr_available ? '' : ' — OCR unavailable'})</span>`;
    }
    if (stats && stats.truncated) {
        warning += ` <span class="upload-chip-warn">(document was long — only the first pages were read)</span>`;
    }

    chip.innerHTML = `
        <span class="upload-chip-icon">📄</span>
        <span class="upload-chip-name">${escapeHtml(filename)}</span>${warning}
        <button type="button" class="upload-chip-remove" title="Remove attached file">&times;</button>
    `;
    chip.querySelector('.upload-chip-remove').addEventListener('click', removeAttachedFile);
}

async function removeAttachedFile() {
    try {
        const token = await getCsrfToken();
        await fetch('/clear_upload', {
            method: 'POST',
            headers: { 'X-CSRFToken': token }
        });
    } catch (e) { /* non-fatal — chip removal proceeds either way */ }
    attachedFilename = null;
    const chip = document.getElementById('uploadChip');
    if (chip) chip.remove();
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// GUEST QUOTA BANNER
function initGuestBanner() {
    const chatHeader = document.querySelector('.chat-header');
    if (!chatHeader || document.getElementById('guestBanner')) return;
    const banner = document.createElement('div');
    banner.id = 'guestBanner';
    banner.style.cssText = 'font-size:0.8rem;color:#888;margin-top:4px;';
    chatHeader.appendChild(banner);

    fetch('/guest_status').then(r => r.json()).then(data => {
        if (data.is_guest && typeof data.remaining === 'number') {
            updateGuestBanner(data.remaining, data.limit);
        }
    }).catch(() => {});
}

function updateGuestBanner(remaining, limit) {
    const banner = document.getElementById('guestBanner');
    if (!banner) return;
    if (remaining <= 0) {
        banner.innerHTML = `You've used all your free guest questions. <a href="/login">Sign in</a> or <a href="/register">register</a> to keep chatting.`;
    } else {
        banner.textContent = `Guest mode: ${remaining} free question${remaining === 1 ? '' : 's'} left (unsaved). `;
    }
}

function showGuestLimitPrompt(message) {
    // Build as real DOM nodes so links render clickably.
    // Do NOT pass HTML through formatMarkdown() — it escapes < > for XSS safety.
    const cleanMsg = (message || "You've used your free guest questions.")
        .replace(/<[^>]*>/g, '')  // strip any accidental tags from backend text
        .trim();

    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg bot';

    const textSpan = document.createElement('span');
    textSpan.className = 'bubble-text';

    const p = document.createElement('p');
    p.appendChild(document.createTextNode(cleanMsg + ' '));

    const loginLink = document.createElement('a');
    loginLink.href = '/login';
    loginLink.textContent = 'Sign in';
    loginLink.style.color = '#4fc3f7';
    loginLink.style.textDecoration = 'underline';

    p.appendChild(loginLink);
    p.appendChild(document.createTextNode(' or '));

    const regLink = document.createElement('a');
    regLink.href = '/register';
    regLink.textContent = 'create a free account';
    regLink.style.color = '#4fc3f7';
    regLink.style.textDecoration = 'underline';

    p.appendChild(regLink);
    p.appendChild(document.createTextNode(' to keep asking.'));

    textSpan.appendChild(p);
    msgDiv.appendChild(textSpan);

    // Remove welcome/suggestions if still visible
    const welcome = messagesArea.querySelector('.welcome-box');
    if (welcome) welcome.remove();
    const grid = messagesArea.querySelector('.suggestions-grid');
    if (grid) grid.remove();

    messagesArea.appendChild(msgDiv);
    scrollToBottom();
    updateGuestBanner(0, null);
}

// 1. DYNAMIC SUGGESTIONS LOADER
async function loadSuggestions() {
    if (!suggestionsGrid) return;
    try {
        const res = await fetch(`/suggestions?lang=${uiLang}`);
        const data = await res.json();
        if (data.suggestions && data.suggestions.length > 0) {
            suggestionsGrid.innerHTML = '';
            data.suggestions.slice(0, 4).forEach(q => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.textContent = q;
                btn.addEventListener('click', () => {
                    questionInput.value = q;
                    questionInput.focus();
                    handleSend();
                });
                suggestionsGrid.appendChild(btn);
            });
        }
    } catch (e) {
        console.error("Error loading suggestions:", e);
    }
}

// 2. CHAT HISTORY
function setupSessionClickHandlers() {
    sessionsList.addEventListener('click', async (e) => {
        const sessionItem = e.target.closest('.session-item');
        const deleteBtn = e.target.closest('.delete-session');

        if (!sessionItem) return;

        const sessionId = parseInt(sessionItem.dataset.id);

        if (deleteBtn) {
            e.stopPropagation();
            await handleDeleteSession(sessionId);
            return;
        }

        stopSpeaking();

        await loadChatHistory(sessionId);
    });
}

async function loadChatHistory(sessionId) {
    try {
        const res = await fetch(`/get_history/${sessionId}`);
        const data = await res.json();

        if (data.error) {
            console.error("Error fetching history:", data.error);
            return;
        }

        currentSessionId = sessionId;

        document.querySelectorAll('.session-item').forEach(el => {
            el.classList.toggle('active', parseInt(el.dataset.id) === sessionId);
        });

        const activeTitleEl = document.querySelector(`.session-item[data-id="${sessionId}"] .session-title`);
        chatTitle.textContent = activeTitleEl ? activeTitleEl.textContent : "Chat Session";

        messagesArea.innerHTML = '';

        if (data.history && data.history.length > 0) {
            data.history.forEach(msg => {
                appendMessageBubble(msg.sender, msg.message, msg.source_doc, msg.source_page);
            });
        } else {
            messagesArea.innerHTML = '<div style="text-align:center;color:#888;padding:40px">This session is empty. Ask a question below to begin!</div>';
        }

        scrollToBottom();
    } catch (e) {
        console.error("Failed to load history:", e);
    }
}

// 3. INPUT & SEND HANDLING
function setupInputListeners() {
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    sendBtn.addEventListener('click', handleSend);
    newChatBtn.addEventListener('click', startNewChat);
}

async function startNewChat() {
    currentSessionId = null;
    chatTitle.textContent = "New Chat";

    stopSpeaking();

    messagesArea.innerHTML = `
        <div class="welcome-box">
            <h3>እንኳን ደህና መጡ! 👋</h3>
            <p>Ask me anything about Ethiopian census data, economic surveys, livestock metrics, or inflation rates.</p>
        </div>
        <div class="suggestions-grid" id="suggestionsGrid"></div>
    `;

    await loadSuggestions();

    document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
    questionInput.value = '';
    questionInput.focus();
}

async function handleSend() {
    const question = questionInput.value.trim();
    if (!question) return;

    questionInput.value = '';

    stopSpeaking();

    if (!currentSessionId && !window.IS_GUEST) {
        const token = await getCsrfToken();
        try {
            const res = await fetch('/new_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': token
                },
                body: JSON.stringify({ title: question })
            });
            const data = await res.json();
            if (data.success) {
                currentSessionId = data.session_id;
                await refreshSidebar();
            } else {
                alert("Failed to initialize a session.");
                return;
            }
        } catch (e) {
            console.error(e);
            return;
        }
    }

    appendMessageBubble('user', question);
    scrollToBottom();

    const typingIndicator = showTypingIndicator();

    try {
        const token = await getCsrfToken();
        const response = await fetch('/ask_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': token
            },
            // ui_lang tells the backend which language the EN/አማ toggle is
            // set to (see force_lang in rag.py) — without this the toggle
            // only changed cosmetic chrome text and never changed what
            // language the bot actually answered in.
            body: JSON.stringify({ question: question, session_id: currentSessionId, ui_lang: uiLang })
        });

        removeTypingIndicator(typingIndicator);

        if (!response.ok) {
            if (response.status === 403) {
                try {
                    const errData = await response.json();
                    if (errData.code === 'guest_limit_reached') {
                        showGuestLimitPrompt(errData.error);
                        return;
                    }
                } catch (e) { /* fall through to generic error */ }
            }
            appendMessageBubble('bot', 'Failed to communicate with the service. Please try again.');
            return;
        }

        const botBubble = appendMessageBubble('bot', '');
        const bubbleTextContainer = botBubble.querySelector('.bubble-text');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let metaReceived = false;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;

            if (buffer.includes(STREAM_META_SEP)) {
                const parts = buffer.split(STREAM_META_SEP);
                const streamText = parts[0];
                bubbleTextContainer.innerHTML = formatMarkdown(streamText);

                try {
                    const metaData = JSON.parse(parts[1]);
                    metaReceived = true;
                    appendSourceTag(botBubble, metaData.source, metaData.page);
                    if (typeof metaData.elapsed === 'number') {
                        appendTimingTag(botBubble, metaData.elapsed);
                    }
                    if (metaData.language) {
                        botBubble.dataset.answerLang = metaData.language; // 'en' | 'am' | 'om'
                    }
                    if (window.IS_GUEST && typeof metaData.guest_remaining === 'number') {
                        updateGuestBanner(metaData.guest_remaining, null);
                    }
                } catch(e) {
                    console.error("Meta parse error:", e);
                }
            } else if (!metaReceived) {
                bubbleTextContainer.innerHTML = formatMarkdown(buffer);
            }
            scrollToBottom();
        }

        // Auto-detect and render charts if the bot answer contains tabular data/numbers
        detectAndRenderChart(botBubble, buffer);

        updateSessionTitle(currentSessionId, question);

    } catch (e) {
        console.error(e);
        removeTypingIndicator(typingIndicator);
        appendMessageBubble('bot', 'An unexpected connection issue occurred.');
    }
}

// 4. CHAT DOM ELEMENT CREATORS (WITH CHARTS & VOICE BUTTONS)
function appendMessageBubble(sender, text, sourceDoc = null, sourcePage = null) {
    const welcome = messagesArea.querySelector('.welcome-box');
    if (welcome) welcome.remove();
    const grid = messagesArea.querySelector('.suggestions-grid');
    if (grid) grid.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${sender}`;

    const textSpan = document.createElement('span');
    textSpan.className = 'bubble-text';
    textSpan.innerHTML = formatMarkdown(text);
    msgDiv.appendChild(textSpan);

    if (sender === 'bot') {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'msg-actions';

        // Copy Button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.textContent = 'Copy';
        copyBtn.addEventListener('click', () => {
            const targetText = textSpan.innerText || textSpan.textContent;
            fallbackCopyText(targetText, copyBtn);
        });

        // Voice Listen Button
        const speakBtn = document.createElement('button');
        speakBtn.className = 'speak-btn';
        speakBtn.innerHTML = '🔊 Listen';
        speakBtn.style.marginLeft = '10px';
        speakBtn.addEventListener('click', () => {
            const targetText = textSpan.innerText || textSpan.textContent;
            speakText(targetText, msgDiv.dataset.answerLang, speakBtn);
        });

        // Voice Stop Button
        const stopBtn = document.createElement('button');
        stopBtn.className = 'stop-btn';
        stopBtn.innerHTML = '🔇 Stop';
        stopBtn.style.marginLeft = '10px';
        stopBtn.addEventListener('click', stopSpeaking);

        actionsDiv.appendChild(copyBtn);
        actionsDiv.appendChild(speakBtn);
        actionsDiv.appendChild(stopBtn);
        msgDiv.appendChild(actionsDiv);

        if (sourceDoc) {
            appendSourceTag(msgDiv, sourceDoc, sourcePage);
        }

        // Check if existing loaded history content has numbers/tables for charts
        if (text) {
            setTimeout(() => detectAndRenderChart(msgDiv, text), 100);
        }
    }

    messagesArea.appendChild(msgDiv);
    return msgDiv;
}

function appendSourceTag(msgDiv, doc, page) {
    if (!doc || doc === 'Unknown') return;
    const cleanDoc = doc.substring(doc.lastIndexOf('/') + 1);

    if (msgDiv.querySelector('.source-tag')) return;

    const sourceDiv = document.createElement('div');
    sourceDiv.className = 'source-tag';
    sourceDiv.innerHTML = `Source: 📑 <strong>${cleanDoc}</strong> (Page: ${page || 'N/A'})`;
    msgDiv.appendChild(sourceDiv);
}

function appendTimingTag(msgDiv, elapsedSeconds) {
    if (msgDiv.querySelector('.timing-tag')) return;

    const timingDiv = document.createElement('div');
    timingDiv.className = 'timing-tag';
    timingDiv.innerHTML = `⏱ Answered in ${elapsedSeconds}s`;
    msgDiv.appendChild(timingDiv);
}

function showTypingIndicator() {
    const loader = document.createElement('div');
    loader.className = 'msg typing-indicator';
    loader.innerHTML = '<span></span><span></span><span></span>';
    messagesArea.appendChild(loader);
    scrollToBottom();
    return loader;
}

function removeTypingIndicator(el) {
    if (el && el.parentNode) {
        el.parentNode.removeChild(el);
    }
}

function scrollToBottom() {
    messagesArea.scrollTop = messagesArea.scrollHeight;
}

// Markdown table (rows of "| a | b |" with a "|---|---|" separator row)
// -> a real HTML <table>. Consumes the table's lines and returns how many
// lines (starting at `start`) it used, so the caller can skip past them.
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
    if (!text) return "";
    let clean = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Pull fenced code blocks + tables out into placeholders BEFORE the
    // line-by-line <p>-wrap below, so multi-line blocks survive the split
    // intact instead of getting one <p> per line.
    const blocks = [];
    clean = clean.replace(/```([\s\S]*?)```/g, (match, code) => {
        blocks.push(`<pre><code>${code}</code></pre>`);
        return `\u0000BLOCK${blocks.length - 1}\u0000`;
    });
    clean = clean.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    clean = clean.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    const srcLines = clean.split('\n');
    const outLines = [];
    for (let i = 0; i < srcLines.length; i++) {
        const table = _consumeMarkdownTable(srcLines, i);
        if (table) {
            blocks.push(table.html);
            outLines.push(`\u0000BLOCK${blocks.length - 1}\u0000`);
            i += table.linesConsumed - 1;
            continue;
        }
        outLines.push(srcLines[i]);
    }

    let html = outLines.map(line => line.trim() ? `<p>${line}</p>` : '').join('');
    html = html.replace(/<p>\u0000BLOCK(\d+)\u0000<\/p>/g, (match, i) => blocks[Number(i)]);
    return html;
}

// 5. SIDEBAR UPDATES
async function refreshSidebar() {
    try {
        const res = await fetch('/get_sessions');
        const data = await res.json();

        if (data.sessions) {
            sessionsList.innerHTML = '';
            data.sessions.forEach(s => {
                const activeClass = (currentSessionId === s.id) ? 'active' : '';
                const title = s.title || 'New Chat';
                const formattedDate = s.created_at || '';

                const item = document.createElement('div');
                item.className = `session-item ${activeClass}`;
                item.dataset.id = s.id;

                item.innerHTML = `
                    <div class="session-info">
                        <div class="session-title">${title}</div>
                        <div class="session-date">${formattedDate}</div>
                    </div>
                    <button class="delete-session" title="Delete Chat">&times;</button>
                `;
                sessionsList.appendChild(item);
            });
        }
    } catch (e) {
        console.error("Error refreshing sidebar:", e);
    }
}

function updateSessionTitle(sessionId, title) {
    const el = document.querySelector(`.session-item[data-id="${sessionId}"] .session-title`);
    if (el && (el.textContent === 'New Chat' || el.textContent.trim() === '')) {
        el.textContent = title.substring(0, 30) + (title.length >= 30 ? '...' : '');
    }
}

async function handleDeleteSession(sessionId) {
    if (!confirm('Are you sure you want to delete this chat session?')) return;
    const token = await getCsrfToken();
    try {
        const r = await fetch(`/delete_session/${sessionId}`, {
            method: 'DELETE',
            headers: { 'X-CSRFToken': token }
        });
        const d = await r.json();
        if (d.success) {
            document.querySelector(`.session-item[data-id="${sessionId}"]`)?.remove();

            if (!sessionsList.querySelector('.session-item')) {
                sessionsList.innerHTML = EMPTY_SESSIONS_HTML;
            }

            if (currentSessionId === sessionId) {
                await startNewChat();
            }
        }
    } catch (e) {
        console.error("Failed to delete session:", e);
    }
}

// 6. ROBUST CLIPBOARD COPY
function fallbackCopyText(text, buttonElement) {
    const textArea = document.createElement("textarea");
    textArea.value = text;

    textArea.style.position = "absolute";
    textArea.style.left = "-9999px";
    textArea.style.top = "0";
    document.body.appendChild(textArea);

    textArea.select();
    textArea.setSelectionRange(0, 99999);

    try {
        const successful = document.execCommand('copy');
        if (successful) {
            buttonElement.textContent = 'Copied!';
        } else {
            buttonElement.textContent = 'Failed!';
        }
    } catch (err) {
        console.error('Fallback copy failed', err);
        buttonElement.textContent = 'Error!';
    }

    document.body.removeChild(textArea);
    setTimeout(() => { buttonElement.textContent = 'Copy'; }, 1800);
}

// 7. TEXT-TO-SPEECH
// Hybrid strategy:
//  - English almost always has a decent built-in browser voice -> use the
//    free, instant, zero-network Web Speech API for it.
//  - Amharic and (especially) Afaan Oromoo voices are missing on nearly
//    every desktop/Android/iOS browser. Setting utterance.lang = 'am-ET' or
//    'om-ET' when no such voice is installed does NOT fail loudly — it
//    silently substitutes a default (usually English) voice, which is what
//    produced the "wrong sound" bug. So for am/om we ask the backend's
//    /tts endpoint (server-side gTTS — free, no API key) to render real
//    audio instead, and only fall back to the broken browser voice if that
//    request fails, so something still plays rather than nothing.
const LANG_TO_TTS_LOCALE = { en: 'en-US', am: 'am-ET', om: 'om-ET' };
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

function hasBrowserVoiceFor(localePrefix) {
    if (!('speechSynthesis' in window)) return false;
    const voices = window.speechSynthesis.getVoices();
    // Voice list loads async in some browsers; if it's empty we haven't
    // heard back yet — assume yes rather than blocking English playback.
    if (!voices.length) return true;
    return voices.some(v => v.lang && v.lang.toLowerCase().startsWith(localePrefix));
}

async function speakText(text, answerLang, btnEl) {
    stopSpeaking();

    const cleanText = text.replace(/[*_`#]/g, '')
                           .replace(/<\/?[^>]+(>|$)/g, "")
                           .trim();
    if (!cleanText) return;

    // Prefer the language the backend actually answered in (passed through
    // the stream metadata). Afaan Oromoo is written in Latin script, so
    // guessing from character ranges can never distinguish it from English.
    // Script-sniffing is kept only as a fallback for older cached bubbles
    // that don't carry a language tag.
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
            btnEl.innerHTML = '⏳ ...';
            btnEl.disabled = true;
        } else {
            btnEl.innerHTML = btnEl.dataset.origLabel || '🔊 Listen';
            btnEl.disabled = false;
        }
    };

    setBtnLoading(true);
    try {
        const res = await fetch(`${TTS_ENDPOINT}?lang=${lang}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: cleanText })
        });

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
        console.error('Server TTS failed, falling back to browser voice:', e);
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(cleanText);
            utterance.lang = LANG_TO_TTS_LOCALE[lang] || 'en-US';
            window.speechSynthesis.speak(utterance);
        }
    } finally {
        setBtnLoading(false);
    }
}

// 8. MULTI-LANGUAGE INJECTOR & SPEECH-TO-TEXT
function injectLanguageSelector() {
    const inputArea = document.querySelector('.input-area');
    if (!inputArea || document.getElementById('langSelector')) return;

    const selectorDiv = document.createElement('div');
    selectorDiv.style.display = 'flex';
    selectorDiv.style.justifyContent = 'flex-end';
    selectorDiv.style.marginBottom = '8px';
    selectorDiv.style.fontSize = '0.8rem';
    selectorDiv.style.gap = '8px';
    selectorDiv.id = 'langSelectorContainer';

    selectorDiv.innerHTML = `
        <span style="color: #888; align-self: center;">🎙️ Mic Language:</span>
        <select id="langSelector" style="background: #0f3460; border: 1px solid #1e4d8c; color: white; border-radius: 4px; padding: 2px 6px; cursor: pointer;">
            <option value="en-US">English (US)</option>
            <option value="am-ET" selected>Amharic (አማርኛ)</option>
            <option value="om-ET">Afaan Oromoo</option>
        </select>
    `;

    inputArea.insertBefore(selectorDiv, inputArea.firstChild);

    const langSelector = document.getElementById('langSelector');
    langSelector.addEventListener('change', (e) => {
        currentLanguage = e.target.value;
        if (recognition) {
            recognition.lang = currentLanguage;
        }
    });

    currentLanguage = langSelector.value; // set initial to Amharic
}

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    // iOS Safari (and some in-app browsers) have no Web Speech API at all.
    // Don't inject a mic button that can never work — that's a dead control
    // with no feedback, which is worse than no button.
    if (!SpeechRecognition) {
        console.warn("Speech recognition is not supported in this browser.");
        return;
    }

    const inputRow = document.querySelector('.input-row');
    if (inputRow && !document.getElementById('micBtn')) {
        const micBtn = document.createElement('button');
        micBtn.type = 'button';
        micBtn.id = 'micBtn';
        micBtn.className = 'mic-btn';
        micBtn.innerHTML = '🎙️';
        micBtn.title = UI_STRINGS[uiLang].micTitle;

        inputRow.insertBefore(micBtn, questionInput);
        micBtn.addEventListener('click', toggleMic);
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = currentLanguage;
    recognition.interimResults = false;

    recognition.onstart = () => {
        isListening = true;
        const micBtn = document.getElementById('micBtn');
        if (micBtn) {
            micBtn.classList.add('recording');
            micBtn.innerHTML = '🔴';
        }
        questionInput.placeholder = UI_STRINGS[uiLang].listening;
        showMicStatus('');
    };

    recognition.onspeechend = () => {
        recognition.stop();
    };

    recognition.onend = () => {
        isListening = false;
        const micBtn = document.getElementById('micBtn');
        if (micBtn) {
            micBtn.classList.remove('recording');
            micBtn.innerHTML = '🎙️';
        }
        questionInput.placeholder = UI_STRINGS[uiLang].placeholder;
    };

    recognition.onresult = (event) => {
        const speechToTextResult = event.results[0][0].transcript;
        questionInput.value = speechToTextResult;
        questionInput.focus();

        setTimeout(() => {
            handleSend();
        }, 500);
    };

    // Mobile browsers hit these far more than desktop (permission prompts,
    // flaky data connections) — surface them inline instead of just logging.
    recognition.onerror = (event) => {
        console.error("Speech Recognition Error: ", event.error);
        const messages = {
            'not-allowed': 'Microphone access denied — allow it in your browser settings to use voice input.',
            'service-not-allowed': 'Microphone access denied — allow it in your browser settings to use voice input.',
            'audio-capture': 'No microphone found on this device.',
            'no-speech': "Didn't catch that — click the mic and try again.",
            'network': 'Voice input needs an internet connection.'
        };
        showMicStatus(messages[event.error] || 'Voice input failed — please type your question instead.');
        recognition.stop();
    };
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
    recognition.lang = currentLanguage; // sync language
    try {
        recognition.start();
    } catch (err) {
        console.error('Mic start failed:', err);
        showMicStatus('Could not start voice input — please try again.');
    }
}

// 9. LAZY LOAD CHART.JS (Keeps response speed lighting fast!)
function loadChartJS() {
    if (window.Chart) return;
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/chart.js";
    script.async = true;
    document.head.appendChild(script);
}

// 10. INTELLIGENT DATA VISUALIZATION ENGINE
// Only render a chart when we have clear "Category: number" pairs that look
// like real statistics (not random prose fragments like "including COVID").
function detectAndRenderChart(bubbleElement, text) {
    if (bubbleElement.querySelector('.chart-container-wrapper')) return;

    // Stricter pattern: short label (1-4 words), then : or -, then a number
    // that is either a percentage or a reasonably large statistic.
    const dataRegex = /(?:^|\n|\.\s+)([A-Za-z\u1200-\u137F][A-Za-z\u1200-\u137F\s]{1,28}?)[:\-]\s*([\d,]+(?:\.\d+)?%?)\b/g;
    let matches = [...text.matchAll(dataRegex)];

    if (matches.length < 2) return;

    const SKIP_LABELS = new Set([
        'source', 'page', 'answered', 'route', 'including', 'and', 'the',
        'with', 'from', 'for', 'efy', 'e.f.y', 'covid', 'covid-19'
    ]);

    const labels = [];
    const dataValues = [];

    matches.forEach(match => {
        const label = match[1].trim().replace(/\s+/g, ' ');
        const raw = match[2];
        const isPct = raw.includes('%');
        const valueStr = raw.replace(/,/g, '').replace('%', '');
        const val = parseFloat(valueStr);

        if (isNaN(val)) return;
        // Reject tiny integers that are almost always page numbers / years / noise
        if (!isPct && val < 10) return;
        // Reject year-like numbers alone
        if (!isPct && val >= 1900 && val <= 2100) return;

        const lower = label.toLowerCase();
        if (SKIP_LABELS.has(lower)) return;
        if (lower.split(/\s+/).length > 5) return; // too long = prose, not a label

        labels.push(label);
        dataValues.push(val);
    });

    // Need at least 2 clean pairs and values that aren't all identical
    if (labels.length < 2 || !window.Chart) return;
    const uniqueVals = new Set(dataValues);
    if (uniqueVals.size < 2) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'chart-container-wrapper';
    wrapper.innerHTML = `<canvas></canvas>`;
    bubbleElement.appendChild(wrapper);

    const ctx = wrapper.querySelector('canvas').getContext('2d');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Extracted Stat Metrics',
                data: dataValues,
                backgroundColor: [
                    'rgba(79, 195, 247, 0.6)',
                    'rgba(16, 185, 129, 0.6)',
                    'rgba(239, 68, 68, 0.6)',
                    'rgba(245, 158, 11, 0.6)',
                    'rgba(139, 92, 246, 0.6)'
                ],
                borderColor: [
                    '#4fc3f7',
                    '#10b981',
                    '#ef4444',
                    '#f59e0b',
                    '#8b5cf6'
                ],
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#aaa' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#aaa' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}
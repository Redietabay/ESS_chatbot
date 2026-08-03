let currentSessionId = null;

// Marker the backend appends after the streamed answer text
const STREAM_META_SEP = '\x00__ESS_META__\x00';
const EMPTY_SESSIONS_HTML = '<div class="empty-sessions">No active sessions found.<br>Start a new chat below.</div>';

let recognition;
let isListening = false;
let currentLanguage = 'en-US'; // Default Language

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
    loadSuggestions();
    setupSessionClickHandlers();
    setupInputListeners();
    injectLanguageSelector(); // Inject Amharic / Afaan Oromoo / English selector
    initSpeechRecognition();
    loadChartJS(); // Load Chart.js dynamically to save loading time
    if (window.IS_GUEST) initGuestBanner();
});

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
        const res = await fetch('/suggestions');
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

        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }

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

    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }

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

    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }

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
            body: JSON.stringify({ question: question, session_id: currentSessionId })
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
            speakText(targetText, msgDiv.dataset.answerLang);
        });

        // Voice Stop Button
        const stopBtn = document.createElement('button');
        stopBtn.className = 'stop-btn';
        stopBtn.innerHTML = '🔇 Stop';
        stopBtn.style.marginLeft = '10px';
        stopBtn.addEventListener('click', () => {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
            }
        });

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

function formatMarkdown(text) {
    if (!text) return "";
    let clean = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    clean = clean.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    clean = clean.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    clean = clean.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    return clean.split('\n').map(line => line.trim() ? `<p>${line}</p>` : '').join('');
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
const LANG_TO_TTS_LOCALE = { en: 'en-US', am: 'am-ET', om: 'om-ET' };

function speakText(text, answerLang) {
    if (!('speechSynthesis' in window)) {
        console.warn("Speech Synthesis not supported.");
        return;
    }

    window.speechSynthesis.cancel();

    let cleanText = text.replace(/[*_`#]/g, '')
                        .replace(/<\/?[^>]+(>|$)/g, "");

    const utterance = new SpeechSynthesisUtterance(cleanText);

    // Prefer the language the backend actually answered in (passed through
    // the stream metadata). Afaan Oromoo is written in Latin script, so
    // guessing from character ranges can never distinguish it from English —
    // this is the fix for that. Script-sniffing is kept only as a fallback
    // for older cached bubbles that don't carry a language tag.
    if (answerLang && LANG_TO_TTS_LOCALE[answerLang]) {
        utterance.lang = LANG_TO_TTS_LOCALE[answerLang];
    } else if (/[\u1200-\u137F]/.test(cleanText)) {
        utterance.lang = 'am-ET';
    } else {
        utterance.lang = 'en-US';
    }
    utterance.rate = 1.0;

    window.speechSynthesis.speak(utterance);
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
    const inputRow = document.querySelector('.input-row');
    if (inputRow && !document.getElementById('micBtn')) {
        const micBtn = document.createElement('button');
        micBtn.type = 'button';
        micBtn.id = 'micBtn';
        micBtn.className = 'mic-btn';
        micBtn.innerHTML = '🎙️';
        micBtn.title = 'Speak your question';

        inputRow.insertBefore(micBtn, questionInput);
        micBtn.addEventListener('click', toggleMic);
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn("Speech recognition is not supported in this browser.");
        return;
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
        questionInput.placeholder = "Listening... Speak now!";
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
        questionInput.placeholder = "Type your data question here... (Press Enter to send)";
    };

    recognition.onresult = (event) => {
        const speechToTextResult = event.results[0][0].transcript;
        questionInput.value = speechToTextResult;
        questionInput.focus();

        setTimeout(() => {
            handleSend();
        }, 500);
    };

    recognition.onerror = (event) => {
        console.error("Speech Recognition Error: ", event.error);
        recognition.stop();
    };
}

function toggleMic() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        alert("Your browser does not support Speech Recognition. Please use Google Chrome.");
        return;
    }

    if (isListening) {
        recognition.stop();
    } else {
        recognition.lang = currentLanguage; // sync language
        recognition.start();
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
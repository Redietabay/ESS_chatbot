/**
 * ESS AI Assistant — embeddable widget loader.
 *
 * USAGE (for the ESS web team): add this ONE line before </body> on the
 * official ESS website:
 *
 *   <script src="https://YOUR-CHATBOT-DOMAIN/static/ess-embed-loader.js"
 *           data-chatbot-url="https://YOUR-CHATBOT-DOMAIN"></script>
 *
 * That's the entire integration. This script injects a floating chat
 * bubble in the bottom-right corner; clicking it opens/closes an iframe
 * pointing to /widget on your chatbot's own domain.
 *
 * CHATBOT_URL resolution order:
 *   1. data-chatbot-url attribute, if set to an absolute http(s) URL.
 *      Use this ONLY when embedding on a DIFFERENT domain than the
 *      chatbot itself (e.g. the real statsethiopia.gov.et site pointing
 *      at your Flask backend).
 *   2. window.location.origin — the origin of the page that's currently
 *      running this script. Correct by default whenever this script is
 *      loaded from the SAME app that serves /widget (local testing via
 *      /test, or ngrok — the browser is always on that one origin, so
 *      this is always right and needs no extra logic).
 *
 * We deliberately do NOT derive the origin from the <script> tag's own
 * src (via document.currentScript) anymore — that approach broke when
 * testing through ngrok (it resolved to localhost:5000 in practice,
 * probably due to caching/duplicate static files), causing the CSP
 * frame-ancestors error. window.location.origin has no such failure mode.
 */
(function () {
  if (document.getElementById('ess-widget-bubble')) return;

  var CURRENT_SCRIPT = document.currentScript;
  var explicitUrl = CURRENT_SCRIPT && CURRENT_SCRIPT.getAttribute('data-chatbot-url');
  var CHATBOT_URL = (explicitUrl && /^https?:\/\//i.test(explicitUrl))
    ? explicitUrl
    : window.location.origin;

  console.log('[ESS widget] CHATBOT_URL resolved to:', CHATBOT_URL);

  var isOpen = false;

  // ── Floating bubble button ──
  var bubble = document.createElement('button');
  bubble.setAttribute('aria-label', 'Open ESS Assistant');
  bubble.id = 'ess-widget-bubble';
  bubble.innerHTML = bubbleIconSVG();
  document.body.appendChild(bubble);

  // ── Iframe container (hidden until first click, then just toggled) ──
  var frameWrap = document.createElement('div');
  frameWrap.id = 'ess-widget-frame-wrap';
  frameWrap.style.display = 'none';

  var iframe = document.createElement('iframe');
  iframe.id = 'ess-widget-iframe';
  iframe.title = 'ESS AI Assistant';
  iframe.src = CHATBOT_URL.replace(/\/$/, '') + '/widget';
  frameWrap.appendChild(iframe);
  document.body.appendChild(frameWrap);

  injectStyles();

  bubble.addEventListener('click', toggleWidget);

  // Widget's own close (×) button posts this message from inside the iframe
  window.addEventListener('message', function (event) {
    if (event.data && event.data.type === 'ess-widget-close') {
      closeWidget();
    }
  });

  function toggleWidget() {
    isOpen ? closeWidget() : openWidget();
  }

  function openWidget() {
    isOpen = true;
    frameWrap.style.display = 'flex';
    requestAnimationFrame(function () {
      frameWrap.classList.add('ess-widget-open');
    });
    bubble.innerHTML = closeIconSVG();
    if (window.matchMedia('(max-width: 640px)').matches) {
      document.body.style.overflow = 'hidden';
    }
  }

  function closeWidget() {
    isOpen = false;
    frameWrap.classList.remove('ess-widget-open');
    bubble.innerHTML = bubbleIconSVG();
    document.body.style.overflow = '';
    setTimeout(function () {
      if (!isOpen) frameWrap.style.display = 'none';
    }, 200);
  }

  function bubbleIconSVG() {
    return '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
      + '<path d="M4 4h16v12H7l-3 3V4z" fill="white"/></svg>';
  }

  function closeIconSVG() {
    return '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
      + '<path d="M6 6L18 18M6 18L18 6" stroke="white" stroke-width="2.2" stroke-linecap="round"/></svg>';
  }

  function injectStyles() {
    var style = document.createElement('style');
    style.textContent = [
      '#ess-widget-bubble{',
      '  position:fixed;',
      '  bottom:calc(20px + env(safe-area-inset-bottom));',
      '  right:calc(20px + env(safe-area-inset-right));',
      '  z-index:2147483000;',
      '  width:56px; height:56px; border-radius:50%; border:none;',
      '  background:#0B3C5D; box-shadow:0 4px 14px rgba(0,0,0,0.3);',
      '  display:flex; align-items:center; justify-content:center;',
      '  cursor:pointer; transition:transform .15s ease;',
      '}',
      '#ess-widget-bubble:hover{ transform:scale(1.06); }',
      '#ess-widget-frame-wrap{',
      '  position:fixed;',
      '  bottom:calc(88px + env(safe-area-inset-bottom));',
      '  right:calc(20px + env(safe-area-inset-right));',
      '  z-index:2147483000;',
      '  width:380px; height:560px; max-width:calc(100vw - 24px);',
      '  max-height:calc(100vh - 110px);',
      '  border-radius:14px; overflow:hidden;',
      '  box-shadow:0 10px 40px rgba(0,0,0,0.35);',
      '  opacity:0; transform:translateY(12px);',
      '  transition:opacity .2s ease, transform .2s ease;',
      '}',
      '#ess-widget-frame-wrap.ess-widget-open{ opacity:1; transform:translateY(0); }',
      '#ess-widget-iframe{ width:100%; height:100%; border:none; display:block; }',
      '@media (max-width: 640px){',
      '  #ess-widget-frame-wrap{',
      '    inset:0; bottom:0; right:0; top:0; left:0;',
      '    width:100vw; height:100vh; height:100dvh;',
      '    max-width:100vw; max-height:100vh; max-height:100dvh;',
      '    border-radius:0;',
      '    transform:translateY(100%);',
      '  }',
      '  #ess-widget-frame-wrap.ess-widget-open{ transform:translateY(0); }',
      '  #ess-widget-bubble{',
      '    bottom:calc(16px + env(safe-area-inset-bottom));',
      '    right:calc(16px + env(safe-area-inset-right));',
      '  }',
      '}'
    ].join('\n');
    document.head.appendChild(style);
  }
})();
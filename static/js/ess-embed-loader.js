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
 * pointing to /widget on your chatbot's own domain. Because the chat UI
 * itself lives in that iframe (same-origin with your Flask app), no CORS
 * configuration or cross-site cookie setup is needed on either side.
 *
 * On phone-width screens the iframe becomes a full-screen sheet instead
 * of a small anchored box — a fixed small box is what traps the input
 * bar under the on-screen keyboard; a full sheet has room to reflow.
 *
 * If data-chatbot-url is omitted, the script falls back to the domain it
 * was itself loaded from (works if the <script> tag's src already points
 * at your chatbot's domain).
 */
(function () {
  // Guards against the script tag being included twice on the same page
  // (accidental duplicate <script> tag, or injected again by a CMS/tag
  // manager) — without this, a second run appended a second bubble + a
  // second iframe, both listening for clicks/messages independently.
  if (document.getElementById('ess-widget-bubble')) return;

  var CURRENT_SCRIPT = document.currentScript;
  var CHATBOT_URL = (CURRENT_SCRIPT && CURRENT_SCRIPT.getAttribute('data-chatbot-url'))
    || (CURRENT_SCRIPT && new URL(CURRENT_SCRIPT.src).origin)
    || '';

  if (!CHATBOT_URL) {
    console.error('[ESS widget] Could not determine chatbot URL — set data-chatbot-url on the <script> tag.');
    return;
  }

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
    // Small delay so the CSS transition (opacity/transform) actually runs
    // instead of jumping straight to the open state.
    requestAnimationFrame(function () {
      frameWrap.classList.add('ess-widget-open');
    });
    bubble.innerHTML = closeIconSVG();
    // Lock host-page scroll while the full-screen sheet is open on mobile —
    // otherwise the background page can scroll behind it as the keyboard opens.
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
    }, 200); // matches the CSS transition duration below
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
      /* Phone-width screens: full-screen sheet instead of a small anchored
         box, so the chat UI has room to shrink around the on-screen keyboard
         rather than being crushed inside a fixed-size box. */
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
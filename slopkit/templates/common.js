// Shared helpers for slopkit HTML templates. A template defines window.setSpec(spec) and window.renderAt(t)
// (t in seconds on the template's clock). slopkit/htmlrender.py calls SLOP.overflowAt(times) before rendering:
// every visible [data-box] element's text ink must sit inside its padding box, and the box inside the frame.
window.SLOP = (() => {
  const W = window.innerWidth || 1920, H = window.innerHeight || 1080;  // the viewport (9:16 templates too)
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, u) => a + (b - a) * u;
  const easeIO = u => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  const easeOut = u => 1 - Math.pow(1 - u, 3);
  // keys: [{t, <field>: value, ease?: 'linear'|'io'}], sorted by t. Holds the first/last value outside the range.
  function keyed(keys, t, field) {
    if (!keys || !keys.length) return undefined;
    if (t <= keys[0].t) return keys[0][field];
    for (let i = 1; i < keys.length; i++) {
      if (t < keys[i].t) {
        const a = keys[i - 1], b = keys[i];
        const u = (t - a.t) / Math.max(1e-6, b.t - a.t);
        return lerp(a[field], b[field], b.ease === 'linear' ? u : easeIO(u));
      }
    }
    return keys[keys.length - 1][field];
  }
  function over(el) { return el.scrollWidth > el.clientWidth + 0.5 || el.scrollHeight > el.clientHeight + 0.5; }
  // Largest font size in [minPx, maxPx] at which el's content fits el's own box (binary search, 0.5 px).
  function fit(el, maxPx, minPx) {
    el.style.fontSize = maxPx + 'px';
    if (!over(el)) return maxPx;
    let lo = minPx, hi = maxPx;
    while (hi - lo > 0.5) {
      const mid = (lo + hi) / 2;
      el.style.fontSize = mid + 'px';
      if (over(el)) hi = mid; else lo = mid;
    }
    el.style.fontSize = lo + 'px';
    if (over(el)) el.dataset.fitfail = '1';  // still too big at minPx: reported by overflowAt
    return lo;
  }
  function shown(el) {
    for (let e = el; e && e !== document.body; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity) === 0) return false;
    }
    return true;
  }
  const R = r => [r.left, r.top, r.right, r.bottom].map(v => Math.round(v));
  function overflowAt(ts) {
    const bad = [];
    for (const t of ts) {
      window.renderAt(t);
      for (const box of document.querySelectorAll('[data-box]')) {
        if (!shown(box)) continue;
        const saved = box.style.transform;
        box.style.transform = 'none';
        const r = box.getBoundingClientRect(), cs = getComputedStyle(box);
        const inner = {l: r.left + (parseFloat(cs.borderLeftWidth) || 0), r: r.right - (parseFloat(cs.borderRightWidth) || 0),
                       t: r.top + (parseFloat(cs.borderTopWidth) || 0), b: r.bottom - (parseFloat(cs.borderBottomWidth) || 0)};
        if (box.dataset.fitfail || box.querySelector('[data-fitfail]'))
          bad.push({t: +t.toFixed(3), box: box.dataset.box, why: 'text cannot fit at the minimum font size'});
        if (!box.hasAttribute('data-offscreen-ok') && (r.left < -0.5 || r.top < -0.5 || r.right > W + 0.5 || r.bottom > H + 0.5))
          bad.push({t: +t.toFixed(3), box: box.dataset.box, why: 'box outside frame', rect: R(r)});
        const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT);
        let n;
        while ((n = walker.nextNode())) {
          if (!n.textContent.trim() || (n.parentElement && n.parentElement.closest('[data-nocheck]'))) continue;
          if (n.parentElement && !shown(n.parentElement)) continue;
          if (n.parentElement && n.parentElement.closest('[data-box]') !== box) continue;  // nested boxes check themselves
          const rg = document.createRange();
          rg.selectNodeContents(n);
          let hit = null;
          for (const q of rg.getClientRects()) {
            if (!q.width) continue;
            if (q.left < inner.l - 0.5 || q.right > inner.r + 0.5 || q.top < inner.t - 0.5 || q.bottom > inner.b + 0.5) { hit = q; break; }
          }
          if (hit) { bad.push({t: +t.toFixed(3), box: box.dataset.box, why: 'text outside box', text: n.textContent.slice(0, 48),
                               inner: R({left: inner.l, top: inner.t, right: inner.r, bottom: inner.b}), ink: R(hit)}); break; }
        }
        box.style.transform = saved;
      }
      if (bad.length > 20) break;
    }
    return bad;
  }
  // Readable text (class "readable") smaller than minPx at 1080p, sampled at times ts (warning, not a failure).
  // One entry per element: its box id, smallest size seen and a text sample.
  function smallText(ts, minPx) {
    const seen = new Map();
    for (const t of ts) {
      window.renderAt(t);
      document.querySelectorAll('.readable').forEach((el, i) => {
        if (!shown(el) || !el.textContent.trim()) return;
        const px = parseFloat(getComputedStyle(el).fontSize);
        if (px >= minPx - 0.01) return;
        const box = el.closest('[data-box]'), key = (box ? box.dataset.box : '?') + '#' + i;
        const o = seen.get(key);
        if (!o || px < o.px || el.textContent.length > o.text.length)
          seen.set(key, {box: box ? box.dataset.box : null, px: Math.round(Math.min(px, o ? o.px : px) * 10) / 10,
                         text: el.textContent.replace('\u2588', '').slice(0, 48)});
      });
    }
    return [...seen.values()];
  }
  // Deterministic PRNG (mulberry32) so decorative streams render identically in every worker.
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function fontsReady() { return Promise.all([...document.fonts].map(f => f.load())).then(() => document.fonts.ready); }
  return {W, H, clamp, lerp, easeIO, easeOut, keyed, fit, over, overflowAt, smallText, rng, fontsReady};
})();

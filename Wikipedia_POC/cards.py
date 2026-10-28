"""Message cards drawn over the page, so the room can follow the bot.

Rendering is stateless: Python keeps the whole picture and redraws it on
every change, so a page the bot navigates to gets its cards back on the
next update with nothing to re-attach.

Every class is prefixed `bisx` (a host page is free to define `.card`),
and nothing here is a click target - `pointer-events:none` throughout, so
the overlay can never change what the bot is able to click.
"""

from __future__ import annotations

from selenium.common.exceptions import WebDriverException

CSS = r"""
#bisxRoot{position:fixed;bottom:18px;right:22px;width:410px;z-index:2147483646;
  pointer-events:none;font:14px/1.45 "Segoe UI",system-ui,-apple-system,sans-serif;
  color:#EAF1FF;-webkit-font-smoothing:antialiased;text-shadow:0 1px 2px rgba(0,0,0,.65)}
:where(#bisxRoot *){box-sizing:border-box;pointer-events:none;margin:0;padding:0}
#bisxRoot *{pointer-events:none;box-sizing:border-box}
/* See-through: a light tint and a light blur, and brightness() darkens the
   page behind rather than covering it - the site stays visible and the
   text keeps its contrast over Wikipedia's white. */
.bisxGlass{background:linear-gradient(160deg,rgba(17,26,51,.42),rgba(9,14,32,.50));
  border:1px solid rgba(170,200,255,.30);border-radius:18px;
  box-shadow:0 14px 40px rgba(3,8,25,.30),inset 0 1px 0 rgba(255,255,255,.08);
  -webkit-backdrop-filter:blur(2.5px) saturate(1.2) brightness(.52);
  backdrop-filter:blur(2.5px) saturate(1.2) brightness(.52)}
.bisxTop{display:flex;align-items:center;justify-content:center;padding:10px 14px;margin-bottom:12px}
.bisxTop .bisxPill{font-size:13px;padding:9px 18px;letter-spacing:.14em}
.bisxLogo{width:34px;height:34px;border-radius:10px;flex:0 0 auto;
  background:conic-gradient(from 210deg,#22D3EE,#6366F1,#A855F7,#22D3EE);
  display:flex;align-items:center;justify-content:center;font:800 15px/1 "Segoe UI",sans-serif;color:#fff}
.bisxBrand{flex:1}
.bisxBrand b{display:block;font-size:14px;letter-spacing:.02em;color:#fff}
.bisxBrand span{font-size:11.5px;color:#C3D3F2}
.bisxPill{font:800 10.5px/1 "Segoe UI",sans-serif;letter-spacing:.12em;padding:7px 10px;
  border-radius:999px;white-space:nowrap}
.bisxOn{background:rgba(16,185,129,.16);color:#34D399;border:1px solid rgba(52,211,153,.55)}
.bisxOff{background:rgba(239,68,68,.15);color:#F87171;border:1px solid rgba(248,113,113,.55)}
.bisxLearn{background:rgba(99,102,241,.18);color:#A5B4FC;border:1px solid rgba(165,180,252,.5)}
.bisxVer{display:flex;gap:8px;margin:0 0 12px}
.bisxVer div{flex:1;padding:10px 14px;border-radius:14px}
.bisxVer small{display:block;font:700 9.5px/1.2 "Segoe UI",sans-serif;letter-spacing:.14em;
  text-transform:uppercase;color:#C3D3F2;margin-bottom:3px}
.bisxVer b{font-size:13.5px;color:#fff}
.bisxCard{position:relative;overflow:hidden;padding:18px 20px 18px 20px;margin-bottom:12px;
  animation:bisxIn .45s cubic-bezier(.2,.9,.25,1.15)}
.bisxCard:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--acc)}
.bisxCard:after{content:"";position:absolute;right:-60px;top:-60px;width:180px;height:180px;
  border-radius:50%;background:radial-gradient(circle,var(--glow),transparent 70%)}
.bisxHead{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.bisxIcon{width:42px;height:42px;border-radius:50%;flex:0 0 auto;display:flex;align-items:center;
  justify-content:center;font:800 21px/1 "Segoe UI Symbol","Segoe UI",sans-serif;color:#fff;
  background:var(--acc);box-shadow:0 0 0 6px var(--glow)}
.bisxSpin{width:42px;height:42px;border-radius:50%;flex:0 0 auto;border:4px solid rgba(245,158,11,.22);
  border-top-color:#F59E0B;animation:bisxRot .8s linear infinite}
.bisxKick{font:800 10.5px/1.2 "Segoe UI",sans-serif;letter-spacing:.15em;text-transform:uppercase;color:var(--acc)}
.bisxTitle{font-size:19px;line-height:1.25;font-weight:700;color:#fff;margin-top:2px}
.bisxBody{color:#E6EEFF;font-size:14px;margin-top:2px}
.bisxSel{margin-top:12px;display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center}
.bisxSel span{font:700 9.5px/1 "Segoe UI",sans-serif;letter-spacing:.12em;text-transform:uppercase;color:#C3D3F2}
.bisxSel code{border:0;font:12.5px/1.35 Consolas,"Cascadia Mono",monospace;padding:5px 8px;border-radius:7px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bisxOld{background:rgba(239,68,68,.12);color:#FCA5A5;text-decoration:line-through;text-decoration-color:rgba(252,165,165,.7)}
.bisxNew{background:rgba(16,185,129,.14);color:#6EE7B7}
.bisxStats{display:flex;gap:8px;margin-top:14px}
.bisxStat{flex:1;padding:9px 11px;border-radius:12px;background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.08)}
.bisxStat small{display:block;font:700 9px/1.2 "Segoe UI",sans-serif;letter-spacing:.13em;
  text-transform:uppercase;color:#C3D3F2}
.bisxStat b{font-size:20px;color:#fff;font-variant-numeric:tabular-nums}
.bisxStat b.bisxG{color:#34D399}
.bisxStat b.bisxR{color:#F87171}
.bisxSteps{padding:12px 14px}
.bisxSteps h4{font:800 9.5px/1.2 "Segoe UI",sans-serif;letter-spacing:.15em;text-transform:uppercase;
  color:#C3D3F2;margin:0 0 8px 2px}
.bisxRow{display:flex;align-items:center;gap:10px;padding:4px 6px;border-radius:9px;color:#DCE6FA;font-size:13px}
.bisxRow i{width:20px;height:20px;border-radius:50%;flex:0 0 auto;font:800 11px/18px "Segoe UI Symbol",sans-serif;
  text-align:center;font-style:normal;border:1.5px solid #34466F;color:#0B1530}
.bisxRow em{margin-left:auto;font:700 9.5px/1 "Segoe UI",sans-serif;letter-spacing:.1em;font-style:normal;text-transform:uppercase}
.bisxRow.ok i{background:#3B82F6;border-color:#3B82F6;color:#fff}
.bisxRow.ok em{color:#93C5FD}
.bisxRow.healed i{background:#10B981;border-color:#10B981;color:#fff}
.bisxRow.healed em{color:#34D399}
.bisxRow.failed i{background:#EF4444;border-color:#EF4444;color:#fff}
.bisxRow.failed em{color:#F87171}
.bisxRow.learned i{background:#6366F1;border-color:#6366F1;color:#fff}
.bisxRow.learned em{color:#A5B4FC}
.bisxRow.run{background:rgba(99,140,255,.12);color:#fff}
.bisxRow.run i{border-color:#F59E0B;border-top-color:transparent;animation:bisxRot .9s linear infinite}
.bisxRow.skip{opacity:.45}
@keyframes bisxIn{from{opacity:0;transform:translateY(-10px) scale(.97)}to{opacity:1;transform:none}}
@keyframes bisxRot{to{transform:rotate(360deg)}}
@media (max-width:1300px){#bisxRoot{width:360px;right:12px}}
@media (max-height:860px){#bisxRoot{transform:scale(.86);transform-origin:bottom right;bottom:12px}}
@media (max-height:700px){#bisxRoot{transform:scale(.74)}}
"""

ACCENT = {
    "info": ("#818CF8", "rgba(129,140,248,.20)", "i"),
    "run": ("#60A5FA", "rgba(96,165,250,.18)", "&#9654;"),
    "ok": ("#3B82F6", "rgba(59,130,246,.22)", "&#10003;"),
    "learned": ("#6366F1", "rgba(99,102,241,.24)", "&#10003;"),
    "broken": ("#EF4444", "rgba(239,68,68,.22)", "!"),
    "healing": ("#F59E0B", "rgba(245,158,11,.20)", ""),
    "healed": ("#10B981", "rgba(16,185,129,.24)", "&#10003;"),
    "failed": ("#EF4444", "rgba(239,68,68,.24)", "&#10007;"),
    "done": ("#10B981", "rgba(16,185,129,.26)", "&#9733;"),
}

ROW_MARK = {"ok": "&#10003;", "healed": "&#10003;", "learned": "&#10003;",
            "failed": "&#10007;"}

RENDER_JS = r"""
const st = arguments[0], css = arguments[1];
if (!document.body) { return false; }
let sty = document.getElementById('bisxStyle');
if (!sty) { sty = document.createElement('style'); sty.id = 'bisxStyle';
  sty.textContent = css; document.head.appendChild(sty); }
let root = document.getElementById('bisxRoot');
if (!root) { root = document.createElement('div'); root.id = 'bisxRoot';
  document.body.appendChild(root); }
const el = (tag, cls, text) => { const e = document.createElement(tag);
  if (cls) { e.className = cls; } if (text != null) { e.textContent = text; } return e; };
root.textContent = '';

const top = el('div', 'bisxGlass bisxTop');
top.appendChild(el('div', 'bisxPill ' + st.pillClass, st.pill));
root.appendChild(top);

const ver = el('div', 'bisxVer');
for (const v of st.facts) { const box = el('div', 'bisxGlass');
  box.appendChild(el('small', '', v[0])); box.appendChild(el('b', '', v[1]));
  ver.appendChild(box); }
root.appendChild(ver);

if (st.card) {
  const c = st.card;
  const card = el('div', 'bisxGlass bisxCard');
  card.style.setProperty('--acc', c.acc); card.style.setProperty('--glow', c.glow);
  if (c.key === root.dataset.lastCard) { card.style.animation = 'none'; }
  root.dataset.lastCard = c.key;
  const head = el('div', 'bisxHead');
  if (c.spin) { head.appendChild(el('div', 'bisxSpin')); }
  else { const ic = el('div', 'bisxIcon'); ic.innerHTML = c.icon; head.appendChild(ic); }
  const tt = el('div');
  tt.appendChild(el('div', 'bisxKick', c.kicker));
  tt.appendChild(el('div', 'bisxTitle', c.title));
  head.appendChild(tt); card.appendChild(head);
  if (c.body) { card.appendChild(el('div', 'bisxBody', c.body)); }
  if (c.oldSel || c.newSel) {
    const g = el('div', 'bisxSel');
    if (c.oldSel) { g.appendChild(el('span', '', c.oldLabel || 'Old address'));
      g.appendChild(el('code', c.newSel ? 'bisxOld' : 'bisxOld', c.oldSel)); }
    if (c.newSel) { g.appendChild(el('span', '', 'New address'));
      g.appendChild(el('code', 'bisxNew', c.newSel)); }
    card.appendChild(g);
  }
  if (c.stats && c.stats.length) {
    const s = el('div', 'bisxStats');
    for (const x of c.stats) { const b = el('div', 'bisxStat');
      b.appendChild(el('small', '', x[0])); b.appendChild(el('b', x[2] || '', x[1]));
      s.appendChild(b); }
    card.appendChild(s);
  }
  root.appendChild(card);
}

const steps = el('div', 'bisxGlass bisxSteps');
steps.appendChild(el('h4', '', st.stepsTitle));
for (const r of st.steps) {
  const row = el('div', 'bisxRow ' + r.state);
  const i = el('i'); i.innerHTML = r.mark || ''; row.appendChild(i);
  row.appendChild(el('span', '', r.label));
  row.appendChild(el('em', '', r.tag || ''));
  steps.appendChild(row);
}
root.appendChild(steps);
return true;
"""


class Cards(object):
    """The overlay's state, redrawn whole on every change."""

    def __init__(self, driver, subtitle, pill, pill_class, facts, labels,
                 steps_title):
        self.driver = driver
        self.subtitle = subtitle
        self.pill = pill
        self.pill_class = pill_class
        self.facts = facts
        self.steps = [{"label": l, "state": "", "tag": "", "mark": ""}
                      for l in labels]
        self.steps_title = steps_title
        self.card = None
        self.seq = 0

    def step(self, i, state, tag=""):
        """Set one row of the step list: run, ok, healed, failed, learned."""
        self.steps[i].update(state=state, tag=tag,
                             mark=ROW_MARK.get(state, ""))
        self.draw()

    def show(self, kind, kicker, title, body="", old_sel="", new_sel="",
             stats=None, old_label=""):
        """Replace the message card."""
        acc, glow, icon = ACCENT[kind]
        self.seq += 1
        self.card = {"key": str(self.seq), "acc": acc, "glow": glow,
                     "icon": icon, "spin": kind == "healing",
                     "kicker": kicker, "title": title, "body": body,
                     "oldSel": old_sel, "newSel": new_sel,
                     "oldLabel": old_label, "stats": stats or []}
        self.draw()

    def draw(self):
        state = {"subtitle": self.subtitle, "pill": self.pill,
                 "pillClass": self.pill_class, "facts": self.facts,
                 "card": self.card, "steps": self.steps,
                 "stepsTitle": self.steps_title}
        try:
            self.driver.execute_script(RENDER_JS, state, CSS)
        except WebDriverException:
            pass            # a page mid-navigation; the next draw catches up

"""Message cards drawn over the page, so the room can follow the bot.

Three parts: a status panel bottom right (the agent's state, the website
version, the step checklist), the explanation card bottom centre, and - only
when a step fails - a pop-up in the middle of the screen naming that step.

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
#bisxRoot,#bisxMsg,#bisxAlert{z-index:2147483646;pointer-events:none;
  font:14px/1.45 "Segoe UI",system-ui,-apple-system,sans-serif;
  color:#EAF1FF;-webkit-font-smoothing:antialiased;text-shadow:0 1px 2px rgba(0,0,0,.65)}
:where(#bisxRoot *,#bisxMsg *,#bisxAlert *){box-sizing:border-box;pointer-events:none;margin:0;padding:0}
#bisxRoot *,#bisxMsg *,#bisxAlert *{pointer-events:none;box-sizing:border-box}
#bisxRoot{position:fixed;bottom:18px;right:22px;width:410px}
/* Bottom centre, kept clear of the status panel on narrower screens. */
#bisxMsg{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);
  width:clamp(380px,calc(100vw - 940px),600px)}
#bisxAlert{position:fixed;inset:0;z-index:2147483647;display:flex;align-items:center;
  justify-content:center;background:rgba(6,8,18,.50)}
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
.bisxCard{position:relative;overflow:hidden;padding:18px 20px 18px 20px;
  animation:bisxUp .45s cubic-bezier(.2,.9,.25,1.15)}
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
/* Struck through only when a new address replaced it; a working address is
   plain, and one that finds nothing is red but still readable. */
.bisxOld{background:rgba(239,68,68,.12);color:#FCA5A5;text-decoration:line-through;text-decoration-color:rgba(252,165,165,.7)}
.bisxBad{background:rgba(239,68,68,.14);color:#FCA5A5}
.bisxAddr{background:rgba(255,255,255,.10);color:#E6EEFF}
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
.bisxRow.failed{background:rgba(239,68,68,.22);color:#fff}
.bisxRow.failed i{background:#EF4444;border-color:#EF4444;color:#fff}
.bisxRow.failed em{color:#F87171}
.bisxRow.learned i{background:#6366F1;border-color:#6366F1;color:#fff}
.bisxRow.learned em{color:#A5B4FC}
.bisxRow.run{background:rgba(99,140,255,.12);color:#fff}
.bisxRow.run i{border-color:#F59E0B;border-top-color:transparent;animation:bisxRot .9s linear infinite}
.bisxRow.skip{opacity:.45}
/* The failure pop-up: opaque on purpose - it is the one thing on the screen
   the room must read, and the dimmed page around it says the run stopped. */
.bisxAlertCard{position:relative;width:min(640px,88vw);padding:30px 34px 28px;border-radius:24px;
  text-align:center;background:linear-gradient(165deg,#3E0C14,#16060C);
  border:2px solid #EF4444;box-shadow:0 0 0 8px rgba(239,68,68,.20),0 30px 90px rgba(0,0,0,.55);
  animation:bisxPop .5s cubic-bezier(.2,.9,.25,1.2),bisxPulse 1.8s ease-in-out .5s infinite}
.bisxAlertIcon{width:72px;height:72px;margin:0 auto 14px;border-radius:50%;background:#EF4444;
  display:flex;align-items:center;justify-content:center;color:#fff;
  font:800 38px/1 "Segoe UI Symbol","Segoe UI",sans-serif;box-shadow:0 0 0 10px rgba(239,68,68,.22)}
.bisxAlertKick{font:800 12px/1.2 "Segoe UI",sans-serif;letter-spacing:.22em;text-transform:uppercase;color:#FCA5A5}
.bisxAlertTitle{font-size:34px;line-height:1.15;font-weight:800;color:#fff;margin-top:6px}
.bisxAlertStep{display:inline-block;margin-top:14px;padding:10px 20px;border-radius:12px;
  background:rgba(239,68,68,.18);border:1px solid rgba(248,113,113,.6);
  font-size:21px;font-weight:700;color:#fff}
.bisxAlertBody{margin:14px auto 0;max-width:540px;color:#FDE2E2;font-size:15px}
.bisxAlertCard .bisxSel{max-width:520px;margin:16px auto 0;text-align:left}
.bisxAlertCard .bisxStats{max-width:520px;margin:18px auto 0;text-align:left}
@keyframes bisxUp{from{opacity:0;transform:translateY(12px) scale(.97)}to{opacity:1;transform:none}}
@keyframes bisxPop{from{transform:scale(.85)}to{transform:none}}
@keyframes bisxPulse{50%{box-shadow:0 0 0 18px rgba(239,68,68,.08),0 30px 90px rgba(0,0,0,.55)}}
@keyframes bisxRot{to{transform:rotate(360deg)}}
@media (max-width:1300px){#bisxRoot{width:360px;right:12px}}
@media (max-height:860px){#bisxRoot{transform:scale(.86);transform-origin:bottom right;bottom:12px}
  #bisxMsg{transform:translateX(-50%) scale(.9);transform-origin:bottom center;bottom:12px}}
@media (max-height:700px){#bisxRoot{transform:scale(.74)}
  #bisxMsg{transform:translateX(-50%) scale(.8)}}
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
const holder = id => { let e = document.getElementById(id);
  if (!e) { e = document.createElement('div'); e.id = id; document.body.appendChild(e); }
  return e; };
const el = (tag, cls, text) => { const e = document.createElement(tag);
  if (cls) { e.className = cls; } if (text != null) { e.textContent = text; } return e; };
const sel = c => { const g = el('div', 'bisxSel');
  if (c.oldSel) { g.appendChild(el('span', '', c.oldLabel || 'Old address'));
    g.appendChild(el('code', c.oldClass || 'bisxAddr', c.oldSel)); }
  if (c.newSel) { g.appendChild(el('span', '', 'New address'));
    g.appendChild(el('code', 'bisxNew', c.newSel)); }
  return g; };
const stats = list => { const s = el('div', 'bisxStats');
  for (const x of list) { const b = el('div', 'bisxStat');
    b.appendChild(el('small', '', x[0])); b.appendChild(el('b', x[2] || '', x[1]));
    s.appendChild(b); }
  return s; };

/* ---- status panel, bottom right */
const root = holder('bisxRoot');
root.textContent = '';
const bar = el('div', 'bisxGlass bisxTop');
bar.appendChild(el('div', 'bisxPill ' + st.pillClass, st.pill));
root.appendChild(bar);
const ver = el('div', 'bisxVer');
for (const v of st.facts) { const box = el('div', 'bisxGlass');
  box.appendChild(el('small', '', v[0])); box.appendChild(el('b', '', v[1]));
  ver.appendChild(box); }
root.appendChild(ver);
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

/* ---- explanation card, bottom centre */
const msg = holder('bisxMsg');
msg.textContent = '';
if (st.card) {
  const c = st.card;
  const card = el('div', 'bisxGlass bisxCard');
  card.style.setProperty('--acc', c.acc); card.style.setProperty('--glow', c.glow);
  if (c.key === msg.dataset.lastCard) { card.style.animation = 'none'; }
  msg.dataset.lastCard = c.key;
  const head = el('div', 'bisxHead');
  if (c.spin) { head.appendChild(el('div', 'bisxSpin')); }
  else { const ic = el('div', 'bisxIcon'); ic.innerHTML = c.icon; head.appendChild(ic); }
  const tt = el('div');
  tt.appendChild(el('div', 'bisxKick', c.kicker));
  tt.appendChild(el('div', 'bisxTitle', c.title));
  head.appendChild(tt); card.appendChild(head);
  if (c.body) { card.appendChild(el('div', 'bisxBody', c.body)); }
  if (c.oldSel || c.newSel) { card.appendChild(sel(c)); }
  if (c.stats && c.stats.length) { card.appendChild(stats(c.stats)); }
  msg.appendChild(card);
}

/* ---- failure pop-up, centre of the screen */
const old = document.getElementById('bisxAlert');
if (!st.alert) { if (old) { old.remove(); } return true; }
const a = st.alert;
if (old && old.dataset.key === a.key) { return true; }
const box = holder('bisxAlert');
box.textContent = ''; box.dataset.key = a.key;
const pop = el('div', 'bisxAlertCard');
const ic = el('div', 'bisxAlertIcon'); ic.innerHTML = '&#10007;'; pop.appendChild(ic);
pop.appendChild(el('div', 'bisxAlertKick', a.kicker));
pop.appendChild(el('div', 'bisxAlertTitle', a.title));
pop.appendChild(el('div', 'bisxAlertStep', a.step));
if (a.body) { pop.appendChild(el('div', 'bisxAlertBody', a.body)); }
if (a.oldSel) { pop.appendChild(sel(a)); }
if (a.stats && a.stats.length) { pop.appendChild(stats(a.stats)); }
box.appendChild(pop);
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
        self.alert = None
        self.seq = 0

    def step(self, i, state, tag=""):
        """Set one row of the step list: run, ok, healed, failed, learned."""
        self.steps[i].update(state=state, tag=tag,
                             mark=ROW_MARK.get(state, ""))
        self.draw()

    def show(self, kind, kicker, title, body="", old_sel="", new_sel="",
             stats=None, old_label=""):
        """Replace the explanation card at the bottom centre."""
        acc, glow, icon = ACCENT[kind]
        if new_sel:
            old_class = "bisxOld"
        elif kind in ("broken", "healing", "failed"):
            old_class = "bisxBad"
        else:
            old_class = "bisxAddr"
        self.seq += 1
        self.card = {"key": str(self.seq), "acc": acc, "glow": glow,
                     "icon": icon, "spin": kind == "healing",
                     "kicker": kicker, "title": title, "body": body,
                     "oldSel": old_sel, "newSel": new_sel,
                     "oldLabel": old_label, "oldClass": old_class,
                     "stats": stats or []}
        self.draw()

    def fail(self, kicker, title, step, body="", old_sel="", stats=None):
        """Show the failure pop-up in the centre, naming the step that failed.

        The explanation card is cleared: the pop-up says it all, and two
        red cards at once split the room's attention.
        """
        self.seq += 1
        self.card = None
        self.alert = {"key": str(self.seq), "kicker": kicker, "title": title,
                      "step": step, "body": body, "oldSel": old_sel,
                      "oldLabel": "Address", "oldClass": "bisxBad",
                      "stats": stats or []}
        self.draw()

    def draw(self):
        state = {"pill": self.pill, "pillClass": self.pill_class,
                 "facts": self.facts, "card": self.card,
                 "alert": self.alert, "steps": self.steps,
                 "stepsTitle": self.steps_title}
        try:
            self.driver.execute_script(RENDER_JS, state, CSS)
        except WebDriverException:
            pass            # a page mid-navigation; the next draw catches up

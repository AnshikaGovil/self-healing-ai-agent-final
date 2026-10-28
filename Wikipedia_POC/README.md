# Wikipedia POC - self-healing on a real website

A bot written for **Wikipedia's 2010 design** runs on **Wikipedia's 2022
redesign** - the design Wikipedia switched to in 2023. Wikipedia still serves
both, live, at the same address:

| version | URL switch | what the bot meets |
| --- | --- | --- |
| 2010 design - Vector legacy | `?useskin=vector` | all 7 steps work |
| 2022 redesign - Vector 2022 | `?useskin=vector-2022` | 3 of the 7 steps broken |

What the redesign really changed, and what the Healing Agent repaired each
break to through the public healing service and its LLM (measured
2026-10-28, the LLM on a Modal T4):

| step | selector written for 2010 | on the 2022 design | repaired to | healing time |
| --- | --- | --- | --- | --- |
| Log in link | `#pt-login a` | still in the page, but hidden in a menu | `#pt-login-2 > a:nth-of-type(1)` | 20.4 s |
| Search box | `input.vector-search-box-input` | class removed | `input[aria-label="Search Wikipedia"]` | 15.6 s |
| Search button | `#searchButton` | gone; a new button with no id | `#searchform > div:nth-of-type(1) > button:nth-of-type(1)` | 15.6 s |

Title, section headings, last-edited date and the search result use
selectors that survived the redesign, so they pass untouched.

Files: `wiki_bot.py` (the bot), `cards.py` (the message cards) and this
README. The bot imports `../rpa-toolkit/bis.py`, the one-file client, and
writes only `memory/`, which git ignores.

## What the audience sees

Chrome opens maximised. See-through message cards in the bottom-right
corner narrate the run:

* the **website version** (2010 design or 2022 redesign) and the stage
  (before or after the release);
* **HEALING AGENT ON** or **HEALING AGENT OFF** on every after-release card
  (**LEARNING** before the release);
* every broken step as three cards - **broken by the release** (red),
  **self-healing** (amber, while the healing service and its LLM decide) and
  **healed** (green) with the old address struck through, the new one, and
  the **healing time**;
* a checklist of the bot's seven steps, and a summary card at the end.

Every element the bot uses gets a ring on the page: blue when the old
address worked, green when it was healed.

---

## Run the demo against the public service

Nothing to start: the bot uses the public healing service,

    https://anshikagovil--bis-healing.us-east.modal.run

whose LLM runs on a cloud GPU behind it. Every repair is an HTTP call to
it. You need Python 3.9 or later, Google Chrome and an internet connection.
From the repository root:

```bash
pip install selenium
cd Wikipedia_POC
```

The public service sleeps when nobody uses it. The first run after a quiet
spell prints `waiting for the healing service and its AI` for up to a
minute or two while it and its GPU wake, then starts by itself.

**Step 1 - before the release: learn on the 2010 design**

```bash
python wiki_bot.py --design old
```

**LEARNING** cards; all 7 steps pass and every element the bot uses is
fingerprinted for the healing service. *"This is the bot as it runs today,
on the design it was written for."*

**Step 2 - after the release, Healing Agent OFF**

```bash
python wiki_bot.py --design new --plain
```

**HEALING AGENT OFF**. Steps 1-3 pass, then **BOT STOPPED** at step 4 (Log
in link). *"Same bot, same article, Wikipedia's new design. One moved link
and the run is dead until an engineer fixes the script."*

**Step 3 - after the release, Healing Agent ON**

```bash
python wiki_bot.py --design new
```

**HEALING AGENT ON**. Each broken step goes broken - self-healing - healed:
the bot sends the page's elements to the service, which ranks them against
what step 1 learned and asks its LLM; the bot then builds the new selector
in its own browser and checks it finds exactly that element. Ends with **7
of 7 steps passed, 3 healed**. A repair on the public service takes about
20 s - most of it ranking Wikipedia's ~900 elements on the service's CPU,
3-4 s of it the LLM on the GPU.

**Step 4 - the same command again**

```bash
python wiki_bot.py --design new
```

The three repairs now come from **fleet memory**, without asking the LLM
again. *"The first robot paid for the repair; every robot after it gets it
free."*

`--pace` sets how long each card stays up (default `1.8` s; `3` for a slow
walkthrough, `1` for a quick rehearsal).

---

## Or against your own healing service

On a machine with the healing service set up (`../setup.md`), start it and
leave it running:

```bash
cd healing-service
python -m healing_service --host 127.0.0.1 --port 8077 --gpu-layers -1 --context 4096
```

Then, in a second terminal, point the bot at it and run steps 1-4 as above:

```bash
cd Wikipedia_POC
export BIS_SERVICE_URL=http://127.0.0.1:8077      # PowerShell: $env:BIS_SERVICE_URL = "http://127.0.0.1:8077"
```

---

## Before the next demo

Step 3 shows live LLM repairs only once in a session; after that they come
from memory. To show the LLM again, run step 1 again: it starts a new
session on the service - a memory of its own, named by the id saved in
`memory/session.txt` - and steps 2-4 then use that one. One person's demo
never warms another's.

On the public service a session's memory lasts while the service is awake
- twenty minutes after its last call - so run steps 1 to 4 in one sitting.

**Keep to the default article.** The healer reads at most 900 elements a
page, and on a much longer article the Log in link is not among them; the
bot then stops with `no safe repair found - refused, not guessed`. That is
the safe outcome - it says it does not know instead of clicking something
else - but it is not the story to open with.

## Options

| option | default | what it does |
| --- | --- | --- |
| `--design old` / `new` | `new` | before the release (learns) / after it |
| `--plain` | off | Healing Agent OFF - plain Selenium, the control run |
| `--pace SECONDS` | `1.8` | how long each message card stays up |
| `--article NAME` | `Robotic_process_automation` | the article to start on |
| `--query TEXT` | `Machine learning` | what the bot searches for |
| `--headless` | off | no browser window |
| `--service URL` | `BIS_SERVICE_URL`, else the public service | which healing service to use |
| `--session ID` | a new one for `--design old`, then the one in `memory/session.txt` | which of the service's memories to use |
| `--shared` | off | use the service's shared memory instead of a session |

## If you are asked

**"What happens when it is not sure?"** It does not guess. An unsure answer
is held for a person (`repair held for review`); no answer is refused (`no
safe repair found`). Both stop the step instead of clicking the wrong
thing.

**"Can the LLM invent a selector?"** No. The new selectors are built inside
the browser from the live page; the LLM only ever answers with the number of
an element that really exists.

**"Is any of this specific to Wikipedia?"** No. The bot is ordinary Selenium
with its lookups going through `bis.find`, which takes the selector plus a
plain-English description of the element. Nothing in the bot or the service
knows about Wikipedia.

## If something goes wrong

| what you see | do this |
| --- | --- |
| `waiting for the healing service ...` for a while | the service and its GPU are waking; the bot carries on by itself (it waits up to 4 minutes) |
| `The healing service ... is not answering` | check the internet connection, or the address given with `--service` |
| `service : ready - ranker only, no AI answered` | no GPU answered - the run still works, but expect `repair held for review` at step 4 |
| step 3 heals from fleet memory straight away | this session already healed once - run step 1 again for a fresh one |
| `FAILED` with `repair held for review` or `no safe repair found` | step 1 was not run against this service, or more than twenty quiet minutes passed since - run steps 1 and 3 |

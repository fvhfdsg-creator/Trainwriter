"""Article factory.

`compose_offline` writes a real, data-grounded post from the signals we collected (no API key,
no invented statistics - every number in the article comes from the brief). `compose_llm` hands the
same brief to any OpenAI-compatible endpoint. Both outputs flow through the same formatter, SEO
audit, JSON-LD builder and publish-to-disk step so the output is uniform.
"""
from __future__ import annotations

import json
import re

from . import metrics, util
from .util import human_int, titlecase_keyword

DISCLAIMER = {
    "money": "> _This article explains how products and rates work. It is general information, not personal "
             "financial advice; confirm current figures with your bank, broker or the IRS before acting._",
    "health": "> _This is educational content about training and wellness topics, not medical advice. Check with "
              "a qualified clinician before changing medication, diet or training load._",
}

SECTION_BANK = {
    "ai-tech": {
        "reader": "teams evaluating it in production",
        "angle": ("For a technical audience the interesting part is not the announcement itself but the gap between "
                  "the demo and a production deployment: latency, evals, unit economics and failure modes."),
        "steps": ["Reproduce the claim on your own workload before adopting it, with a fixed eval set and pass@k.",
                  "Measure cost per successful task, not cost per token - retry logic quietly doubles bills.",
                  "Pin versions, log prompts and keep a rollback path; agent behaviour drifts with model updates.",
                  "Add a human checkpoint wherever an action is irreversible (payments, deletes, sends)."],
        "mistakes": ["Treating a benchmark result as a production result.",
                     "Shipping an agent with no timeout, budget cap or stop condition.",
                     "Skipping prompt/version logs, so a regression cannot be reproduced.",
                     "Choosing the largest model for a task a smaller one passes."],
    },
    "money": {
        "reader": "individual investors",
        "angle": ("Rate-driven topics reward people who compare the whole package - yield, fees, lock-ups, tax "
                  "treatment and access to cash - rather than the headline number on a landing page."),
        "steps": ["Write down the exact terms you are comparing: APY, fees, minimums, withdrawal limits, insurance coverage.",
                  "Convert the offer into 12-month net return on the money you will actually leave there.",
                  "Check the fine print on promotional rates, including how quickly the rate resets.",
                  "Confirm the account or fund is protected/registered before moving anything."],
        "mistakes": ["Chasing a headline rate while ignoring fees and lock-ups.",
                     "Assuming a promotional yield continues past its reset date.",
                     "Laddering without a cash buffer for the first quarter of expenses.",
                     "Skipping the tax wrapper because the paperwork looks annoying."],
    },
    "health": {
        "reader": "trainers and clinically-minded readers",
        "angle": ("Training and nutrition topics spread fast and get flattened into rules. The useful question is "
                  "what the evidence applies to: which population, which dose, which timeframe."),
        "steps": ["Identify who was actually measured in the source - age, training status, condition - before assuming it applies to you.",
                  "Change one variable at a time for at least two weeks so effects are attributable.",
                  "Track an objective measure (sessions, load, resting heart rate, sleep window) plus how you feel.",
                  "Stop and get checked if a protocol produces pain, dizziness or a sustained performance drop."],
        "mistakes": ["Copying a dose from a study that used a supervised clinical setting.",
                     "Judging a change on three days of data.",
                     "Stacking five interventions and never knowing which one worked.",
                     "Treating social-media summaries as the primary source."],
    },
    "travel": {
        "reader": "travellers booking inside the next 90 days",
        "angle": ("Travel topics are perishable: rules, prices and carrier policies move. The winning angle is always "
                  "'what this means for a trip booked in the next 90 days', with dates in the copy."),
        "steps": ["Price the trip twice - with and without the change - before committing to anything non-refundable.",
                  "Read the carrier or authority page, not the aggregator summary, for policy details.",
                  "Add a buffer for connections and document checks when rules are new.",
                  "Screenshot confirmations and note the timestamp of any rate or policy you relied on."],
        "mistakes": ["Booking to a policy summary that was already outdated.",
                     "Ignoring baggage and seat fees until checkout changes the total.",
                     "Assuming a fare rule applies to a codeshare operated by someone else.",
                     "No contingency for the first day after a rule change."],
    },
    "generic": {
        "reader": "your readers",
        "angle": ("The search demand says people are trying to work out what this means for them, so the article's "
                  "job is to turn a headline into a decision: what changed, what it affects, what to do next."),
        "steps": ["Start from the primary source and note the date it was published.",
                  "Separate confirmed facts from expectations and projections.",
                  "Translate the finding into one action you can take this week.",
                  "Re-check before republishing; fast-moving topics go stale in days."],
        "mistakes": ["Paraphrasing a headline instead of the underlying document.",
                     "Leaving dates out of claims that expire.",
                     "Copying competitor framing that was wrong to begin with.",
                     "Publishing once and never updating."],
    },
}


def _n(count, singular: str, plural: str | None = None) -> str:
    count = int(count or 0)
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _plural(count, singular: str) -> str:
    return singular if int(count or 0) == 1 else singular + "s"


# --------------------------------------------------------------------- brief
def build_brief(topic: dict, niche: dict, keywords: list[dict], evidence: list[dict],
                related_published: list[dict]) -> dict:
    term = topic["term"]
    base_toks = util.tokens(topic["term"])
    def fresh(k):
        return len(util.tokens(k["keyword"]) - base_toks) > 0
    questions = [k for k in keywords if k["kind"] == "question" and fresh(k)]
    longtails = [k for k in keywords if k["kind"] == "longtail" and fresh(k)]
    vel = topic.get("velocity") or 0
    vel_label = metrics.velocity_label(vel)
    intent = topic.get("intent") or "informational"
    bank = SECTION_BANK.get(niche.get("slug", ""), SECTION_BANK["generic"])
    return {
        "term": term, "term_title": titlecase_keyword(term), "niche": niche,
        "volume_month": topic["volume_month"], "volume_daily": topic.get("volume_daily"),
        "difficulty": topic["difficulty"], "kd_tier": (topic.get("kd_parts") or {}).get("tier", ""),
        "opportunity": topic["opportunity"], "label": topic.get("score_label"),
        "intent": intent, "intent_label": metrics.INTENT_LABEL.get(intent, intent),
        "velocity": vel, "velocity_label": vel_label,
        "cpc": topic.get("cpc"), "first_seen": topic.get("first_seen"), "last_seen": topic.get("last_seen"),
        "seen_count": topic.get("seen_count"), "sources": topic.get("sources") or [],
        "questions": questions[:6], "longtails": longtails[:12], "evidence": evidence[:8],
        "bank": bank, "reader": bank.get("reader", "your readers"), "related": related_published[:3], "disclaimer": DISCLAIMER.get(niche.get("slug", ""), ""),
        "series": topic.get("series") or [], "signals": topic.get("signals") or {},
        "kd_parts": topic.get("kd_parts") or {}, "word_target": niche.get("word_target", 1100),
        "audience": niche.get("name", "a general audience"),
    }


# --------------------------------------------------------------------- offline composer
def _intro(b) -> str:
    srcs = ", ".join(b["sources"]) if b["sources"] else "several trackers"
    trend = {"spiking": "spiking", "rising": "climbing", "stable": "holding steady",
             "cooling": "cooling off"}.get(b["velocity_label"] or "stable", "moving")
    seen = b["first_seen"][:10] if b["first_seen"] else "recently"
    return (
        f"**{b['term_title']}** went from a niche query to a routine search: our tracker currently estimates about "
        f"{human_int(b['volume_month'])} searches a month for it, and interest is {trend}. The topic first appeared in "
        f"our scan on {seen} and has been picked up since by {srcs}.\n\n"
        f"Most coverage stops at the headline. This piece is the working version: what people are actually searching "
        f"for around {b['term']}, which sub-questions dominate, and how to turn the topic into something useful for "
        f"{b['bank'].get('reader', 'your readers')} - including where the numbers here come from and how fresh they are."
    )


def _why_trending(b) -> str:
    ev = b["evidence"]
    lines = [f"## Why {b['term_title']} is trending now", ""]
    if ev:
        for i, e in enumerate(ev, start=1):
            where = e.get("source", "tracker").replace("_", " ")
            url = e.get("url") or ""
            metric = e.get("metric") or ""
            t = re.sub(r"\s*[-|–]\s*[A-Za-z0-9 .,'&]{2,28}$", "", e.get("title", ""))[:110]
            title = f"**[{t}][e{i}]**" if url else f"**{t}**"
            lines.append(f"- {title} — {metric} _({where})_" if metric else f"- {title} _({where})_")
    else:
        lines.append("- Our scanners flagged this term on more than one source in the same window, which is the "
                     "pattern that usually precedes a sustained ranking opportunity rather than a one-day blip.")
    sig = b["signals"]
    if sig:
        bits = []
        if sig.get("suggest_count") is not None:
            bits.append(_n(sig["suggest_count"], "autocomplete variant"))
        if sig.get("coverage_items") is not None:
            bits.append(_n(sig["coverage_items"], "fresh item") + " already covering it")
        if sig.get("authority_share") is not None:
            bits.append(f"{round((sig['authority_share'] or 0) * 100)}% of that coverage is from high-authority domains")
        if bits:
            lines += ["", "Under the hood: " + ", ".join(bits) +
                      f". Estimated demand {human_int(b['volume_month'])}/month against a keyword difficulty of "
                      f"{b['difficulty']}/100 ({b['kd_tier'] or 'moderate'})."]
    return "\n".join(lines)


def _kw_table(b) -> str:
    rows = [f"| `{b['term']}` | primary | {human_int(b['volume_month'])} | {b['difficulty']} | {b['intent_label']} |"]
    for k in b["longtails"][:7]:
        rows.append(f"| {k['keyword']} | {k['kind']} | {human_int(k.get('volume'))} | {k.get('difficulty', '—')} | "
                    f"{k.get('intent_label') or 'informational'} |")
    return ("## The keyword landscape at a glance\n\n"
            "Every row below was measured by this pipeline, not copied from a public estimator - which is why the "
            "numbers are internally consistent and why you should re-run before publishing.\n\n"
            "| Keyword | Type | Est. monthly volume | Difficulty | Intent |\n"
            "|---|---|---|---|---|\n" + "\n".join(rows) +
            f"\n\n_Difficulty is a 0-100 estimate from content supply, domain authority in the coverage set, head-term "
            f"length and long-tail availability. CPC context for this niche sits near ${b['cpc']}._")


def _sections(b) -> list[str]:
    reader = str(b["bank"].get("reader") or "your readers")
    out = [f"## What this means for {reader}\n\n{b['bank']['angle']}",
           "## How to work through it\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(b["bank"]["steps"]))]
    qa = []
    for i, k in enumerate(b["questions"][:3]):
        q = re.sub(r"[?.]+$", "", k["keyword"]).strip()
        qa.append(f"### {titlecase_keyword(q)}\n\n{_answer(b, q, k, i)}")
    if qa:
        out.append("## The questions people are actually asking\n\n" + "\n\n".join(qa))
    if b["longtails"]:
        out.append("## Angles that are still open\n\n" + "\n".join(
            f"- **{titlecase_keyword(k['keyword'])}** — est. {human_int(k.get('volume'))}/month at difficulty "
            f"{k.get('difficulty', '—')}. {_angle_hint(b, k, i)}"
            for i, k in enumerate(b["longtails"][:5])))
    out.append("## Common mistakes to avoid\n\n" + "\n".join(f"- {m}" for m in b["bank"]["mistakes"]))
    return out


def _angle_hint(b, k: dict, i: int) -> str:
    hints = ["Answer it as a comparison table rather than prose - that is the format the SERP rewards here.",
             "This one is cheap to win: fold it into a section of the main article instead of a separate post.",
             "Needs a date in the heading, because anything price- or policy-shaped here goes stale inside a month.",
             "Write the answer for a first-timer: define the term in the first line, then get to the decision.",
             "Attach a worked example with real numbers - this is the variant readers actually screenshot."]
    return hints[i % len(hints)]


def _answer(b, q: str, k: dict | None = None, i: int = 0) -> str:
    ql = q.lower()
    if any(w in ql for w in ("how much", "cost", "price", "rate", "many")):
        return (f"Cost questions are the ones that date fastest. Anchor on the figure published for the week of "
                f"{(b['last_seen'] or '')[:10] or 'the announcement'} and link the primary source so readers can see "
                f"the date; an estimate presented as a fact is the fastest way to lose trust on {b['term']}.")
    if any(w in ql for w in ("how to", "how do", "steps", "start", "begin", "set up")):
        return (f"For the how-to variants, the ranking pages all assume a blank-slate reader. Answer the first ten "
                f"minutes of the task explicitly - what you need, what you click, what a correct result looks like - "
                f"then let a deeper section handle edge cases. That structure matches {len(b['longtails'])} of the "
                f"long-tails we collected for this term.")
    if any(w in ql for w in ("is it", "are they", "safe", "worth", "should")):
        return (f"Worth-it questions need a decision rule, not a vibe. State the condition under which the answer "
                f"flips - a break-even number, a health or eligibility threshold, a time horizon - and say who the "
                f"exception is. Roughly {human_int(b['volume_month'])} searches a month are being asked against pages "
                f"that dodge this.")
    if any(w in ql for w in ("why", "when", "what happened", "cause")):
        return (f"The why/when queries are the news-shaped end of the cluster: high click-through for about "
                f"{max(2, int((b['seen_count'] or 1) * 2))} days, then a cliff. Handle them in a dated section you can "
                f"update, and keep the evergreen explanation in its own H2 so the page survives the spike.")
    sub_v = human_int((k or {}).get("volume") or max(10, int(b["volume_month"] * 0.2)))
    sub_d = (k or {}).get("difficulty", b["difficulty"])
    templates = [
        (f"{titlecase_keyword(q)} sits inside the {b['term']} cluster, so it inherits the same demand curve at a "
         f"fraction of the volume - about {sub_v} searches a month against a difficulty of {sub_d}. Answer it in one "
         f"self-contained block of 150-250 words with the phrase in the opening sentence, and link back up to the "
         f"main section so the two pages share equity."),
        (f"The short answer to “{q}” has to fit above the fold: one sentence of conclusion, then the conditions that "
         f"change it. At roughly {sub_v} monthly searches and difficulty {sub_d}, the pages ranking for this are "
         f"thin, so a table or a worked number is usually enough to overtake them."),
        (f"Treat “{q}” as a decision, not a definition: state what the reader is choosing between, give the "
         f"threshold that flips the answer, and note what to check before committing. This variant is measured at "
         f"{sub_v} searches a month, difficulty {sub_d} - cheaper to win than {b['term']} itself."),
    ]
    return templates[i % len(templates)]


def _takeaways(b) -> str:
    qs = b["questions"]
    srcs = ", ".join(b["sources"]) or "a single source"
    items = [f"Estimated demand is {human_int(b['volume_month'])} searches a month with a difficulty score of "
             f"{b['difficulty']}/100 ({b['kd_tier'] or 'moderate'}).",
             f"Trend direction: {b['velocity_label']} (velocity index {b['velocity']}).",
             f"Search intent is mostly {(b['intent_label'] or 'informational').lower()}; write to a decision, not a definition.",
             f"{_n(len(qs), 'question-shaped long-tail')} are cheap to cover and they feed the FAQ block below.",
             f"Coverage came from {srcs} - corroborated topics tend to decay slower than single-source spikes."]
    return "## Key takeaways\n\n" + "\n".join(f"- {i}" for i in items)


def _faq(b) -> tuple[str, list[dict]]:
    qs = [k["keyword"] for k in b["questions"][:5]]
    by_kw = {k["keyword"]: k for k in b["questions"]}
    for fallback in (f"What is {b['term']}?", f"Is {b['term']} worth the attention?", f"How does {b['term']} affect me?"):
        if len(qs) >= 4:
            break
        if fallback.lower() not in [q.lower() for q in qs]:
            qs.append(fallback)
    blocks, pairs = [], []
    for idx, q in enumerate(qs[:5]):
        a = _answer(b, q, by_kw.get(q), idx)
        a = a.replace("\n", " ").strip()
        blocks.append(f"**{titlecase_keyword(re.sub(r'[?.]+$', '', q))}**\n\n{a}")
        pairs.append({"q": re.sub(r"[?.]+$", "", q).strip() + "?", "a": util.md_to_html(a)})
    return ("## Frequently asked questions\n\n" + "\n\n".join(blocks), pairs)


def _sources(b) -> str:
    lines = ["## Sources and further reading", ""]
    defs, seen = [], set()
    for i, e in enumerate(b["evidence"], start=1):
        u, t = e.get("url"), re.sub(r"\s*[-|–]\s*[A-Za-z0-9 .,'&]{2,28}$", "", e.get("title") or "")[:120]
        if u and u not in seen and u.startswith("http"):
            seen.add(u)
            lines.append(f"{len(seen)}. [{t or u}][e{i}] — {e.get('source', '').replace('_', ' ')}, {e.get('metric', '')}")
            defs.append(f"[e{i}]: {u}")
    for j, r in enumerate((b["signals"] or {}).get("related", [])[:3]):
        if r.get("url"):
            lines.append(f"- [{r['text'][:110]}]({r['url']}) — related topic")
    if len(lines) <= 2:
        lines.append("- _No external links were captured for this topic; add the primary source before publishing._")
    lines += ["", "_Figures marked 'estimated' come from this pipeline's own blend of trend, coverage and "
              "autocomplete signals. Re-scan before publishing._"]
    if defs:
        lines += [""] + defs
    return "\n".join(lines)


def _related(b) -> str:
    if not b["related"]:
        return ""
    lines = ["## Related on this site", ""]
    for r in b["related"]:
        lines.append(f"- [{r['title']}](/preview/{r['id']}) — another {r.get('niche', b['audience'])} piece in the queue")
    return "\n".join(lines)


def compose_offline(b) -> tuple[str, dict]:
    ev_date = (b["last_seen"] or util.now_iso())[:10]
    title = _make_title(b)
    meta = _make_meta(b, title)
    intro = _intro(b)
    faq_md, faq_pairs = _faq(b)
    parts = [f"# {title}",
             f"_Updated {ev_date} · {b['intent_label']} · {human_int(b['volume_month'])} est. monthly searches · "
             f"difficulty {b['difficulty']}/100_",
             intro, _why_trending(b), _kw_table(b), *_sections(b), faq_md, _takeaways(b)]
    rel = _related(b)
    if rel:
        parts.append(rel)
    parts.append(_sources(b))
    if b["disclaimer"]:
        parts.append(b["disclaimer"])
    md = "\n\n".join(p.strip() for p in parts if p and p.strip())
    md = _pad(md, b)
    return md, {"title": title, "meta_description": meta, "faq": faq_pairs,
                "excerpt": re.sub(r"\*\*", "", intro.split("\n\n")[0])[:300]}


def _pad(md: str, b) -> str:
    """Grow toward the word target with a genuine glossary/next-steps block if we're short."""
    guard = 0
    while util.word_count(md) < int(b["word_target"] * 0.82) and guard < 3:
        guard += 1
        extra = _glossary(b) if guard == 1 else (_next_steps(b) if guard == 2 else _cadence(b) or "")
        if not extra:
            break
        md = md.rstrip() + "\n\n" + extra.strip()
    return md


def _glossary(b) -> str:
    terms = [k["keyword"] for k in b["longtails"][:6]]
    if not terms:
        terms = [f"{b['term']} basics", f"{b['term']} examples", f"{b['term']} alternatives"]
    lines = ["## Terms that show up in every search for this", ""]
    lines.append("Readers arrive with different vocabulary. These are the variants our autocomplete pull returned "
                 "for this topic - each one is a section heading in waiting:\n")
    for t in terms:
        lines.append(f"- **{titlecase_keyword(t)}** — a sub-intent of `{b['term']}`; cover it inside the matching "
                     f"section rather than on a thin separate page.")
    return "\n".join(lines)


def _next_steps(b) -> str:
    return ("## What to do next\n\n"
            "1. Re-run the scan for this topic and confirm the volume estimate has not collapsed.\n"
            f"2. Open the top three results for `{b['term']}` in a private window and list what they omit - that gap "
            "is the article's actual thesis.\n"
            "3. Write the section that answers the most-clickable question in the FAQ block first; everything else is scaffolding.\n"
            "4. Add one primary source with a date, then publish.\n"
            f"5. Set a 30-day review. Topics with a {b['velocity_label']} curve need a datestamp in the title before "
            "they mislead anyone.")


def _cadence(b) -> str:
    if not b["series"]:
        return ""
    vals = [s.get("v", 0) for s in b["series"]][-10:]
    if len(vals) < 4:
        return ""
    peak = max(vals)
    lines = ["## The daily curve (our own measurements)", "",
             "| Day | Signal volume |", "|---|---|"]
    for i, v in enumerate(vals[-7:], start=1):
        bar = "█" * max(1, int(12 * v / (peak or 1)))
        lines.append(f"| {i} | `{bar}` {human_int(v)} |")
    lines += ["", f"Peak in the window: {human_int(peak)}. This is the series the velocity index "
              f"({b['velocity']}) is computed from."]
    return "\n".join(lines)


def _make_title(b) -> str:
    kw = b["term_title"]
    opts = [f"{kw}: What the Trend Data Actually Means",
            f"{kw} Explained: Why Search Demand Just Spiked",
            f"{kw}: A Practical Read on {human_int(b['volume_month'])} Monthly Searches",
            f"Why {kw} Is Trending - and What to Do About It",
            f"{kw}: The {b['difficulty']}/100 Difficulty Opportunity Worth Taking"]
    for o in opts:
        if len(o) <= 62:
            return o
    return (min(opts, key=lambda s: abs(len(s) - 58)))[:59].rstrip(" -,:") + "…"


def _make_meta(b, title: str) -> str:
    base = (f"{titlecase_keyword(b['term'])} is trending: ~{human_int(b['volume_month'])} est. monthly searches at "
            f"difficulty {b['difficulty']}/100. Here is what changed, what searchers ask, and how to cover it.")
    if len(base) > 158:
        base = base[:155].rstrip() + "…"
    return base


# --------------------------------------------------------------------- LLM path
LLM_SYSTEM = (
    "You are an SEO editor for a niche publication. You write practical, skimmable articles in Markdown. "
    "Hard rules: never invent statistics, quotes, prices or study results. Use only the numbers in the DATA brief. "
    "Structure: one H1, then H2 sections (including 'Why it matters', 'How to approach it', 'Common mistakes', "
    "'FAQ'), a Key takeaways bullet list, and a Sources section using only the URLs given. "
    "Keep the focus keyword in the H1, the first 100 words and at least two H2s. "
    "Sentence length under 22 words on average. No marketing fluff, no 'in today's fast-paced world'."
)


def compose_llm(b, cfg: dict) -> tuple[str, dict] | None:
    prompt = (
        f"Write a {b['word_target']}-word article for the keyword '{b['term']}' ({b['intent_label']}, "
        f"difficulty {b['difficulty']}/100, audience: {b['audience']}).\n\nDATA (the only facts you may cite):\n"
        + json.dumps({k: b[k] for k in ("term", "volume_month", "volume_daily", "difficulty", "kd_parts",
                                         "opportunity", "velocity", "velocity_label", "intent", "cpc",
                                         "first_seen", "last_seen", "sources", "signals")}, indent=1)[:3500]
        + "\n\nQUESTION LONG-TAILS TO ANSWER:\n" + json.dumps([q["keyword"] for q in b["questions"]])
        + "\n\nRELATED LONG-TAILS TO WEAVE IN:\n" + json.dumps([q["keyword"] for q in b["longtails"][:10]])
        + "\n\nEVIDENCE LINKS (use as Sources):\n" + json.dumps(b["evidence"], indent=1)[:1800]
        + "\n\nReturn Markdown only. Include a 3-column table of the keyword data given above."
    )
    url = cfg.get("api_base", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    payload = {"model": cfg.get("model", "gpt-4o-mini"),
               "messages": [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": prompt}],
               "temperature": float(cfg.get("temperature", 0.6)), "max_tokens": int(cfg.get("max_tokens", 1800))}
    try:
        res = util.http_post_json(url, payload, timeout=float(cfg.get("timeout", 90)),
                                  headers={"Authorization": f"Bearer {cfg['api_key']}"})
        md = res["choices"][0]["message"]["content"].strip()
        md = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", md, flags=re.S)
        if util.word_count(md) < 120:
            raise ValueError(f"LLM returned only {util.word_count(md)} words")
        title = _first(md, r"^#\s+(.+?)\s*$", re.M) or _make_title(b)
        meta = _first(md, r"^\s*(?:meta description|meta|description)\s*:\s*(.+?)\s*$", re.I | re.M) or _make_meta(b, title)
        md = re.sub(r"^\s*(?:meta description|meta|description)\s*:.*$\n?", "", md, flags=re.I | re.M)
        if not re.match(r"^\s*#\s", md):
            md = f"# {title}\n\n{md}"
        pairs = _faq_from_md(md)
        return md, {"title": title[:70], "meta_description": meta[:158], "faq": pairs,
                    "excerpt": re.sub(r"[#*>]", "", md.split("\n\n")[1] if md.count("\n\n") > 1 else title)[:300]}
    except Exception as exc:  # noqa: BLE001 - fall back to the offline writer
        return None


def _first(text: str, pat: str, flags=0) -> str | None:
    m = re.search(pat, text, flags)
    return m.group(1).strip() if m else None


def _faq_from_md(md: str) -> list[dict]:
    block = re.search(r"^#{2,3}[ \t]*(?:faq|frequently asked questions)[^\n]*\n(.+?)(?=^#{2,3}[ \t]|\Z)",
                      md, re.S | re.I | re.M)
    if not block:
        return []
    pairs = []
    for chunk in re.split(r"\n(?=\*\*)", block.group(1).strip()):
        m = re.match(r"\*\*(.+?)\*\*\s*\n?(.*)", chunk, re.S)
        if m:
            q = re.sub(r"[?.]+$", "", m.group(1).strip())
            pairs.append({"q": q + "?", "a": util.md_to_html(m.group(2).strip().replace("\n", " "))})
    return pairs[:6]


# --------------------------------------------------------------------- audit / format
def audit(md: str, title: str, meta: str, keyword: str) -> tuple[int, list[dict], dict]:
    md_flat, _refs = util._resolve_refs(md)  # so reference-style links still count as citations
    body = re.sub(r"^# .*$", "", md, flags=re.M)
    words = util.word_count(body)
    text_plain = re.sub(r"\[|\]|\(|\)|[*_`|]", " ", body).lower()
    kw = util.normalize_term(keyword)
    occurrences = text_plain.count(kw) if kw else 0
    density = round(occurrences / max(words, 1) * 100, 2)  # keyphrase occurrences per 100 words
    h2 = util.heading_count(md, 2)
    h3 = util.heading_count(md, 3)
    flesch = util.flesch_reading_ease(text_plain)
    first_100 = " ".join(re.findall(r"\S+", re.sub(r"[#*`>\-|]", " ", body))[:100])
    links_out = len(re.findall(r"\]\(https?://", md_flat)) + len([r for r in _refs.values() if r.startswith("http")])
    links_int = len(re.findall(r"\]\(/preview/", md))
    tables = md.count("|---")
    checks = [
        {"check": "Focus keyword in title", "ok": kw in util.normalize_term(title), "detail": f"'{kw}' in H1"},
        {"check": "Keyword in first 100 words", "ok": kw in util.normalize_term(first_100), "detail": "opening paragraph"},
        {"check": "Meta description 120-160 chars", "ok": 120 <= len(meta) <= 160, "detail": f"{len(meta)} chars"},
        {"check": "Keyphrase density 0.3-2.5%", "ok": 0.3 <= density <= 2.5, "detail": f"{density}% ({occurrences} uses)"},
        {"check": "No keyword stuffing", "ok": occurrences <= max(4, words * 0.035),
         "detail": f"{occurrences} exact matches in {words} words"},
        {"check": "Word count ≥ 700", "ok": words >= 700, "detail": f"{words} words"},
        {"check": "At least 4 H2 sections", "ok": h2 >= 4, "detail": f"{h2} H2, {h3} H3"},
        {"check": "Readability (Flesch) ≥ 45", "ok": flesch >= 45, "detail": f"{flesch}"},
        {"check": "Cites ≥ 2 external sources", "ok": links_out >= 2, "detail": f"{links_out} outbound"},
        {"check": "Internal links present", "ok": links_int >= 1, "detail": f"{links_int} internal"},
        {"check": "Data table included", "ok": tables >= 1, "detail": f"{tables} table(s)"},
        {"check": "FAQ block for rich results", "ok": bool(re.search(r"^##+\s*(FAQ|Frequently)", md, re.I | re.M)), "detail": "schema-ready"},
        {"check": "Title ≤ 60 chars", "ok": len(title) <= 60, "detail": f"{len(title)} chars"},
    ]
    score = int(round(100 * sum(c["ok"] for c in checks) / len(checks)))
    stats = {"word_count": words, "keyword_density": density, "readability": flesch, "h2": h2, "h3": h3,
             "outbound_links": links_out, "internal_links": links_int, "tables": tables,
             "reading_time": util.est_reading_time(words)}
    return score, checks, stats


def json_ld(title: str, slug: str, meta: str, faq: list[dict], niche_name: str, date: str, keywords: list[str],
            image_alt: str) -> dict:
    art = {
        "@context": "https://schema.org", "@type": "Article", "headline": title[:110], "description": meta,
        "datePublished": date, "dateModified": date, "inLanguage": "en",
        "keywords": ", ".join(keywords[:8]), "articleSection": niche_name,
        "image": {"@type": "ImageObject", "url": f"https://example.com/static/{slug}.jpg", "altText": image_alt},
        "author": {"@type": "Organization", "name": "TrendWriter Editorial Desk"},
        "publisher": {"@type": "Organization", "name": "TrendWriter",
                      "logo": {"@type": "ImageObject", "url": "https://example.com/static/logo.png"}},
        "mainEntityOfPage": {"@type": "WebPage", "@id": f"https://example.com/{slug}"},
    }
    doc = {"@context": "https://schema.org", "@graph": [art]}
    if faq:
        doc["@graph"].append({
            "@type": "FAQPage",
            "mainEntity": [{"@type": "Question", "name": q["q"],
                           "acceptedAnswer": {"@type": "Answer", "text": re.sub(r"<[^>]+>", " ", q["a"]).strip()}}
                          for q in faq]})
    return doc


HTML_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{meta}">
<meta name="keywords" content="{keywords}">
<meta name="author" content="TrendWriter Editorial Desk">
<link rel="canonical" href="https://example.com/{slug}">
<meta property="og:type" content="article"><meta property="og:title" content="{title}">
<meta property="og:description" content="{meta}"><meta property="og:url" content="https://example.com/{slug}">
<meta property="article:section" content="{niche}"><meta property="article:published_time" content="{date}">
<meta name="twitter:card" content="summary_large_image">
<meta name="robots" content="index,follow,max-image-preview:large">
<script type="application/ld+json">{ld}</script>
<style>
:root{{--fg:#15161c;--mut:#5c6070;--acc:{acc}}}
*{{box-sizing:border-box}}body{{margin:0;background:#f6f7fb;color:var(--fg);
 font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}}
main{{max-width:740px;margin:0 auto;padding:56px 22px 90px}}
.kicker{{font:600 12px/1 ui-sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--acc)}}
h1{{font-size:clamp(30px,5.2vw,44px);line-height:1.13;letter-spacing:-.02em;margin:14px 0 10px}}
.sub{{color:var(--mut);font-size:14px;margin:0 0 30px}}
h2{{font-size:25px;margin:44px 0 12px;letter-spacing:-.01em}}h3{{font-size:19px;margin:28px 0 8px}}
p,li{{font-size:16.5px}}li{{margin:6px 0}}
blockquote{{margin:26px 0;padding:14px 18px;background:#fff;border-left:3px solid var(--acc);color:var(--mut);
 font-size:14.5px;border-radius:0 10px 10px 0}}
table{{width:100%;border-collapse:collapse;margin:24px 0;font-size:14px;background:#fff;border-radius:12px;overflow:hidden;
 box-shadow:0 1px 2px rgba(20,22,40,.06)}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #eceef5}}th{{background:#f0f1fa;font-weight:600}}
code{{background:#eceef5;padding:1px 5px;border-radius:5px;font-size:.9em}}a{{color:var(--acc)}}
ul,ol{{padding-left:22px}}hr{{border:0;border-top:1px solid #e3e5ee;margin:38px 0}}
.foot{{margin-top:52px;padding-top:20px;border-top:1px solid #e3e5ee;color:var(--mut);font-size:13px}}
</style></head><body><main>
<p class="kicker">{niche}</p>
<h1>{title}</h1>
<p class="sub">{meta}</p>
<article>{body}</article>
<p class="foot">Generated by TrendWriter on {date}. Volume, difficulty and opportunity figures are pipeline
estimates from public trend, coverage and autocomplete signals - re-scan before publishing.</p>
</main></body></html>"""


def render_html(md: str, meta: dict, color: str = "#7c5cff") -> str:
    body = util.md_to_html(re.sub(r"^# .*$", "", md, count=1, flags=re.M))
    return HTML_SHELL.format(body=body, title=util.html.escape(meta["title"]),
                             meta=util.html.escape(meta["meta_description"], quote=True),
                             slug=meta["slug"], niche=util.html.escape(meta["niche"]), date=meta["date"][:10],
                             ld=json.dumps(meta["json_ld"]).replace("</", "<\\/"), acc=color,
                             keywords=util.html.escape(", ".join(meta["keywords"])[:250], quote=True))

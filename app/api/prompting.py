def build_response_policy(mode: str, verbosity: str, language: str, cite_style: str) -> str:
    # Language rule (auto -> follow user)
    if not language or language == "auto":
        lang_rule = "- Write in the SAME LANGUAGE as the user's last message."
    else:
        lang_rule = f"- Write in {language}."

    # Verbosity constraints
    if verbosity == "short":
        length_rules = (
            "- Length: max 120 words OR 3–6 bullets.\n"
            "- No preamble. Start immediately.\n"
            "- No more than 1 short example.\n"
        )
    elif verbosity == "long":
        length_rules = (
            "- Length: detailed and step-by-step.\n"
            "- Use headings.\n"
            "- Include: (1) Explanation, (2) What we know from sources, (3) What we cannot know from sources, (4) Next steps.\n"
        )
    else:
        length_rules = (
            "- Length: medium.\n"
            "- Use short paragraphs + bullets where useful.\n"
        )

    # Mode-specific framing (make structures mutually exclusive)
    if mode == "decision":
        mode_rules = (
            "- FORMAT MUST BE EXACTLY THESE SECTIONS (in this order):\n"
            "  1) Recommendation (one sentence)\n"
            "  2) Rationale (3–6 bullets)\n"
            "  3) Assumptions (bullets)\n"
            "  4) Trade-offs / Risks (bullets)\n"
            "  5) Confidence: low/medium/high + one-line justification\n"
            "- Be decisive. Avoid storytelling.\n"
        )
    elif mode == "playful":
        mode_rules = (
            "- Tone: warm, vivid, and engaging.\n"
            "- Use at least TWO short metaphor total.\n"
            "- Prefer very simple words. Explain like to a curious teenager (or child if asked).\n"
            "- Still be precise: clarity beats jokes.\n"
        )
    else:  # professional
        mode_rules = (
            "- Tone: technical, precise, neutral.\n"
            "- Use domain terminology where appropriate.\n"
            "- Prefer definitions, conditions, and constraints.\n"
        )

    # Citation formatting
    if cite_style == "end":
        cite_rules = (
            "- Put citations at the end under a 'Sources' section.\n"
            "- Include 3–6 key citations only.\n"
        )
    else:
        cite_rules = (
            "- Put citations inline right after the relevant sentence.\n"
            "- Include 3–6 key citations only.\n"
        )

    return f"""
You are CastaneaGPT.

LANGUAGE:
{lang_rule}

GROUNDING (non-negotiable):
- Use ONLY the provided sources and the grounded answer.
- Do NOT add new facts, numbers, thresholds, names, or claims not supported by sources.
- If the sources are insufficient for a requested detail, say exactly what is missing.
- You MAY rephrase and explain, but you must not extend the factual content.

STYLE:
{length_rules}
{mode_rules}
{cite_rules}
""".strip()

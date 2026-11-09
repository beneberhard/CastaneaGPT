# Image assessment prototype

The `/assess-images` endpoint validates images, calls Pl@ntNet, retrieves corpus
passages with the existing hybrid retriever and requests one short narrative from
the existing configured language model. There is no agentic loop.

The server reads `PLANTNET_API_KEY` from the project's ignored `.env`. Keep the key
private; do not expose it in browser settings. Existing OpenAI/model configuration
and the indexed corpus are reused. Rebuild/recreate the API after changes:

```sh
docker compose up -d --build --force-recreate api
```

For development mode include `-f docker-compose.yml -f docker-compose.dev.yml`.
Test uploads at `http://localhost:8000/docs`. Context fields may be left blank.

## Response

- `assessment.summary`: the narrative to display.
- `assessment.possible_causes`: Pl@ntNet's original ranking and scores, supplied
  by Python rather than the LLM. Not a corpus-validated ranking.
- `assessment.sources`: retrieved passages for inspection. Retrieval alone is
  **not** evidence that every passage supports a disease candidate.
- `assessment.status=preliminary`: narrative generated, not diagnosis confirmed.
- `assessment.status=fallback`: a useful provider summary and next steps remain
  available when the corpus/model is unavailable; `error_code` identifies why.
- `assessment.disclaimer`: display once prominently alongside the narrative.

The top-level response status continues to describe Pl@ntNet identification.
The model receives text, not photographs. Its instructions distinguish reported
symptoms from reference descriptions, require relevant corpus support, preserve
uncertainty and exclude treatment recommendations. This is prompt-based grounding,
not a guarantee of factual correctness or automatic citation-entailment checking.
Low provider confidence does not prevent a useful narrative.

Run offline tests (mocked Pl@ntNet, retrieval and model; no quota consumed):

```sh
docker compose exec api python -m unittest discover -s tests -v
```

For live demonstration, check narrative relevance and citations using actual
passages. Test both a plausible candidate and an inconclusive image. Live tests
consume Pl@ntNet and configured model/embedding quota.

## Map overlay

On `/map`, click **Valuta danni / Assess tree damage** below the text-chat controls.
The overlay supports previews, organ selection, optional location/date/symptoms,
loading and error states, and a narrative with a separate provider-score table.
**References for review** expands the retrieved passages; these are not verified
citations for each narrative statement. The narrative's source-number markers are
removed, and the prompt asks the model to preserve biological qualifications.
Closing the panel keeps the form/results while the map and chat remain in place.
The first overlay version and assessment text are in English.

## Species context

`species` is an optional multipart field (up to 200 characters). Blank means
unknown, never an assumed chestnut. The value is passed to retrieval and the
model as an unverified user report; the disease endpoint does not establish the
tree species. The overlay labels this limitation explicitly. Retrieval still
uses the existing chestnut-focused corpus, whose coverage of other hosts may be
limited. Chestnut-specific facts must not automatically be applied to other trees.

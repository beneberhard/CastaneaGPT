# Source catalogue and chat citations

`data/source_catalogue.json` is the authoritative bibliography for the numbered
literature in `data/corpus_raw/`. It was imported from the supplied
`CastaneaGPT_final_reference_list.docx`; its filename and SHA-256 checksum are
recorded under `provenance`. The original reference text is preserved in each
entry's `full_reference` field. No publication metadata is looked up or corrected
online.

## Mappings and bibliography fields

Each entry has a permanent `source_id` and exact relative `file` path, plus
`authors_text`, `first_author`, `multiple_authors`, `title`, `year`, `year_suffix`,
`url`, `doi`, and `full_reference`. There are 69 entries; IDs 7, 8, 9, 50 and 57
are reserved gaps. Source IDs do not change when the library is alphabetized.
Three mapped documents (15, 26 and 32) are DOCX files, despite being under `pdf/`.

Sources 16 and 26 have no explicit publication year in the supplied list, so
citations use `n.d.`. Years mentioned in a title, funding programme or filename
are not treated as publication dates. DOI strings are preserved as supplied,
including any errors in the original bibliography.

Disambiguation suffixes are stored explicitly. The initial import assigns them
by title order within each first-author/year group:

| Source IDs | Citations |
| --- | --- |
| 22 / 3 | Conedera, 2004a / 2004b |
| 28 / 27 | Conedera, 2021a / 2021b |
| 15 / 32 | Castellotti, 2022a / 2022b |

The importer is reproducible:

```bash
python -m scripts.import_reference_catalogue --docx /path/to/reference-list.docx
```

It validates IDs, duplicate/missing file mappings, and the expected numbered
file correspondence. Review the resulting structured fields when importing a
changed document: this importer handles the supplied bibliography's format,
including its irregular entries; it is not a general-purpose citation parser.

## Ingestion and rebuilding

Ingestion validates the catalogue against current raw documents, ignores Word
lock files, and refreshes curated metadata even when cleaned text is reused.
PDF/DOCX embedded titles and authors cannot override curated records. All
bibliographic fields are inherited by chunks but excluded from embedding text;
retrieval context supplies their citation labels explicitly.

Empty cached extractions are retried. Image-only PDF pages use local Tesseract
OCR with English, Italian and German models installed in the Docker image.
A rebuild checks that every curated source produced readable chunks **before**
requesting embeddings. OCR quality depends on scan quality; extracted text
should be reviewed when a scan is poor.

Only cleaned files corresponding to current raw documents and successfully
extracted current URLs are indexed. Old cleaned files remain on disk but do not
enter the new index. `--reuse-web-only` retains matching cached URL extractions
without refetching websites. It excludes URLs without cached usable text.

```bash
# Validate mappings and refresh metadata/text, without embeddings.
docker compose run --rm --build --no-deps reingest python -m scripts.ingest --prepare-only --reuse-web-only

# Rebuild using the configured embedding API; normal API charges apply.
docker compose run --rm --no-deps reingest python -m scripts.ingest --reuse-web-only

# Deploy the catalogue/UI/code changes and load the new index into memory.
docker compose up --build -d --no-deps api
```

The index is built in a staging directory. The previous index is renamed to
`index_v2.backup-<timestamp>` only when the new build succeeds; failed publication
restores the previous directory. `catalogue_manifest.json` records the catalogue
checksum, source IDs, active cleaned files, document/chunk counts, and indexed
web sources. No automated process deletes old index backups.

## Citations and the library

Retrieval assigns internal `[Source N]` tokens once per distinct publication,
so multiple chunks of one publication share a number. These temporary numbers
are independent of permanent catalogue IDs and remain internal to the model
prompt. During streaming and finalization, they are resolved deterministically
to `(Mariotti, 2009)`-style citations. The model does not invent author names,
years or bibliography entries.

The final reference list contains only cited sources, alphabetized by first
author and listed once per publication. Entries use
`Mariotti et al. — Publication title — 2009`, with a publication URL when supplied.
Unknown internal source IDs are marked `[unverified source]`. Answers with no
source markers are not assigned guessed citations. With citation style `none`,
markers and references are omitted.

The chat's **Fonti / Source library** button opens a searchable dialog backed by
`GET /sources`. It shows all 69 curated records alphabetically, original source
IDs, the full supplied bibliography text, and publication links. Indexed web
sources are shown separately because they are not covered by the curated Word
bibliography. Text is rendered with DOM text APIs; only HTTP(S) links are allowed.

## Completed migration on 2026-11-02

The final rebuild contains **124 active documents and 3,400 chunks**: all 69
curated literature records are represented, together with 55 cached web sources.
Every curated chunk was audited against its catalogue record for title, first
author, publication year, suffix, original reference, and DOI. The catalogue
checksum matches the index manifest.

Sources 2, 4, 21, 45 and 72 previously had empty cached text. Local OCR recovered
28 pages in total (9, 4, 4, 5 and 6 pages, respectively). Rebuilds now reject a
curated source with no readable chunks before calling the embedding API.

There were 298 cleaned text files before migration. The final rebuild uses 124
active files; the remaining 174 stale files were left on disk but excluded. One
current URL is a duplicated/malformed Slow Food address with no cached usable
text and was not indexed. No websites were fetched during this migration.

The pre-migration index is retained inside the Docker storage volume at
`/app/storage/index_v2.backup-20261102-175911-046894`. An intermediate index is
also retained at `/app/storage/index_v2.backup-20261102-180322-143163`. The active
index is `/app/storage/index_v2`.

Validation: 47 Python tests passed in the deployed image. The JavaScript DOM
harness verifies opening the dialog, displaying 69 alphabetized sources,
searching, closing, and safe clickable reference URLs. Live `/health`, `/sources`
and `/map` checks passed. Browser-control service availability prevented a live
visual browser inspection; no additional live chat completion was needed.

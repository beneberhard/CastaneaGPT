# CastaneaGPT

CastaneaGPT is a prototype assistant for sweet chestnut (*Castanea sativa*) management and education. It combines answers grounded in a document collection with an interactive map of chestnut stands, stand metadata, vegetation statistics, and learning quizzes.

## Features

- **Document-based chat:** retrieval-augmented generation (RAG) combines vector search and BM25 keyword search through LlamaIndex, with source citations.
- **Interactive stand map:** browse stand polygons and markers, select a stand, and use its metadata as chat context.
- **Vegetation analysis:** calculate stand-level NDVI statistics from local GeoTIFF rasters and compare available dates.
- **Learning tools:** generate and grade quizzes using the document collection.
- **Multilingual interface:** English, Italian, and German options, with a forestry glossary and configurable response settings.

The application uses Python 3.11, FastAPI, PostgreSQL/PostGIS, and a browser interface served by the API. Docker Compose manages the application, ingestion jobs, database, and a Redis service. Chat session memory currently lives in the API process and is lost when it restarts.

## Quick start

### Requirements

- Docker with Docker Compose v2.
- An OpenAI API key with access to the configured chat and embedding models.
- Internet access for container builds, model requests, and ingestion of any URLs in the source collection.

Run the following commands from the project root after cloning or downloading the repository.

### 1. Configure the environment

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_api_key_here
LLM_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Keep your actual key out of Git. The repository's `.gitignore` excludes `.env`.

The API service currently sets `OPENAI_MODEL=gpt-4o-mini` directly in `docker-compose.yml`. Change that service setting to select a different OpenAI chat model; a value in `.env` alone will not override it. Compose also supplies the database URL, Redis URL, and shared index path.

### 2. Prepare the document collection

The current version of this repository provides the application source code but does not include the documents used to build the RAG corpus. Users must obtain source documents separately, subject to their respective licenses and access conditions, and place them in `data/corpus_raw/` before building the index. This directory is excluded from Git tracking.

A catalogue identifying the documents used in the study, with bibliographic references and source links, is planned for a future update. Until it is available, the repository alone does not provide everything needed to reconstruct the study's exact corpus.

Ensure `data/corpus_raw/` contains readable source documents before starting. The ingestion script supports:

| Source | Location |
| --- | --- |
| PDFs with extractable text | `data/corpus_raw/pdf/*.pdf` |
| Plain text | `data/corpus_raw/files/*.txt` or `data/corpus_raw/txt/*.txt` |
| Word documents | `*.docx` anywhere under `data/corpus_raw/` |
| Web pages | One URL per line in `data/corpus_raw/urls.txt` |

PDF ingestion uses local Tesseract OCR for image-only pages in the Docker image. The numbered literature uses the curated metadata in `data/source_catalogue.json`; extracted text and citation metadata are written to `data/corpus_clean/`. Embeddings and the retrieval index are stored separately. See [source catalogue and citations](docs/source-catalogue.md) for mappings, citation formatting, the chat source library, and rebuilding. Model calls, including embedding the collection, can incur API charges.

### 3. Build and start

```bash
docker compose up --build -d
```

On first startup, the worker creates the index before the API starts. Ingestion time depends on the document collection and external services.

- Application: [http://localhost:8000/map](http://localhost:8000/map)
- Interactive API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health endpoint: [http://localhost:8000/health](http://localhost:8000/health)

Inspect startup progress with:

```bash
docker compose logs -f worker api
```

A fresh database contains the stand table but **no demonstration stands**. Import your own stand data to populate the map. NDVI rasters are also supplied separately; see below.

Stop the services with:

```bash
docker compose down
```

Named Docker volumes retain the database and index. Using `docker compose down -v` deletes those volumes and their data.

## Development

Use the development override to mount the project into the containers and enable API live reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

The default configuration copies project files into the image during the build. Rebuild after changing code or data when using that configuration.

After changing `.env`, recreate the API container so it receives the updated values:

```bash
docker compose up -d --force-recreate api
```

If using development mode, include both Compose files in that command as well. A simple container restart does not reload environment configuration.

## Updating the document index

After adding source documents, rebuild the shared index explicitly:

```bash
docker compose run --rm --build reingest
docker compose restart api
```

When adding new items, the following command skips ingested documents (provided the embedding configuration hasn't changed.)

```bash
docker compose exec app python -m scripts.ingest_rag_corpus
```


The `reingest` service reads the current project files through a bind mount and writes to the same index volume used by the API. Restarting the API clears its cached index and retriever.

**Changed document content needs re-extraction:** nonempty cleaned text is reused, but curated bibliography metadata is refreshed on every ingestion. Remove the affected generated `.txt` and `.txt.meta.json` files from `data/corpus_clean/` before reingesting changed content. Empty cached text is retried automatically, including local OCR when needed. Removed sources and stale cleaned files are excluded from the rebuilt index automatically. To re-extract the whole collection, clear the generated contents of that directory first.

The normal worker checks whether the index directory exists, rather than whether its contents are complete. If ingestion failed or ran with no documents, correct the underlying problem and run `reingest` explicitly. Keep `OPENAI_EMBEDDING_MODEL` consistent between ingestion and querying; rebuild the index when changing it.

## Stand and NDVI data

Initialization SQL in `db/init/` creates the PostGIS extension and `stands` table. These scripts run when the database volume is first initialized.

Import stand polygons into `public.stands` using SQL or a GIS client such as QGIS. The table expects `Polygon` geometries in **EPSG:4326**, with optional fields for `name`, `species`, `management_type`, `age_class`, `altitude_m`, and `notes`.

For the supplied local Compose configuration, a GIS client connects to `localhost:5432`, database `castanea`, user `postgres`, password `postgres`.

Place optional NDVI GeoTIFFs in `data/NDVI_rasters/`. Use stand-specific filenames such as:

```text
stand3_ndvi_06_2016.tif
stand3_ndvi_07_2017.tiff
```

The number after `stand` must match the database stand ID; the final numbers specify month and year. Rasters must contain NDVI values, cover the relevant polygon, and have valid CRS metadata. The application transforms stand geometry into the raster CRS when needed. Missing raster data leaves NDVI information unavailable without preventing general chat.

In default Docker mode, rebuild the API image after adding rasters. Development mode reads them through the project mount.

## Experimental local chat models

The code includes configuration for a local OpenAI-compatible chat endpoint, for example LM Studio:

```dotenv
LLM_PROVIDER=local
LOCAL_BASE_URL=http://host.docker.internal:1234/v1
LOCAL_API_KEY=lm_studio
LOCAL_MODEL=your_loaded_model_id
```

`host.docker.internal` addresses the host from Docker Desktop containers. Other Docker installations may require additional host networking configuration.

Local chat compatibility is experimental and depends on the server, model ID, and LlamaIndex wrapper. **Embeddings still use OpenAI**, so `OPENAI_API_KEY` remains required. This configuration does not provide a fully offline application. Set `LLM_PROVIDER=openai` and recreate the API container to return to the default provider.

## Repository layout

```text
app/
  api/                  FastAPI routes, browser interface, and session memory
  rag/                  Retrieval and answer generation
  ui/                   Index path configuration
  utils/                Glossary, quizzes, language, and stand/NDVI context
  llm_config.py         Chat and embedding model configuration
scripts/ingest.py        Document extraction and index generation
db/init/                Database initialization SQL
data/corpus_raw/        User-supplied documents and URL list (ignored by Git)
data/glossary/          Forestry glossary
data/NDVI_rasters/      User-supplied rasters (ignored by Git)
data/corpus_clean/      Generated text and metadata (ignored by Git)
storage/                Local generated storage (ignored by Git)
Dockerfile              Python application image
requirements.docker.txt Container dependencies
requirements.txt        Additional pinned Python environment snapshot
docker-compose.yml      Default service configuration
docker-compose.dev.yml  Development mounts and live reload
```

In Docker, the persisted index lives in the `castanea_index_data` volume at `/app/storage/index_v2`; it is not the host's `storage/` directory.

## Project status and contributions

CastaneaGPT is a research and demonstration prototype. Answers depend on the source collection and may contain errors; check cited material before applying recommendations. The supplied Compose configuration uses development database credentials and exposes service ports. It is intended for local use and requires deployment hardening before public hosting.

To report a problem, open a GitHub issue with reproduction steps, expected behavior, and relevant logs with credentials removed. For a contribution, describe the change and how you verified it in a pull request.

## License and source material

No software license is currently included in this repository. A license must be selected and added to define permissions for reuse and redistribution.

Documents, web content, and geospatial datasets may have separate licenses or copyright restrictions. Their inclusion in a retrieval corpus does not grant redistribution rights; verify those rights before publishing source material on GitHub.

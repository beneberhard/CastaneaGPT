# CastaneaGPT

## Docker Compose Modes

Stable default runtime:

```bash
docker compose up --build
```

This uses the built image, disables `uvicorn --reload`, and shares the persisted index through a named Docker volume.

Development runtime with live reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

This adds the project bind mount and re-enables `uvicorn --reload` for iterative development on macOS.

## Dedicated/explicit Force Re-ingestion

When you add or change source files under `data/corpus_raw`, force a corpus refresh with:

```bash
docker compose run --rm --build reingest
```

This one-off service:

- uses the current repo contents from your host, so refreshed `data/corpus_clean` files are visible locally
- rebuilds the persisted index in the shared Docker volume
- does not depend on the normal worker skip logic

After that, start or restart the app normally:

```bash
docker compose up
```

18.5.2026:
In Quiz, especially after questions on pests and diseases, sources related to Turkey appear excessively often. This was fixed in utils/quiz.py -Generate quiz -def generate... in the -prompt:Avoid overusing any single regional case study or country background across the quiz.

The citation style in Chat is poor: e.g. the model saw context chunks labeled like Source 1, Source 2... but the API later renumbered the sources
changed: added a citation label formatter in app/rag/rag.py (line 140):
def _source_year(meta: dict[str, Any]) -> str:...

## LLM Provider

The model currently running is the OpenAI model. Below see the documentation how it was attempted to switch to the local model (Qwen 3.5 running in LM Studio).

Local model not working due to the following issues (the chat model works, but the embedding model doesn't):
LM Studio model id/base URL,
the local LlamaIndex wrapper,
and the Qwen-oriented quiz prompt changes. 

Documentation:
Use the local LM Studio model for development:
see app/llm_config.py
for this purpose changed:

- app/rag/rag.py (line28)
- app/utils/quiz.py (line 15)
- scripts/ingest.py (line 279) (index creation explicitly uses the same embedding model 'text-embedding-3-small')

```bash
LLM_PROVIDER=local
LOCAL_BASE_URL=http://localhost:1234/v1
#Because the app is running inside a container, http://localhost:1234/v1 means "inside the #container", not my Mac. If (it is not, only in case) LM Studio is running on my Mac host, change #to: LOCAL_BASE_URL=http://host.docker.internal:1234/v1
LOCAL_API_KEY=lm_studio
LOCAL_MODEL=local-model
```

Switch back to OpenAI for demos or production:
Switch the model (openai/local) in .env: LLM_PROVIDER=...(openai or local)

after switching run:

```bash
docker compose restart api
```

or:

```bash
docker compose up --build -d api
```

for development mode:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart api
```

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Keep `text-embedding-3-small` unless you rebuild the index with a different embedding model.

19.5.2026
Several files were highlighted in red in the explorer. This meant: nothing is wrong. Red means VS Code/Git sees it as modified after the last commit.

also the README.md file in red: so it was committed:

```bash
git add README.md
git commit -m "Update README notes"
```

User name/email for git commits was changed:

```bash
git config --global user.email "bennoeberhard@gmail.com"
```

After

```bash
git push
```

the the commit is on the remote repository too.

4.6.2026:
Integrated stands Kissos (Greece) and Nazilli (Turkey). Uploaded NDVI raster layer for Nazilli (June 2016) based on Copernicus Data Space Ecosystem.
Stand-scoped rasters are filtered by stand-id, so that e.g. Stand 3 only sees Stand 3 rasters etc.

.tif and .tiff are both supported

Single-stand information can be edited directly in QGis DB Manager, as long as it is connected to the PostGIS database used by the app.
In the Browser panel in QGis open PostgreSQL -> public -> stands i.e. bring stands to the Layers panel. There, open the attribute table, and edit the stand metadata.

When generating a NDVI file for a stand, make sure it has the same CRS as the stand polygon itself. Here EPSG 4326 WGS 84

Changed the chat panel:
Each button contains a stand's name.

Added a red dot for each stand on the map, with tooltips and click behavior matching the polygons.

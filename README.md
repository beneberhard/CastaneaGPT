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

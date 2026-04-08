CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS stands (
  id SERIAL PRIMARY KEY,
  name TEXT,
  geom geometry(Polygon, 4326)
);

CREATE INDEX IF NOT EXISTS stands_geom_gix ON stands USING GIST (geom);

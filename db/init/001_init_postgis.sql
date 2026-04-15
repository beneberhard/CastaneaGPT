CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS stands (
  id SERIAL PRIMARY KEY,
  name TEXT,
  species TEXT,
  management_type TEXT,
  age_class TEXT,
  altitude_m DOUBLE PRECISION,
  notes TEXT,
  geom geometry(Polygon, 4326)
);

CREATE INDEX IF NOT EXISTS stands_geom_gix ON stands USING GIST (geom);

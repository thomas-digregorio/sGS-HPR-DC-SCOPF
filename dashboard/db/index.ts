import { env } from "cloudflare:workers";
import { drizzle } from "drizzle-orm/d1";
import * as schema from "./schema";

const CREATE_TABLE = `
CREATE TABLE IF NOT EXISTS paper_annotations (
  id TEXT PRIMARY KEY NOT NULL,
  owner_key TEXT NOT NULL,
  document_id TEXT NOT NULL,
  section_id TEXT NOT NULL,
  section_title TEXT NOT NULL,
  selected_text TEXT NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset INTEGER NOT NULL,
  prefix TEXT NOT NULL DEFAULT '',
  suffix TEXT NOT NULL DEFAULT '',
  comment TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)`;

const CREATE_INDEX = `
CREATE INDEX IF NOT EXISTS idx_paper_annotations_owner_document_created
ON paper_annotations(owner_key, document_id, created_at)`;

let schemaReady: Promise<void> | null = null;

async function ensureSchema(database: D1Database) {
  if (!schemaReady) {
    schemaReady = database
      .batch([
        database.prepare(CREATE_TABLE),
        database.prepare(CREATE_INDEX),
        database.prepare("PRAGMA optimize"),
      ])
      .then(() => undefined)
      .catch((error) => {
        schemaReady = null;
        throw error;
      });
  }
  await schemaReady;
}

export async function getDb() {
  if (!env.DB) {
    throw new Error("The paper feedback database is unavailable.");
  }
  await ensureSchema(env.DB);
  return drizzle(env.DB, { schema });
}

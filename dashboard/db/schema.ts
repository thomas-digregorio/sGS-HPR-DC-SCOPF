import { sql } from "drizzle-orm";
import { index, integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const paperAnnotations = sqliteTable(
  "paper_annotations",
  {
    id: text("id").primaryKey(),
    ownerKey: text("owner_key").notNull(),
    documentId: text("document_id").notNull(),
    sectionId: text("section_id").notNull(),
    sectionTitle: text("section_title").notNull(),
    selectedText: text("selected_text").notNull(),
    startOffset: integer("start_offset").notNull(),
    endOffset: integer("end_offset").notNull(),
    prefix: text("prefix").notNull().default(""),
    suffix: text("suffix").notNull().default(""),
    comment: text("comment").notNull(),
    status: text("status").notNull().default("open"),
    createdAt: text("created_at").notNull().default(sql`CURRENT_TIMESTAMP`),
    updatedAt: text("updated_at").notNull().default(sql`CURRENT_TIMESTAMP`),
  },
  (table) => [
    index("idx_paper_annotations_owner_document_created").on(
      table.ownerKey,
      table.documentId,
      table.createdAt,
    ),
  ],
);

export type PaperAnnotation = typeof paperAnnotations.$inferSelect;

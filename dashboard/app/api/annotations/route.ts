import { and, desc, eq } from "drizzle-orm";
import { getDb } from "../../../db";
import { paperAnnotations } from "../../../db/schema";

const DOCUMENT_ID_PATTERN = /^[a-z0-9-]{1,80}$/;
const CLIENT_ID_PATTERN = /^[a-zA-Z0-9-]{8,100}$/;

type AnnotationInput = {
  id?: unknown;
  documentId?: unknown;
  sectionId?: unknown;
  sectionTitle?: unknown;
  selectedText?: unknown;
  startOffset?: unknown;
  endOffset?: unknown;
  prefix?: unknown;
  suffix?: unknown;
  comment?: unknown;
  status?: unknown;
};

function ownerKey(request: Request) {
  const email = request.headers.get("oai-authenticated-user-email")?.trim().toLowerCase();
  if (email) return `user:${email}`;

  const session = request.headers.get("x-paper-session")?.trim() ?? "";
  return CLIENT_ID_PATTERN.test(session) ? `session:${session}` : null;
}

function boundedText(value: unknown, maximum: number) {
  if (typeof value !== "string") return null;
  const text = value.trim();
  return text && text.length <= maximum ? text : null;
}

function boundedContext(value: unknown) {
  if (typeof value !== "string") return "";
  return value.slice(0, 240);
}

function safeDocumentId(value: unknown) {
  return typeof value === "string" && DOCUMENT_ID_PATTERN.test(value) ? value : null;
}

function apiError(error: unknown) {
  const message = error instanceof Error ? error.message : "Unexpected feedback service error";
  return Response.json({ error: message }, { status: 500 });
}

export async function GET(request: Request) {
  try {
    const owner = ownerKey(request);
    const documentId = safeDocumentId(new URL(request.url).searchParams.get("document"));
    if (!owner || !documentId) {
      return Response.json({ error: "A valid reader session and document are required." }, { status: 400 });
    }

    const db = await getDb();
    const annotations = await db
      .select()
      .from(paperAnnotations)
      .where(
        and(
          eq(paperAnnotations.ownerKey, owner),
          eq(paperAnnotations.documentId, documentId),
        ),
      )
      .orderBy(desc(paperAnnotations.createdAt), desc(paperAnnotations.id));

    return Response.json({ annotations });
  } catch (error) {
    return apiError(error);
  }
}

export async function POST(request: Request) {
  try {
    const owner = ownerKey(request);
    const payload = (await request.json()) as AnnotationInput;
    const id = typeof payload.id === "string" && CLIENT_ID_PATTERN.test(payload.id) ? payload.id : null;
    const documentId = safeDocumentId(payload.documentId);
    const sectionId = boundedText(payload.sectionId, 120);
    const sectionTitle = boundedText(payload.sectionTitle, 180);
    const selectedText = boundedText(payload.selectedText, 1200);
    const comment = boundedText(payload.comment, 4000);
    const startOffset = Number(payload.startOffset);
    const endOffset = Number(payload.endOffset);

    if (
      !owner ||
      !id ||
      !documentId ||
      !sectionId ||
      !sectionTitle ||
      !selectedText ||
      !comment ||
      !Number.isInteger(startOffset) ||
      !Number.isInteger(endOffset) ||
      startOffset < 0 ||
      endOffset <= startOffset
    ) {
      return Response.json({ error: "The selected passage or feedback is invalid." }, { status: 400 });
    }

    const db = await getDb();
    const [annotation] = await db
      .insert(paperAnnotations)
      .values({
        id,
        ownerKey: owner,
        documentId,
        sectionId,
        sectionTitle,
        selectedText,
        startOffset,
        endOffset,
        prefix: boundedContext(payload.prefix),
        suffix: boundedContext(payload.suffix),
        comment,
      })
      .returning();

    return Response.json({ annotation }, { status: 201 });
  } catch (error) {
    return apiError(error);
  }
}

export async function PATCH(request: Request) {
  try {
    const owner = ownerKey(request);
    const payload = (await request.json()) as AnnotationInput;
    const id = typeof payload.id === "string" && CLIENT_ID_PATTERN.test(payload.id) ? payload.id : null;
    const documentId = safeDocumentId(payload.documentId);
    const comment = boundedText(payload.comment, 4000);
    const status = payload.status === "open" || payload.status === "applied" ? payload.status : null;

    if (!owner || !id || !documentId || !comment || !status) {
      return Response.json({ error: "The feedback update is invalid." }, { status: 400 });
    }

    const db = await getDb();
    const [annotation] = await db
      .update(paperAnnotations)
      .set({ comment, status, updatedAt: new Date().toISOString() })
      .where(
        and(
          eq(paperAnnotations.id, id),
          eq(paperAnnotations.ownerKey, owner),
          eq(paperAnnotations.documentId, documentId),
        ),
      )
      .returning();

    if (!annotation) {
      return Response.json({ error: "Feedback note not found." }, { status: 404 });
    }
    return Response.json({ annotation });
  } catch (error) {
    return apiError(error);
  }
}

export async function DELETE(request: Request) {
  try {
    const owner = ownerKey(request);
    const url = new URL(request.url);
    const id = url.searchParams.get("id");
    const documentId = safeDocumentId(url.searchParams.get("document"));
    if (!owner || !id || !CLIENT_ID_PATTERN.test(id) || !documentId) {
      return Response.json({ error: "A valid feedback note is required." }, { status: 400 });
    }

    const db = await getDb();
    const deleted = await db
      .delete(paperAnnotations)
      .where(
        and(
          eq(paperAnnotations.id, id),
          eq(paperAnnotations.ownerKey, owner),
          eq(paperAnnotations.documentId, documentId),
        ),
      )
      .returning({ id: paperAnnotations.id });

    return Response.json({ deleted: deleted.length === 1 });
  } catch (error) {
    return apiError(error);
  }
}

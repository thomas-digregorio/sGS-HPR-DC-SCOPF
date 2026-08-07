"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

export type PaperSection = {
  id: string;
  title: string;
  depth: number;
};

type AnnotationStatus = "open" | "applied";

type Annotation = {
  id: string;
  documentId: string;
  sectionId: string;
  sectionTitle: string;
  selectedText: string;
  startOffset: number;
  endOffset: number;
  prefix: string;
  suffix: string;
  comment: string;
  status: AnnotationStatus;
  createdAt: string;
  updatedAt: string;
};

type DraftSelection = {
  sectionId: string;
  sectionTitle: string;
  selectedText: string;
  startOffset: number;
  endOffset: number;
  prefix: string;
  suffix: string;
  left: number;
  top: number;
};

type HighlightRegistry = {
  set(name: string, highlight: unknown): void;
  delete(name: string): void;
};

type HighlightConstructor = new (...ranges: Range[]) => unknown;

const sessionStorageKey = "hpr-paper-reader-session";

function rangeForOffsets(root: HTMLElement, startOffset: number, endOffset: number) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let cursor = 0;
  let startNode: Text | null = null;
  let endNode: Text | null = null;
  let localStart = 0;
  let localEnd = 0;
  let node = walker.nextNode();

  while (node) {
    const textNode = node as Text;
    const nextCursor = cursor + textNode.data.length;
    if (!startNode && startOffset >= cursor && startOffset <= nextCursor) {
      startNode = textNode;
      localStart = Math.min(startOffset - cursor, textNode.data.length);
    }
    if (endOffset >= cursor && endOffset <= nextCursor) {
      endNode = textNode;
      localEnd = Math.min(endOffset - cursor, textNode.data.length);
      break;
    }
    cursor = nextCursor;
    node = walker.nextNode();
  }

  if (!startNode || !endNode) return null;
  const range = document.createRange();
  range.setStart(startNode, localStart);
  range.setEnd(endNode, localEnd);
  return range;
}

function offsetBefore(root: HTMLElement, node: Node, offset = 0) {
  const range = document.createRange();
  range.selectNodeContents(root);
  range.setEnd(node, offset);
  return range.toString().length;
}

function sectionAtOffset(root: HTMLElement, startOffset: number) {
  let sectionId = "abstract";
  let sectionTitle = "Abstract";
  for (const heading of root.querySelectorAll<HTMLElement>("[data-paper-heading='true']")) {
    const headingOffset = offsetBefore(root, heading);
    if (headingOffset > startOffset) break;
    sectionId = heading.id;
    sectionTitle = heading.textContent?.trim() || sectionTitle;
  }
  return { sectionId, sectionTitle };
}

function feedbackMarkdown(annotations: Annotation[], release: string) {
  const generated = new Date().toISOString();
  const entries = annotations
    .map(
      (annotation, index) => `## ${index + 1}. ${annotation.sectionTitle}\n\n` +
        `**Status:** ${annotation.status}\n\n` +
        `> ${annotation.selectedText.replaceAll("\n", "\n> ")}\n\n` +
        `**Requested change:** ${annotation.comment}\n`,
    )
    .join("\n");

  return `# Annotated paper feedback\n\n` +
    `Paper release: ${release}\n\n` +
    `Exported: ${generated}\n\n` +
    `Please apply these edits to the LaTeX paper, rebuild the PDF, and preserve the evidence boundaries.\n\n` +
    entries;
}

function downloadText(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function InteractivePaper({
  documentId,
  html,
  release,
  sections,
}: {
  documentId: string;
  html: string;
  release: string;
  sections: PaperSection[];
}) {
  const paperRef = useRef<HTMLElement>(null);
  const sessionRef = useRef("");
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [draft, setDraft] = useState<DraftSelection | null>(null);
  const [comment, setComment] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingComment, setEditingComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [syncState, setSyncState] = useState<"loading" | "saved" | "error">("loading");
  const [message, setMessage] = useState("Select a passage to begin.");

  const request = useCallback(async (input: string, init?: RequestInit) => {
    const response = await fetch(input, {
      ...init,
      headers: {
        "content-type": "application/json",
        "x-paper-session": sessionRef.current,
        ...init?.headers,
      },
    });
    const payload = (await response.json()) as {
      annotations?: Annotation[];
      annotation?: Annotation;
      deleted?: boolean;
      error?: string;
    };
    if (!response.ok) throw new Error(payload.error || "Feedback could not be saved.");
    return payload;
  }, []);

  useEffect(() => {
    const stored = window.localStorage.getItem(sessionStorageKey);
    const session = stored || window.crypto.randomUUID();
    if (!stored) window.localStorage.setItem(sessionStorageKey, session);
    sessionRef.current = session;

    void request(`/api/annotations?document=${encodeURIComponent(documentId)}`)
      .then((payload) => {
        setAnnotations(payload.annotations ?? []);
        setSyncState("saved");
      })
      .catch((error: Error) => {
        setSyncState("error");
        setMessage(error.message);
      });
  }, [documentId, request]);

  useEffect(() => {
    const root = paperRef.current;
    const registry = (CSS as unknown as { highlights?: HighlightRegistry }).highlights;
    const Highlight = (window as unknown as { Highlight?: HighlightConstructor }).Highlight;
    if (!root || !registry || !Highlight) return;

    const openRanges = annotations
      .filter((annotation) => annotation.status === "open")
      .map((annotation) => rangeForOffsets(root, annotation.startOffset, annotation.endOffset))
      .filter((range): range is Range => Boolean(range));
    const appliedRanges = annotations
      .filter((annotation) => annotation.status === "applied")
      .map((annotation) => rangeForOffsets(root, annotation.startOffset, annotation.endOffset))
      .filter((range): range is Range => Boolean(range));

    registry.set("paper-feedback-open", new Highlight(...openRanges));
    registry.set("paper-feedback-applied", new Highlight(...appliedRanges));
    return () => {
      registry.delete("paper-feedback-open");
      registry.delete("paper-feedback-applied");
    };
  }, [annotations, html]);

  const captureSelection = useCallback(() => {
    const root = paperRef.current;
    const selection = window.getSelection();
    if (!root || !selection || selection.rangeCount === 0 || selection.isCollapsed) {
      return;
    }

    const range = selection.getRangeAt(0);
    if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) return;

    const selectedText = range.toString().replace(/\s+/g, " ").trim();
    if (selectedText.length < 3) return;
    if (selectedText.length > 1200) {
      setMessage("Please select a shorter passage (1,200 characters or fewer).");
      return;
    }

    const startOffset = offsetBefore(root, range.startContainer, range.startOffset);
    const endOffset = offsetBefore(root, range.endContainer, range.endOffset);
    const fullText = root.textContent ?? "";
    const section = sectionAtOffset(root, startOffset);
    const rectangle = range.getBoundingClientRect();
    const left = Math.max(12, Math.min(window.innerWidth - 190, rectangle.left));
    const top = Math.max(12, Math.min(window.innerHeight - 58, rectangle.bottom + 10));

    setDraft({
      ...section,
      selectedText,
      startOffset,
      endOffset,
      prefix: fullText.slice(Math.max(0, startOffset - 160), startOffset),
      suffix: fullText.slice(endOffset, endOffset + 160),
      left,
      top,
    });
    setComment("");
    setMessage("Selection ready. Add the change you want.");
  }, []);

  const saveDraft = async () => {
    if (!draft || !comment.trim()) return;
    setBusy(true);
    try {
      const payload = await request("/api/annotations", {
        method: "POST",
        body: JSON.stringify({
          id: window.crypto.randomUUID(),
          documentId,
          ...draft,
          comment: comment.trim(),
        }),
      });
      if (payload.annotation) setAnnotations((current) => [payload.annotation!, ...current]);
      setDraft(null);
      setComment("");
      window.getSelection()?.removeAllRanges();
      setSyncState("saved");
      setMessage("Feedback saved and highlighted.");
    } catch (error) {
      setSyncState("error");
      setMessage(error instanceof Error ? error.message : "Feedback could not be saved.");
    } finally {
      setBusy(false);
    }
  };

  const updateAnnotation = async (annotation: Annotation, status = annotation.status) => {
    const nextComment = editingId === annotation.id ? editingComment.trim() : annotation.comment;
    if (!nextComment) return;
    setBusy(true);
    try {
      const payload = await request("/api/annotations", {
        method: "PATCH",
        body: JSON.stringify({ id: annotation.id, documentId, comment: nextComment, status }),
      });
      if (payload.annotation) {
        setAnnotations((current) =>
          current.map((item) => (item.id === annotation.id ? payload.annotation! : item)),
        );
      }
      setEditingId(null);
      setEditingComment("");
      setSyncState("saved");
      setMessage(status === "applied" ? "Feedback marked as applied." : "Feedback updated.");
    } catch (error) {
      setSyncState("error");
      setMessage(error instanceof Error ? error.message : "Feedback could not be updated.");
    } finally {
      setBusy(false);
    }
  };

  const removeAnnotation = async (annotation: Annotation) => {
    if (!window.confirm("Delete this feedback note?")) return;
    setBusy(true);
    try {
      await request(
        `/api/annotations?id=${encodeURIComponent(annotation.id)}&document=${encodeURIComponent(documentId)}`,
        { method: "DELETE" },
      );
      setAnnotations((current) => current.filter((item) => item.id !== annotation.id));
      setSyncState("saved");
      setMessage("Feedback deleted.");
    } catch (error) {
      setSyncState("error");
      setMessage(error instanceof Error ? error.message : "Feedback could not be deleted.");
    } finally {
      setBusy(false);
    }
  };

  const jumpToAnnotation = (annotation: Annotation) => {
    const root = paperRef.current;
    if (!root) return;
    const range = rangeForOffsets(root, annotation.startOffset, annotation.endOffset);
    const target = range?.startContainer.parentElement;
    target?.scrollIntoView({ behavior: "smooth", block: "center" });

    const registry = (CSS as unknown as { highlights?: HighlightRegistry }).highlights;
    const Highlight = (window as unknown as { Highlight?: HighlightConstructor }).Highlight;
    if (range && registry && Highlight) {
      registry.set("paper-feedback-active", new Highlight(range));
      window.setTimeout(() => registry.delete("paper-feedback-active"), 1800);
    }
  };

  const exportedMarkdown = useMemo(
    () => feedbackMarkdown(annotations, release),
    [annotations, release],
  );

  const copyForCodex = async () => {
    await navigator.clipboard.writeText(exportedMarkdown);
    setMessage("Feedback copied. Paste it into this Codex task when you are ready.");
  };

  const openCount = annotations.filter((annotation) => annotation.status === "open").length;

  return (
    <main className="paper-reader-shell">
      <header className="paper-toolbar">
        <Link className="paper-back-link" href="/">← Research dashboard</Link>
        <div className="paper-toolbar-title">Interactive paper</div>
        <div className={`paper-sync-state ${syncState}`}>
          {syncState === "loading" ? "Loading notes" : syncState === "error" ? "Sync issue" : `${openCount} open notes`}
        </div>
      </header>

      <section className="paper-reader-hero">
        <p className="paper-kicker">Read · highlight · revise</p>
        <h1>Evidence-Bounded Structural Reproduction of a GPU sGS-HPR Solver</h1>
        <p>
          Select any passage, attach a concrete revision request, and export the complete
          annotation set for Codex. Notes are saved privately to this reader.
        </p>
        <div className="paper-reader-meta">
          <span>Thomas DiGregorio</span>
          <span>{release}</span>
          <span>Scientific paper</span>
        </div>
      </section>

      <div className="paper-reader-layout">
        <nav className="paper-contents" aria-label="Paper contents">
          <div className="paper-rail-label">Contents</div>
          {sections
            .filter((section) => section.depth === 2)
            .map((section) => (
              <a href={`#${section.id}`} key={section.id}>{section.title}</a>
            ))}
        </nav>

        <article
          className="interactive-paper"
          onKeyUp={() => window.setTimeout(captureSelection, 0)}
          onPointerUp={() => window.setTimeout(captureSelection, 0)}
          ref={paperRef}
          dangerouslySetInnerHTML={{ __html: html }}
        />

        <aside className="feedback-panel" aria-label="Paper feedback">
          <div className="feedback-panel-head">
            <div>
              <div className="paper-rail-label">Feedback</div>
              <h2>{annotations.length} annotations</h2>
            </div>
            <span className="feedback-open-count">{openCount} open</span>
          </div>

          <p className="feedback-message" aria-live="polite">{message}</p>

          {draft && (
            <section className="feedback-composer" aria-labelledby="feedback-composer-title">
              <div className="annotation-section">{draft.sectionTitle}</div>
              <blockquote>{draft.selectedText}</blockquote>
              <label htmlFor="paper-feedback" id="feedback-composer-title">
                What should change?
              </label>
              <textarea
                autoFocus
                id="paper-feedback"
                onChange={(event) => setComment(event.target.value)}
                placeholder="Be specific: shorten this, clarify the claim, add a citation…"
                rows={5}
                value={comment}
              />
              <div className="feedback-composer-actions">
                <button className="feedback-save" disabled={busy || !comment.trim()} onClick={saveDraft}>
                  Save highlight
                </button>
                <button className="feedback-cancel" onClick={() => setDraft(null)}>Cancel</button>
              </div>
            </section>
          )}

          <div className="feedback-export-row">
            <button disabled={annotations.length === 0} onClick={copyForCodex}>Copy for Codex</button>
            <button
              disabled={annotations.length === 0}
              onClick={() => downloadText("paper-feedback.md", exportedMarkdown, "text/markdown")}
            >
              Download Markdown
            </button>
          </div>

          <div className="annotation-list">
            {annotations.map((annotation) => (
              <article className={`annotation-card ${annotation.status}`} key={annotation.id}>
                <button className="annotation-jump" onClick={() => jumpToAnnotation(annotation)}>
                  <span>{annotation.sectionTitle}</span>
                  <strong>Jump to text ↗</strong>
                </button>
                <blockquote>{annotation.selectedText}</blockquote>
                {editingId === annotation.id ? (
                  <textarea
                    autoFocus
                    onChange={(event) => setEditingComment(event.target.value)}
                    rows={4}
                    value={editingComment}
                  />
                ) : (
                  <p>{annotation.comment}</p>
                )}
                <div className="annotation-actions">
                  {editingId === annotation.id ? (
                    <button disabled={busy || !editingComment.trim()} onClick={() => updateAnnotation(annotation)}>
                      Save edit
                    </button>
                  ) : (
                    <button onClick={() => { setEditingId(annotation.id); setEditingComment(annotation.comment); }}>
                      Edit
                    </button>
                  )}
                  <button
                    disabled={busy}
                    onClick={() => updateAnnotation(annotation, annotation.status === "open" ? "applied" : "open")}
                  >
                    {annotation.status === "open" ? "Mark applied" : "Reopen"}
                  </button>
                  <button disabled={busy} onClick={() => removeAnnotation(annotation)}>Delete</button>
                </div>
              </article>
            ))}
            {annotations.length === 0 && !draft && (
              <div className="feedback-empty">
                Highlight a sentence in the paper. Your note will appear here and remain
                attached to that passage.
              </div>
            )}
          </div>
        </aside>
      </div>

      {draft && (
        <button
          className="selection-feedback-button"
          onClick={() => document.querySelector<HTMLTextAreaElement>("#paper-feedback")?.focus()}
          style={{ left: draft.left, top: draft.top }}
        >
          Add feedback
        </button>
      )}
    </main>
  );
}

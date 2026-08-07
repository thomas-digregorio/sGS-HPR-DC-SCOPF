import type { Metadata } from "next";
import katex from "katex";
import { marked } from "marked";
import { InteractivePaper, type PaperSection } from "./InteractivePaper";
import paperMarkdown from "./final_reproduction_report.md?raw";

export const metadata: Metadata = {
  title: "Interactive paper",
  description:
    "Read, highlight, and annotate the evidence-bounded structural reproduction paper.",
  openGraph: {
    title: "Interactive paper · GPU sGS-HPR",
    description:
      "Highlight passages, attach revision notes, and export a structured brief for Codex.",
    images: ["/og-interactive-paper.png"],
  },
  twitter: {
    card: "summary_large_image",
    title: "Interactive paper · GPU sGS-HPR",
    description: "Highlight. Comment. Revise.",
    images: ["/og-interactive-paper.png"],
  },
};

const documentId = "structural-reproduction-paper-v5";

function renderMath(source: string) {
  const withBlocks = source.replace(/\$\$([\s\S]*?)\$\$/g, (_match, expression: string) => {
    const rendered = katex.renderToString(expression.trim(), {
      displayMode: true,
      throwOnError: false,
      strict: "ignore",
    });
    return `\n\n<div class="paper-equation">${rendered}</div>\n\n`;
  });

  return withBlocks.replace(/\$([^$\n]+?)\$/g, (_match, expression: string) =>
    katex.renderToString(expression.trim(), {
      displayMode: false,
      throwOnError: false,
      strict: "ignore",
    }),
  );
}

function stripHtml(value: string) {
  return value
    .replace(/<[^>]+>/g, "")
    .replaceAll("&amp;", "&")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .trim();
}

function slugify(value: string) {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "section";
}

function renderPaper(source: string) {
  const start = source.indexOf("## Abstract");
  const body = start >= 0 ? source.slice(start) : source;
  const rawHtml = String(marked.parse(renderMath(body), { gfm: true }));
  const duplicateCounts = new Map<string, number>();
  const sections: PaperSection[] = [];

  const html = rawHtml.replace(
    /<h([1-6])>([\s\S]*?)<\/h\1>/g,
    (_match, depthText: string, inner: string) => {
      const depth = Number(depthText);
      const title = stripHtml(inner);
      const baseId = slugify(title);
      const duplicate = duplicateCounts.get(baseId) ?? 0;
      duplicateCounts.set(baseId, duplicate + 1);
      const id = duplicate === 0 ? baseId : `${baseId}-${duplicate + 1}`;
      if (depth <= 3) sections.push({ id, title, depth });
      return `<h${depth} id="${id}" data-paper-heading="true">${inner}</h${depth}>`;
    },
  );

  return { html, sections };
}

export default function PaperPage() {
  const { html, sections } = renderPaper(paperMarkdown);
  return (
    <InteractivePaper
      documentId={documentId}
      html={html}
      release="reproduction-paper-v5"
      sections={sections}
    />
  );
}

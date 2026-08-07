import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(new URL(pathname, "http://localhost/"), {
      headers: { accept: "text/html" },
    }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the research dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  const renderedText = html.replaceAll("<!-- -->", "");
  assert.match(html, /<title>HPR Reproduction Control Room<\/title>/i);
  assert.match(html, /Stage 9 concludes: structural reproduction/);
  assert.match(html, /Research roadmap/);
  assert.match(html, /STAGE_9_COMPLETE/);
  assert.match(html, /STAGE 10 LOCKED/);
  assert.match(html, /Held for separate approval/);
  assert.match(renderedText, /4 \/ 5/);
  assert.match(renderedText, /8 \/ 8/);
  assert.match(html, /13 \/ 13/);
  assert.match(html, /12 \/ 12 PASS/);
  assert.match(html, /Stage 8 large-case campaign/);
  assert.match(html, /Unified-memory guard \/ smaller live budget wins/);
  assert.match(html, /Combined planning footprint/);
  assert.match(html, /Correctness and solver-core median seconds/);
  assert.match(html, /65\.50/);
  assert.match(html, /65\.784/);
  assert.match(html, /65\.496/);
  assert.match(html, /case2868rte/);
  assert.match(html, /case9241pegase/);
  for (const value of ["T48", "T64", "T96", "T4", "T6", "T16", "T24", "T32"]) {
    assert.ok(html.includes(value), `missing Stage 8 campaign horizon ${value}`);
  }
  for (const value of ["24.114", "48.303", "94.435", "141.663", "188.897"]) {
    assert.ok(html.includes(value), `missing Stage 8 combined-memory estimate ${value}`);
  }
  for (const value of [
    "76.238645", "804.863493", "49.968351",
    "108.232479", "1078.891699", "68.383103",
    "163.592526", "1621.905045", "105.022444",
    "142.478889", "3087.217721", "193.631896",
    "963.956952", "357.543837",
  ]) {
    assert.ok(html.includes(value), `missing terminal Stage 8 timing ${value}`);
  }
  assert.match(html, /CPU FP64 correctness returned SolveTimeLimit after 3,600\.093 s/);
  assert.match(html, /included final original-space residual evaluation/);
  assert.match(html, /T6 has no CPU median/);
  assert.match(html, /HiGHS and GPU FP64 passed/);
  assert.match(html, /evaluated T16 live and classified T24[\s\S]*T32 statically/);
  assert.match(html, /Memory-blocked/);
  assert.match(html, /Index-blocked/);
  assert.match(html, /zero[\s\S]*new allocations/i);
  assert.match(html, /paper timings stay[\s\S]*separate/i);
  assert.match(html, /no formal speedup is computed/i);
  assert.match(html, /Stage 9 complete/i);
  assert.match(html, /Stage 10[\s\S]*?locked/i);
  assert.match(html, /D - structural reproduction/i);
  assert.match(html, /docs\/final_reproduction_report\.tex/);
  assert.match(html, /results\/stage_9_result_index\.json/);
  assert.match(html, /configs\/benchmarks\/stage_8_large\.json/);
  assert.match(html, /docs\/stage_reports\/stage_8_report\.md/);
  assert.match(html, /results\/raw\/stage_8\/stage_8_validation\.json/);
  assert.match(html, /results\/raw\/stage_8\/stage_8_checks\.json/);
  assert.match(html, /configs\/benchmarks\/stage_8_gpu_only_completion\.json/);
  assert.match(
    html,
    /results\/raw\/stage_8\/gpu_only_completion\/stage_8_gpu_only_completion_validation\.json/,
  );
  assert.match(
    html,
    /results\/raw\/stage_8\/gpu_only_completion\/stage_8_gpu_only_completion_checks\.json/,
  );
  assert.doesNotMatch(html, /APPROVE STAGE 8 AND RUN STAGE 9/);
  assert.doesNotMatch(html, /APPROVE STAGE 7 AND RUN STAGE 8/);
  assert.doesNotMatch(html, /DGX campaign running/);
  assert.match(html, /DGX Spark/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Starter Project/i);
});

test("starter preview is removed and product metadata is present", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  await assert.rejects(
    access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)),
  );
  assert.match(page, /ReproductionDashboard/);
  assert.match(layout, /HPR Reproduction Control Room/);
  assert.match(layout, /\/og-stage9\.png/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton|site-creator-vinext-starter/);
  assert.doesNotMatch(layout, /Starter Project|codex-preview/);
});

test("server-renders the interactive scientific paper", async () => {
  const response = await render("/paper");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  const renderedText = html.replaceAll("<!-- -->", "");
  assert.match(html, /<title>Interactive paper/i);
  assert.match(renderedText, /Read · highlight · revise/);
  assert.match(html, /Select any passage/);
  assert.match(html, /Copy for Codex/);
  assert.match(html, /Download Markdown/);
  assert.match(html, /Loading notes/);
  assert.match(html, /Structural equality correction/);
  assert.match(html, /No speedup is claimed/);
  assert.match(html, /class="katex"/);
  assert.match(html, /reproduction-paper-v5/);
  assert.doesNotMatch(html, /APPROVE STAGE|Stage 10 LOCKED/);
});

test("annotation persistence is bound to D1 and ships a migration", async () => {
  const [hosting, schema, route, migration] = await Promise.all([
    readFile(new URL("../.openai/hosting.json", import.meta.url), "utf8"),
    readFile(new URL("../db/schema.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/annotations/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0000_brown_cassandra_nova.sql", import.meta.url), "utf8"),
  ]);

  assert.match(hosting, /"d1"\s*:\s*"DB"/);
  assert.match(schema, /paper_annotations/);
  assert.match(route, /ownerKey/);
  assert.match(route, /x-paper-session/);
  assert.match(route, /export async function (GET|POST|PATCH|DELETE)/);
  assert.match(migration, /CREATE TABLE `paper_annotations`/);
  assert.match(migration, /idx_paper_annotations_owner_document_created/);
});

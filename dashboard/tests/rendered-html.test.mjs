import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the research dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>HPR Reproduction Control Room<\/title>/i);
  assert.match(html, /From paper to verified GPU solver/);
  assert.match(html, /Research roadmap/);
  assert.match(html, /APPROVE STAGE 5 AND RUN STAGE 6/);
  assert.match(html, /Stage 4 acceptance approved/);
  assert.match(html, /Stage 5 passed \/ Stage 6 awaits approval/);
  assert.match(html, /Stage 5 evidence brief/);
  assert.match(html, /Ten Ruiz passes, Pock-Chambolle alpha one/);
  assert.match(html, /410 \/ 1,032/);
  assert.match(html, /All gates PASS/);
  assert.match(html, /Fixed \/ no restart/);
  assert.match(html, /Adaptive \/ no restart/);
  assert.match(html, /Fixed \/ restart/);
  assert.match(html, /Adaptive \/ restart/);
  assert.match(html, /sourced proxy rather than an\s*author-identical implementation/i);
  assert.match(html, /Stage 6 remains locked/);
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
  assert.doesNotMatch(packageJson, /react-loading-skeleton|site-creator-vinext-starter/);
  assert.doesNotMatch(layout, /Starter Project|codex-preview/);
});

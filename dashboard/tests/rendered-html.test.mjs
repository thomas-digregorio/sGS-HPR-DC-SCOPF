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
  assert.match(html, /APPROVE STAGE 6 AND RUN STAGE 7/);
  assert.match(html, /Controls, preprocessing ablations, and acceptance approval/);
  assert.match(html, /Stage 6 passed \/ Stage 7 awaits approval/);
  assert.match(html, /Stage 6 FP64 parity brief/);
  assert.match(html, /410 \/ 410/);
  assert.match(html, /1,032 \/ 1,032/);
  assert.match(html, /2.62 × 10⁻¹⁴/);
  assert.match(html, /6.90e-15/);
  assert.match(html, /CSR_ALG2/);
  assert.match(html, /FP64 PASS/);
  assert.match(html, /FP32 diagnostic/);
  assert.match(html, /no speedup claim/i);
  assert.match(html, /Stage 7 remains locked/);
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

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
  assert.match(html, /Six benchmark runs, structurally verified/);
  assert.match(html, /Research roadmap/);
  assert.match(html, /APPROVE STAGE 7 AND RUN STAGE 8/);
  assert.match(html, /Controls, preprocessing ablations, and acceptance approval/);
  assert.match(html, /Stage 7 passed \/ Stage 8 awaits approval/);
  assert.match(html, /Stage 7 structural benchmark brief/);
  assert.match(html, /Accepted benchmark rows/);
  assert.match(html, /6 \/ 6/);
  assert.match(html, /19 \/ 19/);
  assert.match(html, /0 \/ 18/);
  assert.match(html, /18 \/ 18 differ/);
  assert.match(html, /4\.1797236508e-6/);
  assert.match(html, /0\.0096347433/);
  assert.match(html, /0\.00622103995/);
  assert.match(html, /4\.2849939e-8/);
  assert.match(html, /537\.835392 s/);
  assert.match(html, /Solver-core median seconds/);
  for (const value of [
    "1.463376", "13.190169", "1.013252",
    "7.265354", "47.398903", "2.924268",
    "32.095478", "153.046368", "9.480950",
    "101.242296", "336.657478", "21.084461",
    "5.366418", "60.349654", "3.938934",
    "22.138002", "250.601316", "15.408042",
  ]) {
    assert.ok(html.includes(value), `missing Stage 7 timing median ${value}`);
  }
  assert.match(html, /paper timings are context only/i);
  assert.match(html, /Stage 8 large runs remain locked/);
  assert.match(html, /docs\/stage_reports\/stage_7_report\.md/);
  assert.doesNotMatch(html, /APPROVE STAGE 6 AND RUN STAGE 7/);
  assert.doesNotMatch(html, /Stage 7 remains locked/);
  assert.doesNotMatch(html, /Stage 6 FP64 parity brief/);
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
  assert.match(layout, /\/og-stage7\.png/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton|site-creator-vinext-starter/);
  assert.doesNotMatch(layout, /Starter Project|codex-preview/);
});

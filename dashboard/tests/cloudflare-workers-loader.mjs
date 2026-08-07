const MOCK_URL =
  "data:text/javascript," +
  encodeURIComponent("export const env = Object.freeze({});");

export async function resolve(specifier, context, nextResolve) {
  if (specifier === "cloudflare:workers") {
    return { url: MOCK_URL, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}

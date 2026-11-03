import { T } from "./strings.js";

export class ApiError extends Error {
  constructor(message, kind) {
    super(message);
    this.kind = kind; // "down" | "http"
  }
}

async function get(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, value);
  });
  let response;
  try {
    response = await fetch(url);
  } catch {
    throw new ApiError(T.apiDown, "down");
  }
  if (response.status === 502 || response.status === 504) throw new ApiError(T.apiDown, "down");
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {}
    throw new ApiError(detail, "http");
  }
  return response.json();
}

async function post(path, body) {
  let response;
  try {
    response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new ApiError(T.apiDown, "down");
  }
  if (response.status === 502 || response.status === 504) throw new ApiError(T.apiDown, "down");
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {}
    throw new ApiError(detail, "http");
  }
  return response.json();
}

export const api = {
  stats: () => get("/api/stats"),
  categories: () => get("/api/categories"),
  search: (q, k, all, nProbe, category) =>
    get("/api/search", { q, k, all: all ? "true" : "", n_probe: nProbe || "", category: category || "" }),
  classify: (text) => post("/api/classify", { text }),
  ask: (question) => post("/api/ask", { question }),
  document: (id, q) => get(`/api/documents/${encodeURIComponent(id)}`, { q }),
  similar: (id, measure, k = 8) => get(`/api/documents/${encodeURIComponent(id)}/similar`, { measure, k }),
};

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

export const api = {
  stats: () => get("/api/stats"),
  search: (q, k, all) => get("/api/search", { q, k, all: all ? "true" : "" }),
  document: (id, q) => get(`/api/documents/${encodeURIComponent(id)}`, { q }),
  similar: (id, measure, k = 8) => get(`/api/documents/${encodeURIComponent(id)}/similar`, { measure, k }),
};

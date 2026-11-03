import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { T } from "../strings.js";
import { formatDate } from "../format.js";
import Marked from "../components/Marked.jsx";
import ErrorNote from "../components/ErrorNote.jsx";

const PAGE = 10;

export default function SearchPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") || "";
  const all = params.get("all") === "1";
  const k = Number(params.get("k")) || PAGE;
  const category = params.get("category") || "";
  const scope = Number(params.get("scope")) || 0; // 0 = every decision, n = the n best groups

  const [draft, setDraft] = useState(q);
  const [data, setData] = useState(null);
  const [stats, setStats] = useState(null);
  const [categories, setCategories] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => setDraft(q), [q]);

  useEffect(() => {
    api.stats().then(setStats).catch(() => {});
    api.categories().then((d) => setCategories(d.categories)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!q.trim()) {
      setData(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .search(q, k, all, scope, category)
      .then((d) => !cancelled && (setData(d), setError(null)))
      .catch((e) => !cancelled && setError(e))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [q, all, k, scope, category]);

  const go = (next) => {
    const p = {};
    if (next.q) p.q = next.q;
    if (next.all) p.all = "1";
    if (next.scope) p.scope = String(next.scope);
    const chosen = next.category === undefined ? category : next.category;
    if (chosen) p.category = chosen;
    if (next.k && next.k !== PAGE) p.k = String(next.k);
    setParams(p);
  };
  const submit = (e) => {
    e.preventDefault();
    go({ q: draft.trim(), all, scope });
  };

  const topScore = data?.results[0]?.score || 1;

  return (
    <>
      <form className="searchbox" onSubmit={submit} role="search">
        <input
          type="search"
          dir="rtl"
          lang="ar"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={T.searchPlaceholder}
          aria-label="عبارة البحث"
          autoFocus
        />
        <button type="submit">{T.searchButton}</button>
        <label className="toggle" title={T.matchAllHint}>
          <input type="checkbox" checked={all} onChange={(e) => go({ q, all: e.target.checked, scope })} />
          {T.matchAll}
        </label>
        {categories.length > 0 && (
          <label className="scope">
            {T.category}
            <select value={category} onChange={(e) => go({ q, all, scope, category: e.target.value })}>
              <option value="">{T.categoryAll}</option>
              {categories.map((c) => (
                <option key={c.name} value={c.name}>{`${c.name_ar} (${c.count})`}</option>
              ))}
            </select>
          </label>
        )}
        {stats?.partitions && (
          <label className="scope" title={T.scopeHint}>
            {T.scope}
            <select value={scope} onChange={(e) => go({ q, all, scope: Number(e.target.value) })}>
              <option value={0}>{T.scopeAll}</option>
              <option value={1}>{T.scopeBest}</option>
              <option value={2}>{T.scopeBest2}</option>
            </select>
          </label>
        )}
      </form>

      {error && <ErrorNote error={error} />}

      {!q.trim() && !error && (
        <section className="intro">
          <h2>{T.examples}</h2>
          <ul className="examples">
            {T.examplesList.map((example) => (
              <li key={example}>
                <button type="button" lang="ar" dir="rtl" onClick={() => go({ q: example, all, scope })}>
                  {example}
                </button>
              </li>
            ))}
          </ul>
          {stats && <p className="corpus">{T.corpus(stats.documents, stats.terms, stats.mode)}</p>}
        </section>
      )}

      {data && (
        <section aria-live="polite" className={loading ? "is-loading" : ""}>
          <Reading data={data} onPick={(term) => go({ q: term, all, scope })} />

          {data.results.length > 0 && (
            <>
              <p className="count">
                <span>{T.resultCount(data.results.length, data.total)}</span>
                {stats && (
                  <span className="scan" title={T.scopeHint}>
                    {data.routing
                      ? data.routing.fallback
                        ? T.searchedEverything
                        : T.searchedGroups(data.routing.groups)
                      : null}{" "}
                    {T.scoredInfo(data.scored, stats.documents, data.took_ms)}
                  </span>
                )}
              </p>
              <ol className="results">
                {data.results.map((r) => (
                  <li key={r.doc_id} className="result">
                    <Link to={`/doc/${r.doc_id}?q=${encodeURIComponent(q)}`} className="result-link">
                      <h3 lang="ar" dir="rtl" className={r.subject ? "" : "missing"}>
                        {r.subject || T.subjectMissing}
                      </h3>
                      <p lang="ar" dir="rtl" className="snippet">
                        <Marked text={r.snippet} />
                      </p>
                    </Link>
                    <div className="meta">
                      <span>
                        {T.decision} <b>{r.decision_no || r.doc_id}</b>
                      </span>
                      {r.case_no && (
                        <span>
                          {T.case} <b>{r.case_no}</b>
                        </span>
                      )}
                      {r.date && <span>{formatDate(r.date)}</span>}
                      {r.category && <span className="group-tag" lang="ar" dir="rtl">{categories.find((c) => c.name === r.category)?.name_ar || r.category}</span>}
                      {r.group !== null && r.group !== undefined && <span className="group-tag">{T.group(r.group + 1)}</span>}
                    </div>
                    <div className="score" title={`${T.score}: ${r.score.toLocaleString("ar-MA", { minimumFractionDigits: 3, maximumFractionDigits: 3 })}`}>
                      <span className="bar" aria-hidden="true">
                        <span style={{ width: `${Math.max(4, (r.score / topScore) * 100)}%` }} />
                      </span>
                      <span className="score-value">{r.score.toLocaleString("ar-MA", { minimumFractionDigits: 3, maximumFractionDigits: 3 })}</span>
                    </div>
                  </li>
                ))}
              </ol>
              {data.total > data.results.length && (
                <button className="more" onClick={() => go({ q, all, scope, k: k + PAGE })}>
                  {T.moreResults}
                </button>
              )}
            </>
          )}

          {data.results.length === 0 && data.terms.length > 0 && data.unknown.length === 0 && (
            <div className="note">
              <strong>{T.noResults}</strong>
              {all && <p>{T.tryOr}</p>}
            </div>
          )}
        </section>
      )}
    </>
  );
}

// The part that shows how the system understood the query: the stems it searched for.
function Reading({ data, onPick }) {
  if (data.terms.length === 0) return <p className="note">{T.noTerms}</p>;
  return (
    <div className="reading">
      <p className="reading-line">
        <span>{T.readAs}</span>
        <span className="chips" lang="ar" dir="rtl">
          {data.terms.map((t) => {
            const kind = data.unknown.includes(t) ? "unknown" : data.ignored.includes(t) ? "ignored" : "";
            return (
              <span key={t} className={`chip ${kind}`}>
                {t}
              </span>
            );
          })}
        </span>
      </p>
      {data.ignored.length > 0 && (
        <p className="reading-note">
          {T.ignoredNote}{" "}
          <span lang="ar" dir="rtl">
            {data.ignored.join("، ")}
          </span>
        </p>
      )}
      {data.unknown.map((term) => (
        <p key={term} className="reading-note">
          {T.noResultsFor} <span lang="ar" dir="rtl" className="inline-ar">{term}</span>.{" "}
          {(data.suggestions[term] || []).length > 0 && (
            <>
              {T.didYouMean}{" "}
              {data.suggestions[term].map((s, i) => (
                <span key={s.term}>
                  {i > 0 && ", "}
                  <button className="linklike" lang="ar" dir="rtl" onClick={() => onPick(s.term)}>
                    {s.term}
                  </button>
                </span>
              ))}
              ?
            </>
          )}
        </p>
      ))}
    </div>
  );
}

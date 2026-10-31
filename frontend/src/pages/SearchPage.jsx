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

  const [draft, setDraft] = useState(q);
  const [data, setData] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => setDraft(q), [q]);

  useEffect(() => {
    api.stats().then(setStats).catch(() => {});
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
      .search(q, k, all)
      .then((d) => !cancelled && (setData(d), setError(null)))
      .catch((e) => !cancelled && setError(e))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [q, all, k]);

  const go = (next) => {
    const p = {};
    if (next.q) p.q = next.q;
    if (next.all) p.all = "1";
    if (next.k && next.k !== PAGE) p.k = String(next.k);
    setParams(p);
  };
  const submit = (e) => {
    e.preventDefault();
    go({ q: draft.trim(), all });
  };

  const topScore = data?.results[0]?.score || 1;

  return (
    <div className={`search-page${q.trim() ? "" : " search-page--landing"}`}>
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
          <input type="checkbox" checked={all} onChange={(e) => go({ q, all: e.target.checked })} />
          {T.matchAll}
        </label>
      </form>

      {error && <ErrorNote error={error} />}

      {!q.trim() && !error && (
        <section className="intro">
          <h2>{T.examples}</h2>
          <ul className="examples">
            {T.examplesList.map((example) => (
              <li key={example}>
                <button type="button" lang="ar" dir="rtl" onClick={() => go({ q: example, all })}>
                  {example}
                </button>
              </li>
            ))}
          </ul>
          {stats && <p className="corpus">{T.corpus(stats.documents, stats.terms, stats.mode)}</p>}
        </section>
      )}

      {data && (
        <section aria-live="polite" aria-busy={loading} className={`search-results${loading ? " is-loading" : ""}`}>
          <Reading data={data} onPick={(term) => go({ q: term, all })} />

          {data.results.length > 0 && (
            <>
              <p className="count">{T.resultCount(data.results.length, data.total)}</p>
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
                    </div>
                    <div className="score" title={`${T.score}: ${r.score.toFixed(3)}`}>
                      <span className="bar" aria-hidden="true">
                        <span style={{ width: `${Math.max(4, (r.score / topScore) * 100)}%` }} />
                      </span>
                      <span className="score-value">{r.score.toFixed(3)}</span>
                    </div>
                  </li>
                ))}
              </ol>
              {data.total > data.results.length && (
                <button className="more" onClick={() => go({ q, all, k: k + PAGE })}>
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
    </div>
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

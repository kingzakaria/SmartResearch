import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { T } from "../strings.js";
import { formatDate } from "../format.js";
import ErrorNote from "../components/ErrorNote.jsx";

export default function DocumentPage() {
  const { id } = useParams();
  const [params] = useSearchParams();
  const q = params.get("q") || "";
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setDoc(null);
    setError(null);
    window.scrollTo(0, 0);
    api.document(id, q).then(setDoc).catch(setError);
  }, [id, q]);

  const back = q ? `/?q=${encodeURIComponent(q)}` : "/";

  if (error) {
    return (
      <>
        <Link to={back} className="backlink">{T.back}</Link>
        <ErrorNote error={error} />
      </>
    );
  }
  if (!doc) return <p className="muted">{T.loading}</p>;

  return (
    <>
      <Link to={back} className="backlink">{q ? T.back : T.home}</Link>
      <div className="doc-layout">
        <article className="doc">
          <h1 lang="ar" dir="rtl" className={doc.subject ? "" : "missing"}>
            {doc.subject
              ? doc.subject_segments.map((s, i) => (s.m ? <mark key={i}>{s.t}</mark> : <span key={i}>{s.t}</span>))
              : T.subjectMissing}
          </h1>
          <dl className="facts">
            <div><dt>{T.decisionNo}</dt><dd>{doc.decision_no || "—"}</dd></div>
            <div><dt>{T.caseNo}</dt><dd>{doc.case_no || "—"}</dd></div>
            <div><dt>{T.date}</dt><dd>{formatDate(doc.date) || "—"}</dd></div>
            <div><dt>{T.words}</dt><dd>{doc.n_words?.toLocaleString("en") ?? "—"}</dd></div>
            <div><dt>{T.file}</dt><dd>{doc.filename}</dd></div>
          </dl>

          {q && (
            <p className="highlight-info">
              <span className="swatch" aria-hidden="true" />
              {T.highlightInfo(doc.matches)}
            </p>
          )}

          <h2 className="visually-hidden">{T.text}</h2>
          <div className="body" lang="ar" dir="rtl">
            {doc.body.map((s, i) => (s.m ? <mark key={i}>{s.t}</mark> : <span key={i}>{s.t}</span>))}
          </div>

          {doc.closing && (
            <details className="closing">
              <summary>{T.closing}</summary>
              <div lang="ar" dir="rtl">{doc.closing}</div>
            </details>
          )}
        </article>

        <Similar id={id} q={q} />
      </div>
    </>
  );
}

function Similar({ id, q }) {
  const [measure, setMeasure] = useState("cosine");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setData(null);
    api.similar(id, measure).then((d) => (setData(d), setError(null))).catch(setError);
  }, [id, measure]);

  return (
    <aside className="similar">
      <h2>{T.similarTitle}</h2>
      <p className="muted small">{T.similarHint}</p>
      <label className="select">
        {T.measure}
        <select value={measure} onChange={(e) => setMeasure(e.target.value)}>
          {Object.entries(T.measures).map(([key, label]) => (
            <option key={key} value={key}>{label}</option>
          ))}
        </select>
      </label>
      {error && <ErrorNote error={error} />}
      {data && (
        <ol>
          {data.results.map((r) => (
            <li key={r.doc_id}>
              <Link to={`/doc/${r.doc_id}${q ? `?q=${encodeURIComponent(q)}` : ""}`}>
                <span lang="ar" dir="rtl" className="s-title">{r.subject || T.subjectMissing}</span>
                <span className="s-meta">
                  <span>{r.decision_no || r.doc_id}</span>
                  <span title={data.is_distance ? T.distance : T.similarity}>{r.value.toFixed(3)}</span>
                </span>
              </Link>
            </li>
          ))}
        </ol>
      )}
    </aside>
  );
}

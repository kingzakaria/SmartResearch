import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { T } from "../strings.js";
import { formatDate } from "../format.js";
import ErrorNote from "../components/ErrorNote.jsx";

// Turns "[1]" and "[2, 3]" in the answer into links to the passages below.
function AnswerText({ text, turn }) {
  const parts = text.split(/(\[[\d\s,،]+\])/g);
  return (
    <p className="ask-text" lang="ar" dir="auto">
      {parts.map((part, i) => {
        if (!/^\[[\d\s,،]+\]$/.test(part)) return part;
        const numbers = part.match(/\d+/g);
        return numbers.map((n) => (
          <a key={`${i}-${n}`} className="cite" href={`#src-${turn}-${n}`} aria-label={T.askCitation(n)}>
            {n}
          </a>
        ));
      })}
    </p>
  );
}

function Source({ s, turn, question }) {
  return (
    <li id={`src-${turn}-${s.n}`} className={`source ${s.cited ? "cited" : ""}`}>
      <div className="source-head">
        <span className="source-n">{s.n}</span>
        <div>
          <Link className="source-title" to={`/doc/${s.doc_id}?q=${encodeURIComponent(question)}`} lang="ar" dir="rtl">
            {T.decision} {s.decision_no || s.doc_id}
            {s.date ? ` · ${formatDate(s.date)}` : ""}
          </Link>
          {s.subject && <p className="source-subject" lang="ar" dir="rtl">{s.subject}</p>}
          <span className="source-state">{s.cited ? T.askCited : T.askNotCited}</span>
        </div>
      </div>
      <details>
        <summary>{T.askShowPassage}</summary>
        <p className="source-text" lang="ar" dir="rtl">{s.text}</p>
      </details>
    </li>
  );
}

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState([]); // [{question, answer?, sources?, retrieval?, warning?}]
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    if (turns.length) endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, loading]);

  const ask = (text) => {
    const q = text.trim();
    if (!q || loading) return;
    setQuestion("");
    setError(null);
    setLoading(true);
    setTurns((t) => [...t, { question: q }]);
    api
      .ask(q)
      .then((d) => setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, ...d } : x))))
      .catch((err) => {
        setError(err);
        setTurns((t) => t.slice(0, -1));
        setQuestion(q);
      })
      .finally(() => setLoading(false));
  };

  const submit = (e) => {
    e.preventDefault();
    ask(question);
  };
  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(question);
    }
  };

  return (
    <>
      <h1 className="page-title">{T.askTitle}</h1>
      <p className="muted classify-help">{T.askHelp}</p>

      {turns.length === 0 && (
        <section className="examples" aria-label={T.askExamples}>
          <p className="muted">{T.askExamples}</p>
          <div className="chips">
            {T.askExamplesList.map((ex) => (
              <button key={ex} type="button" className="chip example-chip" lang="ar" dir="rtl" onClick={() => ask(ex)}>
                {ex}
              </button>
            ))}
          </div>
        </section>
      )}

      <div className="chat" aria-live="polite">
        {turns.map((t, i) => (
          <article key={i} className="turn">
            <div className="bubble-user" lang="ar" dir="auto">
              <span className="bubble-label">{T.askYou}</span>
              {t.question}
            </div>
            {t.answer ? (
              <div className="bubble-ai">
                <span className="bubble-label">{T.askAnswer}</span>
                <AnswerText text={t.answer} turn={i} />
                <p className="ask-meta">
                  {t.retrieval === "hybrid" ? T.askHybrid : T.askKeyword}
                  {t.warning ? ` — ${t.warning}` : ""}
                </p>
                {t.sources?.length > 0 && (
                  <section className="sources">
                    <h2>{T.askSources}</h2>
                    <p className="muted">{T.askSourcesHint}</p>
                    <ol>
                      {t.sources.map((s) => (
                        <Source key={s.n} s={s} turn={i} question={t.question} />
                      ))}
                    </ol>
                  </section>
                )}
              </div>
            ) : (
              i === turns.length - 1 && loading && <p className="muted ask-wait">{T.askThinking}</p>
            )}
          </article>
        ))}
        <div ref={endRef} />
      </div>

      {error && <ErrorNote error={error} />}

      <form onSubmit={submit} className="ask-form">
        <textarea
          lang="ar"
          dir="auto"
          rows={2}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKey}
          placeholder={T.askPlaceholder}
          aria-label={T.askTitle}
          maxLength={1000}
        />
        <div className="ask-actions">
          <button type="submit" disabled={loading || question.trim().length < 2}>{T.askButton}</button>
          {turns.length > 0 && (
            <button type="button" className="ghost" onClick={() => (setTurns([]), setError(null))} disabled={loading}>
              {T.askNew}
            </button>
          )}
        </div>
      </form>
      <p className="muted ask-disclaimer">{T.askDisclaimer}</p>
    </>
  );
}

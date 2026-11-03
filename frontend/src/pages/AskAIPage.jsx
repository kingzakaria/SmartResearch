import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import ErrorNote from "../components/ErrorNote.jsx";
import { T } from "../strings.js";

export default function AskAIPage() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || loading) return;

    setLoading(true);
    setError(null);
    try {
      setResult(await api.ask(trimmedQuestion));
    } catch (requestError) {
      setError(requestError);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <h1 className="page-title">{T.askTitle}</h1>
      <p className="muted ask-help">{T.askHelp}</p>
      <form onSubmit={submit} className="ask-form">
        <textarea
          lang="ar"
          dir="rtl"
          rows={4}
          maxLength={2000}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={T.askPlaceholder}
          aria-label={T.askTitle}
          required
        />
        <div className="ask-form-footer">
          <span className="muted small">{question.length}/2000</span>
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? T.askLoading : T.askButton}
          </button>
        </div>
      </form>

      {error && <ErrorNote error={error} />}
      {loading && <p className="note" role="status">{T.askLoading}</p>}
      {result && result.sources.length === 0 && <p className="note">{T.askNoSources}</p>}
      {result && result.sources.length > 0 && (
        <section className="ask-result" aria-live="polite">
          <h2>{T.askAnswer}</h2>
          <p className="ask-answer" dir="auto">{result.answer}</p>
          <h2>{T.askSources}</h2>
          <ol className="ask-sources">
            {result.sources.map((source) => (
              <li key={source.doc_id}>
                <Link to={`/doc/${encodeURIComponent(source.doc_id)}`}>
                  <strong>{source.subject}</strong>
                  <span className="ask-source-meta">
                    [{source.doc_id}] · {T.askSource}
                    {source.decision_no && `${T.decision} ${source.decision_no}`}
                    {source.date && ` · ${source.date}`}
                  </span>
                </Link>
                <p lang="ar" dir="rtl">{source.excerpt}</p>
              </li>
            ))}
          </ol>
        </section>
      )}
    </>
  );
}

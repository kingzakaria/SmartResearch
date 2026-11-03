import { useState } from "react";
import { api } from "../api.js";
import { T } from "../strings.js";
import ErrorNote from "../components/ErrorNote.jsx";

export default function ClassifyPage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    api
      .classify(text)
      .then((d) => (setResult(d.predictions), setError(null)))
      .catch((err) => (setError(err), setResult(null)))
      .finally(() => setLoading(false));
  };

  return (
    <>
      <h1 className="page-title">{T.classifyTitle}</h1>
      <p className="muted classify-help">{T.classifyHelp}</p>
      <form onSubmit={submit} className="classify-form">
        <textarea
          lang="ar"
          dir="rtl"
          rows={9}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={T.classifyPlaceholder}
          aria-label="النص المراد تصنيفه"
        />
        <button type="submit" disabled={loading || !text.trim()}>{T.classifyButton}</button>
      </form>

      {error && <ErrorNote error={error} />}
      {result && result.length === 0 && <p className="note">{T.classifyEmpty}</p>}
      {result && result.length > 0 && (
        <section className="predictions" aria-live="polite">
          <h2>{T.classifyResult}</h2>
          <ol>
            {result.map((p, i) => (
              <li key={p.name} className={i === 0 ? "top" : ""}>
                <span className="p-name" lang="ar" dir="rtl">{p.name_ar}</span>
                <span className="bar" aria-hidden="true"><span style={{ width: `${Math.max(2, p.probability * 100)}%` }} /></span>
                <span className="p-value">{(p.probability * 100).toLocaleString("ar-MA", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}٪</span>
              </li>
            ))}
          </ol>
        </section>
      )}
    </>
  );
}

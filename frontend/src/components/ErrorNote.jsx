import { T } from "../strings.js";

export default function ErrorNote({ error }) {
  const down = error.kind === "down";
  return (
    <div className="note note-error" role="alert">
      <strong>{down ? T.apiDown : T.errorTitle}</strong>
      {down ? <code>{T.apiCommand}</code> : <p>{error.message}</p>}
    </div>
  );
}

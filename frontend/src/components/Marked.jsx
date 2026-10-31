// Renders text where the matched words are wrapped in « » (the snippet format of the API) as <mark>.
export default function Marked({ text }) {
  const parts = text.split(/(«[^»]*»)/);
  return parts.map((part, i) =>
    part.startsWith("«") && part.endsWith("»") ? <mark key={i}>{part.slice(1, -1)}</mark> : part
  );
}

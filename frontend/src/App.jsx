import { Link, Route, Routes } from "react-router-dom";
import { T } from "./strings.js";
import SearchPage from "./pages/SearchPage.jsx";
import DocumentPage from "./pages/DocumentPage.jsx";

export default function App() {
  return (
    <>
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-name">{T.appName}</span>
          <span className="brand-sub">{T.appSub}</span>
        </Link>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/doc/:id" element={<DocumentPage />} />
          <Route path="*" element={<p className="note">{T.notFound}</p>} />
        </Routes>
      </main>
    </>
  );
}

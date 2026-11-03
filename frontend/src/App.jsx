import { Link, NavLink, Route, Routes } from "react-router-dom";
import { T } from "./strings.js";
import SearchPage from "./pages/SearchPage.jsx";
import DocumentPage from "./pages/DocumentPage.jsx";
import ClassifyPage from "./pages/ClassifyPage.jsx";
import AskPage from "./pages/AskPage.jsx";

export default function App() {
  return (
    <>
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-title" lang="ar" dir="rtl">{T.appTitle}</span>
          <span className="brand-sub" lang="ar" dir="rtl">{T.appSub}</span>
        </Link>
      </header>
      <nav className="tabs" aria-label={T.sections}>
        <NavLink to="/" end>{T.navSearch}</NavLink>
        <NavLink to="/ask">{T.navAsk}</NavLink>
        <NavLink to="/classify">{T.navClassify}</NavLink>
      </nav>
      <main className="page">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/ask" element={<AskPage />} />
          <Route path="/classify" element={<ClassifyPage />} />
          <Route path="/doc/:id" element={<DocumentPage />} />
          <Route path="*" element={<p className="note">{T.notFound}</p>} />
        </Routes>
      </main>
    </>
  );
}

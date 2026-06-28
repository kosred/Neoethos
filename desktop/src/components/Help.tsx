import { useState, type ReactNode } from "react";

/**
 * Collapsible "what this screen does + how to use it" panel. Shown expanded the
 * first time; the user's open/closed choice is remembered per-screen so power
 * users can hide it for good.
 */
export function HelpPanel({
  id,
  title = "What this does & how to use it",
  children,
}: {
  id: string;
  title?: string;
  children: ReactNode;
}) {
  const key = `help.${id}.open`;
  const [open, setOpen] = useState(() => localStorage.getItem(key) !== "0");
  const toggle = () => {
    const next = !open;
    setOpen(next);
    localStorage.setItem(key, next ? "1" : "0");
  };
  return (
    <div className={`help-panel${open ? " open" : ""}`}>
      <button className="help-toggle" onClick={toggle} aria-expanded={open}>
        <span className="help-ico">ⓘ</span>
        <span className="help-title">{title}</span>
        <span className="help-caret">{open ? "▾" : "▸"}</span>
      </button>
      {open && <div className="help-body">{children}</div>}
    </div>
  );
}

/** One labelled step/row inside a HelpPanel. */
export function HelpStep({ n, children }: { n: number | string; children: ReactNode }) {
  return (
    <div className="help-step">
      <span className="help-step-n">{n}</span>
      <div>{children}</div>
    </div>
  );
}

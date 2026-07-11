import Settings from "./Settings";
import Advanced from "./Advanced";

// One consolidated configuration screen: the friendly Settings sections
// (mode, risk preset, autopilot loop, news, broker, data) on top, then the
// full Advanced form + federation + raw-YAML power-user controls below. A
// single place to change everything, with each control's own Save.
export default function Configuration() {
  return (
    <div className="config-consolidated">
      <Settings />
      <div style={{ borderTop: "2px solid var(--line, #1e2a3a)", margin: "28px 0" }} />
      <Advanced />
    </div>
  );
}

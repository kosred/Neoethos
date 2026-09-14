import { Component, type ReactNode } from "react";

/** Keep navigation usable when one screen fails to render. */
export class ScreenBoundary extends Component<
  { children: ReactNode; label: string },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (!this.state.error) return this.props.children;
    return <section className="screen" role="alert">
      <h2>{this.props.label} could not be displayed</h2>
      <p className="banner warn" style={{ overflowWrap: "anywhere" }}>{String(this.state.error)}</p>
      <p>The other sections remain available. This display error does not stop running backend jobs or close broker positions.</p>
      <button onClick={() => this.setState({ error: null })}>Retry this section</button>
    </section>;
  }
}

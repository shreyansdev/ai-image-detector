import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(_: Error): State {
    return { hasError: true };
  }

  public componentDidCatch(_error: Error, _errorInfo: ErrorInfo) {
    // Sanitized: internal state and component stack traces are not exposed to console
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '40px', textAlign: 'center', color: 'var(--accent-red)', border: 'var(--border-width-thin) var(--border-style) var(--accent-red)', background: 'var(--banner-error-bg)', borderRadius: 'var(--border-radius-panel)' }}>
          <h2 style={{ fontFamily: 'var(--display-font)', color: 'var(--accent-red-glow)' }}>CRITICAL_SYSTEM_FAILURE</h2>
          <p style={{ marginTop: '10px', fontFamily: 'var(--terminal-font)', fontSize: '0.85rem' }}>THE_NEURAL_INTERFACE_HAS_ENCOUNTERED_AN_UNRECOVERABLE_ERROR.</p>
          <button 
            onClick={() => window.location.reload()} 
            style={{ marginTop: '20px', borderColor: 'var(--accent-red-glow)', color: 'var(--accent-red-glow)', fontFamily: 'var(--display-font)', letterSpacing: '1px' }}
          >
            REBOOT_SYSTEM
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

import React from 'react';

interface ScannerProps {
  preview: string | null;
  logs: string[];
  isBatch?: boolean;
  batchPreviews?: { url: string; file: File }[];
}

export const ScannerView: React.FC<ScannerProps> = ({ preview, logs, isBatch, batchPreviews }) => (
  <div className="scanning-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 'var(--spacing-grid-gap)' }}>
    <div className="hologram-container" style={{ position: 'relative', display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '15px' }}>
      {isBatch && batchPreviews ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '10px', width: '100%' }}>
          {batchPreviews.map((p, i) => (
            <div key={i} className="batch-preview-item scanning" style={{ aspectRatio: '1', position: 'relative' }}>
              <img src={p.url} alt={`Scanning ${p.file.name}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              <div className="scan-line"></div>
            </div>
          ))}
        </div>
      ) : (
        <>
          {preview && <img src={preview} alt="Scanning preview" style={{ width: '100%', display: 'block', filter: 'grayscale(100%) contrast(1.2)' }} />}
          <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '4px', background: 'var(--primary-cyan)', boxShadow: 'var(--glow-cyan)', animation: 'scan 2.5s infinite linear' }}></div>
        </>
      )}
    </div>
    <div className="panel-3d" style={{ background: 'var(--panel-bg-logs)', padding: 'var(--spacing-log-padding)', minHeight: '300px', maxHeight: '400px', overflowY: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <div className="panel-header">NEURAL_PROCESS_LOGS</div>
      <div style={{ flexGrow: 1, fontFamily: 'var(--terminal-font)', fontSize: '0.75rem' }}>
        {logs.map((log, i) => <div key={i} style={{ marginBottom: '4px', color: 'var(--accent-green)' }}>{log}</div>)}
        <div style={{ width: '8px', height: '12px', background: 'var(--primary-cyan)', display: 'inline-block', animation: 'blink 1s infinite' }}></div>
      </div>
    </div>
  </div>
);

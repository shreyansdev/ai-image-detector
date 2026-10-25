import React from 'react';
import type { BatchPredictionItemResult } from '../services/api';

interface ResultsProps {
  results: {
    label: string;
    confidence: number;
    inference_time_ms?: number;
    source?: string;
    analysis?: string[];
  };
  preview: string | null;
  onReset: () => void;
}

export const ResultsView: React.FC<ResultsProps> = ({ results, preview, onReset }) => {
  const isAI = results.label === "AI Generated";
  const labelColor = isAI ? 'var(--accent-red-glow)' : 'var(--accent-green-glow)';
  const confidencePct = (results.confidence * 100).toFixed(1);
  const modelDisplayName = (results as any).model_display_name || (results as any).model_used || 'EFFICIENTNET-B0';

  return (
    <div 
      className="panel-3d results-container" 
      style={{ 
        border: `var(--border-width-thin) var(--border-style) ${labelColor}`, 
        padding: 'var(--spacing-results-padding)', 
        background: 'var(--panel-bg-results)',
        borderRadius: 'var(--border-radius-panel)'
      }}
    >
      <div className="results-grid">
        <div 
          className="hologram-container" 
          style={{ border: `var(--border-width-thin) var(--border-style) ${labelColor}`, padding: 'var(--spacing-hologram-padding)', height: 'fit-content', borderRadius: 'var(--border-radius-hologram)' }}
        >
          {preview && <img src={preview} alt="Analysis result preview" style={{ width: '100%', display: 'block', borderRadius: 'var(--border-radius-image)' }} />}
        </div>
        <div>
          <div style={{ fontSize: '0.8rem', opacity: 0.6, letterSpacing: '1px' }}>CLASSIFICATION_REPORT</div>
          <div style={{ fontSize: '0.75rem', opacity: 0.6, letterSpacing: '1px', marginTop: '2px', marginBottom: '8px' }}>
            ANALYZED_BY: <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>{modelDisplayName.toUpperCase()}</span>
          </div>
          <h2 style={{ color: labelColor, fontSize: 'clamp(1.8rem, 5vw, 2.8rem)', marginBottom: '5px', fontWeight: 'bold' }}>
            {results.label.toUpperCase()}
          </h2>
          <div style={{ fontSize: 'clamp(1.1rem, 3vw, 1.5rem)', marginBottom: '20px', fontFamily: 'var(--display-font)' }}>
            CONFIDENCE: <span style={{ color: 'var(--primary-cyan)' }}>{confidencePct}% confidence</span>
          </div>

          {results.inference_time_ms !== undefined && (
            <div style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.7)', marginBottom: '25px' }}>
              LATENCY: <span style={{ color: 'var(--primary-cyan, #66fcf1)' }}>{results.inference_time_ms} ms</span>
            </div>
          )}

          {results.source && (
            <div style={{ marginBottom: '25px', borderLeft: '4px solid var(--primary-cyan)', paddingLeft: '15px' }}>
              <span style={{ fontSize: '0.7rem', opacity: 0.6, fontFamily: 'var(--terminal-font)' }}>PRIMARY_SIGNATURE:</span>
              <p style={{ fontSize: '1.1rem', letterSpacing: '2px', fontWeight: 'bold' }}>{results.source.toUpperCase()}</p>
            </div>
          )}

          {results.analysis && results.analysis.length > 0 && (
            <>
              <div className="panel-header" style={{ fontSize: '0.8rem', marginBottom: '10px', opacity: 0.7 }}>FORENSIC_OBSERVATIONS</div>
              <ul style={{ listStyle: 'none', padding: 0 }}>
                {results.analysis.map((item, i) => (
                  <li key={i} style={{ fontSize: '0.8rem', marginBottom: '8px', opacity: 0.8, display: 'flex', gap: '10px', fontFamily: 'var(--terminal-font)' }}>
                    <span style={{ color: 'var(--primary-cyan)' }}>[+]</span> {item}
                  </li>
                ))}
              </ul>
            </>
          )}

          <button 
            onClick={onReset} 
            style={{ 
              marginTop: '30px', 
              width: '100%', 
              padding: '12px 20px', 
              fontSize: '1rem', 
              fontWeight: 'bold', 
              cursor: 'pointer',
              fontFamily: 'var(--display-font)'
            }}
          >
            CLEAR_AND_ANALYZE_NEW_IMAGE
          </button>
        </div>
      </div>
    </div>
  );
};


interface BatchResultsProps {
  results: BatchPredictionItemResult[];
  previews: { file: File; url: string }[];
  onReset: () => void;
}

export const BatchResultsView: React.FC<BatchResultsProps> = ({ results, previews, onReset }) => {
  const successCount = results.filter(r => r.success).length;

  return (
    <div 
      className="panel-3d results-container" 
      style={{ 
        border: 'var(--border-width-thin) var(--border-style) var(--primary-cyan)', 
        padding: 'var(--spacing-results-padding)', 
        background: 'var(--panel-bg-results)',
        borderRadius: 'var(--border-radius-panel)',
        width: '100%',
        overflowX: 'auto'
      }}
    >
      <div style={{ fontFamily: 'var(--terminal-font)', fontSize: '0.9rem', fontWeight: 'bold', color: 'var(--primary-cyan)', marginBottom: '20px', letterSpacing: '1px' }}>
        BATCH_COMPLETE: {successCount}/{results.length} ANALYZED_SUCCESSFULLY
      </div>

      <table className="batch-results-table">
        <thead>
          <tr>
            <th>THUMBNAIL</th>
            <th>FILENAME</th>
            <th>MODEL_USED</th>
            <th>RESULT</th>
            <th>CONFIDENCE</th>
            <th>STATUS</th>
          </tr>
        </thead>
        <tbody>
          {results.map((item, i) => {
            const previewUrl = previews[i]?.url;
            const isAI = item.label === "AI Generated";
            const labelColor = isAI ? 'var(--accent-red-glow)' : 'var(--accent-green-glow)';
            const modelName = item.model_display_name || item.model_used || '-';

            return (
              <tr key={i}>
                <td data-label="THUMBNAIL" className="thumbnail-col">
                  {previewUrl ? (
                    <div className="hologram-container" style={{ width: '50px', height: '50px', padding: '2px', border: `1px solid ${item.success ? 'var(--primary-cyan)' : 'var(--accent-red-glow)'}`, borderRadius: 'var(--border-radius-hologram)' }}>
                      <img src={previewUrl} alt={item.filename} style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 'var(--border-radius-image)' }} />
                    </div>
                  ) : (
                    <div style={{ width: '50px', height: '50px', background: 'rgba(255,255,255,0.05)' }}></div>
                  )}
                </td>
                <td data-label="FILENAME" style={{ wordBreak: 'break-all', fontFamily: 'var(--terminal-font)' }}>{item.filename}</td>
                <td data-label="MODEL_USED">
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-white)', fontFamily: 'var(--terminal-font)' }}>
                    {modelName.toUpperCase()}
                  </span>
                </td>
                <td data-label="RESULT">
                  {item.success ? (
                    <span style={{ color: labelColor, fontWeight: 'bold', fontFamily: 'var(--terminal-font)' }}>
                      {item.label?.toUpperCase()}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--accent-red-glow)', fontSize: '0.8rem', fontFamily: 'var(--terminal-font)' }}>
                      {item.error || "UNKNOWN_ERROR"}
                    </span>
                  )}
                </td>
                <td data-label="CONFIDENCE">
                  {item.success && item.confidence !== undefined ? (
                    <span style={{ fontFamily: 'var(--terminal-font)' }}>{(item.confidence * 100).toFixed(1)}%</span>
                  ) : (
                    <span style={{ opacity: 0.5 }}>-</span>
                  )}
                </td>
                <td data-label="STATUS">
                  {item.success ? (
                    <span style={{ color: 'var(--accent-green-glow)', fontWeight: 'bold', fontFamily: 'var(--terminal-font)' }}>SUCCESS</span>
                  ) : (
                    <span style={{ color: 'var(--accent-red-glow)', fontWeight: 'bold', fontFamily: 'var(--terminal-font)' }}>FAILED</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <button 
        onClick={onReset} 
        style={{ 
          marginTop: '30px', 
          width: '100%', 
          padding: '12px 20px', 
          fontSize: '1rem', 
          fontWeight: 'bold', 
          cursor: 'pointer',
          fontFamily: 'var(--display-font)'
        }}
      >
        CLEAR_AND_START_NEW_BATCH
      </button>
    </div>
  );
};

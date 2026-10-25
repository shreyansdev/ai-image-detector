import React from 'react';
import type { ModelInfo } from '../services/api';

interface ModelSelectorProps {
  models: ModelInfo[];
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
  disabled?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModel,
  onSelectModel,
  disabled = false,
}) => {
  if (!models || models.length === 0) return null;

  return (
    <div style={{ marginBottom: '25px' }}>
      <div className="panel-header" style={{ fontSize: '0.8rem', marginBottom: '12px' }}>
        SELECT_DETECTION_ENGINE
      </div>
      <div
        className="model-selector-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '15px',
        }}
      >
        {models.map((m) => {
          const isSelected = selectedModel.toLowerCase() === m.id.toLowerCase();
          const isAvailable = m.available;
          const isInteractive = isAvailable && !disabled;

          return (
            <div
              key={m.id}
              onClick={() => {
                if (isInteractive) {
                  onSelectModel(m.id);
                }
              }}
              role="button"
              tabIndex={isInteractive ? 0 : -1}
              onKeyDown={(e) => {
                if (isInteractive && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault();
                  onSelectModel(m.id);
                }
              }}
              style={{
                border: isSelected
                  ? '1px solid var(--primary-cyan)'
                  : '1px solid var(--border-cyan-dim)',
                background: isSelected
                  ? 'rgba(102, 252, 241, 0.08)'
                  : 'var(--protocol-card-bg)',
                boxShadow: isSelected ? 'var(--glow-cyan)' : 'none',
                padding: '16px',
                borderRadius: 'var(--border-radius-panel)',
                cursor: isInteractive ? 'pointer' : 'not-allowed',
                opacity: isAvailable ? 1 : 0.45,
                transition: 'all 0.3s ease',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                minHeight: '130px',
              }}
            >
              <div>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '6px',
                  }}
                >
                  <span
                    style={{
                      fontFamily: 'var(--display-font)',
                      fontWeight: 'bold',
                      fontSize: '0.9rem',
                      color: isSelected ? 'var(--primary-cyan)' : 'var(--text-white)',
                      textShadow: isSelected ? '0 0 8px var(--primary-cyan)' : 'none',
                      letterSpacing: '1px',
                    }}
                  >
                    {m.display_name.toUpperCase()}
                  </span>
                  {isSelected && isAvailable && (
                    <span
                      style={{
                        fontSize: '0.65rem',
                        fontFamily: 'var(--terminal-font)',
                        color: 'var(--bg-color)',
                        background: 'var(--primary-cyan)',
                        padding: '2px 6px',
                        fontWeight: 'bold',
                        borderRadius: '2px',
                        letterSpacing: '1px',
                      }}
                    >
                      SELECTED
                    </span>
                  )}
                </div>

                <p
                  style={{
                    fontSize: '0.75rem',
                    color: 'var(--text-grey)',
                    marginBottom: '12px',
                    lineHeight: '1.35',
                  }}
                >
                  {m.description}
                </p>

                {!isAvailable ? (
                  <div
                    style={{
                      color: 'var(--warning-yellow)',
                      fontFamily: 'var(--display-font)',
                      fontWeight: 'bold',
                      fontSize: '0.75rem',
                      textAlign: 'center',
                      padding: '8px 0',
                      border: '1px dashed var(--warning-yellow)',
                      background: 'var(--banner-warning-bg)',
                      letterSpacing: '1px',
                      marginTop: '8px',
                      borderRadius: 'var(--border-radius-banner)',
                    }}
                  >
                    ⚠️ MODEL_UNAVAILABLE
                  </div>
                ) : (
                  <div style={{ marginTop: 'auto' }}>
                    <div className="stat-row" style={{ fontSize: '0.75rem', marginBottom: '4px', fontFamily: 'var(--terminal-font)' }}>
                      <span style={{ opacity: 0.6 }}>ACCURACY</span>
                      <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>
                        {(m.test_accuracy * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="stat-row" style={{ fontSize: '0.75rem', marginBottom: '4px', fontFamily: 'var(--terminal-font)' }}>
                      <span style={{ opacity: 0.6 }}>AUC_ROC</span>
                      <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>
                        {m.auc_roc.toFixed(4)}
                      </span>
                    </div>
                    <div className="stat-row" style={{ fontSize: '0.75rem', marginBottom: '0px', fontFamily: 'var(--terminal-font)' }}>
                      <span style={{ opacity: 0.6 }}>GPU_SPEED</span>
                      <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>
                        ~{m.avg_inference_ms_gpu.toFixed(2)} ms
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ModelSelector;

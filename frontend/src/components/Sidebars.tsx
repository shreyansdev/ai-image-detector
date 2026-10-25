import React from 'react';

export const SidebarLeft: React.FC = () => {
  return (
    <aside className="sidebar panel-3d" aria-label="Network and System Status">
      <div>
        <div className="panel-header">NETWORK_STATUS</div>
        <div className="stat-row" style={{ fontFamily: 'var(--terminal-font)' }}><span>ENCRYPTION</span><span>AES-256</span></div>
      </div>
      <div style={{marginTop: 'auto'}}>
        <div className="panel-header">SESSION_DATA</div>
        <div style={{fontSize: '0.65rem', opacity: 0.6, fontFamily: 'var(--terminal-font)'}}>NO_PERSISTENT_STORAGE_DETECTED // ALL_DATA_PURGED_ON_EXIT</div>
      </div>
    </aside>
  );
};

export const SidebarRight: React.FC = () => {
  return (
    <aside className="sidebar panel-3d" aria-label="Global Metrics and Engine Status">
      <div>
        <div className="panel-header">ENGINE_STATUS</div>
        <div className="stat-row" style={{ fontFamily: 'var(--terminal-font)' }}><span>ENGINES</span><span>DUAL_MODEL</span></div>
        <div className="stat-row" style={{ fontFamily: 'var(--terminal-font)' }}><span>MODEL_1</span><span>EFFICIENTNET-B0</span></div>
        <div className="stat-row" style={{ fontFamily: 'var(--terminal-font)' }}><span>MODEL_2</span><span>FORENSICNET</span></div>
        <div className="stat-row" style={{ fontFamily: 'var(--terminal-font)' }}><span>VERSION</span><span>1.00</span></div>
      </div>
    </aside>
  );
};

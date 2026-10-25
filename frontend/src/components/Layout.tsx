import React, { useState, useEffect, Suspense, lazy } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ErrorBoundary } from './ErrorBoundary';

const Background3D = lazy(() => import('../Background3D'));

export const Layout: React.FC = () => {
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const [coords, setCoords] = useState('FETCHING...');
  const [terminalId] = useState(() => Math.random().toString(36).substring(2, 11).toUpperCase());

  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString());
    }, 2000);

    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setCoords(`${position.coords.latitude.toFixed(4)}° N, ${position.coords.longitude.toFixed(4)}° E`);
        },
        () => {
          setCoords('UNKNOWN_LOC');
        }
      );
    }

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-layout-container">
      <ErrorBoundary>
        <Suspense fallback={<div className="3d-loader" />}>
          <Background3D />
        </Suspense>
      </ErrorBoundary>

      <div className="corner-brkt top-left"></div>
      <div className="corner-brkt top-right"></div>
      <div className="corner-brkt bottom-left"></div>
      <div className="corner-brkt bottom-right"></div>

      {/* Mobile Menu Overlay */}
      {isMenuOpen && (
        <div className="mobile-menu-overlay">
          <div className="header-container" style={{ borderBottom: '1px solid var(--border-cyan-dim)', paddingBottom: '15px' }}>
            <div>
              <h1 className="glitch-text" data-text="SYSTEM_MENU" style={{ fontSize: '1.5rem', fontFamily: 'var(--display-font)', color: 'var(--primary-cyan)' }}>SYSTEM_MENU</h1>
              <p style={{ fontSize: '0.65rem', opacity: 0.6, fontFamily: 'var(--terminal-font)' }}>V1.00 // ACCESS_DIRECTIVES</p>
            </div>
            <button 
              onClick={() => setIsMenuOpen(false)}
              className="mobile-nav-toggle"
              style={{ borderColor: 'var(--accent-red-glow)', color: 'var(--accent-red-glow)', textShadow: '0 0 5px var(--accent-red-glow)' }}
            >
              [CLOSE]
            </button>
          </div>
          
          <div className="mobile-menu-links">
            <NavLink to="/" onClick={() => setIsMenuOpen(false)} className={({ isActive }) => `mobile-nav-link ${isActive ? 'active' : ''}`}>[+] HOME</NavLink>
            <NavLink to="/detector" onClick={() => setIsMenuOpen(false)} className={({ isActive }) => `mobile-nav-link ${isActive ? 'active' : ''}`}>[+] DETECTOR</NavLink>
            <NavLink to="/how-it-works" onClick={() => setIsMenuOpen(false)} className={({ isActive }) => `mobile-nav-link ${isActive ? 'active' : ''}`}>[+] HOW_IT_WORKS</NavLink>
            <NavLink to="/about" onClick={() => setIsMenuOpen(false)} className={({ isActive }) => `mobile-nav-link ${isActive ? 'active' : ''}`}>[+] ABOUT</NavLink>
            <NavLink to="/faq" onClick={() => setIsMenuOpen(false)} className={({ isActive }) => `mobile-nav-link ${isActive ? 'active' : ''}`}>[+] FAQ</NavLink>
          </div>

          <div style={{ marginTop: 'auto', borderTop: '1px solid var(--border-cyan-very-dim)', paddingTop: '20px', fontSize: '0.7rem', opacity: 0.7, fontFamily: 'var(--terminal-font)' }}>
            <div>LOCAL_TIME: {currentTime}</div>
            <div style={{ marginTop: '5px' }}>COORD: {coords}</div>
          </div>
        </div>
      )}

      <header className="header-container">
        <div className="header-brand">
          <h1 className="glitch-text" data-text="AI IMAGE DETECTOR" style={{ fontSize: 'clamp(1.1rem, 4vw, 1.8rem)' }}>AI IMAGE DETECTOR</h1>
          <p style={{ fontSize: '0.65rem', opacity: 0.6, marginTop: '3px', fontFamily: 'var(--terminal-font)' }}>V1.00 // AUTHENTICITY_VERIFICATION_ENGINE</p>
          
          {/* Mobile Stacking Metadata Row */}
          <div className="mobile-only-block" style={{ display: 'none', gap: '15px', fontSize: '0.65rem', opacity: 0.7, marginTop: '8px' }}>
            <span>LOCAL_TIME: {currentTime}</span>
            <span>COORD: {coords}</span>
          </div>
        </div>

        {/* DESKTOP NAV */}
        <nav className="desktop-nav">
          <NavLink to="/" className={({ isActive }) => `nav-link-btn ${isActive ? 'active' : ''}`}>HOME</NavLink>
          <NavLink to="/detector" className={({ isActive }) => `nav-link-btn ${isActive ? 'active' : ''}`}>DETECTOR</NavLink>
          <NavLink to="/how-it-works" className={({ isActive }) => `nav-link-btn ${isActive ? 'active' : ''}`}>HOW_IT_WORKS</NavLink>
          <NavLink to="/about" className={({ isActive }) => `nav-link-btn ${isActive ? 'active' : ''}`}>ABOUT</NavLink>
          <NavLink to="/faq" className={({ isActive }) => `nav-link-btn ${isActive ? 'active' : ''}`}>FAQ</NavLink>
        </nav>

        {/* MOBILE NAV TOGGLE */}
        <button className="mobile-nav-toggle" onClick={() => setIsMenuOpen(true)}>
          [MENU]
        </button>

        {/* DESKTOP METADATA */}
        <div className="desktop-only-block" style={{ textAlign: 'right', fontSize: '0.7rem' }}>
          <span style={{ color: 'var(--primary-cyan)' }}>LOCAL_TIME:</span> {currentTime}<br />
          <span style={{ color: 'var(--primary-cyan)' }}>COORD:</span> {coords}
        </div>
      </header>

      <main style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial="initial"
            animate="animate"
            exit="exit"
            variants={{
              initial: {
                opacity: 0,
                y: 10,
                filter: 'brightness(1.4) contrast(1.2)'
              },
              animate: {
                opacity: 1,
                y: 0,
                filter: 'brightness(1) contrast(1)'
              },
              exit: {
                opacity: 0,
                y: -10,
                filter: 'brightness(0.8) contrast(1.4)'
              }
            }}
            transition={{
              duration: 0.22,
              ease: "easeOut"
            }}
            style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              flexGrow: 1,
              width: '100%'
            }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>

      <footer style={{ marginTop: 'var(--spacing-footer-mt)', paddingTop: 'var(--spacing-footer-pt)', borderTop: 'var(--border-width-thin) var(--border-style) var(--border-cyan-very-dim)', display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', opacity: 0.5, fontFamily: 'var(--terminal-font)' }}>
        <div>SYSTEM_ENCRYPTED // NO_DATA_RETENTION_POLICY_ACTIVE</div>
        <div>TERMINAL_ID: {terminalId}</div>
      </footer>
    </div>
  );
};

export default Layout;

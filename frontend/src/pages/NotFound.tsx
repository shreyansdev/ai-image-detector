import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

export const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', flexGrow: 1, padding: '40px 0', textAlign: 'center' }}>
      <Helmet>
        <title>Page Not Found — AI Image Detector</title>
        <meta name="description" content="The requested forensic route or image analysis coordinate cannot be found. Verify coordinates and return to the main console." />
        <link rel="canonical" href="https://ai-image-detector.local/404" />
        
        {/* Open Graph */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="Page Not Found — AI Image Detector" />
        <meta property="og:description" content="The requested forensic route or image analysis coordinate cannot be found. Verify coordinates and return to the main console." />
        <meta property="og:url" content="https://ai-image-detector.local/404" />
        <meta property="og:image" content="https://ai-image-detector.local/og-image.png" />
        
        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Page Not Found — AI Image Detector" />
        <meta name="twitter:description" content="The requested forensic route or image analysis coordinate cannot be found. Verify coordinates and return to the main console." />
        <meta name="twitter:image" content="https://ai-image-detector.local/og-image.png" />
      </Helmet>

      <div className="panel-3d" style={{ padding: '40px', border: 'var(--border-width-thin) var(--border-style) var(--accent-red-glow)', background: 'var(--panel-bg-results)', borderRadius: 'var(--border-radius-panel)', maxWidth: '500px' }}>
        <h2 style={{ color: 'var(--accent-red-glow)', fontFamily: 'var(--display-font)', fontSize: '1.8rem', marginBottom: '20px', letterSpacing: '1px' }}>
          ERR_404 // ROUTE_NOT_FOUND
        </h2>
        <p style={{ fontFamily: 'var(--terminal-font)', fontSize: '0.85rem', marginBottom: '30px', opacity: 0.8, lineHeight: '1.6' }}>
          THE REQUESTED COORDINATES CANNOT BE LOCATED ON THIS INFERENCE LINK. CONFIRM ACCESS DIRECTIVES.
        </p>
        <button 
          onClick={() => navigate('/')} 
          style={{ borderColor: 'var(--accent-red-glow)', color: 'var(--accent-red-glow)', cursor: 'pointer', fontFamily: 'var(--display-font)', letterSpacing: '1px' }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = 'var(--accent-red-glow)';
            e.currentTarget.style.color = 'var(--bg-color)';
            e.currentTarget.style.boxShadow = '0 0 15px var(--accent-red-glow)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = 'var(--accent-red-glow)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          RETURN_TO_BASE
        </button>
      </div>
    </div>
  );
};

export default NotFoundPage;

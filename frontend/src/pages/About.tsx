import React from 'react';
import { Helmet } from 'react-helmet-async';

export const AboutPage: React.FC = () => {
  return (
    <div style={{ padding: '20px 0', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '30px' }}>
      <Helmet>
        <title>About the Platform — Project Specifications & Tech Stack</title>
        <meta name="description" content="Discover the details of the AI Image Detector platform. Review our React, FastAPI, and PyTorch tech stack, security protocols, and dataset credits." />
        <link rel="canonical" href="https://ai-image-detector.local/about" />
        
        {/* Open Graph */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="About the Platform — Project Specifications & Tech Stack" />
        <meta property="og:description" content="Discover the details of the AI Image Detector platform. Review our React, FastAPI, and PyTorch tech stack, security protocols, and dataset credits." />
        <meta property="og:url" content="https://ai-image-detector.local/about" />
        <meta property="og:image" content="https://ai-image-detector.local/og-image.png" />
        
        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="About the Platform — Project Specifications & Tech Stack" />
        <meta name="twitter:description" content="Discover the details of the AI Image Detector platform. Review our React, FastAPI, and PyTorch tech stack, security protocols, and dataset credits." />
        <meta name="twitter:image" content="https://ai-image-detector.local/og-image.png" />
      </Helmet>

      {/* 1. PAGE HEADER */}
      <section 
        className="panel-3d" 
        style={{ 
          background: 'var(--panel-bg-results)', 
          border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', 
          padding: '30px', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">PROJECT_INFO</div>
        <p style={{ lineHeight: '1.6', fontSize: '0.95rem', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          Authenticity verification engine running real-time neural diagnostics to classify generative image signatures.
        </p>
      </section>

      {/* 2. PROJECT OVERVIEW PANEL */}
      <section 
        className="panel-3d" 
        style={{ 
          background: 'var(--panel-bg-dense)', 
          border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', 
          padding: '30px', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">PROJECT_OVERVIEW</div>
        <p style={{ lineHeight: '1.6', fontSize: '0.9rem', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          This project was developed as a neural-first verification platform to address the growing difficulty of distinguishing authentic photography from synthetic, generative AI-generated visual media. By extracting low-level frequency artifacts and pixel discrepancies, the application provides detailed authenticity classification telemetry within milliseconds, enabling automated digital integrity audits at scale.
        </p>
      </section>

      {/* 3. TECH STACK PANEL */}
      <section 
        className="panel-3d" 
        style={{ 
          background: 'var(--panel-bg-dense)', 
          border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', 
          padding: '30px', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">TECH_STACK_SPECIFICATION</div>
        
        <div 
          style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            gap: '10px', 
            maxWidth: '500px',
            fontFamily: 'var(--terminal-font)', 
            fontSize: '0.85rem' 
          }}
        >
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(102, 252, 241, 0.1)', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>FRONTEND</span>
            <span>REACT + VITE (TYPESCRIPT)</span>
          </div>
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(102, 252, 241, 0.1)', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>BACKEND</span>
            <span>FASTAPI + UVICORN (PYTHON)</span>
          </div>
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(102, 252, 241, 0.1)', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>MODEL_FRAMEWORK</span>
            <span>PYTORCH + TORCHVISION</span>
          </div>
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(102, 252, 241, 0.1)', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>CLASSIFIERS</span>
            <span>EFFICIENTNET-B0 & FORENSICNET (DUAL-ENGINE)</span>
          </div>
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>DATABASE</span>
            <span>SQLITE3 (FORENSICS.DB PLACEHOLDER)</span>
          </div>
        </div>
      </section>

      {/* 4. DATASET CREDITS PANEL */}
      <section 
        className="panel-3d" 
        style={{ 
          background: 'var(--panel-bg-dense)', 
          border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', 
          padding: '30px', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">DATASET_CREDITS</div>
        <p style={{ lineHeight: '1.6', fontSize: '0.9rem', marginBottom: '20px', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          The underlying classifier core has been trained and optimized utilizing two main balanced dataset categories from Kaggle:
        </p>
        <ul style={{ listStyleType: 'none', padding: 0, fontFamily: 'var(--terminal-font)', fontSize: '0.85rem', lineHeight: '2' }}>
          <li>
            <span style={{ color: 'var(--primary-cyan)' }}>[+]</span> 
            <a 
              href="https://www.kaggle.com/datasets/ahmedhamada0/ai-generated-images-vs-real-images" 
              target="_blank" 
              rel="noopener noreferrer" 
              style={{ color: 'var(--primary-cyan)', marginLeft: '5px', textDecoration: 'none' }}
              onMouseOver={e => e.currentTarget.style.textShadow = '0 0 5px var(--primary-cyan)'}
              onMouseOut={e => e.currentTarget.style.textShadow = 'none'}
            >
              Kaggle: AI-Generated vs Real Images Dataset (Ahmed Hamada)
            </a>
          </li>
          <li>
            <span style={{ color: 'var(--primary-cyan)' }}>[+]</span> 
            <a 
              href="https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images" 
              target="_blank" 
              rel="noopener noreferrer" 
              style={{ color: 'var(--primary-cyan)', marginLeft: '5px', textDecoration: 'none' }}
              onMouseOver={e => e.currentTarget.style.textShadow = '0 0 5px var(--primary-cyan)'}
              onMouseOut={e => e.currentTarget.style.textShadow = 'none'}
            >
              Kaggle: Deepfake and Real Images Dataset (Manjil Karki)
            </a>
          </li>
        </ul>
      </section>

      {/* 5. CONTACT/LINKS PANEL */}
      <section 
        className="panel-3d" 
        style={{ 
          background: 'var(--panel-bg-dense)', 
          border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', 
          padding: '30px', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">COMMUNICATION_LINK_NODES</div>
        <p style={{ lineHeight: '1.6', fontSize: '0.9rem', marginBottom: '20px', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          To view source files or establish contact coordinates, follow these nodes:
        </p>
        <ul style={{ listStyleType: 'none', padding: 0, fontFamily: 'var(--terminal-font)', fontSize: '0.85rem', lineHeight: '2' }}>
          <li>
            <span style={{ color: 'var(--primary-cyan)' }}>[+]</span> DEVELOPER: <strong>[YOUR_NAME]</strong>
          </li>
          <li>
            <span style={{ color: 'var(--primary-cyan)' }}>[+]</span> GITHUB: 
            <a 
              href="[YOUR_GITHUB_URL]" 
              style={{ color: 'var(--primary-cyan)', marginLeft: '5px', textDecoration: 'none' }}
              onMouseOver={e => e.currentTarget.style.textShadow = '0 0 5px var(--primary-cyan)'}
              onMouseOut={e => e.currentTarget.style.textShadow = 'none'}
            >
              [YOUR_GITHUB_URL]
            </a>
          </li>
          <li>
            <span style={{ color: 'var(--primary-cyan)' }}>[+]</span> PORTFOLIO: 
            <a 
              href="[YOUR_PORTFOLIO_URL]" 
              style={{ color: 'var(--primary-cyan)', marginLeft: '5px', textDecoration: 'none' }}
              onMouseOver={e => e.currentTarget.style.textShadow = '0 0 5px var(--primary-cyan)'}
              onMouseOut={e => e.currentTarget.style.textShadow = 'none'}
            >
              [YOUR_PORTFOLIO_URL]
            </a>
          </li>
        </ul>
      </section>

    </div>
  );
};

export default AboutPage;

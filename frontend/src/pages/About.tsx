import React from 'react';
import { Helmet } from 'react-helmet-async';

interface TeamNode {
  name: string;
  role: string;
  tag: string;
  links: {
    label: string;
    url: string;
  }[];
}

const teamNodes: TeamNode[] = [
  {
    name: 'Shreyans Jain',
    role: 'LEAD_ARCHITECTURE',
    tag: 'NODE_01',
    links: [
      { label: 'GITHUB', url: 'https://github.com/shreyansdev' },
      { label: 'LINKEDIN', url: 'https://linkedin.com/in/shreyansdev' },
      { label: 'X_TWITTER', url: 'https://x.com/shreyansdev' },
    ],
  },
  {
    name: 'Sakib I Sanadi',
    role: 'CORE_ENGINEERING',
    tag: 'NODE_02',
    links: [
      { label: 'GITHUB', url: 'https://github.com/Sakib200413' },
      { label: 'LINKEDIN', url: 'https://www.linkedin.com/in/sakib-sanadi-76248b382/' },
      { label: 'X_TWITTER', url: 'https://x.com/SakibSanad56351' },
    ],
  },
  {
    name: 'Kushal G Jain',
    role: 'CORE_ENGINEERING',
    tag: 'NODE_03',
    links: [
      { label: 'GITHUB', url: 'https://github.com/Kushalgjain2005' },
      { label: 'LINKEDIN', url: 'https://www.linkedin.com/in/kushal-g-jain-073939365/' },
    ],
  },
  {
    name: 'Shagun Kumari',
    role: 'CORE_ENGINEERING',
    tag: 'NODE_04',
    links: [
      { label: 'GITHUB', url: 'https://github.com/Shagunkumari07' },
      { label: 'LINKEDIN', url: 'https://www.linkedin.com/in/shagun-kumari-0574632b4' },
      { label: 'X_TWITTER', url: 'https://x.com/shagun9142' },
    ],
  },
];

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
          padding: 'clamp(16px, 4vw, 30px)', 
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
          padding: 'clamp(16px, 4vw, 30px)', 
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
          padding: 'clamp(16px, 4vw, 30px)', 
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
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px', borderBottom: '1px solid rgba(102, 252, 241, 0.1)', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>FRONTEND</span>
            <span>REACT + VITE (TYPESCRIPT)</span>
          </div>
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px', borderBottom: '1px solid rgba(102, 252, 241, 0.1)', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>BACKEND</span>
            <span>FASTAPI + UVICORN (PYTHON)</span>
          </div>
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px', borderBottom: '1px solid rgba(102, 252, 241, 0.1)', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>MODEL_FRAMEWORK</span>
            <span>PYTORCH + TORCHVISION</span>
          </div>
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px', borderBottom: '1px solid rgba(102, 252, 241, 0.1)', paddingBottom: '6px' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>CLASSIFIERS</span>
            <span style={{ textAlign: 'right' }}>EFFICIENTNET-B0 & FORENSICNET</span>
          </div>
          <div className="stat-row" style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px', paddingBottom: '6px' }}>
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
          padding: 'clamp(16px, 4vw, 30px)', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">DATASET_CREDITS</div>
        <p style={{ lineHeight: '1.6', fontSize: '0.9rem', marginBottom: '20px', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          The underlying classifier core has been trained and optimized utilizing two main balanced dataset categories from Kaggle:
        </p>
        <ul style={{ listStyleType: 'none', padding: 0, fontFamily: 'var(--terminal-font)', fontSize: '0.85rem' }}>
          <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '10px' }}>
            <span style={{ color: 'var(--primary-cyan)', flexShrink: 0 }}>[+]</span> 
            <a 
              href="https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images" 
              target="_blank" 
              rel="noopener noreferrer" 
              style={{ color: 'var(--primary-cyan)', textDecoration: 'none', wordBreak: 'break-word', lineHeight: 1.5 }}
              onMouseOver={e => e.currentTarget.style.textShadow = '0 0 5px var(--primary-cyan)'}
              onMouseOut={e => e.currentTarget.style.textShadow = 'none'}
            >
              Kaggle: AI-Generated vs Real Images Dataset (Tristan Zhang)
            </a>
          </li>
          <li style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '10px' }}>
            <span style={{ color: 'var(--primary-cyan)', flexShrink: 0 }}>[+]</span> 
            <a 
              href="https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images" 
              target="_blank" 
              rel="noopener noreferrer" 
              style={{ color: 'var(--primary-cyan)', textDecoration: 'none', wordBreak: 'break-word', lineHeight: 1.5 }}
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
          padding: 'clamp(16px, 4vw, 30px)', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">COMMUNICATION_LINK_NODES</div>
        <p style={{ lineHeight: '1.6', fontSize: '0.9rem', marginBottom: '20px', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          To view source files or establish contact coordinates with the engineering team, follow these verified nodes:
        </p>

        {/* REPOSITORY NODE */}
        <div 
          style={{ 
            marginBottom: '24px', 
            padding: '14px 18px', 
            background: 'rgba(80, 227, 194, 0.03)', 
            border: '1px solid rgba(80, 227, 194, 0.2)', 
            borderRadius: 'var(--border-radius-hologram)', 
            display: 'flex', 
            flexWrap: 'wrap', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            gap: '12px',
            fontFamily: 'var(--terminal-font)'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '8px', fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>[+] REPO_SOURCE:</span>
            <span style={{ color: 'var(--text-white)' }}>shreyansdev/ai-image-detector</span>
          </div>
          <a 
            href="https://github.com/shreyansdev/ai-image-detector" 
            target="_blank" 
            rel="noopener noreferrer" 
            style={{ 
              color: 'var(--primary-cyan)', 
              textDecoration: 'none', 
              fontSize: '0.8rem',
              letterSpacing: '0.5px'
            }}
            onMouseOver={e => e.currentTarget.style.textShadow = '0 0 8px var(--primary-cyan)'}
            onMouseOut={e => e.currentTarget.style.textShadow = 'none'}
          >
            [ACCESS_REPOSITORY &rarr;]
          </a>
        </div>

        {/* TEAM NODES GRID */}
        <div 
          style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', 
            gap: '16px',
            fontFamily: 'var(--terminal-font)'
          }}
        >
          {teamNodes.map((member) => (
            <div 
              key={member.name}
              style={{ 
                background: 'rgba(0, 0, 0, 0.45)', 
                border: '1px solid rgba(80, 227, 194, 0.15)', 
                borderRadius: 'var(--border-radius-hologram)', 
                padding: '16px', 
                display: 'flex', 
                flexDirection: 'column', 
                gap: '12px',
                transition: 'border-color 0.2s ease, box-shadow 0.2s ease'
              }}
              onMouseOver={e => {
                e.currentTarget.style.borderColor = 'var(--primary-cyan)';
                e.currentTarget.style.boxShadow = '0 0 12px rgba(80, 227, 194, 0.15)';
              }}
              onMouseOut={e => {
                e.currentTarget.style.borderColor = 'rgba(80, 227, 194, 0.15)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(80, 227, 194, 0.1)', paddingBottom: '8px' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--primary-cyan)', letterSpacing: '0.5px' }}>
                  [{member.tag} // {member.role}]
                </span>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--primary-cyan)', boxShadow: '0 0 6px var(--primary-cyan)' }} />
              </div>

              <div>
                <strong style={{ color: 'var(--text-white)', fontSize: '0.95rem', letterSpacing: '0.5px' }}>
                  {member.name}
                </strong>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: 'auto', paddingTop: '6px' }}>
                {member.links.map(link => (
                  <div key={link.label} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem' }}>
                    <span style={{ color: 'var(--primary-cyan)', flexShrink: 0 }}>&gt;</span>
                    <span style={{ color: 'var(--text-grey)', minWidth: '76px', flexShrink: 0 }}>{link.label}:</span>
                    <a 
                      href={link.url} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      style={{ 
                        color: 'var(--primary-cyan)', 
                        textDecoration: 'none', 
                        overflow: 'hidden', 
                        textOverflow: 'ellipsis', 
                        whiteSpace: 'nowrap'
                      }}
                      title={link.url}
                      onMouseOver={e => e.currentTarget.style.textShadow = '0 0 6px var(--primary-cyan)'}
                      onMouseOut={e => e.currentTarget.style.textShadow = 'none'}
                    >
                      {link.url.replace(/^https?:\/\/(www\.)?/, '')}
                    </a>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
};

export default AboutPage;

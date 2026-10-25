import React from 'react';
import { Helmet } from 'react-helmet-async';

export const HowItWorksPage: React.FC = () => {
  return (
    <div style={{ padding: '20px 0', flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '30px' }}>
      <Helmet>
        <title>System Architecture — How AI Image Detection Works</title>
        <meta name="description" content="Learn about the EfficientNet-B0 neural model and transfer learning pipeline used to classify real vs AI images. Review datasets, preprocessing, and metrics." />
        <link rel="canonical" href="https://ai-image-detector.local/how-it-works" />
        
        {/* Open Graph */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="System Architecture — How AI Image Detection Works" />
        <meta property="og:description" content="Learn about the EfficientNet-B0 neural model and transfer learning pipeline used to classify real vs AI images. Review datasets, preprocessing, and metrics." />
        <meta property="og:url" content="https://ai-image-detector.local/how-it-works" />
        <meta property="og:image" content="https://ai-image-detector.local/og-image.png" />
        
        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="System Architecture — How AI Image Detection Works" />
        <meta name="twitter:description" content="Learn about the EfficientNet-B0 neural model and transfer learning pipeline used to classify real vs AI images. Review datasets, preprocessing, and metrics." />
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
        <div className="panel-header">SYSTEM_ARCHITECTURE</div>
        <p style={{ lineHeight: '1.6', fontSize: '0.95rem', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          The Neural Forensic Engine utilizes a multi-stage execution pipeline designed to scan, validate, preprocess, and evaluate digital images. By executing a forward pass through a deep convolutional model, it classifies whether pixels contain synthetic noise distribution patterns characteristic of generative AI networks (like Midjourney, DALL-E, or Stable Diffusion) or authentic real-world photography.
        </p>
      </section>

      {/* 2. THE MODEL PANEL */}
      {/* 2. THE MODEL PANEL */}
      <section 
        className="panel-3d" 
        style={{ 
          background: 'var(--panel-bg-dense)', 
          border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', 
          padding: '30px', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">THE_FORENSIC_CLASSIFIER_MODELS</div>
        <p style={{ lineHeight: '1.6', fontSize: '0.9rem', marginBottom: '25px', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          The classification system features a <strong>dual-engine neural architecture</strong> designed for multi-layered forensic verification. Users can select between two specialized models depending on accuracy requirements and latency constraints:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))', gap: '20px', marginBottom: '25px' }}>
          {/* EfficientNet-B0 Card */}
          <div className="panel-3d" style={{ padding: '20px', background: 'rgba(0,0,0,0.4)', border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', borderRadius: 'var(--border-radius-hologram)' }}>
            <div className="panel-header" style={{ fontSize: '0.9rem', color: 'var(--primary-cyan)', marginBottom: '8px' }}>EFFICIENTNET-B0 // PRIMARY</div>
            <p style={{ fontSize: '0.8rem', lineHeight: '1.5', opacity: 0.85, fontFamily: 'var(--terminal-font)', marginBottom: '15px' }}>
              Transfer-learned CNN backbone (4.66M parameters). Deep feature extractor trained across ~238,000 images with backbone unfreezing at epoch 8 for max accuracy.
            </p>
            <div className="stat-row" style={{ fontSize: '0.75rem', fontFamily: 'var(--terminal-font)' }}><span>ACCURACY</span><span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>97.82%</span></div>
            <div className="stat-row" style={{ fontSize: '0.75rem', fontFamily: 'var(--terminal-font)' }}><span>AUC_ROC</span><span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>0.9979</span></div>
            <div className="stat-row" style={{ fontSize: '0.75rem', fontFamily: 'var(--terminal-font)' }}><span>GPU_LATENCY</span><span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>~3.78 ms</span></div>
          </div>

          {/* ForensicNet Card */}
          <div className="panel-3d" style={{ padding: '20px', background: 'rgba(0,0,0,0.4)', border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', borderRadius: 'var(--border-radius-hologram)' }}>
            <div className="panel-header" style={{ fontSize: '0.9rem', color: 'var(--primary-cyan)', marginBottom: '8px' }}>FORENSICNET // FAST INFERENCE</div>
            <p style={{ fontSize: '0.8rem', lineHeight: '1.5', opacity: 0.85, fontFamily: 'var(--terminal-font)', marginBottom: '15px' }}>
              Custom tri-stream architecture (1.05M parameters) combining spatial pixel analysis, noise residual extraction (SRM filters), and frequency domain (DCT) decomposition.
            </p>
            <div className="stat-row" style={{ fontSize: '0.75rem', fontFamily: 'var(--terminal-font)' }}><span>ACCURACY</span><span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>94.83%</span></div>
            <div className="stat-row" style={{ fontSize: '0.75rem', fontFamily: 'var(--terminal-font)' }}><span>AUC_ROC</span><span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>0.9903</span></div>
            <div className="stat-row" style={{ fontSize: '0.75rem', fontFamily: 'var(--terminal-font)' }}><span>GPU_LATENCY</span><span style={{ color: 'var(--primary-cyan)', fontWeight: 'bold' }}>~1.78 ms</span></div>
          </div>
        </div>
      </section>

      {/* 3. THE PIPELINE PANEL */}
      <section 
        className="panel-3d" 
        style={{ 
          background: 'var(--panel-bg-dense)', 
          border: 'var(--border-width-thin) var(--border-style) var(--border-cyan-dim)', 
          padding: '30px', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header">PIPELINE_EXECUTION_STAGES</div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <div style={{ color: 'var(--primary-cyan)', fontFamily: 'var(--display-font)', fontSize: '0.85rem', marginBottom: '5px' }}>
              01_IMAGE_UPLOAD
            </div>
            <p style={{ fontSize: '0.8rem', lineHeight: '1.5', opacity: 0.85, paddingLeft: '15px', borderLeft: '2px solid var(--secondary-cyan)', fontFamily: 'var(--terminal-font)' }}>
              The client loads the image payload locally in session memory. Files must not exceed 10MB in size and should use JPEG, PNG, or WEBP extensions.
            </p>
          </div>

          <div>
            <div style={{ color: 'var(--primary-cyan)', fontFamily: 'var(--display-font)', fontSize: '0.85rem', marginBottom: '5px' }}>
              02_VALIDATION
            </div>
            <p style={{ fontSize: '0.8rem', lineHeight: '1.5', opacity: 0.85, paddingLeft: '15px', borderLeft: '2px solid var(--secondary-cyan)', fontFamily: 'var(--terminal-font)' }}>
              API middleware validates file structure via MIME type headers and signature verification (magic bytes analysis) to check for binary spoofing.
            </p>
          </div>

          <div>
            <div style={{ color: 'var(--primary-cyan)', fontFamily: 'var(--display-font)', fontSize: '0.85rem', marginBottom: '5px' }}>
              03_PREPROCESSING
            </div>
            <p style={{ fontSize: '0.8rem', lineHeight: '1.5', opacity: 0.85, paddingLeft: '15px', borderLeft: '2px solid var(--secondary-cyan)', fontFamily: 'var(--terminal-font)' }}>
              The input image is resized to 256x256, center-cropped to 224x224 pixels, and normalized using mean/std statistics before entering the neural pathways.
            </p>
          </div>

          <div>
            <div style={{ color: 'var(--primary-cyan)', fontFamily: 'var(--display-font)', fontSize: '0.85rem', marginBottom: '5px' }}>
              04_MODEL_INFERENCE
            </div>
            <p style={{ fontSize: '0.8rem', lineHeight: '1.5', opacity: 0.85, paddingLeft: '15px', borderLeft: '2px solid var(--secondary-cyan)', fontFamily: 'var(--terminal-font)' }}>
              A PyTorch forward pass runs features through the un-pooled convolutional backbone, feeding spatial logits to the Custom Linear Head to evaluate authentic features.
            </p>
          </div>

          <div>
            <div style={{ color: 'var(--primary-cyan)', fontFamily: 'var(--display-font)', fontSize: '0.85rem', marginBottom: '5px' }}>
              05_RESULT_CLASSIFICATION
            </div>
            <p style={{ fontSize: '0.8rem', lineHeight: '1.5', opacity: 0.85, paddingLeft: '15px', borderLeft: '2px solid var(--secondary-cyan)', fontFamily: 'var(--terminal-font)' }}>
              Computes probabilities for "AI Generated" vs "Real" status classes. Latency is logged, and observations are output as classified telemetry.
            </p>
          </div>
        </div>
      </section>

      {/* 4. LIMITATIONS PANEL */}
      <section 
        className="panel-3d" 
        style={{ 
          background: 'var(--panel-bg-dense)', 
          border: 'var(--border-width-thin) var(--border-style) var(--accent-red-glow)', 
          padding: '30px', 
          borderRadius: 'var(--border-radius-panel)' 
        }}
      >
        <div className="panel-header" style={{ color: 'var(--accent-red-glow)', borderBottomColor: 'var(--accent-red-glow)' }}>
          SYSTEM_CONSTRAINTS_AND_KNOWN_LIMITATIONS
        </div>
        <p style={{ lineHeight: '1.6', fontSize: '0.9rem', color: 'var(--text-grey)', fontFamily: 'var(--terminal-font)' }}>
          The Neural Forensic Engine is subject to specific physical limitations. The model may exhibit reduced reliability on heavily airbrushed, retouched, or professionally color-graded authentic photography, as smoothing passes mimic GAN blurring filters. Additionally, prediction accuracy is heavily dependent on the generative model used, and images generated from architectures not represented in the training corpus (e.g., highly custom model checkpoints or newer version releases) may result in confidence drift.
        </p>
      </section>

    </div>
  );
};

export default HowItWorksPage;

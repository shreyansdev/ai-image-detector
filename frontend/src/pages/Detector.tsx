import React, { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { SidebarLeft, SidebarRight } from '../components/Sidebars';
import { ScannerView } from '../components/Scanner';
import { ResultsView, BatchResultsView } from '../components/Results';
import { ModelSelector } from '../components/ModelSelector';
import { analyzeImage, analyzeBatch, checkHealth, getModels, DEFAULT_MODELS } from '../services/api';
import type { BatchPredictionItemResult, ModelInfo } from '../services/api';

type ViewState = 'uploading' | 'scanning' | 'results';

export const DetectorPage: React.FC = () => {
  const [view, setView] = useState<ViewState>('uploading');
  const [preview, setPreview] = useState<string | null>(null);
  const [results, setResults] = useState<any | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  // Model selection states
  const [models, setModels] = useState<ModelInfo[]>(DEFAULT_MODELS);
  const [selectedModel, setSelectedModel] = useState<string>('efficientnet');

  // Batch specific states
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchPreviews, setBatchPreviews] = useState<{ file: File; url: string }[]>([]);
  const [batchResults, setBatchResults] = useState<BatchPredictionItemResult[] | null>(null);

  // Backend integration & warning/error states
  const [modelLoaded, setModelLoaded] = useState<boolean>(true);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [warningMessage, setWarningMessage] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    let pollInterval: ReturnType<typeof setInterval> | null = null;

    const startPolling = () => {
      if (pollInterval) return;
      pollInterval = setInterval(async () => {
        try {
          const health = await checkHealth(8000);
          if (!isMounted) return;
          if (health.model_loaded) {
            setModelLoaded(true);
            if (pollInterval) {
              clearInterval(pollInterval);
              pollInterval = null;
            }
            getModels().then(m => {
              if (isMounted) setModels(m);
            }).catch(() => {});
          }
        } catch {
          // Still connecting or booting
        }
      }, 3500);
    };

    // Health check call on mount
    checkHealth(8000)
      .then((health) => {
        if (!isMounted) return;
        setModelLoaded(health.model_loaded);
        if (!health.model_loaded) {
          startPolling();
        }
      })
      .catch(() => {
        if (!isMounted) return;
        setModelLoaded(false);
        startPolling();
      });

    // Dynamically fetch model metadata on page load
    getModels()
      .then((modelsData) => {
        if (isMounted) setModels(modelsData);
      })
      .catch((err) => {
        console.error('Failed to load model metadata:', err);
      });

    return () => {
      isMounted = false;
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
      batchPreviews.forEach(p => URL.revokeObjectURL(p.url));
    };
  }, [preview, batchPreviews]);

  const addLog = (msg: string) => {
    setLogs(prev => [...prev.slice(-15), `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  const handleSingleFile = (selectedFile: File) => {
    if (isProcessing) return;

    setErrorMessage(null);
    setWarningMessage(null);

    // Client-side pre-validation: check MIME type starts with "image/"
    if (!selectedFile.type.startsWith("image/")) {
      const msg = "Invalid file type. Please select an image file.";
      setErrorMessage(msg);
      addLog(`ERROR: ${msg}`);
      return;
    }

    // Client-side pre-validation: check size over 10MB
    if (selectedFile.size > 10 * 1024 * 1024) {
      const msg = "File too large. Maximum size is 10MB.";
      setErrorMessage(msg);
      addLog(`ERROR: ${msg}`);
      return;
    }

    // Revoke previous single-file preview if one exists to prevent memory leak
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    // Client-side dimension check using browser Image() object (> 4000x4000 pixels)
    const objectUrl = URL.createObjectURL(selectedFile);
    const img = new Image();

    img.onload = () => {
      if (img.width > 4000 || img.height > 4000) {
        const warnText = `Warning: Image resolution is very high (${img.width}x${img.height}). Analysis may take slightly longer.`;
        setWarningMessage(warnText);
        addLog(`WARN: HIGH_RES_IMAGE_${img.width}X${img.height}`);
      }
      setPreview(objectUrl);
      setView('scanning');
      startScanning(selectedFile);
    };

    img.onerror = () => {
      // Proceed with upload even if dimension check fails
      setPreview(objectUrl);
      setView('scanning');
      startScanning(selectedFile);
    };

    img.src = objectUrl;
  };

  const handleFiles = (fileList: FileList) => {
    if (isProcessing) return;
    setErrorMessage(null);
    setWarningMessage(null);

    const files = Array.from(fileList);
    if (files.length === 0) return;

    if (mode === 'single') {
      handleSingleFile(files[0]);
      return;
    }

    // Batch mode validation
    const totalCount = batchFiles.length + files.length;
    if (totalCount > 10) {
      setErrorMessage("MAX_10_IMAGES_PER_BATCH");
      addLog("ERROR: MAX_10_IMAGES_PER_BATCH");
      return;
    }

    const updatedFiles = [...batchFiles];
    const updatedPreviews = [...batchPreviews];

    for (const file of files) {
      if (!file.type.startsWith("image/")) {
        const msg = `Invalid file type for ${file.name}. Please select only image files.`;
        setErrorMessage(msg);
        addLog(`ERROR: ${msg}`);
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        const msg = `File ${file.name} too large. Maximum size is 10MB.`;
        setErrorMessage(msg);
        addLog(`ERROR: ${msg}`);
        return;
      }
      updatedFiles.push(file);
      updatedPreviews.push({
        file,
        url: URL.createObjectURL(file)
      });
    }

    setBatchFiles(updatedFiles);
    setBatchPreviews(updatedPreviews);
  };

  const removeFile = (index: number) => {
    if (isProcessing) return;
    const updatedFiles = [...batchFiles];
    const updatedPreviews = [...batchPreviews];

    URL.revokeObjectURL(updatedPreviews[index].url);

    updatedFiles.splice(index, 1);
    updatedPreviews.splice(index, 1);

    setBatchFiles(updatedFiles);
    setBatchPreviews(updatedPreviews);
  };

  const startScanning = async (selectedFile: File) => {
    setIsProcessing(true);
    setLogs([]);
    addLog('INITIALIZING NEURAL LINK...');

    try {
      addLog(`MODEL_SELECTED: ${selectedModel.toUpperCase()}`);
      addLog('UPLOADING IMAGE TO INFERENCE PIPELINE...');
      addLog('PROCESSING DEEP FEATURE EXTRACTION...');

      const res = await analyzeImage(selectedFile, selectedModel);

      addLog(`CLASSIFICATION COMPLETE: ${res.label.toUpperCase()}`);
      addLog(`CONFIDENCE SCORE: ${(res.confidence * 100).toFixed(1)}%`);
      addLog(`FORWARD PASS LATENCY: ${res.inference_time_ms} ms`);

      setResults(res);
      setIsProcessing(false);
      setView('results');
    } catch (err: any) {
      const friendlyError = err.message || 'Something went wrong. Please try again.';
      setErrorMessage(friendlyError);
      addLog(`CRITICAL_ERROR: ${friendlyError}`);
      setIsProcessing(false);
      setView('uploading');
    }
  };

  const startBatchScanning = async (selectedFiles: File[]) => {
    setIsProcessing(true);
    setLogs([]);
    addLog('INITIALIZING NEURAL LINK...');
    addLog(`MODEL_SELECTED: ${selectedModel.toUpperCase()}`);
    addLog(`QUEUING BATCH PROTOCOL: ${selectedFiles.length} SIGNALS...`);

    try {
      addLog('TRANSMITTING BATCH STREAM TO CLOUD ENGINE...');
      addLog('ANALYZING LAYER BY LAYER FEATURES...');

      const res = await analyzeBatch(selectedFiles, selectedModel);

      const successCount = res.filter(r => r.success).length;
      addLog(`BATCH PROCESS COMPLETE. SUCCESS: ${successCount}/${selectedFiles.length}`);

      setBatchResults(res);
      setIsProcessing(false);
      setView('results');
    } catch (err: any) {
      const friendlyError = err.message || 'Something went wrong. Please try again.';
      setErrorMessage(friendlyError);
      addLog(`CRITICAL_ERROR: ${friendlyError}`);
      setIsProcessing(false);
      setView('uploading');
    }
  };

  const resetUpload = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    setPreview(null);
    setResults(null);
    setBatchFiles([]);
    batchPreviews.forEach(p => URL.revokeObjectURL(p.url));
    setBatchPreviews([]);
    setBatchResults(null);
    setErrorMessage(null);
    setWarningMessage(null);
    setIsProcessing(false);
    setView('uploading');
  };

  return (
    <>
      <Helmet>
        <title>AI Image Scanner — Analyze & Expose AI-Generated Images</title>
        <meta name="description" content="Upload JPEG, PNG, or WEBP images up to 10MB to analyze pixel structures and detect digital manipulation. Real-time neural diagnostics output in milliseconds." />
        <link rel="canonical" href="https://ai-image-detector.local/detector" />
        
        {/* Open Graph */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="AI Image Scanner — Analyze & Expose AI-Generated Images" />
        <meta property="og:description" content="Upload JPEG, PNG, or WEBP images up to 10MB to analyze pixel structures and detect digital manipulation. Real-time neural diagnostics output in milliseconds." />
        <meta property="og:url" content="https://ai-image-detector.local/detector" />
        <meta property="og:image" content="https://ai-image-detector.local/og-image.png" />
        
        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="AI Image Scanner — Analyze & Expose AI-Generated Images" />
        <meta name="twitter:description" content="Upload JPEG, PNG, or WEBP images up to 10MB to analyze pixel structures and detect digital manipulation. Real-time neural diagnostics output in milliseconds." />
        <meta name="twitter:image" content="https://ai-image-detector.local/og-image.png" />
      </Helmet>

      {/* Model loading status warning banner */}
      {!modelLoaded && (
        <div
          className="warning-banner"
          style={{
            background: 'var(--banner-warning-bg)',
            border: 'var(--border-width-thin) var(--border-style) var(--warning-yellow)',
            color: 'var(--warning-yellow)',
            padding: '12px 20px',
            marginBottom: '25px',
            textAlign: 'center',
            borderRadius: 'var(--border-radius-banner)',
            fontSize: '0.85rem',
            fontWeight: 'bold',
            fontFamily: 'var(--terminal-font)'
          }}
        >
          ⚠️ MODEL_LOADING_IN_PROGRESS // PREDICTIONS_PENDING
        </div>
      )}

      {/* Image high resolution warning banner */}
      {warningMessage && (
        <div
          className="warning-banner"
          style={{
            background: 'var(--banner-warning-bg)',
            border: 'var(--border-width-thin) var(--border-style) var(--warning-yellow)',
            color: 'var(--warning-yellow)',
            padding: '12px 20px',
            marginBottom: '25px',
            textAlign: 'center',
            borderRadius: 'var(--border-radius-banner)',
            fontSize: '0.85rem',
            fontWeight: 'bold',
            fontFamily: 'var(--terminal-font)'
          }}
        >
          ⚠️ {warningMessage}
        </div>
      )}

      {/* Error message banner */}
      {errorMessage && (
        <div
          className="error-banner"
          style={{
            background: 'var(--banner-error-bg)',
            border: 'var(--border-width-thin) var(--border-style) var(--banner-error-border)',
            color: 'var(--banner-error-border)',
            padding: '12px 20px',
            marginBottom: '25px',
            textAlign: 'center',
            borderRadius: 'var(--border-radius-banner)',
            fontSize: '0.85rem',
            fontWeight: 'bold',
            fontFamily: 'var(--terminal-font)'
          }}
        >
          ❌ {errorMessage}
        </div>
      )}

      <div className="grid-layout">
        <SidebarLeft />

        <div className="detector-main">
          {/* Mode Tabs */}
          {view === 'uploading' && (
            <div className="mode-toggle-container">
              <button 
                onClick={() => setMode('single')}
                disabled={isProcessing}
                className="mode-toggle-btn"
                style={{
                  background: mode === 'single' ? 'var(--primary-cyan)' : 'transparent',
                  color: mode === 'single' ? 'var(--bg-color)' : 'var(--primary-cyan)',
                  boxShadow: mode === 'single' ? 'var(--glow-cyan)' : 'none',
                  border: '1px solid var(--primary-cyan)',
                  cursor: isProcessing ? 'not-allowed' : 'pointer',
                }}
              >
                SINGLE_SCAN
              </button>
              <button 
                onClick={() => {
                  setMode('batch');
                  setPreview(null);
                }}
                disabled={isProcessing}
                className="mode-toggle-btn"
                style={{
                  background: mode === 'batch' ? 'var(--primary-cyan)' : 'transparent',
                  color: mode === 'batch' ? 'var(--bg-color)' : 'var(--primary-cyan)',
                  boxShadow: mode === 'batch' ? 'var(--glow-cyan)' : 'none',
                  border: '1px solid var(--primary-cyan)',
                  cursor: isProcessing ? 'not-allowed' : 'pointer',
                }}
              >
                BATCH_SCAN
              </button>
            </div>
          )}

          {view === 'uploading' && (
            <div>
              {/* Model Selector above upload zone */}
              <ModelSelector
                models={models}
                selectedModel={selectedModel}
                onSelectModel={setSelectedModel}
                disabled={isProcessing}
              />

              <section 
                onDragOver={e => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (isProcessing) return;
                  if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
                }}
                style={{
                  border: 'var(--border-width-thin) var(--border-style) var(--primary-cyan)',
                  padding: 'var(--spacing-upload-padding)',
                  textAlign: 'center',
                  background: 'var(--dropzone-bg)',
                  cursor: isProcessing ? 'not-allowed' : 'pointer',
                  position: 'relative',
                  opacity: isProcessing ? 0.6 : 1
                }}
                onClick={() => !isProcessing && document.getElementById('fileInput')?.click()}
                aria-label="Upload Image Area"
              >
                <input 
                  id="fileInput" 
                  type="file" 
                  hidden 
                  accept="image/*" 
                  disabled={isProcessing}
                  multiple={mode === 'batch'}
                  aria-label="File Input"
                  onChange={e => e.target.files && handleFiles(e.target.files)} 
                />
                <h2 style={{ marginBottom: '10px' }}>FEED_DATA_STREAM</h2>
                <p style={{ fontFamily: 'var(--terminal-font)', fontSize: '0.85rem' }}>
                  {isProcessing 
                    ? 'PROCESSING_IMAGE...' 
                    : mode === 'batch' 
                      ? 'DRAG_AND_DROP_MULTIPLE_IMAGES_OR_CLICK_TO_BROWSE // MAX_10'
                      : 'DRAG_AND_DROP_IMAGE_OR_CLICK_TO_BROWSE'}
                </p>
                <div style={{ position: 'absolute', bottom: '10px', right: '10px', fontSize: '0.65rem', opacity: 0.4, fontFamily: 'var(--terminal-font)' }}>
                  {isProcessing ? 'PIPELINE_BUSY' : 'AWAITING_INPUT_SIGNAL'}
                </div>
              </section>

              {/* Show preview grid in batch mode if files are selected */}
              {mode === 'batch' && batchPreviews.length > 0 && (
                <div>
                  <div className="batch-preview-grid">
                    {batchPreviews.map((p, index) => (
                      <div key={index} className="batch-preview-item">
                        <img src={p.url} alt={`Preview ${index}`} />
                        <button 
                          className="remove-btn" 
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFile(index);
                          }}
                          title="Remove image"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={() => startBatchScanning(batchFiles)}
                    disabled={isProcessing}
                    style={{
                      width: '100%',
                      padding: '12px 20px',
                      fontSize: '1rem',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                      background: 'var(--primary-cyan)',
                      color: 'var(--bg-color)',
                      boxShadow: 'var(--glow-cyan)',
                      fontFamily: 'var(--display-font)'
                    }}
                  >
                    ANALYZE_BATCH ({batchFiles.length}_IMAGES)
                  </button>
                </div>
              )}
            </div>
          )}

          {view === 'scanning' && (
            <ScannerView 
              preview={preview} 
              logs={logs} 
              isBatch={mode === 'batch'}
              batchPreviews={batchPreviews}
            />
          )}

          {view === 'results' && (
            mode === 'batch' && batchResults ? (
              <BatchResultsView 
                results={batchResults}
                previews={batchPreviews}
                onReset={resetUpload}
              />
            ) : (
              results && (
                <ResultsView 
                  results={results} 
                  preview={preview} 
                  onReset={resetUpload} 
                />
              )
            )
          )}
        </div>

        <SidebarRight />
      </div>
    </>
  );
};

export default DetectorPage;

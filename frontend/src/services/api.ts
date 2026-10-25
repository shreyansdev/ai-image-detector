/**
 * Frontend API service module for communicating with the AI Image Detector backend.
 */

const getBaseUrl = (): string => {
  // 1. Explicit Vite environment variable takes top priority
  if (typeof import.meta !== "undefined" && import.meta.env) {
    if (import.meta.env.VITE_API_URL) {
      return (import.meta.env.VITE_API_URL as string).replace(/\/+$/, "");
    }
    if (import.meta.env.REACT_APP_API_URL) {
      return (import.meta.env.REACT_APP_API_URL as string).replace(/\/+$/, "");
    }
  }

  // 2. Global process.env fallback
  const globalProcess = (globalThis as any).process;
  if (typeof globalProcess !== "undefined" && globalProcess.env && globalProcess.env.REACT_APP_API_URL) {
    return (globalProcess.env.REACT_APP_API_URL as string).replace(/\/+$/, "");
  }

  // 3. Browser window context: if running locally on Vite dev port 5173, point to backend on 8000
  if (typeof window !== "undefined") {
    if (window.location.hostname === "localhost" && window.location.port === "5173") {
      return "http://localhost:8000";
    }
    // Otherwise fallback to same origin (e.g. monolithic FastAPI serving frontend/dist or reverse proxy)
    return window.location.origin.replace(/\/+$/, "");
  }

  return "http://localhost:8000";
};

const BASE_URL = getBaseUrl();

export interface PredictionResult {
  label: "AI Generated" | "Real" | string;
  confidence: number;
  inference_time_ms: number;
  model_used?: string;
  model_display_name?: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device?: string;
}

export interface ModelInfo {
  id: string;
  display_name: string;
  description: string;
  test_accuracy: number;
  auc_roc: number;
  params: string;
  avg_inference_ms_gpu: number;
  available: boolean;
}

export class ApiError extends Error {
  statusCode: number;

  constructor(message: string, statusCode: number) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Helper to execute fetch requests with a 25-second AbortController timeout.
 */
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs: number = 25000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error: any) {
    if (error.name === "AbortError") {
      throw new ApiError("Request timed out. The server may be busy.", 408);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export const DEFAULT_MODELS: ModelInfo[] = [
  {
    id: "efficientnet",
    display_name: "EfficientNet-B0",
    description: "Transfer-learned CNN classifier optimized for highest classification accuracy.",
    test_accuracy: 0.9782,
    auc_roc: 0.9979,
    params: "4.66M",
    avg_inference_ms_gpu: 3.78,
    available: true,
  },
  {
    id: "forensicnet",
    display_name: "ForensicNet",
    description: "Custom tri-stream forensic architecture (spatial, noise residual & frequency analysis), ultra-fast inference.",
    test_accuracy: 0.9483,
    auc_roc: 0.9903,
    params: "1.05M",
    avg_inference_ms_gpu: 1.78,
    available: true,
  }
];

/**
 * Queries the list of available AI models and their performance metrics from the backend.
 */
export async function getModels(): Promise<ModelInfo[]> {
  try {
    const response = await fetchWithTimeout(`${BASE_URL}/api/v1/models`, {
      method: "GET",
    });

    if (!response.ok) {
      return DEFAULT_MODELS;
    }

    const data: ModelInfo[] = await response.json();
    return data && data.length > 0 ? data : DEFAULT_MODELS;
  } catch (error: any) {
    return DEFAULT_MODELS;
  }
}

/**
 * Sends an image file to the backend prediction endpoint for classification using the selected model.
 */
export async function analyzeImage(file: File, model: string = "efficientnet"): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("model", model);

  try {
    const response = await fetchWithTimeout(`${BASE_URL}/api/v1/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let errorDetail = "";
      try {
        const errJson = await response.json();
        errorDetail = errJson.detail || errJson.message || "";
      } catch {
        // Fallback if response body is not JSON
      }

      switch (response.status) {
        case 413:
          throw new ApiError(errorDetail || "File too large. Maximum allowed image size is 10MB.", 413);
        case 415:
          throw new ApiError(errorDetail || "Unsupported format. Only JPEG, PNG, and WEBP images are accepted.", 415);
        case 422:
          throw new ApiError(errorDetail || "Invalid image. The uploaded file is corrupted or not a valid image.", 422);
        case 429:
          throw new ApiError(errorDetail || "Too many requests. Please wait a moment before trying again.", 429);
        case 503:
          throw new ApiError(errorDetail || "Model not ready. The ML detection model is currently initializing.", 503);
        default:
          throw new ApiError("Something went wrong. Please try again.", response.status);
      }
    }

    const data: PredictionResult = await response.json();
    return data;
  } catch (error: any) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError("Something went wrong. Please try again.", 0);
  }
}

/**
 * Interface representing a single image's prediction result in a batch request.
 */
export interface BatchPredictionItemResult {
  filename: string;
  success: boolean;
  label?: "AI Generated" | "Real" | string;
  confidence?: number;
  inference_time_ms?: number;
  model_used?: string;
  model_display_name?: string;
  error?: string;
}

/**
 * Sends a list of image files to the backend batch prediction endpoint for classification using the selected model.
 */
export async function analyzeBatch(files: File[], model: string = "efficientnet"): Promise<BatchPredictionItemResult[]> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("images", file);
  });
  formData.append("model", model);

  try {
    const response = await fetchWithTimeout(`${BASE_URL}/api/v1/predict/batch`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let errorDetail = "";
      try {
        const errJson = await response.json();
        errorDetail = errJson.detail || errJson.message || "";
      } catch {
        // Fallback if response body is not JSON
      }

      switch (response.status) {
        case 400:
          throw new ApiError(errorDetail || "Maximum 10 images per batch request.", 400);
        case 413:
          throw new ApiError(errorDetail || "Payload too large. Maximum allowed batch size is 100MB.", 413);
        case 429:
          throw new ApiError(errorDetail || "Too many requests. Please wait a moment before trying again.", 429);
        case 503:
          throw new ApiError(errorDetail || "Model not ready. The ML detection model is currently initializing.", 503);
        default:
          throw new ApiError("Something went wrong. Please try again.", response.status);
      }
    }

    const data: BatchPredictionItemResult[] = await response.json();
    return data;
  } catch (error: any) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError("Something went wrong. Please try again.", 0);
  }
}

/**
 * Queries the health status of the backend API service.
 */
export async function checkHealth(timeoutMs: number = 10000): Promise<HealthStatus> {
  try {
    const response = await fetchWithTimeout(`${BASE_URL}/health`, {
      method: "GET",
    }, timeoutMs);

    if (!response.ok) {
      throw new ApiError(`Health check failed with status ${response.status}`, response.status);
    }

    const data: HealthStatus = await response.json();
    return {
      status: data.status,
      model_loaded: data.model_loaded,
      device: data.device,
    };
  } catch (error: any) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError("Unable to reach health check endpoint.", 0);
  }
}

import { NextRequest, NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import fs from 'fs';
import path from 'path';

// Configuration for the captioning service
const CAPTION_SERVICE_URL = process.env.CAPTION_SERVICE_URL || 'http://127.0.0.1:5000';

interface CaptionRequest {
  datasetName: string;
  imagePaths?: string[];
  style?: string;
  prompt?: string;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  doSample?: boolean;
  saveToFile?: boolean;
  overwriteExisting?: boolean;
}

interface CaptionResult {
  imagePath: string;
  success: boolean;
  caption?: string;
  error?: string;
  generationTime?: number;
}

async function checkCaptionService(): Promise<boolean> {
  try {
    const response = await fetch(`${CAPTION_SERVICE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    
    if (!response.ok) {
      return false;
    }
    
    const data = await response.json();
    return data.status === 'healthy' && data.model_loaded;
  } catch (error) {
    console.error('Caption service health check failed:', error);
    return false;
  }
}

async function captionSingleImage(
  imagePath: string,
  style: string,
  prompt?: string,
  generationParams?: any
): Promise<CaptionResult> {
  try {
    const requestBody = {
      image_path: imagePath,
      style,
      prompt,
      ...generationParams,
    };

    const response = await fetch(`${CAPTION_SERVICE_URL}/caption`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json();
      return {
        imagePath,
        success: false,
        error: errorData.error || `HTTP ${response.status}`,
      };
    }

    const data = await response.json();
    return {
      imagePath,
      success: data.success,
      caption: data.caption,
      generationTime: data.generation_time,
    };
  } catch (error) {
    return {
      imagePath,
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

async function captionBatchImages(
  imagePaths: string[],
  style: string,
  prompt?: string,
  generationParams?: any
): Promise<CaptionResult[]> {
  try {
    const requestBody = {
      image_paths: imagePaths,
      style,
      prompt,
      ...generationParams,
    };

    const response = await fetch(`${CAPTION_SERVICE_URL}/batch_caption`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `HTTP ${response.status}`);
    }

    const data = await response.json();
    return data.results.map((result: any) => ({
      imagePath: result.image_path,
      success: result.success,
      caption: result.caption,
      error: result.error,
      generationTime: result.generation_time,
    }));
  } catch (error) {
    // If batch fails, return error for all images
    return imagePaths.map(imagePath => ({
      imagePath,
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }));
  }
}

function saveCaption(imagePath: string, caption: string): boolean {
  try {
    const captionPath = imagePath.replace(/\.[^/.]+$/, '.txt');
    fs.writeFileSync(captionPath, caption, 'utf8');
    return true;
  } catch (error) {
    console.error(`Failed to save caption for ${imagePath}:`, error);
    return false;
  }
}

function getImageFiles(datasetFolder: string): string[] {
  try {
    const files = fs.readdirSync(datasetFolder);
    const imageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'];
    
    return files
      .filter(file => {
        const ext = path.extname(file).toLowerCase();
        return imageExtensions.includes(ext);
      })
      .map(file => path.join(datasetFolder, file))
      .sort();
  } catch (error) {
    console.error(`Failed to read dataset folder ${datasetFolder}:`, error);
    return [];
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: CaptionRequest = await request.json();
    
    if (!body.datasetName) {
      return NextResponse.json({ error: 'datasetName is required' }, { status: 400 });
    }

    // Check if caption service is available
    const serviceAvailable = await checkCaptionService();
    if (!serviceAvailable) {
      return NextResponse.json({ 
        error: 'Caption service is not available. Please ensure the captioning service is running.' 
      }, { status: 503 });
    }

    // Get dataset folder
    const datasetsRoot = await getDatasetsRoot();
    const datasetFolder = path.join(datasetsRoot, body.datasetName);
    
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: 'Dataset folder not found' }, { status: 404 });
    }

    // Determine which images to caption
    let imagePaths: string[];
    if (body.imagePaths && body.imagePaths.length > 0) {
      // Use provided image paths
      imagePaths = body.imagePaths.map(imgPath => 
        path.isAbsolute(imgPath) ? imgPath : path.join(datasetFolder, imgPath)
      );
    } else {
      // Get all image files in the dataset
      imagePaths = getImageFiles(datasetFolder);
    }

    if (imagePaths.length === 0) {
      return NextResponse.json({ error: 'No image files found' }, { status: 400 });
    }

    // Filter out images that already have captions if not overwriting
    if (!body.overwriteExisting) {
      imagePaths = imagePaths.filter(imagePath => {
        const captionPath = imagePath.replace(/\.[^/.]+$/, '.txt');
        return !fs.existsSync(captionPath);
      });
    }

    if (imagePaths.length === 0) {
      return NextResponse.json({ 
        error: 'All images already have captions. Set overwriteExisting to true to regenerate.' 
      }, { status: 400 });
    }

    // Prepare generation parameters
    const generationParams = {
      max_new_tokens: body.maxNewTokens || 256,
      temperature: body.temperature || 0.6,
      top_p: body.topP || 0.9,
      top_k: body.topK,
      do_sample: body.doSample !== false,
    };

    // Generate captions
    let results: CaptionResult[];
    const style = body.style || 'descriptive';
    
    if (imagePaths.length === 1) {
      // Single image
      const result = await captionSingleImage(
        imagePaths[0],
        style,
        body.prompt,
        generationParams
      );
      results = [result];
    } else {
      // Batch processing
      results = await captionBatchImages(
        imagePaths,
        style,
        body.prompt,
        generationParams
      );
    }

    // Save captions to files if requested
    if (body.saveToFile !== false) {
      for (const result of results) {
        if (result.success && result.caption) {
          const saved = saveCaption(result.imagePath, result.caption);
          if (!saved) {
            result.error = 'Failed to save caption file';
          }
        }
      }
    }

    // Calculate statistics
    const successful = results.filter(r => r.success).length;
    const failed = results.length - successful;
    const totalTime = results.reduce((sum, r) => sum + (r.generationTime || 0), 0);

    return NextResponse.json({
      success: true,
      results,
      statistics: {
        total: results.length,
        successful,
        failed,
        totalTime,
        averageTime: successful > 0 ? totalTime / successful : 0,
      },
      settings: {
        style,
        prompt: body.prompt,
        generationParams,
        saveToFile: body.saveToFile !== false,
        overwriteExisting: body.overwriteExisting || false,
      },
    });

  } catch (error: any) {
    console.error('Caption API error:', error);
    return NextResponse.json({ 
      error: error.message || 'Internal server error' 
    }, { status: 500 });
  }
}

export async function GET(request: NextRequest) {
  try {
    // Get available prompt styles from the caption service
    const response = await fetch(`${CAPTION_SERVICE_URL}/prompts`);
    
    if (!response.ok) {
      return NextResponse.json({ 
        error: 'Failed to get available prompts from caption service' 
      }, { status: 503 });
    }
    
    const data = await response.json();
    
    // Also check service health
    const serviceAvailable = await checkCaptionService();
    
    return NextResponse.json({
      available: serviceAvailable,
      prompts: data.prompts,
      styles: data.styles,
      serviceUrl: CAPTION_SERVICE_URL,
    });

  } catch (error: any) {
    console.error('Caption API GET error:', error);
    return NextResponse.json({ 
      error: error.message || 'Internal server error' 
    }, { status: 500 });
  }
}

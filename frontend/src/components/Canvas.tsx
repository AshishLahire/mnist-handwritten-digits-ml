import React, { useRef, useState, useEffect } from 'react';
import { Eraser, Pencil, RotateCcw } from 'lucide-react';
import { cn } from '../lib/utils';

interface CanvasProps {
  onImageChange: (data: number[] | null) => void;
}

export const Canvas: React.FC<CanvasProps> = ({ onImageChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isEmpty, setIsEmpty] = useState(true);

  // Set up the canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set line properties
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 32;
    ctx.shadowBlur = 2;
    ctx.shadowColor = 'white';

    // Fill background with black for digit recognition consistency
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (canvas) {
      processImage(canvas);
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = ('touches' in e) ? e.touches[0].clientX - rect.left : (e as React.MouseEvent).clientX - rect.left;
    const y = ('touches' in e) ? e.touches[0].clientY - rect.top : (e as React.MouseEvent).clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsEmpty(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    setIsEmpty(true);
    onImageChange(null);
  };

  const processImage = (sourceCanvas: HTMLCanvasElement) => {
    const ctx = sourceCanvas.getContext('2d');
    if (!ctx) return;

    // 1. Find bounding box of the digit
    const imageData = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
    const pixels = imageData.data;
    let minX = sourceCanvas.width, minY = sourceCanvas.height, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < sourceCanvas.height; y++) {
      for (let x = 0; x < sourceCanvas.width; x++) {
        const index = (y * sourceCanvas.width + x) * 4;
        if (pixels[index] > 20) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }

    if (!found) {
      onImageChange(null);
      return;
    }

    // 2. Crop the digit to a 20x20 box (with padding)
    const width = maxX - minX;
    const height = maxY - minY;
    
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = 20;
    cropCanvas.height = 20;
    const cropCtx = cropCanvas.getContext('2d');
    if (!cropCtx) return;

    // Maintain aspect ratio while resizing to fit in 20x20
    let drawWidth, drawHeight;
    if (width > height) {
      drawWidth = 20;
      drawHeight = (height / width) * 20;
    } else {
      drawHeight = 20;
      drawWidth = (width / height) * 20;
    }
    const dx = (20 - drawWidth) / 2;
    const dy = (20 - drawHeight) / 2;

    cropCtx.fillStyle = 'black';
    cropCtx.fillRect(0, 0, 20, 20);
    cropCtx.drawImage(sourceCanvas, minX, minY, width, height, dx, dy, drawWidth, drawHeight);

    // 3. Calculate Center of Mass of the 20x20 image
    const croppedData = cropCtx.getImageData(0, 0, 20, 20);
    const croppedPixels = croppedData.data;
    let sumX = 0, sumY = 0, totalMass = 0;

    for (let y = 0; y < 20; y++) {
      for (let x = 0; x < 20; x++) {
        const index = (y * 20 + x) * 4;
        const mass = croppedPixels[index];
        sumX += x * mass;
        sumY += y * mass;
        totalMass += mass;
      }
    }

    const centerX = sumX / totalMass;
    const centerY = sumY / totalMass;

    // 4. Place the 20x20 image into the final 28x28 image such that CoM is at (14, 14)
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = 28;
    finalCanvas.height = 28;
    const finalCtx = finalCanvas.getContext('2d');
    if (!finalCtx) return;

    finalCtx.fillStyle = 'black';
    finalCtx.fillRect(0, 0, 28, 28);

    // Shift to center of mass
    const offsetX = 14 - centerX;
    const offsetY = 14 - centerY;

    finalCtx.drawImage(cropCanvas, offsetX, offsetY);

    // 5. Export grayscale data
    const finalImageData = finalCtx.getImageData(0, 0, 28, 28);
    const finalPixels = finalImageData.data;
    const grayscale = [];
    for (let i = 0; i < finalPixels.length; i += 4) {
      grayscale.push(finalPixels[i] / 255);
    }

    onImageChange(grayscale);
  };

  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-md mx-auto animate-fade-in">
      <div className="relative group p-1 rounded-2xl bg-gradient-to-br from-primary-500/20 to-purple-500/20 backdrop-blur-sm border border-white/10 shadow-2xl">
        <canvas
          ref={canvasRef}
          width={400}
          height={400}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="bg-black rounded-xl cursor-crosshair touch-none w-full aspect-square shadow-inner"
        />
        
        {isEmpty && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-40 transition-opacity group-hover:opacity-60">
            <div className="text-center">
              <Pencil className="w-12 h-12 mx-auto mb-2 text-white/50" />
              <p className="text-sm text-white/50 font-medium">Draw a digit here</p>
            </div>
          </div>
        )}
      </div>

      <div className="flex gap-3 w-full">
        <button
          onClick={clearCanvas}
          className="flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all active:scale-95 group"
        >
          <RotateCcw className="w-4 h-4 text-white/70 group-hover:rotate-[-45deg] transition-transform" />
          <span className="font-medium text-white/90">Clear Canvas</span>
        </button>
      </div>
    </div>
  );
};

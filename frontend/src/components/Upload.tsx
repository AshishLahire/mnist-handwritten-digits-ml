import React, { useCallback, useState } from 'react';
import { Upload as UploadIcon, X, Image as ImageIcon } from 'lucide-react';
import { cn } from '../lib/utils';

interface UploadProps {
  onImageChange: (data: number[] | null) => void;
}

export const Upload: React.FC<UploadProps> = ({ onImageChange }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Process the image to 28x28 grayscale
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Draw and convert to grayscale
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, 28, 28);
        
        // Mantain aspect ratio and center
        const scale = Math.min(28 / img.width, 28 / img.height);
        const x = (28 - img.width * scale) / 2;
        const y = (28 - img.height * scale) / 2;
        
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);

        const imageData = ctx.getImageData(0, 0, 28, 28);
        const pixels = imageData.data;
        const grayscale = [];

        for (let i = 0; i < pixels.length; i += 4) {
          // Weighted grayscale conversion or just R channel
          const avg = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
          grayscale.push(avg / 255);
        }

        onImageChange(grayscale);
        setPreview(e.target?.result as string);
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const clearImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreview(null);
    onImageChange(null);
  };

  return (
    <div className="w-full max-w-md mx-auto animate-fade-in">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        className={cn(
          "relative aspect-square rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center cursor-pointer group overflow-hidden",
          isDragging 
            ? "border-primary-500 bg-primary-500/10 scale-[1.02]" 
            : "border-white/20 bg-white/5 hover:bg-white/10 hover:border-white/30",
          preview && "border-none"
        )}
        onClick={() => !preview && document.getElementById('file-upload')?.click()}
      >
        <input
          id="file-upload"
          type="file"
          className="hidden"
          accept="image/*"
          onChange={onFileChange}
        />

        {preview ? (
          <div className="relative w-full h-full group/preview">
            <img 
              src={preview} 
              alt="Preview" 
              className="w-full h-full object-contain bg-black/40 rounded-2xl transition-transform duration-500 group-hover/preview:scale-110" 
            />
            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover/preview:opacity-100 transition-opacity flex items-center justify-center">
              <button
                onClick={clearImage}
                className="p-3 bg-red-500/80 hover:bg-red-500 rounded-full text-white shadow-lg transition-transform hover:scale-110"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="p-6 rounded-full bg-primary-500/10 mb-4 transition-transform duration-500 group-hover:scale-110">
              <UploadIcon className="w-10 h-10 text-primary-400" />
            </div>
            <div className="text-center px-6">
              <p className="text-lg font-semibold text-white/90 mb-1">Upload an image</p>
              <p className="text-sm text-white/50">Drag and drop or click to browse</p>
            </div>
          </>
        )}
      </div>
      
      <p className="mt-4 text-center text-xs text-white/30 italic">
        * Images will be automatically resized to 28x28 and converted to grayscale
      </p>
    </div>
  );
};

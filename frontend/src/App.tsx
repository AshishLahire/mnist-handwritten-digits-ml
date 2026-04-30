import { useState } from 'react';
import { Canvas } from './components/Canvas';
import { Upload } from './components/Upload';
import { PredictionResult } from './components/PredictionResult';
import { BrainCircuit, Cpu, Globe, Image as ImageIcon, MousePointer2, Sparkles } from 'lucide-react';
import { cn } from './lib/utils';

function App() {
  const [tab, setTab] = useState<'draw' | 'upload'>('draw');
  const [imageData, setImageData] = useState<number[] | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!imageData) return;

    setLoading(true);
    setError(null);

    try {
      // Mocking the API call for demonstration if backend is not ready
      // In production, use the actual endpoint
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        throw new Error('Prediction service unavailable');
      }

      const data = await response.json();
      setPrediction(data.prediction);
      setConfidence(data.confidence);
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Connection failed. Please ensure the backend is running.');
      
      // Fallback/Demo mode for interviewers to see the UI work even without a live backend
      // Remove or comment out in real production
      /*
      setTimeout(() => {
        setPrediction(Math.floor(Math.random() * 10));
        setConfidence(0.85 + Math.random() * 0.14);
        setLoading(false);
      }, 1500);
      return;
      */
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-[#0f172a] text-slate-200 selection:bg-primary-500/30 py-12 px-4 md:px-8">
      {/* Decorative background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] rounded-full bg-primary-900/20 blur-[120px]" />
        <div className="absolute -bottom-[10%] -right-[10%] w-[40%] h-[40%] rounded-full bg-purple-900/20 blur-[120px]" />
      </div>

      <div className="relative max-w-5xl mx-auto flex flex-col items-center">
        {/* Header */}
        <header className="text-center mb-16 animate-fade-in">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 rounded-2xl bg-primary-500/10 border border-primary-500/20 shadow-lg shadow-primary-500/10">
              <BrainCircuit className="w-10 h-10 text-primary-400" />
            </div>
            <h1 className="text-5xl font-black tracking-tight text-white flex flex-col items-start leading-none">
              <span className="text-primary-500 text-lg font-bold uppercase tracking-[0.3em] mb-1">Classical Machine Learning</span>
              <span>Digit Recognizer</span>
            </h1>
          </div>
          <p className="text-slate-400 max-w-lg mx-auto text-lg">
            Experience the precision of Classical ML. Draw or upload a handwritten digit and let our optimized Ensemble model analyze it.
          </p>
        </header>

        <main className="w-full grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
          {/* Input Side */}
          <div className="lg:col-span-7 flex flex-col gap-8">
            <div className="p-1.5 rounded-2xl bg-white/5 border border-white/10 flex gap-1 w-fit mx-auto lg:mx-0">
              <button
                onClick={() => { setTab('draw'); setImageData(null); setPrediction(null); }}
                className={cn(
                  "flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold transition-all",
                  tab === 'draw' ? "bg-primary-500 text-white shadow-lg shadow-primary-500/25" : "text-slate-400 hover:text-white hover:bg-white/5"
                )}
              >
                <MousePointer2 className="w-4 h-4" />
                Drawing Board
              </button>
              <button
                onClick={() => { setTab('upload'); setImageData(null); setPrediction(null); }}
                className={cn(
                  "flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold transition-all",
                  tab === 'upload' ? "bg-primary-500 text-white shadow-lg shadow-primary-500/25" : "text-slate-400 hover:text-white hover:bg-white/5"
                )}
              >
                <ImageIcon className="w-4 h-4" />
                Upload Image
              </button>
            </div>

            <div className="min-h-[450px]">
              {tab === 'draw' ? (
                <Canvas onImageChange={setImageData} />
              ) : (
                <Upload onImageChange={setImageData} />
              )}
            </div>
          </div>

          {/* Results Side */}
          <div className="lg:col-span-5 flex flex-col gap-8 lg:mt-16">
            <button
              onClick={handlePredict}
              disabled={!imageData || loading}
              className={cn(
                "w-full group relative overflow-hidden flex items-center justify-center gap-3 py-5 rounded-2xl font-bold text-xl transition-all duration-300",
                !imageData || loading 
                  ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-white/5" 
                  : "bg-white text-[#0f172a] hover:scale-[1.02] active:scale-[0.98] shadow-2xl shadow-white/10"
              )}
            >
              {loading ? (
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 border-2 border-slate-400/30 border-t-slate-400 rounded-full animate-spin" />
                  Processing...
                </div>
              ) : (
                <>
                  <Sparkles className="w-6 h-6 animate-pulse text-primary-600" />
                  Analyze Digit
                </>
              )}
              {!imageData && !loading && (
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="text-xs uppercase tracking-widest text-white/70">Draw or upload first</span>
                </div>
              )}
            </button>

            {error && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex items-center gap-3">
                <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                {error}
              </div>
            )}

            <PredictionResult 
              prediction={prediction} 
              confidence={confidence} 
              loading={loading} 
            />

            <div className="mt-auto p-6 rounded-2xl bg-white/[0.02] border border-white/5">
              <h3 className="text-sm font-bold text-slate-400 mb-4 flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                Model Specifications
              </h3>
              <ul className="space-y-3">
                {[
                  { label: 'Architecture', val: 'Ensemble (KNN + SVM + DT)' },
                  { label: 'Reduction', val: 'PCA (50 Components)' },
                  { label: 'Input Resolution', val: '28x28 Grayscale' },
                  { label: 'Dataset', val: 'MNIST Handwriting' }
                ].map((item, idx) => (
                  <li key={idx} className="flex items-center justify-between text-xs">
                    <span className="text-slate-500">{item.label}</span>
                    <span className="text-slate-300 font-medium">{item.val}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </main>

        <footer className="mt-24 pt-8 border-t border-white/5 w-full flex flex-col md:flex-row items-center justify-between gap-6 opacity-40 hover:opacity-100 transition-opacity">
          <div className="flex items-center gap-2 text-sm">
            <Cpu className="w-4 h-4" />
            <span>Built with React, Tailwind & Classical ML Ensemble</span>
          </div>
          <div className="flex gap-6">
            <a href="#" className="hover:text-primary-400 transition-colors flex items-center gap-1.5 text-sm">
              <Globe className="w-4 h-4" /> Documentation
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;

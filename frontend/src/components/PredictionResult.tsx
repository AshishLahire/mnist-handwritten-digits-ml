import React from 'react';
import { cn } from '../lib/utils';
import { Brain, Percent, Trophy } from 'lucide-react';

interface PredictionResultProps {
  prediction: number | null;
  confidence: number | null;
  loading: boolean;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({ 
  prediction, 
  confidence, 
  loading 
}) => {
  if (loading) {
    return (
      <div className="w-full p-8 rounded-2xl glass flex flex-col items-center justify-center gap-4 animate-pulse">
        <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
        <p className="text-primary-400 font-medium tracking-wide">Analyzing Digit...</p>
      </div>
    );
  }

  if (prediction === null) {
    return (
      <div className="w-full p-8 rounded-2xl glass-light border-dashed border-white/10 flex flex-col items-center justify-center gap-3 text-white/30">
        <Brain className="w-8 h-8 opacity-20" />
        <p className="text-sm font-medium">Prediction will appear here</p>
      </div>
    );
  }

  return (
    <div className="w-full p-6 rounded-2xl glass border-primary-500/20 shadow-[0_0_50px_-12px_rgba(14,165,233,0.3)] animate-slide-up">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-primary-500/10 border border-primary-500/20">
          <Trophy className="w-4 h-4 text-primary-400" />
          <span className="text-xs font-bold text-primary-400 uppercase tracking-wider">Top Prediction</span>
        </div>
        <div className="flex items-center gap-1 text-white/40">
          <Percent className="w-3 h-3" />
          <span className="text-[10px] font-bold uppercase tracking-tighter">Confidence Score</span>
        </div>
      </div>

      <div className="flex items-end justify-between gap-6">
        <div className="flex flex-col">
          <span className="text-sm text-white/50 font-medium mb-1">Digit Identified</span>
          <span className="text-8xl font-black text-transparent bg-clip-text bg-gradient-to-b from-white to-white/40 leading-none">
            {prediction}
          </span>
        </div>

        <div className="flex flex-col items-end flex-1">
          <div className="relative w-full h-24 flex items-end justify-end mb-4">
             {/* Confidence ring mockup or just large text */}
             <div className="text-right">
                <span className="text-4xl font-bold text-primary-400">
                  {(confidence! * 100).toFixed(1)}
                </span>
                <span className="text-xl font-bold text-primary-400/60">%</span>
             </div>
          </div>
          
          <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-primary-600 to-primary-400 transition-all duration-1000 ease-out shadow-[0_0_15px_rgba(14,165,233,0.5)]"
              style={{ width: `${confidence! * 100}%` }}
            />
          </div>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-white/5">
        <p className="text-[10px] text-white/30 text-center uppercase tracking-[0.2em]">
          Powered by Neural Network Model
        </p>
      </div>
    </div>
  );
};

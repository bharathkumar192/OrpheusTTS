import React, { useRef, useEffect, useState, useCallback, useMemo, ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, Volume2, VolumeX, MoreVertical, Waves, BarChart3, Circle } from 'lucide-react';

interface StreamingAudioPlayerProps {
  src: ReadableStream<Uint8Array> | string;
  visualiser?: 'waves' | 'bars' | 'circles';
  height?: number;
  primaryColor?: string;
  accentColor?: string;
  onFftData?: (freqArray: Uint8Array) => void;
  onProgress?: (currentTime: number, duration: number) => void;
  onError?: (error: Error) => void;
}

interface AudioState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  isLoading: boolean;
  bufferedPercentage: number;
  error: string | null;
}

const useAudioStream = (src: ReadableStream<Uint8Array> | string) => {
  const [audioState, setAudioState] = useState<AudioState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    volume: 1,
    isMuted: false,
    isLoading: true,
    bufferedPercentage: 0,
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const audioBufferRef = useRef<AudioBuffer | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const startTimeRef = useRef<number>(0);
  const pauseTimeRef = useRef<number>(0);
  const animationFrameRef = useRef<number>(0);

  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      analyserRef.current.smoothingTimeConstant = 0.7;
      
      gainNodeRef.current = audioContextRef.current.createGain();
      gainNodeRef.current.connect(audioContextRef.current.destination);
      analyserRef.current.connect(gainNodeRef.current);
    }
    
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  }, []);

  const loadAudio = useCallback(async () => {
    try {
      setAudioState((prev: AudioState) => ({ ...prev, isLoading: true, error: null }));
      
      let audioData: ArrayBuffer;
      
      if (typeof src === 'string') {
        const response = await fetch(src, { mode: 'cors' });
        if (!response.ok) throw new Error('Failed to fetch audio');
        audioData = await response.arrayBuffer();
      } else {
        const reader = src.getReader();
        const chunks: Uint8Array[] = [];
        let totalLength = 0;
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          totalLength += value.length;
          
          setAudioState((prev: AudioState) => ({ 
            ...prev, 
            bufferedPercentage: Math.min((totalLength / 1000000) * 100, 95) 
          }));
        }
        
        const tempAudioData = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
          tempAudioData.set(chunk, offset);
          offset += chunk.length;
        }
        audioData = tempAudioData.buffer;
      }

      await initAudioContext();
      if (!audioContextRef.current) throw new Error('Audio context not available');
      
      audioBufferRef.current = await audioContextRef.current.decodeAudioData(audioData);
      
      setAudioState((prev: AudioState) => ({
        ...prev,
        isLoading: false,
        duration: audioBufferRef.current?.duration || 0,
        bufferedPercentage: 100,
      }));
    } catch (error) {
      setAudioState((prev: AudioState) => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load audio',
      }));
    }
  }, [src, initAudioContext]);

  const play = useCallback(async () => {
    if (!audioContextRef.current || !audioBufferRef.current) return;
    
    await initAudioContext();
    
    if (sourceNodeRef.current) {
      sourceNodeRef.current.stop();
    }
    
    sourceNodeRef.current = audioContextRef.current.createBufferSource();
    sourceNodeRef.current.buffer = audioBufferRef.current;
    sourceNodeRef.current.connect(analyserRef.current!);
    
    const currentTime = pauseTimeRef.current;
    sourceNodeRef.current.start(0, currentTime);
    startTimeRef.current = audioContextRef.current.currentTime - currentTime;
    
    setAudioState((prev: AudioState) => ({ ...prev, isPlaying: true }));
  }, [initAudioContext]);

  const pause = useCallback(() => {
    if (sourceNodeRef.current && audioContextRef.current) {
      sourceNodeRef.current.stop();
      pauseTimeRef.current = audioContextRef.current.currentTime - startTimeRef.current;
      setAudioState((prev: AudioState) => ({ ...prev, isPlaying: false }));
    }
  }, []);

  const seek = useCallback((time: number) => {
    pauseTimeRef.current = Math.max(0, Math.min(time, audioState.duration));
    if (audioState.isPlaying) {
      pause();
      setTimeout(play, 10);
    }
  }, [audioState.duration, audioState.isPlaying, pause, play]);

  const setVolume = useCallback((volume: number) => {
    if (gainNodeRef.current) {
      const clampedVolume = Math.max(0, Math.min(1, volume));
      gainNodeRef.current.gain.value = clampedVolume;
      setAudioState((prev: AudioState) => ({ ...prev, volume: clampedVolume }));
    }
  }, []);

  const toggleMute = useCallback(() => {
    if (gainNodeRef.current) {
      const newMutedState = !audioState.isMuted;
      gainNodeRef.current.gain.value = newMutedState ? 0 : audioState.volume;
      setAudioState((prev: AudioState) => ({ ...prev, isMuted: newMutedState }));
    }
  }, [audioState.isMuted, audioState.volume]);

  useEffect(() => {
    loadAudio();
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (sourceNodeRef.current) {
        sourceNodeRef.current.stop();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [loadAudio]);

  useEffect(() => {
    const updateTime = () => {
      if (audioState.isPlaying && audioContextRef.current) {
        const currentTime = audioContextRef.current.currentTime - startTimeRef.current;
        setAudioState((prev: AudioState) => ({ ...prev, currentTime }));
        
        if (currentTime >= audioState.duration) {
          setAudioState((prev: AudioState) => ({ ...prev, isPlaying: false, currentTime: 0 }));
          pauseTimeRef.current = 0;
        }
      }
      animationFrameRef.current = requestAnimationFrame(updateTime);
    };
    
    animationFrameRef.current = requestAnimationFrame(updateTime);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [audioState.isPlaying, audioState.duration]);

  return {
    audioState,
    play,
    pause,
    seek,
    setVolume,
    toggleMute,
    analyserNode: analyserRef.current,
  };
};

const useVisualiser = (
  analyserNode: AnalyserNode | null,
  canvasRef: React.RefObject<HTMLCanvasElement>,
  visualiser: 'waves' | 'bars' | 'circles',
  isPlaying: boolean,
  primaryColor: string,
  accentColor: string,
  onFftData?: (freqArray: Uint8Array) => void
) => {
  const animationFrameRef = useRef<number>(0);
  const waveDataRef = useRef<number[]>([]);
  const phaseRef = useRef<number>(0);

  useEffect(() => {
    if (!canvasRef.current || !analyserNode) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const freqData = new Uint8Array(analyserNode.frequencyBinCount);
    const timeData = new Uint8Array(analyserNode.frequencyBinCount);

    const animate = () => {
      analyserNode.getByteFrequencyData(freqData);
      analyserNode.getByteTimeDomainData(timeData);
      
      onFftData?.(freqData);

      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      // Create gradient
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, primaryColor);
      gradient.addColorStop(1, accentColor);

      if (visualiser === 'waves') {
        drawWaves(ctx, timeData, width, height, gradient, isPlaying);
      } else if (visualiser === 'bars') {
        drawBars(ctx, freqData, width, height, gradient, isPlaying);
      } else if (visualiser === 'circles') {
        drawCircles(ctx, freqData, width, height, gradient, isPlaying);
      }

      phaseRef.current += 0.05;
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [analyserNode, visualiser, isPlaying, primaryColor, accentColor, onFftData]);

  const drawWaves = (
    ctx: CanvasRenderingContext2D,
    timeData: Uint8Array,
    width: number,
    height: number,
    gradient: CanvasGradient,
    isPlaying: boolean
  ) => {
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Primary wave
    ctx.beginPath();
    ctx.globalAlpha = isPlaying ? 1 : 0.3;
    
    const sliceWidth = width / timeData.length;
    let x = 0;

    for (let i = 0; i < timeData.length; i++) {
      const v = (timeData[i] - 128) / 128;
      const y = (v * height * 0.3) + height / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.stroke();

    // Secondary wave with phase offset
    if (isPlaying) {
      ctx.beginPath();
      ctx.globalAlpha = 0.4;
      x = 0;

      for (let i = 0; i < timeData.length; i++) {
        const v = (timeData[i] - 128) / 128;
        const phase = Math.sin(phaseRef.current + i * 0.02) * 0.3;
        const y = (v * height * 0.2 + phase * 30) + height / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.stroke();
    }
  };

  const drawBars = (
    ctx: CanvasRenderingContext2D,
    freqData: Uint8Array,
    width: number,
    height: number,
    gradient: CanvasGradient,
    isPlaying: boolean
  ) => {
    const barCount = 64;
    const barWidth = width / barCount;
    const barSpacing = barWidth * 0.1;
    const actualBarWidth = barWidth - barSpacing;

    ctx.fillStyle = gradient;
    ctx.globalAlpha = isPlaying ? 1 : 0.3;

    for (let i = 0; i < barCount; i++) {
      const freqIndex = Math.floor((i / barCount) * freqData.length);
      const barHeight = (freqData[freqIndex] / 255) * height * 0.8;
      
      const x = i * barWidth + barSpacing / 2;
      const y = height - barHeight;

      // Add slight wave motion
      const wave = isPlaying ? Math.sin(phaseRef.current + i * 0.2) * 5 : 0;
      
      ctx.fillRect(x, y + wave, actualBarWidth, barHeight);
      
      // Add glow effect
      if (isPlaying && barHeight > height * 0.1) {
        ctx.globalAlpha = 0.2;
        ctx.fillRect(x - 2, y + wave - 2, actualBarWidth + 4, barHeight + 4);
        ctx.globalAlpha = isPlaying ? 1 : 0.3;
      }
    }
  };

  const drawCircles = (
    ctx: CanvasRenderingContext2D,
    freqData: Uint8Array,
    width: number,
    height: number,
    gradient: CanvasGradient,
    isPlaying: boolean
  ) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) / 2 - 20;

    // Calculate RMS for overall amplitude
    let rms = 0;
    for (let i = 0; i < freqData.length; i++) {
      rms += freqData[i] * freqData[i];
    }
    rms = Math.sqrt(rms / freqData.length) / 255;

    ctx.strokeStyle = gradient;
    ctx.globalAlpha = isPlaying ? 1 : 0.3;

    // Draw concentric circles
    const numCircles = 8;
    for (let i = 0; i < numCircles; i++) {
      const baseRadius = (maxRadius / numCircles) * (i + 1);
      const amplitude = rms * 30;
      const phase = phaseRef.current + i * 0.5;
      
      ctx.beginPath();
      ctx.lineWidth = 2 + (rms * 5);
      
      // Create wavy circle
      for (let angle = 0; angle <= Math.PI * 2; angle += 0.1) {
        const freqIndex = Math.floor((angle / (Math.PI * 2)) * 64);
        const freqValue = freqData[freqIndex] || 0;
        const radius = baseRadius + 
          (isPlaying ? Math.sin(phase + angle * 4) * amplitude * 0.5 : 0) +
          (freqValue / 255) * 20;
        
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        if (angle === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.closePath();
      ctx.stroke();
      
      // Add inner glow
      if (isPlaying && rms > 0.1) {
        ctx.globalAlpha = 0.1;
        ctx.lineWidth = 6 + (rms * 10);
        ctx.stroke();
        ctx.globalAlpha = isPlaying ? 1 : 0.3;
      }
    }
  };
};

export const StreamingAudioPlayer: React.FC<StreamingAudioPlayerProps> = ({
  src,
  visualiser = 'waves',
  height = 300,
  primaryColor = '#3b82f6',
  accentColor = '#8b5cf6',
  onFftData,
  onProgress,
  onError,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentVisualiser, setCurrentVisualiser] = useState(visualiser);
  const [isDragging, setIsDragging] = useState(false);

  const { audioState, play, pause, seek, setVolume, toggleMute, analyserNode } = useAudioStream(src);

  useVisualiser(analyserNode, canvasRef as React.RefObject<HTMLCanvasElement>, currentVisualiser, audioState.isPlaying, primaryColor, accentColor, onFftData);

  // Canvas setup with device pixel ratio
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const observer = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(dpr, dpr);
      }
      
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    });

    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      
      switch (e.key) {
        case ' ':
          e.preventDefault();
          audioState.isPlaying ? pause() : play();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          seek(audioState.currentTime - 5);
          break;
        case 'ArrowRight':
          e.preventDefault();
          seek(audioState.currentTime + 5);
          break;
        case 'm':
        case 'M':
          e.preventDefault();
          toggleMute();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [audioState.isPlaying, audioState.currentTime, play, pause, seek, toggleMute]);

  // Progress callback
  useEffect(() => {
    onProgress?.(audioState.currentTime, audioState.duration);
  }, [audioState.currentTime, audioState.duration, onProgress]);

  // Error callback
  useEffect(() => {
    if (audioState.error) {
      onError?.(new Error(audioState.error));
    }
  }, [audioState.error, onError]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (audioState.bufferedPercentage < 95) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * audioState.duration;
    seek(newTime);
  };

  const handleProgressDrag = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || audioState.bufferedPercentage < 95) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, clickX / rect.width));
    const newTime = percentage * audioState.duration;
    seek(newTime);
  };

  const progressPercentage = audioState.duration > 0 ? (audioState.currentTime / audioState.duration) * 100 : 0;
  const canSeek = audioState.bufferedPercentage >= 95;

  return (
    <div 
      className="streaming-audio-player"
      style={{
        '--player-primary': primaryColor,
        '--player-accent': accentColor,
        '--player-height': `${height}px`,
      } as React.CSSProperties}
    >
      <div className="player-container">
        {/* Error State */}
        <AnimatePresence>
          {audioState.error && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="error-banner"
            >
              {audioState.error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Player */}
        <div className="player-main">
          {/* Canvas Visualiser */}
          <div className="visualiser-container">
            <canvas
              ref={canvasRef}
              className="visualiser-canvas"
              aria-label="Audio visualization"
            />
            
            {/* Loading Overlay */}
            <AnimatePresence>
              {audioState.isLoading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="loading-overlay"
                >
                  <div className="loading-spinner" />
                  <p>Loading audio...</p>
                  {audioState.bufferedPercentage > 0 && (
                    <div className="loading-progress">
                      <div 
                        className="loading-progress-bar"
                        style={{ width: `${audioState.bufferedPercentage}%` }}
                      />
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Controls */}
          <div className="player-controls">
            {/* Play/Pause Button */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={audioState.isPlaying ? pause : play}
              disabled={audioState.isLoading}
              className="play-pause-btn"
              aria-label={audioState.isPlaying ? 'Pause' : 'Play'}
            >
              <motion.div
                initial={false}
                animate={{ rotate: audioState.isPlaying ? 0 : 0 }}
                transition={{ type: 'spring', stiffness: 200, damping: 20 }}
              >
                {audioState.isPlaying ? (
                  <Pause size={24} />
                ) : (
                  <Play size={24} />
                )}
              </motion.div>
            </motion.button>

            {/* Progress Section */}
            <div className="progress-section">
              <span className="time-display">
                {formatTime(audioState.currentTime)}
              </span>
              
              <div 
                className={`progress-container ${canSeek ? 'seekable' : 'disabled'}`}
                onClick={handleProgressClick}
                onMouseDown={() => canSeek && setIsDragging(true)}
                onMouseMove={handleProgressDrag}
                onMouseUp={() => setIsDragging(false)}
                onMouseLeave={() => setIsDragging(false)}
              >
                <div className="progress-track">
                  <div 
                    className="progress-buffer"
                    style={{ width: `${audioState.bufferedPercentage}%` }}
                  />
                  <div 
                    className="progress-fill"
                    style={{ width: `${progressPercentage}%` }}
                  />
                  <div 
                    className="progress-thumb"
                    style={{ left: `${progressPercentage}%` }}
                  />
                </div>
              </div>
              
              <span className="time-display">
                {formatTime(audioState.duration)}
              </span>
            </div>

            {/* Volume Control */}
            <div className="volume-section">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={toggleMute}
                className="volume-btn"
                aria-label={audioState.isMuted ? 'Unmute' : 'Mute'}
              >
                {audioState.isMuted ? (
                  <VolumeX size={20} />
                ) : (
                  <Volume2 size={20} />
                )}
              </motion.button>
              
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={audioState.isMuted ? 0 : audioState.volume}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setVolume(parseFloat(e.target.value))}
                className="volume-slider"
                aria-label="Volume"
              />
            </div>

            {/* Visualiser Selector */}
            <div className="visualiser-selector">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setCurrentVisualiser(currentVisualiser === 'waves' ? 'bars' : currentVisualiser === 'bars' ? 'circles' : 'waves')}
                className="visualiser-btn"
                aria-label="Switch visualizer"
              >
                {currentVisualiser === 'waves' && <Waves size={20} />}
                {currentVisualiser === 'bars' && <BarChart3 size={20} />}
                {currentVisualiser === 'circles' && <Circle size={20} />}
              </motion.button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
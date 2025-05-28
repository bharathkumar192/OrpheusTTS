// streaming.ts - Real-time Audio Visualization
import { animate } from 'motion';

export class AudioStreamer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private waveformData: number[] = [];
    private animationFrame: number | null = null;
    private isStreaming = false;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private streamingStartTime: number = 0;
    
    constructor() {
        this.canvas = document.getElementById('waveform-canvas') as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d')!;
        this.initializeCanvas();
        this.initializeAudioContext();
    }
    
    private initializeCanvas() {
        // Set canvas size for high DPI displays
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        // Set canvas style size
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        
        this.drawInitialWaveform();
    }
    
    private async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;
        } catch (error) {
            console.warn('AudioContext initialization failed:', error);
        }
    }
    
    private drawInitialWaveform() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // Draw placeholder waveform with darker theme
        this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        for (let x = 0; x < width; x += 4) {
            const y = height / 2 + Math.sin(x * 0.02) * 8;
            if (x === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
        
        // Add "Ready" text
        this.ctx.fillStyle = 'rgba(163, 163, 163, 0.6)';
        this.ctx.font = '14px -apple-system, BlinkMacSystemFont, system-ui, sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Ready to generate audio...', width / 2, height / 2 + 40);
    }
    
    updateWaveform(chunk: Uint8Array) {
        // Convert audio chunk to waveform data
        const samples = this.extractAudioSamples(chunk);
        
        if (samples.length > 0) {
            this.waveformData.push(...samples);
            
            // Keep only recent data to prevent memory issues
            if (this.waveformData.length > 2000) {
                this.waveformData = this.waveformData.slice(-2000);
            }
            
            // Start visualization if not already running
            if (!this.isStreaming) {
                this.streamingStartTime = performance.now();
                this.startWaveformAnimation();
            }
        }
    }
    
    private extractAudioSamples(chunk: Uint8Array): number[] {
        const samples: number[] = [];
        let startOffset = 0;
        
        // Check if this chunk contains a WAV header (first chunk typically does)
        if (chunk.length > 44 && 
            chunk[0] === 82 && chunk[1] === 73 && chunk[2] === 70 && chunk[3] === 70) { // "RIFF"
            startOffset = 44; // Skip WAV header
        }
        
        // Process as 16-bit PCM data
        for (let i = startOffset; i < chunk.length - 1; i += 2) {
            const sample = (chunk[i] | (chunk[i + 1] << 8));
            // Convert to signed 16-bit
            const signedSample = sample > 32767 ? sample - 65536 : sample;
            // Normalize to -1 to 1
            samples.push(signedSample / 32768);
        }
        
        // More aggressive downsampling for smoother visualization
        return samples.filter((_, index) => index % 8 === 0);
    }
    
    private startWaveformAnimation() {
        this.isStreaming = true;
        
        const animate = () => {
            if (!this.isStreaming) return;
            
            this.drawRealtimeWaveform();
            this.animationFrame = requestAnimationFrame(animate);
        };
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        
        this.animationFrame = requestAnimationFrame(animate);
    }
    
    stopWaveformAnimation() {
        this.isStreaming = false;
        
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        
        // Draw final static waveform
        setTimeout(() => {
            this.drawFinalWaveform();
        }, 100);
    }
    
    private drawRealtimeWaveform() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        this.ctx.clearRect(0, 0, width, height);
        
        if (this.waveformData.length === 0) {
            this.drawStreamingIndicator();
            return;
        }
        
        // Create gradient for the waveform
        const gradient = this.ctx.createLinearGradient(0, 0, width, 0);
        const elapsed = (performance.now() - this.streamingStartTime) / 1000;
        
        // Animate gradient colors during streaming
        const hue1 = (elapsed * 30) % 360;
        const hue2 = (elapsed * 30 + 60) % 360;
        gradient.addColorStop(0, `hsl(${hue1}, 70%, 60%)`);
        gradient.addColorStop(0.5, '#6366f1');
        gradient.addColorStop(1, `hsl(${hue2}, 70%, 60%)`);
        
        // Draw streaming waveform with flowing effect
        this.ctx.strokeStyle = gradient;
        this.ctx.lineWidth = 3;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.ctx.beginPath();
        
        const samplesPerPixel = Math.max(1, Math.floor(this.waveformData.length / width));
        const scrollOffset = elapsed * 50; // Scrolling effect
        
        for (let x = 0; x < width; x++) {
            const dataIndex = Math.floor(((x + scrollOffset) / width) * this.waveformData.length) % this.waveformData.length;
            
            if (dataIndex < this.waveformData.length) {
                // Calculate RMS with some smoothing
                let sum = 0;
                let count = 0;
                
                for (let i = 0; i < samplesPerPixel && dataIndex + i < this.waveformData.length; i++) {
                    sum += Math.abs(this.waveformData[dataIndex + i]);
                    count++;
                }
                
                const rms = count > 0 ? sum / count : 0;
                const amplitude = rms * height * 0.35;
                const y = height / 2 - amplitude;
                
                if (x === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
        }
        
        this.ctx.stroke();
        
        // Draw mirror for bottom half
        this.ctx.beginPath();
        for (let x = 0; x < width; x++) {
            const dataIndex = Math.floor(((x + scrollOffset) / width) * this.waveformData.length) % this.waveformData.length;
            
            if (dataIndex < this.waveformData.length) {
                let sum = 0;
                let count = 0;
                
                for (let i = 0; i < samplesPerPixel && dataIndex + i < this.waveformData.length; i++) {
                    sum += Math.abs(this.waveformData[dataIndex + i]);
                    count++;
                }
                
                const rms = count > 0 ? sum / count : 0;
                const amplitude = rms * height * 0.35;
                const y2 = height / 2 + amplitude;
                
                if (x === 0) {
                    this.ctx.moveTo(x, y2);
                } else {
                    this.ctx.lineTo(x, y2);
                }
            }
        }
        
        this.ctx.stroke();
        
        // Add glow effect
        this.ctx.shadowColor = '#6366f1';
        this.ctx.shadowBlur = 20;
        this.ctx.globalAlpha = 0.6;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
        this.ctx.globalAlpha = 1;
        
        // Add streaming indicator and status
        this.drawStreamingIndicator();
        this.drawStreamingStatus();
    }
    
    private drawStreamingIndicator() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        // Animated streaming line
        const time = Date.now() * 0.005;
        const x = (Math.sin(time) * 0.3 + 0.7) * width;
        
        this.ctx.strokeStyle = '#10b981';
        this.ctx.lineWidth = 4;
        this.ctx.setLineDash([8, 8]);
        this.ctx.lineDashOffset = time * 20;
        this.ctx.globalAlpha = 0.8;
        this.ctx.beginPath();
        this.ctx.moveTo(x, 20);
        this.ctx.lineTo(x, height - 20);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        this.ctx.globalAlpha = 1;
    }
    
    private drawStreamingStatus() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        // "LIVE" indicator
        this.ctx.fillStyle = '#ef4444';
        this.ctx.fillRect(width - 80, 20, 8, 8);
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, system-ui, sans-serif';
        this.ctx.textAlign = 'left';
        this.ctx.fillText('LIVE', width - 65, 28);
        
        // Streaming status
        const elapsed = (performance.now() - this.streamingStartTime) / 1000;
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        this.ctx.font = '11px -apple-system, BlinkMacSystemFont, system-ui, sans-serif';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`Streaming: ${elapsed.toFixed(1)}s`, 20, height - 25);
        this.ctx.fillText(`Samples: ${this.waveformData.length}`, 20, height - 10);
    }
    
    private drawFinalWaveform() {
        if (this.waveformData.length === 0) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // Create final gradient
        const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, '#8b5cf6');
        gradient.addColorStop(0.5, '#6366f1');
        gradient.addColorStop(1, '#3b82f6');
        
        this.ctx.strokeStyle = gradient;
        this.ctx.lineWidth = 2;
        this.ctx.lineCap = 'round';
        this.ctx.beginPath();
        
        const samplesPerPixel = Math.max(1, Math.floor(this.waveformData.length / width));
        
        for (let x = 0; x < width; x++) {
            const dataIndex = Math.floor((x / width) * this.waveformData.length);
            
            if (dataIndex < this.waveformData.length) {
                let sum = 0;
                let count = 0;
                
                for (let i = 0; i < samplesPerPixel && dataIndex + i < this.waveformData.length; i++) {
                    sum += Math.abs(this.waveformData[dataIndex + i]);
                    count++;
                }
                
                const rms = count > 0 ? sum / count : 0;
                const amplitude = rms * height * 0.4;
                const y1 = height / 2 - amplitude;
                const y2 = height / 2 + amplitude;
                
                this.ctx.moveTo(x, y1);
                this.ctx.lineTo(x, y2);
            }
        }
        
        this.ctx.stroke();
        
        // Add completion status
        this.ctx.fillStyle = 'rgba(16, 185, 129, 0.8)';
        this.ctx.font = '12px -apple-system, BlinkMacSystemFont, system-ui, sans-serif';
        this.ctx.textAlign = 'right';
        this.ctx.fillText('✓ Complete', width - 20, 30);
    }
    
    // Static waveform for completed audio
    drawStaticWaveform(audioBuffer: AudioBuffer) {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // Create gradient for the waveform
        const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, '#8b5cf6');
        gradient.addColorStop(0.5, '#6366f1');
        gradient.addColorStop(1, '#3b82f6');
        
        const channelData = audioBuffer.getChannelData(0);
        const samplesPerPixel = Math.floor(channelData.length / width);
        
        this.ctx.strokeStyle = gradient;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        for (let x = 0; x < width; x++) {
            const start = x * samplesPerPixel;
            const end = start + samplesPerPixel;
            
            let min = 0;
            let max = 0;
            
            for (let i = start; i < end && i < channelData.length; i++) {
                const sample = channelData[i];
                min = Math.min(min, sample);
                max = Math.max(max, sample);
            }
            
            const yMin = height / 2 + (min * height * 0.4);
            const yMax = height / 2 + (max * height * 0.4);
            
            this.ctx.moveTo(x, yMin);
            this.ctx.lineTo(x, yMax);
        }
        
        this.ctx.stroke();
        
        // Add glow effect
        this.ctx.shadowColor = '#6366f1';
        this.ctx.shadowBlur = 15;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
    }
    
    destroy() {
        this.stopWaveformAnimation();
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}

// Helper function to create stagger animation
function stagger(delay: number) {
    return (i: number) => i * delay;
}
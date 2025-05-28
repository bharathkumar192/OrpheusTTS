// streaming.ts - Real-time Audio Streaming Implementation
import { animate } from 'motion';

interface Analytics {
    ttft_s: number;
    ttfa_s: number;
    total_time_s: number;
    tokens_per_second: number;
    snac_tokens: number;
    audio_chunks: number;
}

export class AudioStreamer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private waveformData: number[] = [];
    private animationFrame: number | null = null;
    private isStreaming = false;
    private audioContext: AudioContext | null = null;
    private analyser: AnalyserNode | null = null;
    private startTime: number = 0;
    private firstChunkTime: number = 0;
    
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
        } catch (error) {
            console.warn('AudioContext initialization failed:', error);
        }
    }
    
    private drawInitialWaveform() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // Draw placeholder waveform
        this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        for (let x = 0; x < width; x += 4) {
            const y = height / 2 + Math.sin(x * 0.02) * 10;
            if (x === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
    }
    
    async streamGeneration(
        text: string, 
        speakerId: string, 
        onChunk: (chunk: Uint8Array) => void,
        onAnalytics: (analytics: Analytics) => void
    ): Promise<Blob> {
        this.startTime = performance.now();
        this.firstChunkTime = 0;
        this.isStreaming = true;
        
        // Show streaming UI
        this.showStreamingState();
        
        const API_BASE = 'https://fuchr--veena-veenattsapi-asgi-app.modal.run';
        
        try {
            const response = await fetch(`${API_BASE}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text,
                    speaker_id: speakerId,
                    streaming: true,
                    max_new_tokens: 1536,
                    temperature: 0.4,
                    repetition_penalty: 1.05,
                    top_p: 0.9
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const reader = response.body!.getReader();
            const chunks: Uint8Array[] = [];
            let totalBytes = 0;
            let chunkCount = 0;
            
            // Read the stream
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                if (value) {
                    // Record first chunk time for TTFA
                    if (this.firstChunkTime === 0) {
                        this.firstChunkTime = performance.now();
                    }
                    
                    chunks.push(value);
                    totalBytes += value.length;
                    chunkCount++;
                    
                    // Process chunk for real-time visualization
                    onChunk(value);
                    
                    // Update streaming visualization
                    this.updateStreamingVisualization(value);
                }
            }
            
            // Calculate analytics
            const endTime = performance.now();
            const totalTime = (endTime - this.startTime) / 1000;
            const ttfa = this.firstChunkTime ? (this.firstChunkTime - this.startTime) / 1000 : 0;
            
            const analytics: Analytics = {
                ttft_s: ttfa, // For audio, TTFT is essentially TTFA
                ttfa_s: ttfa,
                total_time_s: totalTime,
                tokens_per_second: chunkCount / totalTime,
                snac_tokens: chunkCount * 7, // Approximate
                audio_chunks: chunkCount
            };
            
            onAnalytics(analytics);
            
            // Combine all chunks into a single blob
            const audioBlob = new Blob(chunks, { type: 'audio/wav' });
            
            // Hide streaming state
            this.hideStreamingState();
            
            return audioBlob;
            
        } catch (error) {
            this.hideStreamingState();
            throw error;
        } finally {
            this.isStreaming = false;
        }
    }
    
    private showStreamingState() {
        const overlay = document.getElementById('waveform-overlay') as HTMLElement;
        const canvas = this.canvas;
        
        // Show overlay
        overlay.style.display = 'flex';
        animate(overlay, { opacity: [0, 1] }, { duration: 0.3 });
        
        // Start wave animation
        const dots = overlay.querySelectorAll('.dot');
        animate(dots, 
            { 
                scale: [1, 1.2, 1],
                opacity: [0.5, 1, 0.5]
            }, 
            { 
                duration: 1.5, 
                repeat: Infinity,
                delay: stagger(0.2)
            }
        );
        
        // Clear canvas for streaming
        const rect = this.canvas.getBoundingClientRect();
        this.ctx.clearRect(0, 0, rect.width, rect.height);
        
        // Start real-time waveform animation
        this.startWaveformAnimation();
    }
    
    private hideStreamingState() {
        const overlay = document.getElementById('waveform-overlay') as HTMLElement;
        
        animate(overlay, { opacity: [1, 0] }, { duration: 0.3 }).finished.then(() => {
            overlay.style.display = 'none';
        });
        
        this.stopWaveformAnimation();
    }
    
    private updateStreamingVisualization(chunk: Uint8Array) {
        // Convert audio chunk to waveform data
        const samples = this.extractAudioSamples(chunk);
        this.waveformData.push(...samples);
        
        // Keep only recent data to prevent memory issues
        if (this.waveformData.length > 1000) {
            this.waveformData = this.waveformData.slice(-1000);
        }
    }
    
    private extractAudioSamples(chunk: Uint8Array): number[] {
        // For WAV format, we need to skip the header and extract PCM data
        const samples: number[] = [];
        
        // Simple approach: treat as 16-bit PCM data
        for (let i = 44; i < chunk.length - 1; i += 2) {
            const sample = (chunk[i] | (chunk[i + 1] << 8));
            // Convert to signed 16-bit
            const signedSample = sample > 32767 ? sample - 65536 : sample;
            // Normalize to -1 to 1
            samples.push(signedSample / 32768);
        }
        
        return samples.filter((_, index) => index % 10 === 0); // Downsample for visualization
    }
    
    private startWaveformAnimation() {
        const animate = () => {
            if (!this.isStreaming) return;
            
            this.drawRealtimeWaveform();
            this.animationFrame = requestAnimationFrame(animate);
        };
        animate();
    }
    
    private stopWaveformAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }
    
    private drawRealtimeWaveform() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        this.ctx.clearRect(0, 0, width, height);
        
        if (this.waveformData.length === 0) return;
        
        // Draw streaming waveform
        this.ctx.strokeStyle = '#6366f1';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        const samplesPerPixel = Math.max(1, Math.floor(this.waveformData.length / width));
        
        for (let x = 0; x < width; x++) {
            const dataIndex = Math.floor((x / width) * this.waveformData.length);
            
            if (dataIndex < this.waveformData.length) {
                // Get RMS of samples for this pixel
                let sum = 0;
                let count = 0;
                
                for (let i = 0; i < samplesPerPixel && dataIndex + i < this.waveformData.length; i++) {
                    sum += Math.abs(this.waveformData[dataIndex + i]);
                    count++;
                }
                
                const rms = count > 0 ? sum / count : 0;
                const y = height / 2 - (rms * height * 0.4);
                const y2 = height / 2 + (rms * height * 0.4);
                
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
            const dataIndex = Math.floor((x / width) * this.waveformData.length);
            
            if (dataIndex < this.waveformData.length) {
                let sum = 0;
                let count = 0;
                
                for (let i = 0; i < samplesPerPixel && dataIndex + i < this.waveformData.length; i++) {
                    sum += Math.abs(this.waveformData[dataIndex + i]);
                    count++;
                }
                
                const rms = count > 0 ? sum / count : 0;
                const y2 = height / 2 + (rms * height * 0.4);
                
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
        this.ctx.shadowBlur = 10;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
        
        // Add streaming indicator
        this.drawStreamingIndicator();
    }
    
    private drawStreamingIndicator() {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        // Animated streaming line
        const time = Date.now() * 0.003;
        const x = (Math.sin(time) * 0.5 + 0.5) * width;
        
        this.ctx.strokeStyle = '#10b981';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(x, 0);
        this.ctx.lineTo(x, height);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }
    
    updateWaveform(chunk: Uint8Array) {
        // This method is called from the main app for each chunk
        // We handle the visualization in updateStreamingVisualization
        this.updateStreamingVisualization(chunk);
    }
    
    // Static waveform for completed audio
    drawStaticWaveform(audioBuffer: AudioBuffer) {
        const rect = this.canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        
        this.ctx.clearRect(0, 0, width, height);
        
        const channelData = audioBuffer.getChannelData(0);
        const samplesPerPixel = Math.floor(channelData.length / width);
        
        this.ctx.strokeStyle = '#6366f1';
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
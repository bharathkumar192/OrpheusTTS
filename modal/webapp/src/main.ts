// main.ts - Maya Research TTS Web Application
import { animate, timeline, stagger } from 'motion';
import { UIManager } from './ui';
// React imports for the streaming audio player
import React from 'react';
import ReactDOM from 'react-dom/client';
import { StreamingAudioPlayer } from './streamingAudioPlayer';

// API Configuration
const API_BASE = 'https://fuchr--veena-veenattsapi-asgi-app.modal.run';

// Types
interface Speaker {
    id: string;
    name: string;
    description: string;
    gender: 'male' | 'female';
    language: string;
    use_cases: string[];
    voice_characteristics: string[];
}

interface SystemStatus {
    ready: boolean;
    state: 'ready' | 'warming_up' | 'failed' | 'initializing';
    gpu: string;
    version: string;
    warmup_started: boolean;
    error?: string;
}

interface Analytics {
    ttft_s: number;
    ttfa_s: number;
    total_time_s: number;
    tokens_per_second: number;
    snac_tokens: number;
    audio_chunks: number;
}

// Application State
class AppState {
    private static instance: AppState;
    
    public speakers: Speaker[] = [];
    public selectedSpeaker: Speaker | null = null;
    public systemStatus: SystemStatus | null = null;
    public isGenerating = false;
    public analytics: Analytics | null = null;
    
    static getInstance(): AppState {
        if (!AppState.instance) {
            AppState.instance = new AppState();
        }
        return AppState.instance;
    }
    
    // State change callbacks
    private listeners: { [key: string]: Function[] } = {};
    
    on(event: string, callback: Function) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }
    
    emit(event: string, data?: any) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }
    
    updateSystemStatus(status: SystemStatus) {
        this.systemStatus = status;
        this.emit('systemStatusChanged', status);
    }
    
    updateAnalytics(analytics: Analytics) {
        this.analytics = analytics;
        this.emit('analyticsChanged', analytics);
    }
}

// API Service
class APIService {
    private baseUrl: string;
    
    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }
    
    async getSystemStatus(): Promise<SystemStatus> {
        try {
            const response = await fetch(`${this.baseUrl}/status`);
            if (!response.ok) throw new Error('Status check failed');
            return await response.json();
        } catch (error) {
            console.error('Failed to get system status:', error);
            return {
                ready: false,
                state: 'failed',
                gpu: 'Unknown',
                version: '1.0.0',
                warmup_started: false,
                error: 'Connection failed'
            };
        }
    }
    
    async getSpeakers(): Promise<Speaker[]> {
        try {
            const response = await fetch(`${this.baseUrl}/speakers`);
            if (!response.ok) throw new Error('Failed to fetch speakers');
            const data = await response.json();
            return data.speakers;
        } catch (error) {
            console.error('Failed to get speakers:', error);
            return [];
        }
    }
    
    async generateSpeech(text: string, speakerId: string): Promise<ReadableStream<Uint8Array>> {
        // Get advanced options values
        const maxTokens = parseInt((document.getElementById('max-tokens') as HTMLInputElement).value);
        const temperature = parseFloat((document.getElementById('temperature') as HTMLInputElement).value);
        const repetitionPenalty = parseFloat((document.getElementById('repetition-penalty') as HTMLInputElement).value);
        const topP = parseFloat((document.getElementById('top-p') as HTMLInputElement).value);
        
        console.log('🔧 Using advanced options:', {
            max_new_tokens: maxTokens,
            temperature,
            repetition_penalty: repetitionPenalty,
            top_p: topP
        });
        
        const response = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text,
                speaker_id: speakerId,
                streaming: true,
                max_new_tokens: maxTokens,
                temperature: temperature,
                repetition_penalty: repetitionPenalty,
                top_p: topP
            })
        });
        
        if (!response.ok) {
            throw new Error(`Generation failed: ${response.statusText}`);
        }
        
        return response.body!;
    }
}

// Main Application Class
class MayaApp {
    private state: AppState;
    private api: APIService;
    private ui: UIManager;
    private reactRoot: ReactDOM.Root | null = null;
    private statusCheckInterval: number | null = null;
    // PCM playback format (updated after parsing WAV header)
    private audioSampleRate: number = 24000;
    private audioNumChannels: number = 1;
    
    constructor() {
        this.state = AppState.getInstance();
        this.api = new APIService(API_BASE);
        this.ui = new UIManager();
        
        this.initializeApp();
    }
    
    private async initializeApp() {
        console.log('🎤 Initializing Maya TTS Application');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize UI
        this.ui.initialize();
        
        // Ensure React container exists
        this.ensureReactContainer();
        
        // Load speakers
        await this.loadSpeakers();
        
        // Check system status
        await this.checkSystemStatus();
        
        // Start status monitoring
        this.startStatusMonitoring();
        
        console.log('✅ Maya TTS Application Ready');
    }
    
    private ensureReactContainer() {
        const audioSectionEl = document.getElementById('audio-section');
        if (audioSectionEl && !document.getElementById('react-audio-player-root')) {
            // The HTML should already have the react-audio-player-root div
            console.warn('React container should already exist in HTML');
        }
    }
    
    private setupEventListeners() {
        // Text input events
        const textInput = document.getElementById('text-input') as HTMLTextAreaElement;
        const charCount = document.getElementById('char-count') as HTMLElement;
        const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
        const clearBtn = document.getElementById('clear-btn') as HTMLButtonElement;
        
        textInput.addEventListener('input', (e) => {
            const text = (e.target as HTMLTextAreaElement).value;
            const count = text.length;
            charCount.textContent = `${count} / 5000`;
            
            // Update generate button state
            const hasText = text.trim().length > 0;
            const isReady = this.state.systemStatus?.ready || false;
            
            if (hasText && isReady && !this.state.isGenerating) {
                generateBtn.classList.remove('disabled');
            } else {
                generateBtn.classList.add('disabled');
            }
            
            // Animate input border
            if (hasText) {
                textInput.classList.add('has-content');
            } else {
                textInput.classList.remove('has-content');
            }
        });
        
        clearBtn.addEventListener('click', () => {
            textInput.value = '';
            textInput.dispatchEvent(new Event('input'));
            animate(textInput, { scale: [1, 0.98, 1] }, { duration: 0.2 });
        });
        
        // Generate button
        generateBtn.addEventListener('click', () => {
            if (!generateBtn.classList.contains('disabled')) {
                this.generateSpeech();
            }
        });
        
        // Speaker dropdown
        this.setupSpeakerDropdown();
        
        // Advanced options
        this.setupAdvancedOptions();
        
        // State change listeners
        this.state.on('systemStatusChanged', (status: SystemStatus) => {
            this.updateStatusDisplay(status);
        });
        
        this.state.on('analyticsChanged', (analytics: Analytics) => {
            this.updateAnalyticsDisplay(analytics);
        });
    }
    
    private setupSpeakerDropdown() {
        const dropdown = document.getElementById('speaker-dropdown') as HTMLElement;
        const trigger = dropdown.querySelector('.dropdown-trigger') as HTMLElement;
        const menu = dropdown.querySelector('.dropdown-menu') as HTMLElement;
        
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = dropdown.classList.contains('open');
            
            if (isOpen) {
                dropdown.classList.remove('open');
                animate(menu, { opacity: 0, y: -10 }, { duration: 0.2 }).finished.then(() => {
                    menu.style.display = 'none';
                });
            } else {
                menu.style.display = 'block';
                dropdown.classList.add('open');
                animate(menu, { opacity: [0, 1], y: [-10, 0] }, { duration: 0.3 });
            }
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!dropdown.contains(e.target as Node)) {
                dropdown.classList.remove('open');
                animate(menu, { opacity: 0, y: -10 }, { duration: 0.2 }).finished.then(() => {
                    menu.style.display = 'none';
                });
            }
        });
    }
    
    private setupAdvancedOptions() {
        const advancedToggle = document.getElementById('advanced-toggle') as HTMLButtonElement;
        const advancedOptions = document.querySelector('.advanced-options') as HTMLElement;
        const advancedContent = document.getElementById('advanced-content') as HTMLElement;
        const resetBtn = document.getElementById('reset-advanced') as HTMLButtonElement;
        
        // Toggle expand/collapse
        advancedToggle.addEventListener('click', () => {
            const isExpanded = advancedOptions.classList.contains('expanded');
            
            if (isExpanded) {
                advancedOptions.classList.remove('expanded');
                advancedContent.classList.remove('expanded');
            } else {
                advancedOptions.classList.add('expanded');
                advancedContent.classList.add('expanded');
            }
            
            // Animate toggle
            animate(advancedToggle, { scale: [1, 0.96, 1] }, { duration: 0.2 });
        });
        
        // Setup slider value updates
        const sliders = [
            { id: 'max-tokens', valueId: 'max-tokens-value', formatter: (v: number) => v.toString() },
            { id: 'temperature', valueId: 'temperature-value', formatter: (v: number) => v.toFixed(2) },
            { id: 'repetition-penalty', valueId: 'repetition-penalty-value', formatter: (v: number) => v.toFixed(2) },
            { id: 'top-p', valueId: 'top-p-value', formatter: (v: number) => v.toFixed(2) }
        ];
        
        sliders.forEach(({ id, valueId, formatter }) => {
            const slider = document.getElementById(id) as HTMLInputElement;
            const valueElement = document.getElementById(valueId) as HTMLElement;
            
            slider.addEventListener('input', () => {
                const value = parseFloat(slider.value);
                valueElement.textContent = formatter(value);
                
                // Animate value update
                animate(valueElement, { 
                    scale: [1, 1.1, 1],
                    color: ['#a3a3a3', '#6366f1', '#a3a3a3']
                }, { duration: 0.3 });
            });
        });
        
        // Reset to defaults
        resetBtn.addEventListener('click', () => {
            const defaults = {
                'max-tokens': '1536',
                'temperature': '0.4',
                'repetition-penalty': '1.05',
                'top-p': '0.9'
            };
            
            Object.entries(defaults).forEach(([id, value]) => {
                const slider = document.getElementById(id) as HTMLInputElement;
                slider.value = value;
                slider.dispatchEvent(new Event('input'));
            });
            
            // Animate reset
            animate(resetBtn, { scale: [1, 0.9, 1] }, { duration: 0.2 });
        });
    }
    
    private async loadSpeakers() {
        try {
            const speakers = await this.api.getSpeakers();
            this.state.speakers = speakers;
            
            // Set default speaker
            this.state.selectedSpeaker = speakers.find(s => s.id === 'vinaya_assist') || speakers[0];
            
            // Populate dropdown
            this.populateSpeakerDropdown(speakers);
            
        } catch (error) {
            console.error('Failed to load speakers:', error);
        }
    }
    
    private populateSpeakerDropdown(speakers: Speaker[]) {
        const menu = document.querySelector('.dropdown-menu') as HTMLElement;
        menu.innerHTML = '';
        
        speakers.forEach(speaker => {
            const item = document.createElement('div');
            item.className = 'dropdown-item';
            item.innerHTML = `
                <div class="speaker-avatar ${speaker.gender}"></div>
                <div class="speaker-details">
                    <div class="speaker-name">${speaker.name}</div>
                    <div class="speaker-desc">${speaker.description}</div>
                    <div class="speaker-tags">
                        ${speaker.voice_characteristics.slice(0, 2).map(char => 
                            `<span class="tag">${char}</span>`
                        ).join('')}
                    </div>
                </div>
            `;
            
            item.addEventListener('click', () => {
                this.selectSpeaker(speaker);
                document.getElementById('speaker-dropdown')?.classList.remove('open');
                animate(menu, { opacity: 0, y: -10 }, { duration: 0.2 }).finished.then(() => {
                    menu.style.display = 'none';
                });
            });
            
            menu.appendChild(item);
        });
    }
    
    private selectSpeaker(speaker: Speaker) {
        this.state.selectedSpeaker = speaker;
        
        // Update dropdown trigger
        const trigger = document.querySelector('.dropdown-trigger .speaker-info') as HTMLElement;
        trigger.innerHTML = `
            <span class="speaker-name">${speaker.name}</span>
            <span class="speaker-desc">${speaker.description}</span>
        `;
        
        // Update avatar
        const avatar = document.querySelector('.dropdown-trigger .speaker-avatar') as HTMLElement;
        avatar.className = `speaker-avatar ${speaker.gender}`;
        
        // Animate selection
        animate(trigger, { scale: [1, 0.98, 1] }, { duration: 0.3 });
    }
    
    private async checkSystemStatus() {
        const status = await this.api.getSystemStatus();
        this.state.updateSystemStatus(status);
        
        // If not ready and not warming up, we might need to trigger warmup
        if (!status.ready && status.state === 'initializing') {
            console.log('System not ready, warmup may be needed');
        }
    }
    
    private startStatusMonitoring() {
        this.statusCheckInterval = window.setInterval(async () => {
            if (!this.state.systemStatus?.ready) {
                await this.checkSystemStatus();
            }
        }, 2000);
    }
    
    private updateStatusDisplay(status: SystemStatus) {
        const indicator = document.querySelector('.status-indicator') as HTMLElement;
        const statusText = document.querySelector('.status-text') as HTMLElement;
        const gpuStatus = document.getElementById('gpu-status') as HTMLElement;
        
        // Update GPU info
        gpuStatus.textContent = status.gpu;
        
        // Update status indicator
        indicator.className = 'status-indicator';
        
        switch (status.state) {
            case 'ready':
                indicator.classList.add('ready');
                statusText.textContent = 'System Ready';
                break;
            case 'warming_up':
                indicator.classList.add('warming');
                statusText.textContent = 'Warming Up...';
                break;
            case 'failed':
                indicator.classList.add('error');
                statusText.textContent = status.error || 'System Error';
                break;
            default:
                indicator.classList.add('loading');
                statusText.textContent = 'Initializing...';
        }
        
        // Enable/disable generate button based on system status
        const textInput = document.getElementById('text-input') as HTMLTextAreaElement;
        textInput.dispatchEvent(new Event('input'));
    }
    
    private updateAnalyticsDisplay(analytics: Analytics) {
        const ttftValue = document.getElementById('ttft-value') as HTMLElement;
        const tokensValue = document.getElementById('tokens-value') as HTMLElement;
        
        if (ttftValue) ttftValue.textContent = `${analytics.ttfa_s.toFixed(2)}s`;
        if (tokensValue) tokensValue.textContent = `${Math.round(analytics.tokens_per_second)}`;
        
        // Animate metrics update
        if (ttftValue && tokensValue) {
            animate([ttftValue, tokensValue], { 
                scale: [1, 1.1, 1],
                color: ['#ffffff', '#6366f1', '#ffffff']
            }, { duration: 0.5 });
        }
        
        console.log("Analytics updated:", analytics);
    }
    
    // New method to render the React player
    private renderReactPlayer(audioStream: ReadableStream<Uint8Array>, text: string) {
        const playerRootEl = document.getElementById('react-audio-player-root');
        if (!playerRootEl) {
            console.error("React player root element not found!");
            this.ui.showError("Audio player UI could not be initialized.");
            return;
        }

        // Clean up previous React instance to ensure fresh state
        if (this.reactRoot) {
            this.reactRoot.unmount();
            this.reactRoot = null;
        }

        // Create new React root for fresh state
        this.reactRoot = ReactDOM.createRoot(playerRootEl);

        // Generate unique key to force component re-mount
        const componentKey = `audio-player-${Date.now()}`;

        this.reactRoot.render(
            React.createElement(StreamingAudioPlayer, {
                key: componentKey, // Force re-mount with new key
                src: audioStream,
                visualiser: 'waves',
                height: 250,
                primaryColor: '#6366f1',
                accentColor: '#8b5cf6',
                onProgress: (currentTime, duration) => {
                    // Progress is handled by React component itself
                    console.log(`🎵 Playback progress: ${currentTime.toFixed(1)}s / ${duration.toFixed(1)}s`);
                },
                onError: (error) => {
                    console.error('Error from StreamingAudioPlayer:', error);
                    this.ui.showError(`Audio Player Error: ${error.message}`);
                },
                onFftData: (fftData) => {
                    // React component handles its own visualization
                    // Could feed data to external visualizer if needed
                }
            })
        );

        console.log(`🎛️ React player rendered with key: ${componentKey}`);
    }
    
    private async generateSpeech() {
        const textInput = document.getElementById('text-input') as HTMLTextAreaElement;
        const text = textInput.value.trim();
        
        if (!text || !this.state.selectedSpeaker || this.state.isGenerating) {
            return;
        }
        
        // Cleanup any previous audio playback
        this.cleanupPreviousAudio();
        
        this.state.isGenerating = true;
        const generationStartTime = performance.now();
        
        try {
            // Animate UI transition to side-by-side layout
            await this.ui.transitionToWorkspace();
            
            // Start generation
            const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
            generateBtn.classList.add('loading');
            
            console.log('🎵 Starting speech generation...', {
                text: text.substring(0, 50) + '...',
                speaker: this.state.selectedSpeaker.name,
                timestamp: new Date().toISOString()
            });
            
            // Get the audio stream
            const rawAudioStream = await this.api.generateSpeech(
                text, 
                this.state.selectedSpeaker.id
            );
            
            console.log('✅ Audio stream received, initializing real-time player...');
            
            // Create multiple tees for different purposes
            const [streamForRealTimePlayer, streamForReactPlayer, streamForAnalytics] = this.teeStream(rawAudioStream, 3);
            
            // Start real-time audio playback immediately
            this.startRealTimeAudioPlayback(streamForRealTimePlayer, generationStartTime);
            
            // Render the React player for UI (but audio comes from real-time player)
            this.renderReactPlayer(streamForReactPlayer, text);
            
            // Process analytics stream separately
            await this.processStreamForAnalyticsAndDownload(
                streamForAnalytics, 
                text, 
                generationStartTime
            );
            
        } catch (error) {
            console.error('Generation failed:', error);
            this.ui.showError('Generation failed. Please try again.');
        } finally {
            this.state.isGenerating = false;
            const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
            generateBtn.classList.remove('loading');
        }
    }
    
    // Cleanup previous audio instances
    private cleanupPreviousAudio() {
        console.log('🧹 Cleaning up previous audio instances...');
        
        // Stop any existing fallback audio
        const currentFallbackAudio = (window as any).currentFallbackAudio as HTMLAudioElement;
        if (currentFallbackAudio) {
            currentFallbackAudio.pause();
            currentFallbackAudio.currentTime = 0;
            if (currentFallbackAudio.src) {
                URL.revokeObjectURL(currentFallbackAudio.src);
            }
            (window as any).currentFallbackAudio = null;
            console.log('🧹 Previous fallback audio stopped');
        }
        
        // Stop any Web Audio API sources (they'll be cleaned up by garbage collector)
        // The individual sources can't be stopped once started, but the contexts will be closed
        
        console.log('🧹 Audio cleanup complete');
    }
    
    // Helper method to create multiple tees from a stream
    private teeStream(stream: ReadableStream<Uint8Array>, count: number): ReadableStream<Uint8Array>[] {
        let currentStream = stream;
        const streams: ReadableStream<Uint8Array>[] = [];
        
        for (let i = 0; i < count - 1; i++) {
            const [stream1, stream2] = currentStream.tee();
            streams.push(stream1);
            currentStream = stream2;
        }
        streams.push(currentStream);
        
        return streams;
    }
    
    /* ---------------------------------------
       WAV / PCM REAL-TIME STREAMING HELPERS
    -----------------------------------------*/

    /** Parse a 44-byte WAV header and extract sample-rate, channels, bits-per-sample. */
    private parseWavHeader(headerBytes: Uint8Array): boolean {
        if (headerBytes.length < 44) return false;

        const riff = String.fromCharCode(...headerBytes.slice(0, 4));
        const wave = String.fromCharCode(...headerBytes.slice(8, 12));
        if (riff !== 'RIFF' || wave !== 'WAVE') {
            console.warn('Invalid WAV header signatures', riff, wave);
            return false;
        }

        const dv = new DataView(headerBytes.buffer, headerBytes.byteOffset);
        this.audioNumChannels = dv.getUint16(22, true);
        this.audioSampleRate  = dv.getUint32(24, true);
        const bitsPerSample   = dv.getUint16(34, true);

        console.log(`🔊 WAV header → SR=${this.audioSampleRate}, CH=${this.audioNumChannels}, Bits=${bitsPerSample}`);
        if (bitsPerSample !== 16) console.warn('Only 16-bit PCM is fully supported for real-time playback.');
        return this.audioSampleRate > 0 && this.audioNumChannels > 0;
    }

    /** Convert 16-bit signed little-endian PCM bytes → Float32Array [-1,1]. */
    private pcmBytesToFloat32(pcmBytes: Uint8Array): Float32Array {
        const samples = pcmBytes.length / 2;
        const floats = new Float32Array(samples);
        const dv = new DataView(pcmBytes.buffer, pcmBytes.byteOffset);
        for (let i = 0; i < samples; i++) {
            floats[i] = dv.getInt16(i * 2, true) / 32768;
        }
        return floats;
    }

    private concatFloat32(a: Float32Array, b: Float32Array): Float32Array {
        const out = new Float32Array(a.length + b.length);
        out.set(a, 0);
        out.set(b, a.length);
        return out;
    }

    /** Schedule a chunk of PCM Float32 samples for playback and return new nextStart time */
    private playPcm(audioCtx: AudioContext, pcm: Float32Array, nextStart: number): number {
        if (pcm.length === 0) return nextStart;
        const frameCount = Math.floor(pcm.length / this.audioNumChannels);
        const buffer = audioCtx.createBuffer(this.audioNumChannels, frameCount, this.audioSampleRate);

        if (this.audioNumChannels === 1) {
            buffer.getChannelData(0).set(pcm);
        } else if (this.audioNumChannels === 2) {
            const left  = new Float32Array(frameCount);
            const right = new Float32Array(frameCount);
            for (let i = 0; i < frameCount; i++) {
                left[i]  = pcm[i*2];
                right[i] = pcm[i*2+1];
            }
            buffer.getChannelData(0).set(left);
            buffer.getChannelData(1).set(right);
        } else {
            // Fallback: interleave channels into mono
            buffer.getChannelData(0).set(pcm.filter((_, idx)=> idx % this.audioNumChannels === 0));
        }

        const src = audioCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(audioCtx.destination);
        const startAt = Math.max(audioCtx.currentTime, nextStart);
        src.start(startAt);
        return startAt + buffer.duration;
    }

    /**
     * Real-time WAV streaming: parse header once, then treat all following bytes as 16-bit PCM and
     * schedule them for playback in ~100 ms slices.
     */
    private async startRealTimeAudioPlayback(audioStream: ReadableStream<Uint8Array>, generationStartTime: number) {
        const reader = audioStream.getReader();
        let audioCtx: AudioContext | null = null;
        let nextPlay = 0;
        let headerParsed = false;
        let pcmBuffer = new Float32Array(0);
        const allChunks: Uint8Array[] = [];

        const PCM_SLICE_SEC = 0.12; // ~120 ms slices

        try {
            audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
            if (audioCtx.state === 'suspended') await audioCtx.resume();
            nextPlay = audioCtx.currentTime;
            console.log('🔊 AudioContext ready, currentTime =', nextPlay.toFixed(3));

            let chunkIndex = 0;
            while (true) {
                const { done, value } = await reader.read();
                if (value) {
                    allChunks.push(value);
                    chunkIndex++;

                    // First chunk expected to contain WAV header
                    if (!headerParsed) {
                        if (value.length >= 44 && this.parseWavHeader(value.slice(0, 44))) {
                            headerParsed = true;
                            const afterHeader = value.slice(44);
                            if (afterHeader.length) {
                                pcmBuffer = this.concatFloat32(pcmBuffer, this.pcmBytesToFloat32(afterHeader));
                            }
                            const ttfa = (performance.now() - generationStartTime) / 1000;
                            console.log(`⚡ TTFA (header parsed) = ${ttfa.toFixed(3)} s`);
                        } else {
                            console.warn('Waiting for full WAV header...');
                        }
                    } else {
                        // Already parsed header ⇒ treat as PCM
                        pcmBuffer = this.concatFloat32(pcmBuffer, this.pcmBytesToFloat32(value));
                    }

                    // Play whenever we have enough PCM samples buffered
                    const sliceSamples = Math.floor(this.audioSampleRate * PCM_SLICE_SEC) * this.audioNumChannels;
                    if (headerParsed && pcmBuffer.length >= sliceSamples) {
                        const slice = pcmBuffer.slice(0, sliceSamples);
                        pcmBuffer = pcmBuffer.slice(sliceSamples);
                        nextPlay = this.playPcm(audioCtx, slice, nextPlay);
                    }
                }

                if (done) {
                    console.log('🔚 Stream ended. Scheduling remaining PCM...');
                    if (headerParsed && pcmBuffer.length > 0) {
                        nextPlay = this.playPcm(audioCtx, pcmBuffer, nextPlay);
                    }
                    break;
                }
            }
        } catch (err) {
            console.error('Real-time PCM playback failed:', err);
            if (allChunks.length) this.setupFallbackAudioPlayback(allChunks);
        } finally {
            // Close context after last chunk played
            if (audioCtx) {
                const wait = Math.max(0, (nextPlay - audioCtx.currentTime) * 1000 + 500);
                setTimeout(()=>audioCtx?.close(), wait);
            }
        }
    }
    
    // Fallback HTML5 audio playback
    private setupFallbackAudioPlayback(audioChunks: Uint8Array[]) {
        try {
            // Combine all chunks
            const totalLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
            const combinedBuffer = new Uint8Array(totalLength);
            
            let offset = 0;
            for (const chunk of audioChunks) {
                combinedBuffer.set(chunk, offset);
                offset += chunk.length;
            }
            
            // Create blob and audio element
            const audioBlob = new Blob([combinedBuffer], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Create and configure audio element
            const audioElement = new Audio();
            audioElement.src = audioUrl;
            audioElement.controls = false;
            audioElement.autoplay = true;
            
            // Event listeners
            audioElement.addEventListener('loadeddata', () => {
                console.log(`🎵 Fallback audio loaded: ${audioElement.duration.toFixed(2)}s duration`);
            });
            
            audioElement.addEventListener('canplay', () => {
                console.log('🎵 Fallback audio ready to play');
                audioElement.play().catch(e => console.warn('Autoplay blocked:', e));
            });
            
            audioElement.addEventListener('error', (e) => {
                console.error('🚫 Fallback audio error:', e);
                URL.revokeObjectURL(audioUrl);
            });
            
            audioElement.addEventListener('ended', () => {
                console.log('🎵 Fallback audio playback finished');
                URL.revokeObjectURL(audioUrl);
            });
            
            // Store reference for cleanup
            (window as any).currentFallbackAudio = audioElement;
            
            console.log('🔄 HTML5 fallback audio initialized');
            
        } catch (error) {
            console.error('Fallback audio setup failed:', error);
        }
    }
    
    // Kept for analytics and download functionality
    private async processStreamForAnalyticsAndDownload(
        audioStream: ReadableStream<Uint8Array>, 
        text: string, 
        generationStartTime: number
    ) {
        const chunks: Uint8Array[] = [];
        const reader = audioStream.getReader();
        let chunkCount = 0;
        let isFirstChunk = true;
        let firstChunkTime = 0;
        let totalBytes = 0;

        try {
            console.log('📊 Starting analytics processing...');
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                if (value) {
                    if (isFirstChunk) {
                        isFirstChunk = false;
                        firstChunkTime = performance.now();
                        console.log(`⚡ First chunk received: ${(firstChunkTime - generationStartTime).toFixed(2)}ms`);
                    }
                    chunks.push(value);
                    chunkCount++;
                    totalBytes += value.length;
                }
            }

            const endTime = performance.now();
            const totalStreamReadTimeS = (endTime - generationStartTime) / 1000;
            const ttfaS = firstChunkTime ? (firstChunkTime - generationStartTime) / 1000 : 0;

            const analyticsData: Analytics = {
                ttft_s: ttfaS,
                ttfa_s: ttfaS,
                total_time_s: totalStreamReadTimeS,
                tokens_per_second: chunkCount > 0 && totalStreamReadTimeS > 0 ? chunkCount / totalStreamReadTimeS : 0,
                snac_tokens: chunkCount * 7,
                audio_chunks: chunkCount
            };
            
            console.log('📈 Analytics calculated:', analyticsData);
            this.state.updateAnalytics(analyticsData);

            const audioBlob = new Blob(chunks, { type: 'audio/wav' });
            this.setupAudioDownload(audioBlob);
            
            console.log(`🎵 Audio processing complete: ${totalBytes} bytes, ${chunkCount} chunks`);

        } catch (error) {
            console.error('Stream processing for analytics/download failed:', error);
            this.ui.showError('Audio data processing for download failed.');
        }
    }
    
    private setupAudioDownload(audioBlob: Blob) {
        const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;
        if (!downloadBtn) {
            console.warn("#download-btn not found. Download will not be available.");
            return;
        }
        
        downloadBtn.onclick = () => {
            const audioUrl = URL.createObjectURL(audioBlob);
            const link = document.createElement('a');
            link.href = audioUrl;
            link.download = `maya-tts-${Date.now()}.wav`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(audioUrl);
            console.log('💾 Audio download initiated');
        };
    }
    
    private formatTime(seconds: number): string {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    // Cleanup
    destroy() {
        console.log('🧹 Destroying Maya TTS Application...');
        
        if (this.statusCheckInterval) {
            window.clearInterval(this.statusCheckInterval);
        }
        
        // Cleanup audio resources
        this.cleanupPreviousAudio();
        
        // Cleanup React components
        if (this.reactRoot) {
            this.reactRoot.unmount();
        }
        
        console.log('🧹 Maya TTS Application destroyed');
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new MayaApp();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    // Cleanup resources
});

export default MayaApp;
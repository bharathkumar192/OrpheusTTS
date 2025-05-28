// main.ts - Veena TTS Professional Web Application
import { animate, timeline, stagger } from 'motion';
import { AudioStreamer } from './streaming';
import { UIManager } from './ui';

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
    public currentAudio: HTMLAudioElement | null = null;
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
        const response = await fetch(`${this.baseUrl}/generate`, {
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
            throw new Error(`Generation failed: ${response.statusText}`);
        }
        
        return response.body!;
    }
}

// Main Application Class
class VeenaApp {
    private state: AppState;
    private api: APIService;
    private ui: UIManager;
    private audioStreamer: AudioStreamer;
    private statusCheckInterval: NodeJS.Timeout | null = null;
    
    constructor() {
        this.state = AppState.getInstance();
        this.api = new APIService(API_BASE);
        this.ui = new UIManager();
        this.audioStreamer = new AudioStreamer();
        
        this.initializeApp();
    }
    
    private async initializeApp() {
        console.log('🎤 Initializing Veena TTS Application');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize UI
        this.ui.initialize();
        
        // Load speakers
        await this.loadSpeakers();
        
        // Check system status
        await this.checkSystemStatus();
        
        // Start status monitoring
        this.startStatusMonitoring();
        
        console.log('✅ Veena TTS Application Ready');
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
        
        // Quick templates
        document.querySelectorAll('.template-pill').forEach(pill => {
            pill.addEventListener('click', (e) => {
                const text = (e.target as HTMLElement).dataset.text || '';
                textInput.value = text;
                textInput.dispatchEvent(new Event('input'));
                
                // Animate template selection
                animate(pill, { scale: [1, 0.95, 1] }, { duration: 0.2 });
                animate(textInput, { 
                    boxShadow: ['0 0 0 0 rgba(99, 102, 241, 0.4)', '0 0 0 4px rgba(99, 102, 241, 0.4)', '0 0 0 0 rgba(99, 102, 241, 0)']
                }, { duration: 0.6 });
            });
        });
        
        // Speaker dropdown
        this.setupSpeakerDropdown();
        
        // Audio controls
        this.setupAudioControls();
        
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
        
        trigger.addEventListener('click', () => {
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
    
    private setupAudioControls() {
        const playBtn = document.getElementById('play-btn') as HTMLButtonElement;
        const progressBar = document.getElementById('progress-bar') as HTMLElement;
        const volumeSlider = document.getElementById('volume-slider') as HTMLInputElement;
        const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;
        
        playBtn.addEventListener('click', () => {
            if (this.state.currentAudio) {
                if (this.state.currentAudio.paused) {
                    this.state.currentAudio.play();
                    playBtn.classList.add('playing');
                } else {
                    this.state.currentAudio.pause();
                    playBtn.classList.remove('playing');
                }
            }
        });
        
        volumeSlider.addEventListener('input', (e) => {
            const volume = parseInt((e.target as HTMLInputElement).value) / 100;
            if (this.state.currentAudio) {
                this.state.currentAudio.volume = volume;
            }
        });
        
        downloadBtn.addEventListener('click', () => {
            if (this.state.currentAudio) {
                const link = document.createElement('a');
                link.href = this.state.currentAudio.src;
                link.download = `veena-tts-${Date.now()}.wav`;
                link.click();
            }
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
        this.statusCheckInterval = setInterval(async () => {
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
        
        ttftValue.textContent = `${analytics.ttft_s}s`;
        tokensValue.textContent = `${analytics.tokens_per_second}`;
        
        // Animate metrics update
        animate([ttftValue, tokensValue], { 
            scale: [1, 1.1, 1],
            color: ['#ffffff', '#6366f1', '#ffffff']
        }, { duration: 0.5 });
    }
    
    private async generateSpeech() {
        const textInput = document.getElementById('text-input') as HTMLTextAreaElement;
        const text = textInput.value.trim();
        
        if (!text || !this.state.selectedSpeaker || this.state.isGenerating) {
            return;
        }
        
        this.state.isGenerating = true;
        
        try {
            // Animate UI transition to side-by-side layout
            await this.ui.transitionToWorkspace();
            
            // Start generation
            const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
            generateBtn.classList.add('loading');
            
            // Stream audio
            const audioBlob = await this.audioStreamer.streamGeneration(
                text, 
                this.state.selectedSpeaker.id,
                (chunk) => this.handleAudioChunk(chunk),
                (analytics) => this.state.updateAnalytics(analytics)
            );
            
            // Setup audio playback
            this.setupAudioPlayback(audioBlob);
            
        } catch (error) {
            console.error('Generation failed:', error);
            this.ui.showError('Generation failed. Please try again.');
        } finally {
            this.state.isGenerating = false;
            const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
            generateBtn.classList.remove('loading');
        }
    }
    
    private handleAudioChunk(chunk: Uint8Array) {
        // Update waveform visualization
        this.audioStreamer.updateWaveform(chunk);
    }
    
    private setupAudioPlayback(audioBlob: Blob) {
        const audioPlayer = document.getElementById('audio-player') as HTMLAudioElement;
        const audioUrl = URL.createObjectURL(audioBlob);
        
        audioPlayer.src = audioUrl;
        this.state.currentAudio = audioPlayer;
        
        // Setup audio event listeners
        audioPlayer.addEventListener('loadedmetadata', () => {
            const totalTime = document.getElementById('total-time') as HTMLElement;
            totalTime.textContent = this.formatTime(audioPlayer.duration);
        });
        
        audioPlayer.addEventListener('timeupdate', () => {
            const currentTime = document.getElementById('current-time') as HTMLElement;
            const progressFill = document.querySelector('.progress-fill') as HTMLElement;
            
            currentTime.textContent = this.formatTime(audioPlayer.currentTime);
            const progress = (audioPlayer.currentTime / audioPlayer.duration) * 100;
            progressFill.style.width = `${progress}%`;
        });
        
        audioPlayer.addEventListener('ended', () => {
            const playBtn = document.getElementById('play-btn') as HTMLButtonElement;
            playBtn.classList.remove('playing');
        });
        
        // Auto-play the generated audio
        audioPlayer.play().then(() => {
            const playBtn = document.getElementById('play-btn') as HTMLButtonElement;
            playBtn.classList.add('playing');
        });
    }
    
    private formatTime(seconds: number): string {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    // Cleanup
    destroy() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }
        this.audioStreamer.destroy();
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new VeenaApp();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    // Cleanup resources
});

export default VeenaApp;
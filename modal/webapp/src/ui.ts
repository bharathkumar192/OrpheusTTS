// ui.ts - UI Manager and Smooth Animations
import { animate, timeline, stagger } from 'motion';

export class UIManager {
    private isWorkspaceMode = false;
    private heroSection: HTMLElement;
    private audioSection: HTMLElement;
    private appContainer: HTMLElement;
    
    constructor() {
        this.heroSection = document.getElementById('hero-section') as HTMLElement;
        this.audioSection = document.getElementById('audio-section') as HTMLElement;
        this.appContainer = document.getElementById('app-container') as HTMLElement;
    }
    
    initialize() {
        // Set initial state
        this.setupInitialAnimations();
        this.setupParticleBackground();
        this.initializeResponsive();
    }
    
    private setupInitialAnimations() {
        // Animate hero section elements on load
        const heroTitle = document.querySelector('.hero-title') as HTMLElement;
        const inputContainer = document.querySelector('.input-container') as HTMLElement;
        const speakerContainer = document.querySelector('.speaker-container') as HTMLElement;
        const generateBtn = document.getElementById('generate-btn') as HTMLElement;
        
        // Staggered entrance animation
        timeline([
            [heroTitle, { opacity: [0, 1], y: [-20, 0] }, { duration: 0.6 }],
            [speakerContainer, { opacity: [0, 1], y: [15, 0] }, { duration: 0.5, at: 0.3 }],
            [inputContainer, { opacity: [0, 1], y: [20, 0], scale: [0.95, 1] }, { duration: 0.6, at: 0.5 }],
            [generateBtn, { opacity: [0, 1], scale: [0.9, 1] }, { duration: 0.4, at: 0.7 }]
        ]);
        
        // Floating animation for generate button
        this.startFloatingAnimation(generateBtn);
    }
    
    private startFloatingAnimation(element: HTMLElement) {
        const float = () => {
            animate(element, 
                { y: [0, -2, 0] }, 
                { 
                    duration: 3,
                    repeat: Infinity,
                    easing: 'ease-in-out'
                }
            );
        };
        float();
    }
    
    private setupParticleBackground() {
        // Create animated background particles
        const particleContainer = document.createElement('div');
        particleContainer.className = 'particle-background';
        document.body.appendChild(particleContainer);
        
        // Create floating particles
        for (let i = 0; i < 12; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 10 + 's';
            particle.style.animationDuration = (Math.random() * 20 + 10) + 's';
            particleContainer.appendChild(particle);
        }
    }
    
    async transitionToWorkspace(): Promise<void> {
        if (this.isWorkspaceMode) return;
        
        this.isWorkspaceMode = true;
        
        // Add workspace class to container
        this.appContainer.classList.add('workspace-mode');
        
        // Show audio section first (hidden)
        this.audioSection.classList.remove('hidden');
        this.audioSection.style.opacity = '0';
        this.audioSection.style.transform = 'translateX(100px)';
        
        // Create the transition timeline
        await timeline([
            // Phase 1: Move hero section to the left
            [
                this.heroSection,
                { 
                    x: [0, -50],
                    scale: [1, 0.95],
                    opacity: [1, 0.9]
                },
                { duration: 0.6, easing: 'ease-in-out' }
            ],
            
            // Phase 2: Slide in audio section from right
            [
                this.audioSection,
                {
                    x: [100, 0],
                    opacity: [0, 1],
                    scale: [0.95, 1]
                },
                { duration: 0.6, at: 0.3 }
            ],
            
            // Phase 3: Adjust hero section to final position
            [
                this.heroSection,
                {
                    x: [-50, 0],
                    scale: [0.95, 1],
                    opacity: [0.9, 1]
                },
                { duration: 0.5, at: 0.8 }
            ]
        ]).finished;
        
        // Animate individual elements in the audio section
        this.animateAudioSectionElements();
    }
    
    private animateAudioSectionElements() {
        const audioHeader = document.querySelector('.audio-header') as HTMLElement;
        const waveformContainer = document.querySelector('.waveform-container') as HTMLElement;
        const audioControls = document.querySelector('.audio-controls') as HTMLElement;
        const analyticsPanel = document.querySelector('.analytics-panel') as HTMLElement;
        
        // Staggered animation for audio section elements
        timeline([
            [audioHeader, { opacity: [0, 1], y: [-10, 0] }, { duration: 0.3 }],
            [waveformContainer, { opacity: [0, 1], scale: [0.95, 1] }, { duration: 0.4, at: 0.2 }],
            [audioControls, { opacity: [0, 1], y: [10, 0] }, { duration: 0.3, at: 0.4 }],
            [analyticsPanel, { opacity: [0, 1], y: [10, 0] }, { duration: 0.3, at: 0.5 }]
        ]);
    }
    
    async transitionBackToHero(): Promise<void> {
        if (!this.isWorkspaceMode) return;
        
        this.isWorkspaceMode = false;
        
        // Reverse the transition
        await timeline([
            // Fade out audio section
            [
                this.audioSection,
                {
                    x: [0, 100],
                    opacity: [1, 0],
                    scale: [1, 0.95]
                },
                { duration: 0.5 }
            ],
            
            // Move hero back to center
            [
                this.heroSection,
                {
                    x: [0, 0],
                    scale: [1, 1],
                    opacity: [1, 1]
                },
                { duration: 0.6, at: 0.2 }
            ]
        ]).finished;
        
        // Hide audio section and remove workspace mode
        this.audioSection.classList.add('hidden');
        this.appContainer.classList.remove('workspace-mode');
    }
    
    showError(message: string) {
        // Create error notification
        const errorEl = document.createElement('div');
        errorEl.className = 'error-notification';
        errorEl.innerHTML = `
            <div class="error-content">
                <svg class="error-icon" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                <span>${message}</span>
            </div>
            <button class="error-close">×</button>
        `;
        
        document.body.appendChild(errorEl);
        
        // Animate in
        animate(errorEl, { 
            opacity: [0, 1], 
            y: [-50, 0],
            scale: [0.9, 1]
        }, { duration: 0.4 });
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            this.hideError(errorEl);
        }, 5000);
        
        // Manual close button
        errorEl.querySelector('.error-close')?.addEventListener('click', () => {
            this.hideError(errorEl);
        });
    }
    
    private hideError(errorEl: HTMLElement) {
        animate(errorEl, { 
            opacity: [1, 0], 
            y: [0, -30],
            scale: [1, 0.9]
        }, { duration: 0.3 }).finished.then(() => {
            errorEl.remove();
        });
    }
    
    showSuccess(message: string) {
        // Create success notification
        const successEl = document.createElement('div');
        successEl.className = 'success-notification';
        successEl.innerHTML = `
            <div class="success-content">
                <svg class="success-icon" viewBox="0 0 24 24">
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                </svg>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(successEl);
        
        // Animate in
        animate(successEl, { 
            opacity: [0, 1], 
            y: [50, 0],
            scale: [0.9, 1]
        }, { duration: 0.4 });
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            animate(successEl, { 
                opacity: [1, 0], 
                y: [0, 30],
                scale: [1, 0.9]
            }, { duration: 0.3 }).finished.then(() => {
                successEl.remove();
            });
        }, 3000);
    }
    
    pulseElement(element: HTMLElement, color = '#6366f1') {
        animate(element, {
            boxShadow: [
                `0 0 0 0 ${color}40`,
                `0 0 0 10px ${color}00`,
                `0 0 0 0 ${color}00`
            ]
        }, { duration: 0.6 });
    }
    
    // Loading states
    showButtonLoading(button: HTMLElement) {
        button.classList.add('loading');
        const btnContent = button.querySelector('.btn-content') as HTMLElement;
        const btnLoading = button.querySelector('.btn-loading') as HTMLElement;
        
        animate(btnContent, { opacity: [1, 0] }, { duration: 0.2 });
        animate(btnLoading, { opacity: [0, 1] }, { duration: 0.2, at: 0.2 });
    }
    
    hideButtonLoading(button: HTMLElement) {
        button.classList.remove('loading');
        const btnContent = button.querySelector('.btn-content') as HTMLElement;
        const btnLoading = button.querySelector('.btn-loading') as HTMLElement;
        
        animate(btnLoading, { opacity: [1, 0] }, { duration: 0.2 });
        animate(btnContent, { opacity: [0, 1] }, { duration: 0.2, at: 0.2 });
    }
    
    // Theme transitions
    async switchTheme(isDark: boolean) {
        const root = document.documentElement;
        
        if (isDark) {
            root.classList.add('dark-theme');
        } else {
            root.classList.remove('dark-theme');
        }
        
        // Animate theme transition
        animate(document.body, {
            opacity: [1, 0.8, 1]
        }, { duration: 0.4 });
    }
    
    // Responsive adjustments
    handleResize() {
        const isMobile = window.innerWidth < 768;
        
        if (isMobile && this.isWorkspaceMode) {
            // Stack vertically on mobile
            this.appContainer.classList.add('mobile-stack');
        } else {
            this.appContainer.classList.remove('mobile-stack');
        }
    }
    
    // Initialize resize handler
    initializeResponsive() {
        window.addEventListener('resize', () => this.handleResize());
        this.handleResize(); // Initial call
    }
}
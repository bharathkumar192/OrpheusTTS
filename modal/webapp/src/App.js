import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import WaveSurfer from 'wavesurfer.js';

const SAMPLE_HINDI_SENTENCES = [
  { text: "नमस्ते, आपका दिन कैसा रहा?", description: "Hello, how was your day?" },
  { text: "भारत एक विविधतापूर्ण देश है", description: "India is a diverse country" },
  { text: "मैं हिंदी में बात कर रहा हूँ", description: "I am speaking in Hindi" },
  { text: "क्या आप मेरी आवाज़ सुन सकते हैं?", description: "Can you hear my voice?" },
  { text: "कृपया मुझे वह किताब दे दीजिए", description: "Please give me that book" },
  { text: "आज मौसम बहुत अच्छा है", description: "The weather is very nice today" },
  { text: "मुझे संगीत सुनना पसंद है", description: "I like listening to music" },
  { text: "क्या आप मुझे समझ सकते हैं?", description: "Can you understand me?" },
  { text: "माया रिसर्च में आपका स्वागत है", description: "Welcome to Maya Research" },
  { text: "आर्टिफिशियल इंटेलिजेंस का भविष्य उज्जवल है", description: "The future of artificial intelligence is bright" },
];

function App() {
  // State variables
  const [text, setText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [speakerId, setSpeakerId] = useState('aisha');
  const [audioUrl, setAudioUrl] = useState(null);
  const [showHealthCheck, setShowHealthCheck] = useState(false);
  const [healthStatus, setHealthStatus] = useState(null);
  const [advancedOptions, setAdvancedOptions] = useState(false);
  const [maxNewTokens, setMaxNewTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [speedAdjustment, setSpeedAdjustment] = useState(1.0);
  const [audioQualityPreset, setAudioQualityPreset] = useState('high');
  const [isHealthy, setIsHealthy] = useState(null);
  
  // Refs
  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);
  const parallaxRef = useRef(null);
  const audioRef = useRef(null);
  
  // Parallax effect on scroll
  useEffect(() => {
    const handleScroll = () => {
      if (parallaxRef.current) {
        const scrollPosition = window.scrollY;
        parallaxRef.current.style.transform = `translateY(${scrollPosition * 0.5}px)`;
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  // Initialize WaveSurfer
  useEffect(() => {
    if (audioUrl && waveformRef.current) {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
      }
      
      const wavesurfer = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#4eff91',
        progressColor: '#1db954',
        cursorColor: '#1db954',
        barWidth: 2,
        barRadius: 3,
        cursorWidth: 1,
        height: 100,
        barGap: 3,
        responsive: true,
        normalize: true,
      });
      
      wavesurfer.load(audioUrl);
      wavesurferRef.current = wavesurfer;
      
      wavesurfer.on('ready', () => {
        wavesurfer.play();
      });
    }
  }, [audioUrl]);
  
  // Check API health status
  const checkHealth = async () => {
    try {
      const response = await fetch('/health');
      const data = await response.json();
      setHealthStatus(data);
      setIsHealthy(data.status === 'healthy');
    } catch (error) {
      setHealthStatus({ error: error.message });
      setIsHealthy(false);
    }
  };
  
  // Handle text generation
  const generateSpeech = async () => {
    if (!text.trim()) return;
    
    setIsGenerating(true);
    
    const requestBody = {
      text,
      speaker_id: speakerId,
      max_new_tokens: maxNewTokens,
      temperature,
      top_p: topP,
      repetition_penalty: repetitionPenalty,
      speed_adjustment: speedAdjustment,
      audio_quality_preset: audioQualityPreset,
    };
    
    try {
      const response = await fetch('/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate speech');
      }
      
      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      
      setAudioUrl(url);
      if (audioRef.current) {
        audioRef.current.src = url;
      }
    } catch (error) {
      console.error('Error generating speech:', error);
      alert('Error generating speech. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };
  
  useEffect(() => {
    if (showHealthCheck) {
      checkHealth();
    }
  }, [showHealthCheck]);
  
  const selectSampleSentence = (sentence) => {
    setText(sentence.text);
  };
  
  return (
    <div className="app">
      <div className="stars"></div>
      <div className="twinkling"></div>
      
      <div className="parallax-container">
        <div className="parallax-background" ref={parallaxRef}></div>
        
        <header className="app-header">
          <div className="logo-container">
            <div className="logo">माया</div>
            <div className="logo-subtitle">RESEARCH</div>
          </div>
          <h1 className="title">Orpheus Hindi TTS</h1>
          <p className="subtitle">Neural Text-to-Speech Synthesis for Hindi</p>
        </header>
        
        <div className="content-container">
          <div className="main-content">
            <div className="text-input-section">
              <h2>Enter Text to Synthesize</h2>
              <div className="input-container">
                <textarea
                  className="text-input"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Type Hindi text here or select a sample sentence below..."
                  rows={4}
                />
                
                <div className="samples-section">
                  <h3>Sample Sentences</h3>
                  <div className="sample-grid">
                    {SAMPLE_HINDI_SENTENCES.map((sentence, index) => (
                      <div
                        key={index}
                        className="sample-card"
                        onClick={() => selectSampleSentence(sentence)}
                      >
                        <div className="sample-text">{sentence.text}</div>
                        <div className="sample-description">{sentence.description}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="speaker-selection">
                <label htmlFor="speaker-select">Voice:</label>
                <select 
                  id="speaker-select"
                  value={speakerId}
                  onChange={(e) => setSpeakerId(e.target.value)}
                >
                  <option value="aisha">Aisha</option>
                </select>
              </div>
              
              <div className="advanced-toggle">
                <button
                  className={`toggle-button ${advancedOptions ? 'active' : ''}`}
                  onClick={() => setAdvancedOptions(!advancedOptions)}
                >
                  {advancedOptions ? 'Hide Advanced Options' : 'Show Advanced Options'}
                </button>
              </div>
              
              {advancedOptions && (
                <div className="advanced-options">
                  <div className="option-row">
                    <div className="option-group">
                      <label htmlFor="max-tokens">Max New Tokens:</label>
                      <input
                        id="max-tokens"
                        type="range"
                        min="100"
                        max="4096"
                        step="1"
                        value={maxNewTokens}
                        onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
                      />
                      <span className="value-display">{maxNewTokens}</span>
                    </div>
                    
                    <div className="option-group">
                      <label htmlFor="temperature">Temperature:</label>
                      <input
                        id="temperature"
                        type="range"
                        min="0"
                        max="2"
                        step="0.05"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      />
                      <span className="value-display">{temperature.toFixed(2)}</span>
                    </div>
                  </div>
                  
                  <div className="option-row">
                    <div className="option-group">
                      <label htmlFor="top-p">Top P:</label>
                      <input
                        id="top-p"
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={topP}
                        onChange={(e) => setTopP(parseFloat(e.target.value))}
                      />
                      <span className="value-display">{topP.toFixed(2)}</span>
                    </div>
                    
                    <div className="option-group">
                      <label htmlFor="repetition-penalty">Repetition Penalty:</label>
                      <input
                        id="repetition-penalty"
                        type="range"
                        min="1"
                        max="2"
                        step="0.05"
                        value={repetitionPenalty}
                        onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                      />
                      <span className="value-display">{repetitionPenalty.toFixed(2)}</span>
                    </div>
                  </div>
                  
                  <div className="option-row">
                    <div className="option-group">
                      <label htmlFor="speed-adjustment">Speed Adjustment:</label>
                      <input
                        id="speed-adjustment"
                        type="range"
                        min="0.5"
                        max="2"
                        step="0.05"
                        value={speedAdjustment}
                        onChange={(e) => setSpeedAdjustment(parseFloat(e.target.value))}
                      />
                      <span className="value-display">{speedAdjustment.toFixed(2)}x</span>
                    </div>
                    
                    <div className="option-group">
                      <label htmlFor="audio-quality">Audio Quality:</label>
                      <select
                        id="audio-quality"
                        value={audioQualityPreset}
                        onChange={(e) => setAudioQualityPreset(e.target.value)}
                      >
                        <option value="low">Low (16kHz)</option>
                        <option value="medium">Medium (24kHz)</option>
                        <option value="high">High (24kHz, best quality)</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="generate-button-container">
                <button
                  className="generate-button"
                  onClick={generateSpeech}
                  disabled={isGenerating || !text.trim()}
                >
                  {isGenerating ? (
                    <>
                      <span className="spinner"></span>
                      Generating...
                    </>
                  ) : (
                    "Generate Speech"
                  )}
                </button>
              </div>
            </div>
            
            <div className={`audio-output-section ${audioUrl ? 'active' : ''}`}>
              <h2>Generated Audio</h2>
              {audioUrl && (
                <>
                  <div className="waveform-container" ref={waveformRef}></div>
                  <div className="audio-controls">
                    <audio ref={audioRef} controls>
                      <source src={audioUrl} type="audio/wav" />
                      Your browser does not support the audio element.
                    </audio>
                    <a
                      href={audioUrl}
                      download="maya_tts_output.wav"
                      className="download-button"
                    >
                      Download Audio
                    </a>
                  </div>
                </>
              )}
              {!audioUrl && !isGenerating && (
                <div className="placeholder-message">
                  Generated audio will appear here
                </div>
              )}
            </div>
          </div>
          
          <div className="system-status-section">
            <div className="health-check-toggle">
              <button 
                className={`health-toggle ${showHealthCheck ? 'active' : ''}`}
                onClick={() => setShowHealthCheck(!showHealthCheck)}
              >
                {showHealthCheck ? 'Hide System Status' : 'Show System Status'}
              </button>
            </div>
            
            {showHealthCheck && (
              <div className="health-status">
                <h3>System Status</h3>
                {isHealthy === null ? (
                  <div className="loading">Checking status...</div>
                ) : (
                  <>
                    <div className={`status-indicator ${isHealthy ? 'healthy' : 'unhealthy'}`}>
                      {isHealthy ? 'System Operational' : 'System Experiencing Issues'}
                    </div>
                    
                    {healthStatus && (
                      <div className="status-details">
                        <div className="status-item">
                          <span className="status-label">Status:</span>
                          <span className="status-value">{healthStatus.status || 'unknown'}</span>
                        </div>
                        
                        {healthStatus.timestamp && (
                          <div className="status-item">
                            <span className="status-label">Last Updated:</span>
                            <span className="status-value">
                              {new Date(healthStatus.timestamp * 1000).toLocaleString()}
                            </span>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
        
        <footer className="app-footer">
          <div className="footer-content">
            <p>© 2023 Maya Research - Neural Text-to-Speech System</p>
            <p>Powered by Orpheus and SNAC</p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App; 
import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import './App.css';
import { FaPlay, FaPause, FaDownload, FaSpinner, FaExclamationTriangle, FaCog, FaChevronDown, FaChevronUp, FaHeartbeat } from 'react-icons/fa';
import { FiExternalLink } from 'react-icons/fi';

// --- Configuration ---
const API_ENDPOINT = process.env.REACT_APP_TTS_API_ENDPOINT || 'https://fuchr--orpheus-tts-api-refined-v2-orpheusttsapi-run-fastapi-app.modal.run';

// Speaker configuration
const AVAILABLE_SPEAKERS = [
  { id: 'aisha', displayName: 'Aisha (Female, Default)' },
  { id: 'anika', displayName: 'Anika (Female)' },
  { id: 'ivanna', displayName: 'Ivanna (Female)' },
  { id: 'raju', displayName: 'Raju (Male)' },
  { id: 'sia', displayName: 'Sia (Female)' },
  { id: 'sangeeta', displayName: 'Sangeeta (Female)' }
];
const DEFAULT_SPEAKER_ID = AVAILABLE_SPEAKERS[0].id;

// Sample Hindi sentences
const PRESET_SENTENCES = [
  { id: 's1', text: 'कहाँ तक पहुँचे? मैं इंतज़ार कर रहा हूँ।' },
  { id: 's2', text: 'वीडियो कॉल कर लेंगे, ठीक है?' },
  { id: 's3', text: 'क्या तुमने आज का मेमो पढ़ा?' },
  { id: 's4', text: 'अगर ऑर्डर डिलीवर नहीं होता, तो आपका पूरा पैसा वापस कर दिया जाएगा।' },
  { id: 's5', text: 'पेट्रोल के दाम फिर बढ़ गए क्या?' },
  { id: 's6', text: 'चलो, वीकेंड में पहाड़ चलते हैं!' },
  { id: 's7', text: 'यह एक ओपन-सोर्स टेक्स्ट-टू-स्पीच मॉडल है।' },
  { id: 's8', text: 'नमस्ते, आप कैसे हैं?' },
  { id: 's9', text: 'आज मौसम बहुत अच्छा है।' },
  { id: 's10', text: 'यह एक परीक्षण नमूना है।' },
];

// Toggle for health check visibility (you can set this to false to hide it)
const SHOW_HEALTH_CHECK = true;

function App() {
  // Basic state management
  const [text, setText] = useState(PRESET_SENTENCES[0].text);
  const [selectedPreset, setSelectedPreset] = useState(PRESET_SENTENCES[0].id);
  const [speakerId, setSpeakerId] = useState(DEFAULT_SPEAKER_ID);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState("00:00");
  const [currentTime, setCurrentTime] = useState("00:00");

  // Health status state
  const [healthStatus, setHealthStatus] = useState({ status: 'unknown', timestamp: null });
  const [showHealthCheck, setShowHealthCheck] = useState(SHOW_HEALTH_CHECK);

  // Advanced Settings State
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [speedAdjustment, setSpeedAdjustment] = useState(1.0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [seed, setSeed] = useState('');
  const [earlyStopping, setEarlyStopping] = useState(true);
  const [topP, setTopP] = useState(0.9);
  const [audioQuality, setAudioQuality] = useState('high');

  // Refs
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const audioControlRef = useRef(new Audio());

  // API endpoint check
  useEffect(() => {
    if (!API_ENDPOINT) {
      setError('API endpoint (REACT_APP_TTS_API_ENDPOINT) is not configured. Please set it in your environment.');
      console.error('CRITICAL: REACT_APP_TTS_API_ENDPOINT is not set.');
    } else if (showHealthCheck) {
      // Initial health check
      checkApiHealth();
      
      // Set up periodic health checks
      const healthCheckInterval = setInterval(checkApiHealth, 60000); // Check every minute
      return () => clearInterval(healthCheckInterval);
    }
  }, [showHealthCheck]);
  
  // Health check function
  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_ENDPOINT}/health`);
      setHealthStatus({
        status: response.data.status === 'healthy' ? 'healthy' : 'unhealthy',
        timestamp: response.data.timestamp || Date.now()
      });
    } catch (err) {
      console.warn('Health check failed:', err);
      setHealthStatus({
        status: 'error',
        timestamp: Date.now(),
        message: err.message
      });
    }
  };

  // Time formatting helper
  const formatTime = (time) => {
    if (isNaN(time) || time === Infinity) return "00:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  // Initialize WaveSurfer
  useEffect(() => {
    if (waveformRef.current && !wavesurfer.current) {
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'rgba(54, 255, 77, 0.3)',
        progressColor: '#36ff4d',
        cursorColor: '#ffffff',
        barWidth: 3,
        barRadius: 3,
        responsive: true,
        height: 100,
        normalize: true,
        media: audioControlRef.current,
      });

      wavesurfer.current.on('ready', () => {
        setDuration(formatTime(wavesurfer.current.getDuration()));
        if (audioBlob && wavesurfer.current) {
          wavesurfer.current.play().catch(e => console.warn("Audio autoplay prevented on ready:", e));
        }
      });
      wavesurfer.current.on('audioprocess', (time) => setCurrentTime(formatTime(time)));
      wavesurfer.current.on('play', () => setIsPlaying(true));
      wavesurfer.current.on('pause', () => setIsPlaying(false));
      wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
        wavesurfer.current.seekTo(0);
        setCurrentTime("00:00");
      });
    }
    return () => {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
        wavesurfer.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load audio into WaveSurfer when blob changes
  useEffect(() => {
    if (audioBlob && wavesurfer.current) {
      const audioUrl = URL.createObjectURL(audioBlob);
      wavesurfer.current.load(audioUrl);
      return () => URL.revokeObjectURL(audioUrl);
    } else if (!audioBlob && wavesurfer.current) {
        wavesurfer.current.empty();
        audioControlRef.current.src = "";
        setDuration("00:00");
        setCurrentTime("00:00");
        setIsPlaying(false);
    }
  }, [audioBlob]);

  // Event handlers
  const handleTextChange = (event) => {
    setText(event.target.value);
    setSelectedPreset('');
    if (error) setError(null);
  };

  const handlePresetChange = (event) => {
    const sentenceId = event.target.value;
    setSelectedPreset(sentenceId);
    const selectedSentence = PRESET_SENTENCES.find(s => s.id === sentenceId);
    if (selectedSentence) setText(selectedSentence.text);
    if (error) setError(null);
  };

  const handleSpeakerChange = (event) => {
    setSpeakerId(event.target.value);
    if (error) setError(null);
  };

  // Handlers for Advanced Settings
  const handleTemperatureChange = (e) => setTemperature(parseFloat(e.target.value));
  const handleSpeedAdjustmentChange = (e) => setSpeedAdjustment(parseFloat(e.target.value));
  const handleRepetitionPenaltyChange = (e) => setRepetitionPenalty(parseFloat(e.target.value));
  const handleTopPChange = (e) => setTopP(parseFloat(e.target.value));
  const handleSeedChange = (e) => setSeed(e.target.value);
  const handleEarlyStoppingChange = (e) => setEarlyStopping(e.target.checked);
  const handleAudioQualityChange = (e) => setAudioQuality(e.target.value);

  // Form submission handler
  const handleSubmit = async (event) => {
    if (event) event.preventDefault();
    if (!API_ENDPOINT) {
      setError('API endpoint is not configured.');
      return;
    }
    if (!text.trim()) {
      setError('Please enter or select some text to synthesize.');
      return;
    }

    setIsLoading(true);
    setError(null);
    if (wavesurfer.current && wavesurfer.current.isPlaying()) {
        wavesurfer.current.stop();
    }
    setAudioBlob(null); 
    setIsPlaying(false);
    setCurrentTime("00:00");
    setDuration("00:00");

    const payload = {
      text: text,
      speaker_id: speakerId,
      output_sample_rate: 24000,
      temperature: temperature,
      speed_adjustment: speedAdjustment,
      repetition_penalty: repetitionPenalty,
      top_p: topP,
      audio_quality_preset: audioQuality,
      early_stopping: earlyStopping,
    };

    if (seed && !isNaN(parseInt(seed))) {
      payload.seed = parseInt(seed);
    }

    try {
      const response = await axios.post(`${API_ENDPOINT}/generate`, payload, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'audio/wav',
        },
        timeout: 90000,
      });
      setAudioBlob(new Blob([response.data], { type: 'audio/wav' }));
    } catch (err)  {
      console.error('API Error:', err);
      let errorMessageText = 'An unexpected error occurred during TTS generation.';
      if (err.response) {
          if (err.response.data instanceof Blob) {
              try {
                  const errorBlobText = await err.response.data.text();
                  const errorJson = JSON.parse(errorBlobText);
                  errorMessageText = errorJson.error || errorJson.detail || `Server error: ${err.response.status}`;
              } catch (e) {
                  errorMessageText = `Server error: ${err.response.status}. Could not parse error response.`;
              }
          } else {
              errorMessageText = err.response.data.error || err.response.data.detail || `Server error: ${err.response.status}`;
          }
      } else if (err.request) {
          errorMessageText = 'No response from the server. It might be down or unreachable.';
      } else if (err.message) {
          errorMessageText = err.message;
      }
      setError(errorMessageText);
    } finally {
      setIsLoading(false);
    }
  };

  // Audio player controls
  const togglePlayPause = useCallback(() => {
    if (wavesurfer.current && audioBlob) wavesurfer.current.playPause();
  }, [audioBlob]);

  const handleDownload = () => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      const a = document.createElement('a');
      a.href = url;
      const selectedSpeakerName = AVAILABLE_SPEAKERS.find(s => s.id === speakerId)?.displayName.split(' ')[0] || 'veena';
      a.download = `${selectedSpeakerName.toLowerCase()}_tts_maya_research_${text.substring(0, 20).replace(/\s+/g, '_')}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="App">
      <div className="background-animation">
        {Array.from({ length: 50 }).map((_, i) => (
          <div className="line" key={i} style={{
            left: `${Math.random() * 100}%`,
            animationDuration: `${Math.random() * 5 + 5}s`,
            animationDelay: `${Math.random() * 5}s`,
            height: `${Math.random() * 150 + 50}px`
          }}></div>
        ))}
      </div>
      
      {/* Health status bar - Conditionally shown */}
      {showHealthCheck && (
        <div className={`health-status ${healthStatus.status === 'healthy' ? 'healthy' : 'unhealthy'}`}>
          <div className="status-indicator">
            <FaHeartbeat className="heartbeat-icon" />
          </div>
          <div className="status-text">
            API Status: {healthStatus.status === 'healthy' ? 'Healthy' : healthStatus.status === 'error' ? 'Error' : 'Unhealthy'} 
            {healthStatus.timestamp && (
              <span className="timestamp">
                Last checked: {new Date(healthStatus.timestamp * 1000).toLocaleTimeString()}
              </span>
            )}
          </div>
          <button className="health-refresh" onClick={checkApiHealth} title="Refresh Health Status">
            ↻
          </button>
        </div>
      )}
      
      <div className="content-wrapper">
        <header className="App-header">
          <h1>ORPHEUS HINDI TTS</h1>
          <p>Building AI for India</p>
        </header>

        <main className="App-main">
          <div className="tts-container">
            <div className="tts-controls-column">
              <h2>Text-to-Speech Synthesis</h2>
              <p className="tagline">Lifelike Hindi speech, available to everyone.</p>

              <form onSubmit={handleSubmit} className="tts-form">
                {/* Preset Sentence Select */}
                <div className="form-group">
                  <label htmlFor="preset-select">Select a sample sentence</label>
                  <select id="preset-select" value={selectedPreset} onChange={handlePresetChange} disabled={isLoading || !API_ENDPOINT} className="styled-select">
                    {PRESET_SENTENCES.map(sentence => (<option key={sentence.id} value={sentence.id}>{sentence.text.substring(0,50)}...</option>))}
                    <option value="">— Type custom text below —</option>
                  </select>
                </div>

                {/* Custom Text Input */}
                <div className="form-group">
                  <label htmlFor="text-input">Or type your custom Hindi text (max 200 chars)</label>
                  <textarea id="text-input" value={text} onChange={handleTextChange} placeholder="यहाँ अपना हिंदी पाठ लिखें..." rows="4" maxLength="200" className="text-input" disabled={isLoading || !API_ENDPOINT} aria-label="Hindi text input"/>
                </div>

                {/* Speaker Select */}
                <div className="form-group">
                  <label htmlFor="speaker-select">Select Voice</label>
                  <select id="speaker-select" value={speakerId} onChange={handleSpeakerChange} disabled={isLoading || !API_ENDPOINT || AVAILABLE_SPEAKERS.length <= 1} className="styled-select" aria-label="Select voice">
                    {AVAILABLE_SPEAKERS.map(speaker => (<option key={speaker.id} value={speaker.id}>{speaker.displayName}</option>))}
                  </select>
                </div>
                
                {/* Generate Button */}
                <button type="submit" className="generate-button" disabled={isLoading || !text.trim() || !API_ENDPOINT} aria-label="Generate audio">
                  {isLoading ? (<><FaSpinner className="spinner-icon" /> GENERATING...</>) : ('SYNTHESIZE SPEECH')}
                </button>

                {/* Advanced Settings Toggle */}
                <button type="button" onClick={() => setShowAdvancedSettings(!showAdvancedSettings)} className="advanced-settings-toggle" aria-expanded={showAdvancedSettings}>
                  <FaCog style={{ marginRight: '8px' }} />
                  Advanced Settings
                  {showAdvancedSettings ? <FaChevronUp style={{ marginLeft: 'auto' }} /> : <FaChevronDown style={{ marginLeft: 'auto' }} />}
                </button>

                {/* Advanced Settings Section */}
                {showAdvancedSettings && (
                  <div className="advanced-settings-container">
                    {/* Temperature */}
                    <div className="form-group">
                      <label htmlFor="temperature-slider">Temperature: <span className="slider-value">{temperature.toFixed(2)}</span></label>
                      <input type="range" id="temperature-slider" min="0.0" max="1.0" step="0.01" value={temperature} onChange={handleTemperatureChange} className="styled-slider" disabled={isLoading} />
                    </div>

                    {/* Speed Adjustment */}
                    <div className="form-group">
                      <label htmlFor="speed-slider">Speed: <span className="slider-value">{speedAdjustment.toFixed(2)}x</span></label>
                      <input type="range" id="speed-slider" min="0.5" max="2.0" step="0.05" value={speedAdjustment} onChange={handleSpeedAdjustmentChange} className="styled-slider" disabled={isLoading} />
                    </div>
                    
                    {/* Top P */}
                    <div className="form-group">
                      <label htmlFor="top-p-slider">Top P: <span className="slider-value">{topP.toFixed(2)}</span></label>
                      <input type="range" id="top-p-slider" min="0.0" max="1.0" step="0.05" value={topP} onChange={handleTopPChange} className="styled-slider" disabled={isLoading} />
                    </div>
                    
                    {/* Repetition Penalty */}
                    <div className="form-group">
                      <label htmlFor="repetition-penalty-slider">Repetition Penalty: <span className="slider-value">{repetitionPenalty.toFixed(2)}</span></label>
                      <input type="range" id="repetition-penalty-slider" min="1.0" max="2.0" step="0.05" value={repetitionPenalty} onChange={handleRepetitionPenaltyChange} className="styled-slider" disabled={isLoading} />
                    </div>

                    {/* Audio Quality */}
                    <div className="form-group">
                      <label htmlFor="audio-quality-select">Audio Quality</label>
                      <select id="audio-quality-select" value={audioQuality} onChange={handleAudioQualityChange} className="styled-select" disabled={isLoading}>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                      </select>
                    </div>

                    {/* Seed */}
                    <div className="form-group">
                      <label htmlFor="seed-input">Seed (Optional)</label>
                      <input type="number" id="seed-input" value={seed} onChange={handleSeedChange} placeholder="e.g., 42" className="text-input" disabled={isLoading} />
                    </div>
                    
                    {/* Early Stopping */}
                    <div className="form-group form-group-checkbox">
                      <input type="checkbox" id="early-stopping-checkbox" checked={earlyStopping} onChange={handleEarlyStoppingChange} className="styled-checkbox" disabled={isLoading} />
                      <label htmlFor="early-stopping-checkbox">Enable Early Stopping</label>
                    </div>
                  </div>
                )}
              </form>
            </div>

            {/* Player Column */}
            <div className="tts-player-column">
              <div className="glass-card">
                {error && (
                  <div className="error-message" role="alert">
                    <FaExclamationTriangle style={{ marginRight: '10px', verticalAlign: 'middle' }}/>
                    <strong>Error:</strong> {error}
                  </div>
                )}
                <div className="sentence-display" title="Synthesized text appears here.">
                  {text || "Your synthesized text will appear here."}
                </div>
                <div className="wavesurfer-wrapper">
                  <div id="waveform" ref={waveformRef}></div>
                </div>
                {(audioBlob || isLoading) && (
                  <div className="audio-player-controls-footer">
                     <button onClick={togglePlayPause} className="play-pause-button" aria-label={isPlaying ? "Pause" : "Play"} disabled={!audioBlob || isLoading}>
                        {isPlaying ? <FaPause /> : <FaPlay />}
                    </button>
                    <div className="time-indicators-footer">
                        <span>{currentTime}</span> / <span>{duration}</span>
                    </div>
                    <button onClick={handleDownload} className="download-button" disabled={!audioBlob || isLoading} aria-label="Download audio">
                        <FaDownload />
                    </button>
                  </div>
                )}
                {!isLoading && !audioBlob && !error && (
                  <div className="player-placeholder">
                      Synthesize speech to listen and see the waveform.
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>

        <div className="section-separator"></div>

        <section className="secondary-content">
          <a href="#" onClick={() => setShowHealthCheck(!showHealthCheck)} className="secondary-link">
            {showHealthCheck ? "Hide Health Status" : "Show Health Status"}
          </a>
          <a href="https://mayaresearch.ai" target="_blank" rel="noopener noreferrer" className="secondary-link">
            Website - Maya Research <FiExternalLink />
          </a>
        </section>

        <footer className="App-footer">
          <p>© {new Date().getFullYear()} Orpheus Hindi TTS</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
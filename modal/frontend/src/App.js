import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import {
  FaPlay, FaSpinner, FaExclamationTriangle, FaVolumeUp, FaCog,
  FaTimes, FaDownload, FaShareAlt, FaGithub, FaDiscord, FaSun, FaMoon, FaWaveSquare
} from 'react-icons/fa'; // Added more icons

// --- Configuration ---
const API_ENDPOINT = process.env.REACT_APP_TTS_API_ENDPOINT || "https://your-modal-app-url"; // Fallback for local dev

// Default values for parameters
const DEFAULT_SPEAKER = 'aisha';
const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_REPETITION_PENALTY = 1.1;
const DEFAULT_SPEED_ADJUSTMENT = 1.0;
const DEFAULT_TOP_P = 0.9; // Added from API spec
const DEFAULT_MAX_NEW_TOKENS = 2048; // Added from API spec

// Quality presets mapped to sample rates (as per design "Quality preset chips")
const QUALITY_PRESETS = [
  { id: 'low', name: 'Low', sampleRate: 16000, icon: <FaWaveSquare /> }, // Placeholder icon
  { id: 'medium', name: 'Medium', sampleRate: 24000, icon: <FaWaveSquare /> },
  { id: 'high', name: 'High', sampleRate: 48000, icon: <FaWaveSquare /> }, // Orpheus default 24k, API takes 48k
];
const DEFAULT_QUALITY_PRESET = 'medium';


const AVAILABLE_SPEAKERS = [
  { id: 'aisha', name: 'Aisha' },
  { id: 'anika', name: 'Anika' },
  { id: 'ivanna', name: 'Ivanna' },
  { id: 'raju', name: 'Raju' },
  { id: 'sia', name: 'Sia' },
  { id: 'sangeeta', name: 'Sangeeta' },
];

function App() {
  const [text, setText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [audioSrc, setAudioSrc] = useState(null);
  const audioRef = useRef(null);

  // Modal State
  const [showInputConsole, setShowInputConsole] = useState(false);

  // TTS Parameters States
  const [speakerId, setSpeakerId] = useState(DEFAULT_SPEAKER);
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [repetitionPenalty, setRepetitionPenalty] = useState(DEFAULT_REPETITION_PENALTY);
  const [speedAdjustment, setSpeedAdjustment] = useState(DEFAULT_SPEED_ADJUSTMENT);
  const [topP, setTopP] = useState(DEFAULT_TOP_P);
  const [maxNewTokens, setMaxNewTokens] = useState(DEFAULT_MAX_NEW_TOKENS);
  const [selectedQuality, setSelectedQuality] = useState(DEFAULT_QUALITY_PRESET);

  const [showAdvanced, setShowAdvanced] = useState(false); // Keep for advanced section within modal
  const [isDarkMode, setIsDarkMode] = useState(true); // Default to dark mode
  const [apiHealth, setApiHealth] = useState(null); // 'healthy', 'unhealthy', null (pending)

  const MAX_TEXT_LENGTH = 1000; // Example character limit for textarea

  // API Health Check
  const checkApiHealth = useCallback(async () => {
    if (!API_ENDPOINT) return;
    try {
      const response = await axios.get(`${API_ENDPOINT}/health`, { timeout: 5000 });
      if (response.data && response.data.status === 'healthy') {
        setApiHealth('healthy');
      } else {
        setApiHealth('unhealthy');
      }
    } catch (err) {
      console.error('API health check failed:', err);
      setApiHealth('unhealthy');
    }
  }, []);

  useEffect(() => {
    if (!API_ENDPOINT || API_ENDPOINT === "https://your-modal-app-url") {
      setError('API endpoint (REACT_APP_TTS_API_ENDPOINT) is not configured. Please set it in your environment or .env file.');
      console.error('CRITICAL: REACT_APP_TTS_API_ENDPOINT is not set.');
      setApiHealth('unhealthy');
    } else {
      checkApiHealth();
      const intervalId = setInterval(checkApiHealth, 60000); // Check every minute
      return () => clearInterval(intervalId);
    }
  }, [checkApiHealth]);

  useEffect(() => {
    document.body.className = isDarkMode ? 'dark-mode' : 'light-mode';
    // For this example, we'll just toggle a class on body. A full theme switch is more involved.
    // The CSS provided is dark-mode first. A light-mode would require more CSS.
  }, [isDarkMode]);


  const handleTextChange = (event) => {
    const newText = event.target.value;
    if (newText.length <= MAX_TEXT_LENGTH) {
      setText(newText);
    }
    if (error) setError(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!API_ENDPOINT || apiHealth === 'unhealthy') {
      setError('API is not available or not configured correctly.');
      return;
    }
    if (!text.trim()) {
      setError('Please enter some text to synthesize.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setAudioSrc(null);

    const qualityPreset = QUALITY_PRESETS.find(p => p.id === selectedQuality);

    const payload = {
      text: text,
      speaker_id: speakerId,
      temperature: parseFloat(temperature),
      repetition_penalty: parseFloat(repetitionPenalty),
      speed_adjustment: parseFloat(speedAdjustment),
      top_p: parseFloat(topP),
      max_new_tokens: parseInt(maxNewTokens, 10),
      output_sample_rate: qualityPreset ? qualityPreset.sampleRate : 24000,
      audio_quality_preset: selectedQuality, // as string "low", "medium", "high"
      // seed: null, // Optional
      // early_stopping: true, // Default in API
    };

    console.log("Sending payload:", payload);

    try {
      const response = await axios.post(API_ENDPOINT + '/generate', payload, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'audio/wav',
        },
        timeout: 90000,
      });

      const audioBlob = new Blob([response.data], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      setAudioSrc(audioUrl);
      // setShowInputConsole(false); // Optionally close modal on success

    } catch (err) {
      console.error('API Error:', err);
      let errorMessage = 'An unexpected error occurred while generating audio.';
      if (err.response) {
        try {
          const errorDataText = await err.response.data.text();
          const errorJson = JSON.parse(errorDataText);
          errorMessage = errorJson.error || errorJson.detail || `Server error: ${err.response.status}`;
        } catch (parseError) {
          errorMessage = `${err.response.status}: ${err.response.statusText || 'Server Error'}`;
        }
      } else if (err.request) {
        errorMessage = 'No response from the server. It might be down or unreachable.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (audioSrc && audioRef.current) {
      audioRef.current.play().catch(e => console.warn("Audio autoplay prevented:", e));
    }
  }, [audioSrc]);

  const ParameterSlider = ({ label, value, onChange, min, max, step, unit = "" }) => (
    <div className="parameter-slider">
      <label htmlFor={label.toLowerCase().replace(/\s+/g, '-')}>
        {label}
        <span className="slider-value">{value}{unit}</span>
      </label>
      <input
        type="range"
        id={label.toLowerCase().replace(/\s+/g, '-')}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={isLoading}
        aria-label={`${label} slider`}
      />
    </div>
  );

  const renderInputConsole = () => (
    <div className={`modal-overlay ${showInputConsole ? 'active' : ''}`} onClick={() => setShowInputConsole(false)}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">Create Your Audio</h2>
          <button className="close-modal-button" onClick={() => setShowInputConsole(false)} aria-label="Close console">
            <FaTimes />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="input-console-grid">
            <div className="text-input-column">
              <div className="text-input-wrapper">
                <textarea
                  value={text}
                  onChange={handleTextChange}
                  placeholder="Type or paste text… (Max 1000 characters)"
                  className="text-input"
                  disabled={isLoading || apiHealth !== 'healthy'}
                  aria-label="Text to synthesize"
                  rows="10"
                />
                <div className="char-counter">{text.length}/{MAX_TEXT_LENGTH}</div>
              </div>
            </div>

            <div className="parameters-column">
              <div className="parameter-item">
                <label htmlFor="speakerId">Voice:</label>
                <select
                  id="speakerId"
                  value={speakerId}
                  onChange={(e) => setSpeakerId(e.target.value)}
                  disabled={isLoading}
                  className="styled-select"
                  aria-label="Select voice"
                >
                  {AVAILABLE_SPEAKERS.map(speaker => (
                    <option key={speaker.id} value={speaker.id}>{speaker.name}</option>
                  ))}
                </select>
              </div>

              <div className="parameter-item">
                <label>Quality Preset:</label>
                <div className="quality-presets">
                  {QUALITY_PRESETS.map(preset => (
                    <button
                      type="button"
                      key={preset.id}
                      className={`quality-chip ${selectedQuality === preset.id ? 'active' : ''}`}
                      onClick={() => setSelectedQuality(preset.id)}
                      disabled={isLoading}
                      aria-pressed={selectedQuality === preset.id}
                    >
                      {preset.icon} {preset.name}
                    </button>
                  ))}
                </div>
              </div>
              
              <ParameterSlider
                label="Speed"
                value={speedAdjustment}
                onChange={setSpeedAdjustment}
                min="0.5" max="2.0" step="0.05" unit="×"
              />

              {/* Advanced Settings Toggle - could be a small button or icon */}
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="toggle-advanced-button" // This class needs to be styled according to new theme
                style={{ alignSelf: 'flex-start', background: 'transparent', border:'1px solid var(--glass-border)', color: 'var(--text-muted)', padding:'8px 12px', borderRadius:'8px', cursor:'pointer' }}
              >
                <FaCog style={{ marginRight: '5px' }} /> {showAdvanced ? 'Hide' : 'Show'} Advanced
              </button>

              {showAdvanced && (
                <>
                  <ParameterSlider
                    label="Temperature"
                    value={temperature}
                    onChange={setTemperature}
                    min="0.0" max="2.0" step="0.05"
                  />
                  <ParameterSlider
                    label="Top-P"
                    value={topP}
                    onChange={setTopP}
                    min="0.0" max="1.0" step="0.01"
                  />
                   <ParameterSlider
                    label="Repetition Penalty"
                    value={repetitionPenalty}
                    onChange={setRepetitionPenalty}
                    min="1.0" max="2.0" step="0.05"
                  />
                  <div className="parameter-item"> {/* max_new_tokens as number input */}
                     <label htmlFor="maxNewTokens">Max New Tokens: <span className="slider-value">{maxNewTokens}</span></label>
                     <input
                        type="number"
                        id="maxNewTokens"
                        className="styled-input"
                        value={maxNewTokens}
                        onChange={(e) => setMaxNewTokens(Math.max(100, Math.min(4096, parseInt(e.target.value,10))))}
                        min="100" max="4096" step="10"
                        disabled={isLoading}
                     />
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="generate-button-wrapper">
            <button
              type="submit"
              className={`generate-button ${isLoading ? 'loading' : ''}`}
              disabled={isLoading || !text.trim() || apiHealth !== 'healthy'}
              aria-label="Generate audio"
            >
              {isLoading ? <span className="button-text sr-only">Generating...</span> : <span className="button-text">Generate Speech</span>}
            </button>
          </div>
        </form>

        {error && (
          <div className="error-message" role="alert" style={{ marginTop: '20px' }}>
            <FaExclamationTriangle style={{ marginRight: '10px', verticalAlign: 'middle' }} />
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="App">
      {/* Hero Section */}
      <section className="hero-section" aria-labelledby="hero-title-text">
        <div className="hero-content">
          <h1 id="hero-title-text" className="hero-title">Hear Your Words Come Alive</h1>
          <p className="hero-subtitle">
            Experience cutting-edge Text-to-Speech in Hindi, English, and more. Any voice, any emotion, brought to life with AI.
          </p>
          <button className="hero-cta-button" onClick={() => setShowInputConsole(true)} aria-haspopup="dialog">
            Generate Speech <FaVolumeUp style={{ marginLeft: '8px' }} />
          </button>
        </div>
      </section>

      {/* Input Console Modal */}
      {renderInputConsole()}

      {/* Audio Player Section - appears when audio is ready */}
      {audioSrc && (
        <section className="audio-player-section" aria-label="Audio Playback">
          <h2>Generated Speech</h2>
          <audio controls src={audioSrc} ref={audioRef} className="audio-player" aria-label="Generated audio player">
            Your browser does not support the audio element.
          </audio>
          <div className="audio-actions">
            <button 
              onClick={() => audioRef.current && audioRef.current.play()} 
              className="download-button" // Reusing style for simplicity
              disabled={!audioSrc || isLoading}
              aria-label="Play generated audio again"
            >
              <FaPlay style={{ marginRight: '8px' }} /> Play Again
            </button>
            <a href={audioSrc} download="orpheus_tts_output.wav" className="download-button">
              <FaDownload style={{ marginRight: '8px' }} /> Download WAV
            </a>
            {/* Share button functionality would require more logic (e.g., copying URL) */}
            {/* <button className="share-button" onClick={() => alert('Share functionality to be implemented')}>
              <FaShareAlt style={{ marginRight: '8px' }} /> Share
            </button> */}
          </div>
        </section>
      )}

      {/* Footer */}
      <footer className="App-footer">
        <div className="footer-content">
          <div className="footer-logo">Orpheus<span className="neon">TTS</span></div>
          <div className="health-indicator">
            <span className={`health-dot ${apiHealth !== 'healthy' ? 'unhealthy' : ''}`} aria-label={`API status: ${apiHealth || 'checking'}`}></span>
            API Status: {apiHealth || 'Checking...'}
          </div>
          <div className="footer-icons">
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" aria-label="GitHub"><FaGithub /></a>
            <a href="https://discord.com" target="_blank" rel="noopener noreferrer" aria-label="Discord"><FaDiscord /></a>
            <button onClick={() => setIsDarkMode(!isDarkMode)} className="light-mode-toggle" aria-label="Toggle light/dark mode">
              {isDarkMode ? <FaSun /> : <FaMoon />}
            </button>
          </div>
        </div>
        <p>© {new Date().getFullYear()} हिन्दी वाणी (Orpheus TTS). All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
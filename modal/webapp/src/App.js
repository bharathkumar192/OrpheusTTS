import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import WaveSurfer from 'wavesurfer.js';
import config from './config';

// Hindi sample sentences without translations
const SAMPLE_HINDI_SENTENCES = [
  "नमस्ते, आपका दिन कैसा रहा?",
  "भारत एक विविधतापूर्ण देश है",
  "मैं हिंदी में बात कर रहा हूँ",
  "क्या आप मेरी आवाज़ सुन सकते हैं?",
  "कृपया मुझे वह किताब दे दीजिए",
  "आज मौसम बहुत अच्छा है",
  "मुझे संगीत सुनना पसंद है",
  "क्या आप मुझे समझ सकते हैं?",
  "माया रिसर्च में आपका स्वागत है",
  "आर्टिफिशियल इंटेलिजेंस का भविष्य उज्जवल है",
  "यह एक नई तकनीक है जो भाषा को आवाज़ में बदलती है",
  "आपकी आवाज़ बहुत सुरीली है",
  "मैं आपकी सहायता करने के लिए यहां हूं",
  "आज हम एक नई यात्रा पर निकलेंगे",
  "हिंदी भारत की प्रमुख भाषाओं में से एक है"
];

function App() {
  // State variables
  const [text, setText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [speakerId, setSpeakerId] = useState('');
  const [availableSpeakers, setAvailableSpeakers] = useState([]);
  const [defaultSpeaker, setDefaultSpeaker] = useState('');
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
  const [isWarmingUp, setIsWarmingUp] = useState(true);
  const [apiInfo, setApiInfo] = useState(null);
  
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
  
  // Initialize WaveSurfer when audio URL changes
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
        height: 150,
        barGap: 3,
        responsive: true,
        normalize: true,
        media: audioRef.current // Connect to audio element for better sync
      });
      
      wavesurfer.load(audioUrl);
      wavesurferRef.current = wavesurfer;
      
      wavesurfer.on('ready', () => {
        // Wait a moment for the audio to be fully ready
        setTimeout(() => {
          if (audioRef.current) {
            // Sync wavesurfer to audio element
            audioRef.current.addEventListener('play', () => wavesurfer.play());
            audioRef.current.addEventListener('pause', () => wavesurfer.pause());
            audioRef.current.addEventListener('seeked', () => {
              wavesurfer.seekTo(audioRef.current.currentTime / audioRef.current.duration);
            });
            
            // Start playback
            audioRef.current.play();
          }
        }, 100);
      });
    }
  }, [audioUrl]);
  
  // Fetch API info (including available speakers)
  const fetchApiInfo = async () => {
    try {
      const response = await fetch(`${config.apiUrl}/`);
      const data = await response.json();
      setApiInfo(data);
      if (data.available_speakers && data.available_speakers.length > 0) {
        setAvailableSpeakers(data.available_speakers);
        setDefaultSpeaker(data.default_speaker || data.available_speakers[0]);
        // Set the speakerId if it's not already set
        if (!speakerId) {
          setSpeakerId(data.default_speaker || data.available_speakers[0]);
        }
      }
      return data;
    } catch (error) {
      console.error('Error fetching API info:', error);
      // Fallback to default speakers if API info can't be fetched
      const fallbackSpeakers = ['shayana', 'raju'];
      setAvailableSpeakers(fallbackSpeakers);
      setDefaultSpeaker('shayana');
      if (!speakerId) {
        setSpeakerId('shayana');
      }
      return null;
    }
  };
  
  // Check API health status
  const checkHealth = async () => {
    try {
      setIsWarmingUp(true);
      const response = await fetch(`${config.apiUrl}/health`);
      const data = await response.json();
      setHealthStatus(data);
      setIsHealthy(data.status === 'healthy');
      setIsWarmingUp(false);
      
      // Also fetch API info to get available speakers
      await fetchApiInfo();
    } catch (error) {
      setHealthStatus({ error: error.message });
      setIsHealthy(false);
      setIsWarmingUp(false);
    }
  };
  
  // Initial health check and API info fetch on page load
  useEffect(() => {
    checkHealth();
  }, []);
  
  // Health check when toggle is clicked
  useEffect(() => {
    if (showHealthCheck && !isWarmingUp) {
      checkHealth();
    }
  }, [showHealthCheck]);
  
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
      const response = await fetch(`${config.apiUrl}/generate`, {
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
  
  const selectSampleSentence = (sentence) => {
    setText(sentence);
  };
  
  // Capitalize first letter of speaker name for display
  const formatSpeakerName = (name) => {
    return name.charAt(0).toUpperCase() + name.slice(1);
  };
  
  return (
    <div className="app">
      <div className="stars"></div>
      <div className="twinkling"></div>
      
      <div className="parallax-container">
        <div className="parallax-background" ref={parallaxRef}></div>
        
        <div className="app-container">
          {/* Top Bar with Title and Health Check */}
          <div className="top-bar">
            <h1 className="brand-name">Maya Research</h1>
            
            <div className="health-check-container">
              <button 
                className={`health-toggle ${showHealthCheck ? 'active' : ''} ${isHealthy === true ? 'healthy-indicator' : ''}`}
                onClick={() => setShowHealthCheck(!showHealthCheck)}
              >
                {isWarmingUp ? 'Warming Up...' : (isHealthy ? 'Server Online' : 'Check Status')}
              </button>
              
              {showHealthCheck && (
                <div className="health-dropdown">
                  {isWarmingUp ? (
                    <div className="loading">Fetching API Health...</div>
                  ) : (
                    <>
                      <div className={`status-indicator ${isHealthy ? 'healthy' : 'unhealthy'}`}>
                        {isHealthy ? 'API Operational' : 'API Issues Detected'}
                      </div>
                      
                      {healthStatus && healthStatus.timestamp && (
                        <div className="status-timestamp">
                          Updated: {new Date(healthStatus.timestamp * 1000).toLocaleTimeString()}
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
          
          {/* Main Content - Two Column Layout */}
          <div className="main-grid">
            {/* Left Column - Input */}
            <div className="input-column">
              <div className="text-input-section">
                <h2>Text to Synthesize</h2>
                <textarea
                  className="text-input"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Type Hindi text here or select a sample sentence below..."
                  rows={5}
                />
                
                <div className="speaker-selection">
                  <label htmlFor="speaker-select">Voice:</label>
                  <select 
                    id="speaker-select"
                    value={speakerId}
                    onChange={(e) => setSpeakerId(e.target.value)}
                    disabled={availableSpeakers.length === 0}
                  >
                    {availableSpeakers.length > 0 ? (
                      availableSpeakers.map(speaker => (
                        <option key={speaker} value={speaker}>
                          {formatSpeakerName(speaker)}
                        </option>
                      ))
                    ) : (
                      <option value="">Loading voices...</option>
                    )}
                  </select>
                </div>
                
                <div className="advanced-toggle">
                  <button
                    className={`toggle-button ${advancedOptions ? 'active' : ''}`}
                    onClick={() => setAdvancedOptions(!advancedOptions)}
                  >
                    {advancedOptions ? 'Hide Parameters' : 'Show Parameters'}
                  </button>
                </div>
                
                {advancedOptions && (
                  <div className="advanced-options">
                    <div className="param-grid">
                      <div className="param-group">
                        <label htmlFor="max-tokens">Max Tokens: <span className="value-display">{maxNewTokens}</span></label>
                        <input
                          id="max-tokens"
                          type="range"
                          min="100"
                          max="4096"
                          step="1"
                          value={maxNewTokens}
                          onChange={(e) => setMaxNewTokens(parseInt(e.target.value))}
                        />
                      </div>
                      
                      <div className="param-group">
                        <label htmlFor="temperature">Temperature: <span className="value-display">{temperature.toFixed(2)}</span></label>
                        <input
                          id="temperature"
                          type="range"
                          min="0"
                          max="2"
                          step="0.05"
                          value={temperature}
                          onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="param-group">
                        <label htmlFor="top-p">Top P: <span className="value-display">{topP.toFixed(2)}</span></label>
                        <input
                          id="top-p"
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={topP}
                          onChange={(e) => setTopP(parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="param-group">
                        <label htmlFor="repetition-penalty">Repetition Penalty: <span className="value-display">{repetitionPenalty.toFixed(2)}</span></label>
                        <input
                          id="repetition-penalty"
                          type="range"
                          min="1"
                          max="2"
                          step="0.05"
                          value={repetitionPenalty}
                          onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="param-group">
                        <label htmlFor="speed-adjustment">Speed: <span className="value-display">{speedAdjustment.toFixed(2)}x</span></label>
                        <input
                          id="speed-adjustment"
                          type="range"
                          min="0.5"
                          max="2"
                          step="0.05"
                          value={speedAdjustment}
                          onChange={(e) => setSpeedAdjustment(parseFloat(e.target.value))}
                        />
                      </div>
                      
                      <div className="param-group">
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
                    disabled={isGenerating || !text.trim() || !speakerId}
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
            </div>
            
            {/* Right Column - Output */}
            <div className="output-column">
              <div className={`audio-output-section ${audioUrl ? 'active' : ''}`}>
                <h2>Generated Audio</h2>
                
                {audioUrl ? (
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
                ) : (
                  <div className="placeholder-container">
                    {isGenerating ? (
                      <div className="processing-indicator">
                        <div className="large-spinner"></div>
                        <div className="processing-text">Processing audio...</div>
                      </div>
                    ) : (
                      <div className="placeholder-message">
                        Generated audio visualization will appear here
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Sample Sentences at Bottom */}
          <div className="sample-sentences-section">
            <h2>Sample Hindi Sentences</h2>
            <div className="sentence-carousel">
              {SAMPLE_HINDI_SENTENCES.map((sentence, index) => (
                <div
                  key={index}
                  className="sample-pill"
                  onClick={() => selectSampleSentence(sentence)}
                >
                  {sentence}
                </div>
              ))}
            </div>
          </div>
          
          <footer className="minimal-footer">
            <p>Built by <a href="https://www.dheemanthreddy.com/" target="_blank" rel="noopener noreferrer">Dheemanth Reddy</a> | <a href="https://www.linkedin.com/in/bharath-kumar92" target="_blank" rel="noopener noreferrer">Bharath Kumar</a></p>
          </footer>
        </div>
      </div>
    </div>
  );
}

export default App; 
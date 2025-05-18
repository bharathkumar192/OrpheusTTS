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
      // First, clean up any previous instance and event listeners
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
      
      if (audioRef.current) {
        // Remove all existing event listeners to prevent duplicates
        const oldAudio = audioRef.current;
        oldAudio.pause();
        oldAudio.removeAttribute('src');
        oldAudio.load();
        
        // Clean up any existing event listeners
        const newAudio = oldAudio.cloneNode(false);
        oldAudio.parentNode.replaceChild(newAudio, oldAudio);
        audioRef.current = newAudio;
        
        // Set the new source
        audioRef.current.src = audioUrl;
      }
      
      // Create new WaveSurfer instance
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
      });
      
      wavesurfer.load(audioUrl);
      wavesurferRef.current = wavesurfer;
      
      // Wait for WaveSurfer to be ready before setting up event handlers
      wavesurfer.on('ready', () => {
        if (audioRef.current) {
          // Sync audio element to wavesurfer
          audioRef.current.addEventListener('play', () => {
            if (!wavesurfer.isPlaying()) wavesurfer.play();
          });
          
          audioRef.current.addEventListener('pause', () => {
            if (wavesurfer.isPlaying()) wavesurfer.pause();
          });
          
          audioRef.current.addEventListener('seeked', () => {
            const currentTime = audioRef.current.currentTime;
            const duration = audioRef.current.duration;
            if (duration > 0) {
              wavesurfer.seekTo(currentTime / duration);
            }
          });
          
          // Sync wavesurfer to audio element
          wavesurfer.on('play', () => {
            if (audioRef.current.paused) audioRef.current.play();
          });
          
          wavesurfer.on('pause', () => {
            if (!audioRef.current.paused) audioRef.current.pause();
          });
          
          wavesurfer.on('seek', (progress) => {
            if (audioRef.current.duration) {
              audioRef.current.currentTime = progress * audioRef.current.duration;
            }
          });
          
          // Start playback with a slight delay to ensure everything is loaded
          setTimeout(() => {
            try {
              audioRef.current.play()
                .catch(err => console.log("Autoplay prevented:", err));
            } catch (e) {
              console.log("Error during autoplay:", e);
            }
          }, 300);
        }
      });
      
      // Cleanup for this effect
      return () => {
        if (wavesurferRef.current) {
          wavesurferRef.current.destroy();
          wavesurferRef.current = null;
        }
      };
    }
  }, [audioUrl]);
  
  // Fetch API info (including available speakers)
  const fetchApiInfo = async () => {
    try {
      console.log("Fetching API info from:", config.apiUrl);
      const response = await fetch(`${config.apiUrl}/`, {
        headers: {
          'Accept': 'application/json',
          'Cache-Control': 'no-cache'
        }
      });
      
      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }
      
      const data = await response.json();
      console.log("API info received:", data);
      setApiInfo(data);
      
      // Check if the API returned the expected speaker data
      if (data.available_speakers && Array.isArray(data.available_speakers) && data.available_speakers.length > 0) {
        console.log("Available speakers from API:", data.available_speakers);
        setAvailableSpeakers(data.available_speakers);
        setDefaultSpeaker(data.default_speaker || data.available_speakers[0]);
        
        // Set the speakerId if it's not already set
        setSpeakerId(prevSpeakerId => {
          if (!prevSpeakerId) {
            console.log("Setting speaker ID to:", data.default_speaker || data.available_speakers[0]);
            return data.default_speaker || data.available_speakers[0];
          }
          return prevSpeakerId;
        });
      } else {
        console.warn("API response missing expected speaker data, using fallback speakers");
        // Use fallback values
        setFallbackSpeakers();
      }
      return data;
    } catch (error) {
      console.error('Error fetching API info:', error);
      // Fallback to default speakers if API info can't be fetched
      setFallbackSpeakers();
      return null;
    }
  };
  
  // Helper function to set default speakers when API fails
  const setFallbackSpeakers = () => {
    const fallbackSpeakers = ['shayana', 'raju'];
    console.log("Using fallback speakers:", fallbackSpeakers);
    setAvailableSpeakers(fallbackSpeakers);
    setDefaultSpeaker('shayana');
    setSpeakerId(prevSpeakerId => {
      if (!prevSpeakerId) {
        return 'shayana';
      }
      return prevSpeakerId;
    });
  };
  
  // Check API health status
  const checkHealth = async () => {
    setIsWarmingUp(true);
    
    try {
      console.log("Checking API health at:", `${config.apiUrl}/health`);
      const response = await fetch(`${config.apiUrl}/health`, {
        headers: {
          'Accept': 'application/json',
          'Cache-Control': 'no-cache'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Health check failed with status ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Health check response:", data);
      setHealthStatus(data);
      setIsHealthy(data.status === 'healthy');
    } catch (error) {
      console.error('Health check failed:', error);
      setHealthStatus({ error: error.message, status: 'unhealthy' });
      setIsHealthy(false);
    } finally {
      setIsWarmingUp(false);
      
      // Always try to fetch API info, even if health check fails
      try {
        await fetchApiInfo();
      } catch (error) {
        console.error("Failed to fetch API info after health check:", error);
      }
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
    
    // If we have an existing audio, pause it and clear its source
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.removeAttribute('src');
      audioRef.current.load();
    }
    
    // If we have an existing wavesurfer, destroy it
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
      wavesurferRef.current = null;
    }
    
    // Reset the audio URL to null to ensure the useEffect will trigger properly
    setAudioUrl(null);
    
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
      
      // Revoke any existing URL object to prevent memory leaks
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
      
      const url = URL.createObjectURL(audioBlob);
      
      // Set the new audio URL, which will trigger the useEffect hook
      setAudioUrl(url);
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
                  <div className="select-with-refresh">
                    <select 
                      id="speaker-select"
                      value={speakerId}
                      onChange={(e) => setSpeakerId(e.target.value)}
                      disabled={availableSpeakers.length === 0 || isWarmingUp}
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
                    <button 
                      className="refresh-button" 
                      onClick={fetchApiInfo} 
                      title="Refresh speakers list"
                      disabled={isWarmingUp}
                    >
                      ⟳
                    </button>
                  </div>
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
                  
                  {/* Debug info for API connection issues */}
                  {availableSpeakers.length === 0 && !isWarmingUp && (
                    <div className="debug-info">
                      <p>Having trouble connecting to the API? Try these steps:</p>
                      <ol>
                        <li>Click the refresh button next to the voice dropdown</li>
                        <li>Check if your browser is blocking cross-origin requests</li>
                        <li>Ensure the API is running (check Server Status)</li>
                      </ol>
                      <button 
                        className="manual-fallback-btn"
                        onClick={setFallbackSpeakers}
                      >
                        Use Fallback Speakers
                      </button>
                    </div>
                  )}
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
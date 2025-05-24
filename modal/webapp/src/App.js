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

// New Speaker Metadata Map
// Internal IDs (keys) map to display names and descriptions
const SPEAKER_METADATA_MAP = {
  'aisha': { displayName: 'Charu Soft', defaultDescription: 'Warm and intimate voice', gender: 'female' },
  'anika': { displayName: 'Ishana Spark', defaultDescription: 'Energetic and friendly voice', gender: 'female' },
  'arfa': { displayName: 'Kyra Prime', defaultDescription: 'Confident and clear voice', gender: 'female' },
  'asmr': { displayName: 'Mohini Whispers', defaultDescription: 'Soft and calming whisper voice', gender: 'special' },
  'nikita': { displayName: 'Keerti Joy', defaultDescription: 'Bright and cheerful voice', gender: 'female' },
  'raju': { displayName: 'Varun Chat', defaultDescription: 'Casual and approachable voice', gender: 'male' },
  'rhea': { displayName: 'Soumya Calm', defaultDescription: 'Sultry and soothing voice', gender: 'female' },
  'ruhaan': { displayName: 'Agastya Impact', defaultDescription: 'Deep and authoritative voice', gender: 'male' },
  'sangeeta': { displayName: 'Maitri Connect', defaultDescription: 'Gentle and nurturing voice', gender: 'female' }, // Mapped from user's "Sia"
  'shayana': { displayName: 'Vinaya Assist', defaultDescription: 'Polite and helpful voice (default)', gender: 'female' }
};


function App() {
  // State variables
  const [text, setText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [speakerId, setSpeakerId] = useState(''); // Stores internal ID, e.g., "shayana"
  const [availableSpeakers, setAvailableSpeakers] = useState([]); // Stores internal IDs
  const [defaultSpeaker, setDefaultSpeaker] = useState(''); // Stores internal ID
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
  const [speakerDetails, setSpeakerDetails] = useState(null); // From /speakers API endpoint
  
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
        wavesurferRef.current = null;
      }
      
      if (audioRef.current) {
        const oldAudio = audioRef.current;
        oldAudio.pause();
        oldAudio.removeAttribute('src');
        oldAudio.load();
        
        const newAudio = oldAudio.cloneNode(false);
        oldAudio.parentNode.replaceChild(newAudio, oldAudio);
        audioRef.current = newAudio;
        audioRef.current.src = audioUrl;
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
      });
      
      wavesurfer.load(audioUrl);
      wavesurferRef.current = wavesurfer;
      
      wavesurfer.on('ready', () => {
        if (audioRef.current) {
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
          setTimeout(() => {
            try {
              audioRef.current.play().catch(err => console.log("Autoplay prevented:", err));
            } catch (e) {
              console.log("Error during autoplay:", e);
            }
          }, 300);
        }
      });
      
      return () => {
        if (wavesurferRef.current) {
          wavesurferRef.current.destroy();
          wavesurferRef.current = null;
        }
      };
    }
  }, [audioUrl]);

  // Enhanced speaker metadata function
  const getSpeakerMetadata = (internalSpeakerId) => {
    const localMeta = SPEAKER_METADATA_MAP[internalSpeakerId] || { 
      displayName: internalSpeakerId.charAt(0).toUpperCase() + internalSpeakerId.slice(1), 
      defaultDescription: 'Hindi voice', 
      gender: 'unknown' 
    };

    let apiDescription = null;
    if (speakerDetails && speakerDetails.speaker_descriptions && speakerDetails.speaker_descriptions[internalSpeakerId]) {
      apiDescription = speakerDetails.speaker_descriptions[internalSpeakerId];
    }
    
    const descriptionToUse = apiDescription || localMeta.defaultDescription;
    
    // Gender primarily from local map, API description can refine if needed but local map is source of truth for grouping
    let genderToUse = localMeta.gender;
    // Example: if API description has strong gender cues, you could refine genderToUse here,
    // but for simplicity, we'll rely on SPEAKER_METADATA_MAP's gender for grouping.

    return {
      name: localMeta.displayName, // Always use the mapped display name
      description: descriptionToUse,
      gender: genderToUse 
    };
  };
  
  // Fetch API info (including available speakers)
  const fetchApiInfo = async () => {
    try {
      console.log("Fetching API info from:", config.apiUrl);
      const response = await fetch(`${config.apiUrl}/`, {
        headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
      });
      if (!response.ok) throw new Error(`API returned status ${response.status}`);
      
      const data = await response.json();
      console.log("API info received:", data);
      setApiInfo(data);
      
      if (data.available_speakers && Array.isArray(data.available_speakers) && data.available_speakers.length > 0) {
        console.log("Internal Available speakers from API:", data.available_speakers);
        setAvailableSpeakers(data.available_speakers); // Stores internal IDs
        const defaultInternalSpeaker = data.default_speaker || data.available_speakers[0];
        setDefaultSpeaker(defaultInternalSpeaker); // Stores internal ID
        
        setSpeakerId(prevSpeakerId => {
          if (!prevSpeakerId || !data.available_speakers.includes(prevSpeakerId)) {
            console.log("Setting speaker internal ID to:", defaultInternalSpeaker);
            return defaultInternalSpeaker; // Set to internal ID
          }
          return prevSpeakerId;
        });
      } else {
        console.warn("API response missing expected speaker data, using fallback speakers");
        setFallbackSpeakers();
      }
      return data;
    } catch (error) {
      console.error('Error fetching API info:', error);
      setFallbackSpeakers();
      return null;
    }
  };
  
  // Helper function to set default speakers when API fails
  const setFallbackSpeakers = () => {
    const fallbackInternalSpeakers = Object.keys(SPEAKER_METADATA_MAP);
    console.log("Using fallback internal speakers:", fallbackInternalSpeakers);
    setAvailableSpeakers(fallbackInternalSpeakers);
    const defaultFallbackInternalSpeaker = 'shayana'; // Assuming 'shayana' is always a safe fallback
    setDefaultSpeaker(defaultFallbackInternalSpeaker);
    setSpeakerId(prevSpeakerId => {
      if (!prevSpeakerId || !fallbackInternalSpeakers.includes(prevSpeakerId)) {
        return defaultFallbackInternalSpeaker;
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
        headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
      });
      if (!response.ok) throw new Error(`Health check failed with status ${response.status}`);
      
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
      try {
        await fetchApiInfo();
        await fetchSpeakerDetails();
      } catch (error) {
        console.error("Failed to fetch API info/speaker details after health check:", error);
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
    
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.removeAttribute('src');
      audioRef.current.load();
    }
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
      wavesurferRef.current = null;
    }
    setAudioUrl(null);
    
    const requestBody = {
      text,
      speaker_id: speakerId, // Uses internal ID
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      if (!response.ok) throw new Error('Failed to generate speech');
      
      const audioBlob = await response.blob();
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      const url = URL.createObjectURL(audioBlob);
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
  
  // Fetch detailed speaker information from API
  const fetchSpeakerDetails = async () => {
    try {
      console.log("Fetching speaker details from:", `${config.apiUrl}/speakers`);
      const response = await fetch(`${config.apiUrl}/speakers`, {
        headers: { 'Accept': 'application/json', 'Cache-Control': 'no-cache' }
      });
      if (response.ok) {
        const data = await response.json();
        console.log("Speaker details received:", data);
        setSpeakerDetails(data); // This contains API's descriptions
        return data;
      } else {
        console.warn("Speaker details endpoint not available or failed.");
        setSpeakerDetails(null); // Clear if fetch fails
        return null;
      }
    } catch (error) {
      console.warn('Error fetching speaker details:', error);
      setSpeakerDetails(null);
      return null;
    }
  };
  
  return (
    <div className="app">
      <div className="stars"></div>
      <div className="twinkling"></div>
      
      <div className="parallax-container">
        <div className="parallax-background" ref={parallaxRef}></div>
        
        <div className="app-container">
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
          
          <div className="main-grid">
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
                      value={speakerId} // This is the internal ID
                      onChange={(e) => setSpeakerId(e.target.value)}
                      disabled={availableSpeakers.length === 0 || isWarmingUp}
                      className="enhanced-speaker-select"
                    >
                      {availableSpeakers.length > 0 ? (
                        <>
                          <optgroup label="Female Voices">
                            {availableSpeakers
                              .filter(internalSpkId => getSpeakerMetadata(internalSpkId).gender === 'female')
                              .map(internalSpkId => {
                                const metadata = getSpeakerMetadata(internalSpkId);
                                return (
                                  <option key={internalSpkId} value={internalSpkId}> {/* value is internal ID */}
                                    {metadata.name} - {metadata.description} {/* Display new name & description */}
                                  </option>
                                );
                              })
                            }
                          </optgroup>
                          <optgroup label="Male Voices">
                            {availableSpeakers
                              .filter(internalSpkId => getSpeakerMetadata(internalSpkId).gender === 'male')
                              .map(internalSpkId => {
                                const metadata = getSpeakerMetadata(internalSpkId);
                                return (
                                  <option key={internalSpkId} value={internalSpkId}>
                                    {metadata.name} - {metadata.description}
                                  </option>
                                );
                              })
                            }
                          </optgroup>
                          <optgroup label="Special Voices">
                            {availableSpeakers
                              .filter(internalSpkId => getSpeakerMetadata(internalSpkId).gender === 'special')
                              .map(internalSpkId => {
                                const metadata = getSpeakerMetadata(internalSpkId);
                                return (
                                  <option key={internalSpkId} value={internalSpkId}>
                                    {metadata.name} - {metadata.description}
                                  </option>
                                );
                              })
                            }
                          </optgroup>
                          {availableSpeakers
                            .filter(internalSpkId => !['female', 'male', 'special'].includes(getSpeakerMetadata(internalSpkId).gender))
                            .map(internalSpkId => {
                              const metadata = getSpeakerMetadata(internalSpkId);
                              return (
                                <option key={internalSpkId} value={internalSpkId}>
                                  {metadata.name} - {metadata.description}
                                </option>
                              );
                            })
                          }
                        </>
                      ) : (
                        <option value="">Loading voices...</option>
                      )}
                    </select>
                    <button 
                      className="refresh-button" 
                      onClick={() => { fetchApiInfo(); fetchSpeakerDetails(); }}
                      title="Refresh speakers list"
                      disabled={isWarmingUp}
                    >
                      ⟳
                    </button>
                  </div>
                  {speakerId && ( // speakerId is an internal ID
                    <div className="speaker-info">
                      <span className="speaker-details">
                        Selected: {getSpeakerMetadata(speakerId).name} ({getSpeakerMetadata(speakerId).description})
                      </span>
                    </div>
                  )}
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
                        <input id="max-tokens" type="range" min="100" max="4096" step="1" value={maxNewTokens} onChange={(e) => setMaxNewTokens(parseInt(e.target.value))} />
                      </div>
                      <div className="param-group">
                        <label htmlFor="temperature">Temperature: <span className="value-display">{temperature.toFixed(2)}</span></label>
                        <input id="temperature" type="range" min="0" max="2" step="0.05" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} />
                      </div>
                      <div className="param-group">
                        <label htmlFor="top-p">Top P: <span className="value-display">{topP.toFixed(2)}</span></label>
                        <input id="top-p" type="range" min="0" max="1" step="0.05" value={topP} onChange={(e) => setTopP(parseFloat(e.target.value))} />
                      </div>
                      <div className="param-group">
                        <label htmlFor="repetition-penalty">Repetition Penalty: <span className="value-display">{repetitionPenalty.toFixed(2)}</span></label>
                        <input id="repetition-penalty" type="range" min="1" max="2" step="0.05" value={repetitionPenalty} onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))} />
                      </div>
                      <div className="param-group">
                        <label htmlFor="speed-adjustment">Speed: <span className="value-display">{speedAdjustment.toFixed(2)}x</span></label>
                        <input id="speed-adjustment" type="range" min="0.5" max="2" step="0.05" value={speedAdjustment} onChange={(e) => setSpeedAdjustment(parseFloat(e.target.value))} />
                      </div>
                      <div className="param-group">
                        <label htmlFor="audio-quality">Audio Quality:</label>
                        <select id="audio-quality" value={audioQualityPreset} onChange={(e) => setAudioQualityPreset(e.target.value)}>
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
                    {isGenerating ? (<><span className="spinner"></span>Generating...</>) : ("Generate Speech")}
                  </button>
                  {availableSpeakers.length === 0 && !isWarmingUp && (
                    <div className="debug-info">
                      <p>Having trouble connecting? Try these steps:</p>
                      <ol>
                        <li>Click the refresh button ⟳ next to the voice dropdown.</li>
                        <li>Check the Server Status button above.</li>
                      </ol>
                      <button className="manual-fallback-btn" onClick={setFallbackSpeakers}>
                        Use Fallback Voices
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
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
                      <a href={audioUrl} download="maya_tts_output.wav" className="download-button">
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
          
          <div className="sample-sentences-section">
            <h2>Sample Hindi Sentences</h2>
            <div className="sentence-carousel">
              {SAMPLE_HINDI_SENTENCES.map((sentence, index) => (
                <div key={index} className="sample-pill" onClick={() => selectSampleSentence(sentence)}>
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
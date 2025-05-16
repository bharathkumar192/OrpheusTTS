import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import './App.css';
import { FaPlay, FaPause, FaDownload, FaSpinner, FaExclamationTriangle } from 'react-icons/fa'; // Keep FaGithub if used
import { FiExternalLink } from 'react-icons/fi';

// --- Configuration ---
const API_ENDPOINT = process.env.REACT_APP_TTS_API_ENDPOINT || 'https://fuchr--orpheus-tts-api-refined-v2-orpheusttsapi-run-fastapi-app.modal.run'; // Replace with your actual endpoint
const VEENA_SPEAKER_ID = 'aisha';

const PRESET_SENTENCES = [
  { id: 's1', text: 'कहाँ तक पहुँचे? मैं इंतज़ार कर रहा हूँ।' },
  { id: 's2', text: 'वीडियो कॉल कर लेंगे, ठीक है?' },
  { id: 's3', text: 'क्या तुमने आज का मेमो पढ़ा?' },
  { id: 's4', text: 'अगर ऑर्डर डिलीवर नहीं होता, तो आपका पूरा पैसा वापस कर दिया जाएगा।' },
  { id: 's5', text: 'पेट्रोल के दाम फिर बढ़ गए क्या?' },
  { id: 's6', text: 'चलो, वीकेंड में पहाड़ चलते हैं!' },
  { id: 's7', text: 'यह एक ओपन-सोर्स टेक्स्ट-टू-स्पीच मॉडल है।' },
  { id: 's8', text: 'माया रिसर्च भारत के लिए AI बना रहा है।' },
];

function App() {
  const [text, setText] = useState(PRESET_SENTENCES[0].text);
  const [selectedPreset, setSelectedPreset] = useState(PRESET_SENTENCES[0].id);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState("00:00");
  const [currentTime, setCurrentTime] = useState("00:00");

  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const audioControlRef = useRef(new Audio()); // Renamed for clarity, this is our control audio element

  useEffect(() => {
    if (!API_ENDPOINT) {
      setError('API endpoint (REACT_APP_TTS_API_ENDPOINT) is not configured. Please set it in your environment.');
      console.error('CRITICAL: REACT_APP_TTS_API_ENDPOINT is not set.');
    }
  }, []);
  
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
        // Provide the audio element that WaveSurfer will control for playback
        media: audioControlRef.current, 
      });

      wavesurfer.current.on('ready', () => {
        // Duration is now available from the controlled media element
        setDuration(formatTime(wavesurfer.current.getDuration()));
        // Auto-play if a blob was just loaded
        if (audioBlob && wavesurfer.current) { // Check audioBlob again here
          wavesurfer.current.play().catch(e => console.warn("Audio autoplay prevented on ready:", e));
        }
      });

      wavesurfer.current.on('audioprocess', (time) => {
        setCurrentTime(formatTime(time));
      });
      
      // No need for 'interaction' listener if media is correctly linked,
      // WaveSurfer handles seeking on the media element.

      wavesurfer.current.on('play', () => setIsPlaying(true));
      wavesurfer.current.on('pause', () => setIsPlaying(false));
      wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
        wavesurfer.current.seekTo(0);
        setCurrentTime("00:00"); // Reset current time display
      });
    }

    return () => {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
        wavesurfer.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // audioBlob removed from dependencies here, as loading is handled in the next effect

   // Load audio blob into WaveSurfer and the controlled Audio element
   useEffect(() => {
    if (audioBlob && wavesurfer.current) {
      const audioUrl = URL.createObjectURL(audioBlob);
      // audioControlRef.current.src = audioUrl; // WaveSurfer's load will handle this for its media element
      wavesurfer.current.load(audioUrl); // This also sets the src on audioControlRef.current
      
      return () => {
        URL.revokeObjectURL(audioUrl);
      };
    } else if (!audioBlob && wavesurfer.current) {
        wavesurfer.current.empty();
        audioControlRef.current.src = ""; // Clear src of our audio element too
        setDuration("00:00");
        setCurrentTime("00:00");
        setIsPlaying(false);
    }
  }, [audioBlob]);


  const handleTextChange = (event) => {
    setText(event.target.value);
    setSelectedPreset('');
    if (error) setError(null);
  };

  const handlePresetChange = (event) => {
    const sentenceId = event.target.value;
    setSelectedPreset(sentenceId);
    const selectedSentence = PRESET_SENTENCES.find(s => s.id === sentenceId);
    if (selectedSentence) {
      setText(selectedSentence.text);
    }
    if (error) setError(null);
  };

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
    // Setting audioBlob to null will trigger the useEffect above to clear WaveSurfer
    if (wavesurfer.current && wavesurfer.current.isPlaying()) {
        wavesurfer.current.stop();
    }
    setAudioBlob(null); 
    setIsPlaying(false);
    setCurrentTime("00:00");
    setDuration("00:00");


    const payload = {
      text: text,
      speaker_id: VEENA_SPEAKER_ID,
      output_sample_rate: 24000,
    };

    try {
      const response = await axios.post(API_ENDPOINT + '/generate', payload, {
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

  const togglePlayPause = useCallback(() => {
    if (wavesurfer.current && audioBlob) { // Ensure audioBlob is present
      wavesurfer.current.playPause();
    }
  }, [audioBlob]);


  const handleDownload = () => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `veena_tts_maya_research_${text.substring(0, 20).replace(/\s+/g, '_')}.wav`;
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
      <div className="content-wrapper">
        <header className="App-header">
          <h1>MAYA RESEARCH</h1>
          <p>Building AI for India</p>
        </header>

        <main className="App-main">
          <div className="tts-container">
            <div className="tts-controls-column">
              <h2>Veena – open-source TTS</h2>
              <p className="tagline">Lifelike Hindi speech, available to everyone.</p>

              <form onSubmit={handleSubmit} className="tts-form">
                <div className="form-group">
                  <label htmlFor="preset-select">Select a sample sentence</label>
                  <select
                    id="preset-select"
                    value={selectedPreset}
                    onChange={handlePresetChange}
                    disabled={isLoading || !API_ENDPOINT}
                    className="styled-select"
                  >
                    {PRESET_SENTENCES.map(sentence => (
                      <option key={sentence.id} value={sentence.id}>{sentence.text.substring(0,50)}...</option>
                    ))}
                    <option value="">— Type custom text below —</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="text-input">Or type your custom Hindi text (max 200 chars)</label>
                  <textarea
                    id="text-input"
                    value={text}
                    onChange={handleTextChange}
                    placeholder="यहाँ अपना हिंदी पाठ लिखें..."
                    rows="4"
                    maxLength="200"
                    className="text-input"
                    disabled={isLoading || !API_ENDPOINT}
                    aria-label="Hindi text input"
                  />
                </div>
                
                <button
                  type="submit"
                  className="generate-button"
                  disabled={isLoading || !text.trim() || !API_ENDPOINT}
                  aria-label="Generate audio"
                >
                  {isLoading ? (
                    <>
                      <FaSpinner className="spinner-icon" /> GENERATING...
                    </>
                  ) : (
                    'SYNTHESIZE SPEECH'
                  )}
                </button>
              </form>
            </div>

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
                
                <div id="waveform" ref={waveformRef}></div>
                
                {(audioBlob || isLoading) && (
                  <div className="audio-player-controls-footer">
                     <button 
                        onClick={togglePlayPause} 
                        className="play-pause-button" 
                        aria-label={isPlaying ? "Pause" : "Play"}
                        disabled={!audioBlob || isLoading} // Disable if no blob or loading
                    >
                        {isPlaying ? <FaPause /> : <FaPlay />}
                    </button>
                    <div className="time-indicators-footer">
                        <span>{currentTime}</span> / <span>{duration}</span>
                    </div>
                    <button 
                        onClick={handleDownload} 
                        className="download-button" 
                        disabled={!audioBlob || isLoading} 
                        aria-label="Download audio"
                    >
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
          {/* Ensure you have FaGithub imported if you use it here */}
          <a href="https://mayaresearch.ai" target="_blank" rel="noopener noreferrer" className="secondary-link">
            Maya Research <FiExternalLink />
          </a>
           {/* If you want a GitHub link:
           <a href="https://github.com/your-repo" target="_blank" rel="noopener noreferrer" className="github-button">
             <FaGithub /> Contribute on GitHub
           </a>
           */}
        </section>

        <footer className="App-footer">
          <p>© {new Date().getFullYear()} Maya Research</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import './App.css';
import { FaPlay, FaPause, FaDownload, FaSpinner, FaExclamationTriangle, FaGithub } from 'react-icons/fa';
import { FiExternalLink } from 'react-icons/fi';

// --- Configuration ---
const API_ENDPOINT = process.env.REACT_APP_TTS_API_ENDPOINT;
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

  const waveformRef = useRef(null); // For WaveSurfer container
  const wavesurfer = useRef(null); // For WaveSurfer instance
  const audioRef = useRef(new Audio()); // Use a detached Audio element for playback control

  useEffect(() => {
    if (!API_ENDPOINT) {
      setError('API endpoint (REACT_APP_TTS_API_ENDPOINT) is not configured. Please set it in your environment.');
      console.error('CRITICAL: REACT_APP_TTS_API_ENDPOINT is not set.');
    }
  }, []);
  
  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  // Initialize WaveSurfer
  useEffect(() => {
    if (waveformRef.current && !wavesurfer.current) {
      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'rgba(54, 255, 77, 0.3)', // Neon green with transparency
        progressColor: '#36ff4d', // Solid neon green
        cursorColor: '#ffffff',
        barWidth: 3,
        barRadius: 3,
        responsive: true,
        height: 100,
        normalize: true,
        backend: 'MediaElement', // Use MediaElement backend with our audioRef
      });

      // Sync WaveSurfer with the detached audio element
      wavesurfer.current.loadElement(audioRef.current, audioBlob ? [audioBlob] : undefined);


      wavesurfer.current.on('ready', () => {
        if (audioRef.current) {
          setDuration(formatTime(audioRef.current.duration));
          // Auto-play after ready and blob is set
          if(audioBlob) {
            audioRef.current.play().catch(e => console.warn("Audio autoplay prevented:", e));
          }
        }
      });

      wavesurfer.current.on('audioprocess', () => {
        if (audioRef.current) {
          setCurrentTime(formatTime(audioRef.current.currentTime));
        }
      });
      
      wavesurfer.current.on('interaction', () => {
         if (audioRef.current && wavesurfer.current) {
            audioRef.current.currentTime = wavesurfer.current.getCurrentTime();
         }
      });

      wavesurfer.current.on('play', () => setIsPlaying(true));
      wavesurfer.current.on('pause', () => setIsPlaying(false));
      wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
        wavesurfer.current.seekTo(0); // Reset to beginning
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
  }, []); // Run once on mount to setup WaveSurfer container

   // Load audio blob into WaveSurfer and Audio element
   useEffect(() => {
    if (audioBlob && wavesurfer.current) {
      const audioUrl = URL.createObjectURL(audioBlob);
      audioRef.current.src = audioUrl; // Set src for the detached audio element
      
      // Wavesurfer needs to be reloaded with the new audio element context if src changes
      // Or, if it's already loaded with the element, just playing it should be fine.
      // Let's ensure it's loaded with the blob data directly if possible.
      wavesurfer.current.load(audioUrl);

      return () => {
        URL.revokeObjectURL(audioUrl);
      };
    } else if (!audioBlob && wavesurfer.current) {
        // Clear waveform if no audio blob
        wavesurfer.current.empty();
        setDuration("00:00");
        setCurrentTime("00:00");
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
    setAudioBlob(null); // This will trigger useEffect to clear WaveSurfer
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
    } catch (err) {
      console.error('API Error:', err);
      let errorMessage = 'An unexpected error occurred.';
      if (err.response) {
        try {
          const errorDataText = await err.response.data.text();
          const errorJson = JSON.parse(errorDataText);
          errorMessage = errorJson.error || errorJson.detail || `Server error: ${err.response.status}`;
        } catch (parseError) {
          errorMessage = `Server error: ${err.response.status} ${err.response.statusText || ''}`;
        }
      } else if (err.request) {
        errorMessage = 'No response from server.';
      } else {
        errorMessage = err.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const togglePlayPause = useCallback(() => {
    if (wavesurfer.current && audioBlob) {
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
        {/* Lines for background effect - can be generated dynamically or be static SVGs */}
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
                
                <div className="sentence-display" title="Phonetic transcription could appear here on hover.">
                  {text || "Your synthesized text will appear here."}
                </div>
                
                <div id="waveform" ref={waveformRef}></div>
                
                {/* Controls below waveform */}
                {(audioBlob || isLoading) && (
                  <div className="audio-player-controls-footer">
                     <button 
                        onClick={togglePlayPause} 
                        className="play-pause-button" 
                        aria-label={isPlaying ? "Pause" : "Play"}
                        disabled={!audioBlob || isLoading}
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
          <a href="https://huggingface.co/spaces/sahilp/Hindi_TTS" target="_blank" rel="noopener noreferrer" className="secondary-link"> {/* Example link */}
            Model Comparison <FiExternalLink />
          </a>
          <a href="https://github.com/gitgithan/ModalTTS" target="_blank" rel="noopener noreferrer" className="github-button"> {/* Example link */}
            <FaGithub /> Contribute on GitHub
          </a>
        </section>

        <footer className="App-footer">
          <p>© {new Date().getFullYear()} Maya Research</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
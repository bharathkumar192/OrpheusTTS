import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { FaPlay, FaPause, FaDownload, FaSpinner, FaExclamationTriangle, FaGithub } from 'react-icons/fa';
import { FiExternalLink } from 'react-icons/fi'; // For model comparison link

// --- Configuration ---
const API_ENDPOINT = process.env.REACT_APP_TTS_API_ENDPOINT;
const VEENA_SPEAKER_ID = 'aisha'; // Assuming 'aisha' is the speaker ID for Veena, or adjust as needed

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
  const [audioSrc, setAudioSrc] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState("00:00");
  const [currentTime, setCurrentTime] = useState("00:00");
  const [progress, setProgress] = useState(0);

  const audioRef = useRef(null);

  useEffect(() => {
    if (!API_ENDPOINT) {
      setError('API endpoint (REACT_APP_TTS_API_ENDPOINT) is not configured. Please set it in your environment.');
      console.error('CRITICAL: REACT_APP_TTS_API_ENDPOINT is not set.');
    }
  }, []);

  const handleTextChange = (event) => {
    setText(event.target.value);
    setSelectedPreset(''); // Deselect preset if typing custom text
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
    setAudioSrc(null);
    setAudioBlob(null);
    setIsPlaying(false);
    setProgress(0);
    setCurrentTime("00:00");
    setDuration("00:00");

    const payload = {
      text: text,
      speaker_id: VEENA_SPEAKER_ID, // Using the fixed speaker ID for Veena
      // Using backend defaults for other parameters like temperature, etc.
      // Ensure app.py's TTSRequest has appropriate defaults.
      output_sample_rate: 24000, // Or let backend decide from its default
    };

    console.log("Sending payload:", payload);

    try {
      const response = await axios.post(API_ENDPOINT + '/generate', payload, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'audio/wav',
        },
        timeout: 90000, // 90 seconds timeout
      });

      const blob = new Blob([response.data], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(blob);
      setAudioBlob(blob);
      setAudioSrc(audioUrl);

    } catch (err) {
      console.error('API Error:', err);
      let errorMessage = 'An unexpected error occurred while generating audio.';
      if (err.response) {
        try {
          const errorDataText = await err.response.data.text();
          const errorJson = JSON.parse(errorDataText);
          errorMessage = errorJson.error || errorJson.detail || `Server error: ${err.response.status}`;
        } catch (parseError) {
          if (err.response.statusText) {
            errorMessage = `${err.response.status}: ${err.response.statusText}`;
          } else {
            errorMessage = `Server error: ${err.response.status}`;
          }
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

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      const setAudioData = () => {
        setDuration(formatTime(audio.duration));
        setCurrentTime(formatTime(audio.currentTime));
      }
      const setAudioTime = () => {
        setCurrentTime(formatTime(audio.currentTime));
        setProgress((audio.currentTime / audio.duration) * 100);
      }

      audio.addEventListener('loadeddata', setAudioData);
      audio.addEventListener('timeupdate', setAudioTime);
      audio.addEventListener('play', () => setIsPlaying(true));
      audio.addEventListener('pause', () => setIsPlaying(false));
      audio.addEventListener('ended', () => setIsPlaying(false));

      // Autoplay when src changes
      if (audioSrc) {
        audio.play().catch(e => console.warn("Audio autoplay prevented:", e));
      }

      return () => {
        audio.removeEventListener('loadeddata', setAudioData);
        audio.removeEventListener('timeupdate', setAudioTime);
        audio.removeEventListener('play', () => setIsPlaying(true));
        audio.removeEventListener('pause', () => setIsPlaying(false));
        audio.removeEventListener('ended', () => setIsPlaying(false));
      }
    }
  }, [audioSrc]);

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
    }
  };

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
  
  const handleSeek = (event) => {
    if (audioRef.current && audioRef.current.duration) {
        const newTime = (event.target.value / 100) * audioRef.current.duration;
        audioRef.current.currentTime = newTime;
    }
  };


  return (
    <div className="App">
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

              {audioSrc && (
                <div className="audio-player-controls">
                  <button onClick={togglePlayPause} className="play-pause-button" aria-label={isPlaying ? "Pause" : "Play"}>
                    {isPlaying ? <FaPause /> : <FaPlay />}
                  </button>
                  <div className="seek-bar-container">
                    <input 
                        type="range" 
                        min="0" 
                        max="100" 
                        value={progress} 
                        onChange={handleSeek} 
                        className="seek-bar"
                        aria-label="Audio seek bar"
                    />
                  </div>
                  <span className="time-indicator">{currentTime} / {duration}</span>
                  <button onClick={handleDownload} className="download-button" disabled={!audioBlob} aria-label="Download audio">
                    <FaDownload />
                  </button>
                </div>
              )}
              <audio ref={audioRef} src={audioSrc} style={{display: 'none'}} />
              
              {!isLoading && !audioSrc && !error && (
                 <div className="player-placeholder">
                    Synthesize speech to listen.
                 </div>
              )}

              <p className="quality-notice">Quality on par with ElevenLabs — fully open-source.</p>
            </div>
          </div>
        </div>
      </main>

      <div className="section-separator"></div>

      <section className="secondary-content">
        <a href="https://example.com/model-comparison" target="_blank" rel="noopener noreferrer" className="secondary-link">
          Model Comparison <FiExternalLink />
        </a>
        <a href="https://github.com/your-repo/veena-tts" target="_blank" rel="noopener noreferrer" className="github-button">
          <FaGithub /> Contribute on GitHub
        </a>
      </section>

      <footer className="App-footer">
        <p>© {new Date().getFullYear()} Maya Research</p>
      </footer>
    </div>
  );
}

export default App;
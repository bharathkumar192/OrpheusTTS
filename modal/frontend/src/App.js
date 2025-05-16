import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // We'll refine this
import { FaPlay, FaSpinner, FaExclamationTriangle } from 'react-icons/fa'; // For icons

// --- Configuration ---
const API_ENDPOINT = process.env.REACT_APP_TTS_API_ENDPOINT;

// Default values for parameters - can be adjusted
const DEFAULT_SPEAKER = 'aisha';
const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_REPETITION_PENALTY = 1.1;
const DEFAULT_SPEED_ADJUSTMENT = 1.0;
const DEFAULT_OUTPUT_SAMPLE_RATE = 24000;

// Available speakers (expand this list as you add more)
const AVAILABLE_SPEAKERS = [
  { id: 'aisha', name: 'Aisha (Default)' },
  { id: 'anika', name: 'Anika' }, // Example, add your actual speaker IDs
  { id: 'ivanna', name: 'Ivanna' }, // Example
  // Add more speakers here
];

function App() {
  const [text, setText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [audioSrc, setAudioSrc] = useState(null);
  const audioRef = useRef(null);

  // TTS Parameters States
  const [speakerId, setSpeakerId] = useState(DEFAULT_SPEAKER);
  const [temperature, setTemperature] = useState(DEFAULT_TEMPERATURE);
  const [repetitionPenalty, setRepetitionPenalty] = useState(DEFAULT_REPETITION_PENALTY);
  const [speedAdjustment, setSpeedAdjustment] = useState(DEFAULT_SPEED_ADJUSTMENT);
  const [outputSampleRate, setOutputSampleRate] = useState(DEFAULT_OUTPUT_SAMPLE_RATE);

  const [showAdvanced, setShowAdvanced] = useState(false);


  useEffect(() => {
    if (!API_ENDPOINT) {
      setError('API endpoint (REACT_APP_TTS_API_ENDPOINT) is not configured. Please set it in your environment.');
      console.error('CRITICAL: REACT_APP_TTS_API_ENDPOINT is not set.');
    }
  }, []);

  const handleTextChange = (event) => {
    setText(event.target.value);
    if (error) setError(null); // Clear error when user types
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!API_ENDPOINT) {
      setError('API endpoint is not configured.');
      return;
    }
    if (!text.trim()) {
      setError('Please enter some text to synthesize.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setAudioSrc(null); // Clear previous audio

    const payload = {
      text: text,
      speaker_id: speakerId,
      temperature: parseFloat(temperature),
      repetition_penalty: parseFloat(repetitionPenalty),
      speed_adjustment: parseFloat(speedAdjustment),
      output_sample_rate: parseInt(outputSampleRate, 10),
      // Add other parameters from TTSRequest model if needed
      // max_new_tokens: 2048, 
      // top_p: 0.9,
      // audio_quality_preset: "high",
      // seed: null,
      // early_stopping: true,
    };

    console.log("Sending payload:", payload);

    try {
      const response = await axios.post(API_ENDPOINT + '/generate', payload, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'audio/wav',
        },
        timeout: 90000, // 90 seconds timeout for potentially long generations
      });

      const audioBlob = new Blob([response.data], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      setAudioSrc(audioUrl);

    } catch (err) {
      console.error('API Error:', err);
      let errorMessage = 'An unexpected error occurred while generating audio.';
      if (err.response) {
        // Attempt to parse error from server if it's JSON blob
        try {
          const errorDataText = await err.response.data.text();
          const errorJson = JSON.parse(errorDataText);
          errorMessage = errorJson.error || errorJson.detail || `Server error: ${err.response.status}`;
        } catch (parseError) {
          // Fallback if error response is not JSON or cannot be parsed
          if (err.response.statusText) {
            errorMessage = `${err.response.status}: ${err.response.statusText}`;
          } else {
            errorMessage = `Server error: ${err.response.status}`;
          }
        }
      } else if (err.request) {
        errorMessage = 'No response from the server. It might be down or unreachable. Check console for details.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // Automatically play audio when src changes and it's not the initial null state
  useEffect(() => {
    if (audioSrc && audioRef.current) {
      audioRef.current.play().catch(e => console.warn("Audio autoplay prevented:", e));
    }
  }, [audioSrc]);

  const ParameterSlider = ({ label, value, onChange, min, max, step, unit = "" }) => (
    <div className="parameter-slider">
      <label htmlFor={label.toLowerCase().replace(" ", "-")}>{label}: {value}{unit}</label>
      <input
        type="range"
        id={label.toLowerCase().replace(" ", "-")}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={isLoading}
      />
    </div>
  );

  return (
    <div className="App">
      <header className="App-header">
        <h1>हिन्दी वाणी <span className="beta-tag">(Hindi TTS)</span></h1>
        <p>Convert Hindi text into natural-sounding speech.</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="tts-form">
          <textarea
            value={text}
            onChange={handleTextChange}
            placeholder="यहाँ अपना हिंदी पाठ लिखें (Enter your Hindi text here)..."
            rows="5"
            className="text-input"
            disabled={isLoading || !API_ENDPOINT}
            aria-label="Hindi text input"
          />

          <div className="parameters-grid">
            <div className="parameter-item">
              <label htmlFor="speakerId">Speaker:</label>
              <select 
                id="speakerId" 
                value={speakerId} 
                onChange={(e) => setSpeakerId(e.target.value)} 
                disabled={isLoading}
                className="styled-select"
              >
                {AVAILABLE_SPEAKERS.map(speaker => (
                  <option key={speaker.id} value={speaker.id}>{speaker.name}</option>
                ))}
              </select>
            </div>

            <div className="parameter-item">
              <label htmlFor="outputSampleRate">Sample Rate (Hz):</label>
              <select 
                id="outputSampleRate" 
                value={outputSampleRate} 
                onChange={(e) => setOutputSampleRate(parseInt(e.target.value, 10))} 
                disabled={isLoading}
                className="styled-select"
              >
                <option value="16000">16000 Hz (Low)</option>
                <option value="24000">24000 Hz (Standard)</option>
                <option value="48000">48000 Hz (High)</option> 
              </select>
            </div>
          </div>
          
          <button 
            type="button" 
            onClick={() => setShowAdvanced(!showAdvanced)} 
            className="toggle-advanced-button"
          >
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
          </button>

          {showAdvanced && (
            <div className="advanced-parameters">
              <ParameterSlider 
                label="Temperature" 
                value={temperature} 
                onChange={setTemperature} 
                min="0.1" max="1.5" step="0.05" 
              />
              <ParameterSlider 
                label="Repetition Penalty" 
                value={repetitionPenalty} 
                onChange={setRepetitionPenalty} 
                min="1.0" max="1.5" step="0.05" 
              />
              <ParameterSlider 
                label="Speed Adjustment" 
                value={speedAdjustment} 
                onChange={setSpeedAdjustment} 
                min="0.5" max="1.5" step="0.05" 
              />
            </div>
          )}

          <button 
            type="submit" 
            className="generate-button" 
            disabled={isLoading || !text.trim() || !API_ENDPOINT}
            aria-label="Generate audio"
          >
            {isLoading ? (
              <>
                <FaSpinner className="spinner-icon" /> Generating...
              </>
            ) : (
              'Generate Audio'
            )}
          </button>
        </form>

        {error && (
          <div className="error-message" role="alert">
            <FaExclamationTriangle style={{ marginRight: '10px', verticalAlign: 'middle' }}/>
            <strong>Error:</strong> {error}
          </div>
        )}

        {audioSrc && (
          <div className="audio-player-container">
            <h2>Generated Speech:</h2>
            <audio controls src={audioSrc} ref={audioRef} className="audio-player" aria-label="Generated audio player">
              Your browser does not support the audio element.
            </audio>
            <button 
              onClick={() => audioRef.current && audioRef.current.play()} 
              className="play-button"
              disabled={!audioSrc || isLoading}
              aria-label="Play generated audio"
            >
              <FaPlay style={{ marginRight: '8px' }} /> Play Again
            </button>
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>Crafted with Orpheus TTS & Modal. हिन्दी वाणी © 2024</p>
      </footer>
    </div>
  );
}

export default App;
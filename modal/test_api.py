import requests
import argparse
import os
import time

def test_health(api_url):
    """Test the API health endpoint"""
    url = f"{api_url}/health"
    print(f"Testing health endpoint: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"Health check successful: {response.json()}")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def generate_speech(api_url, text, output_file, streaming=False, **kwargs):
    """Generate speech using the API"""
    url = f"{api_url}/generate"
    print(f"Generating speech at: {url}")
    
    # Prepare request payload
    payload = {
        "text": text,
        "speaker": "aisha",
        "stream": streaming,
        "temperature": 1.0,
        "repetition_penalty": 1.1,
        "top_p": 0.9,
        **kwargs
    }
    
    print(f"Request parameters: {payload}")
    start_time = time.time()
    
    try:
        if streaming:
            # Handle streaming response
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            # Save streamed content
            with open(output_file, 'wb') as f:
                for i, chunk in enumerate(response.iter_content(chunk_size=8192)):
                    if chunk:
                        print(f"Received chunk {i+1} ({len(chunk)} bytes)")
                        f.write(chunk)
        else:
            # Standard response
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # Save audio to file
            with open(output_file, 'wb') as f:
                f.write(response.content)
        
        elapsed_time = time.time() - start_time
        print(f"Speech generated successfully in {elapsed_time:.2f} seconds")
        print(f"Saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Speech generation failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return False

def test_different_parameters(api_url, base_text):
    """Test different parameter combinations"""
    test_cases = [
        {
            "name": "default",
            "params": {}
        },
        {
            "name": "low_quality",
            "params": {
                "audio_quality": "low",
                "output_sample_rate": 16000
            }
        },
        {
            "name": "fast_speech",
            "params": {
                "speed_adjustment": 1.2
            }
        },
        {
            "name": "slow_speech",
            "params": {
                "speed_adjustment": 0.8
            }
        },
        {
            "name": "creative",
            "params": {
                "temperature": 1.5,
                "top_p": 0.95
            }
        },
        {
            "name": "streaming",
            "params": {
                "stream": True
            }
        }
    ]
    
    results = {}
    for test in test_cases:
        output_file = f"test_output_{test['name']}.wav"
        print(f"\n----- Testing {test['name']} -----")
        success = generate_speech(
            api_url, 
            base_text, 
            output_file, 
            **test["params"]
        )
        results[test['name']] = {
            "success": success,
            "output_file": output_file if success else None
        }
    
    print("\n----- Test Results Summary -----")
    for name, result in results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        file_info = f" -> {result['output_file']}" if result["success"] else ""
        print(f"{name}: {status}{file_info}")

def main():
    parser = argparse.ArgumentParser(description="Test the Orpheus TTS API")
    parser.add_argument("--url", type=str, required=True, help="The base URL of the API (e.g., https://username--orpheus-tts-backend-fastapi-app.modal.run)")
    parser.add_argument("--text", type=str, default="मैं ये सोच रहा था कि ऑनलाइन इतनी इनफॉर्मेशन अवेलेबल होने के बावजूद, कभी कभी ऑथेंटिक सोर्स ढूंढना कितना मुश्किल हो जाता है आज भी।", help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--run-all-tests", action="store_true", help="Run tests with different parameter combinations")
    
    args = parser.parse_args()
    
    # Make sure the API URL doesn't end with a slash
    api_url = args.url.rstrip('/')
    
    # First test health endpoint
    if not test_health(api_url):
        print("Skipping speech generation because health check failed")
        return
    
    # Run all tests or just a single test
    if args.run_all_tests:
        test_different_parameters(api_url, args.text)
    else:
        generate_speech(api_url, args.text, args.output, args.stream)

if __name__ == "__main__":
    main() 
import requests
import os
import json

# --- CONFIGURATION ---
# Make sure to set your RunPod API Key as an environment variable
# or replace os.getenv("RUNPOD_API_KEY") directly with your key.
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "") # Replace with your actual key
ENDPOINT_ID = "" # This is your endpoint ID from the URL
RUNPOD_API_BASE = "https://api.runpod.ai/v2"

# The endpoint URL for synchronous execution
url = f"{RUNPOD_API_BASE}/{ENDPOINT_ID}/runsync"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {RUNPOD_API_KEY}' # Note the f-string and "Bearer " prefix
}

# --- YOUR INPUT DATA ---
# This needs to match the fields in your TTSRequest Pydantic model in app_runpod.py
# All fields from TTSRequest are available here.
input_data = {
    "input": {
        "text": "Hello, this is a test of the RunPod serverless text to speech.",
        "speaker": "aisha",                 # Default is 'aisha', you can change if you have others
        "max_tokens": 2048,               # Default is 2048
        "temperature": 1.0,               # Default is 1.0
        "repetition_penalty": 1.1,        # Default is 1.1
        "top_p": 0.9,                     # Default is 0.9
        "stream": False,                  # Set to False for /runsync to get the full audio back
        "output_sample_rate": 24000,      # Default is 24000
        "audio_quality": "high",          # Default is "high" (options: low, medium, high)
        "speed_adjustment": 1.0,          # Default is 1.0 (0.5-1.5)
        "seed": None,                     # Optional: e.g., 42 for reproducible results
        "early_stopping": True            # Default is True
    }
}

print(f"Sending request to: {url}")
print(f"With data: {json.dumps(input_data, indent=2)}")

try:
    response = requests.post(url, headers=headers, json=input_data, timeout=300) # 300 seconds timeout

    print(f"\nResponse Status Code: {response.status_code}")

    if response.status_code == 200:
        response_json = response.json()
        print("Full JSON Response:")
        print(json.dumps(response_json, indent=2))

        # Your app_runpod.py handler for non-streaming directly returns wav_bytes.
        # The RunPod serverless system will wrap this.
        # The actual audio data might be in response_json.get("output")
        # or if it's direct bytes, response.content should be used.

        # Check if the output is what your handler returns.
        # Your handler, for non-streaming, returns raw wav_bytes.
        # The RunPod /runsync endpoint should return a JSON like:
        # {
        #   "id": "job-id",
        #   "status": "COMPLETED",
        #   "output": <your_handler_output_if_json_serializable_or_base64_encoded>,
        #   "executionTime": ...,
        #   "delayTime": ...
        # }
        # Since your handler returns raw bytes, and RunPod /runsync expects JSON,
        # RunPod *might* automatically base64 encode the binary output.
        # Or, it might be downloadable via a link if the Content-Type was correctly set.
        # Let's check the structure.

        job_output = response_json.get("output")
        job_status = response_json.get("status")

        if job_status == "COMPLETED":
            print("\nJob completed successfully!")
            # If your handler returned raw bytes, and RunPod expects JSON,
            # it might automatically base64 encode it.
            # Or it might provide it in a different way.
            # For now, let's assume it's in 'output' and might be base64.
            # If your handler returns a dictionary like {"audio_bytes": ..., "content_type": "audio/wav"},
            # then job_output would be that dictionary.
            # Your current app_runpod.py non-streaming handler returns `wav_bytes` directly.

            # The Runpod SDK usually handles the raw bytes output by making it downloadable
            # or by returning it in a field if the client (like the Python runpod client) can handle it.
            # When using requests directly, if the API returns raw bytes with an appropriate
            # Content-Type (e.g., audio/wav), then `response.content` would hold the bytes.
            # However, the /runsync endpoint typically wraps the output in a JSON.
            # Let's assume for now your app_runpod.py's raw byte output is handled by RunPod and
            # might be available in a specific way or needs client-side interpretation.

            # The simplest is to check if 'output' exists and print its type.
            if job_output is not None:
                print(f"Type of job_output: {type(job_output)}")
                # If it's a string, it might be base64 encoded audio.
                # If it's a dictionary, inspect its keys.
                # If you were expecting raw audio bytes directly in the response body (not JSON wrapped),
                # you would use response.content and check response.headers['Content-Type'].
                # But /runsync typically gives JSON.

                # Try to save the raw response content if Content-Type suggests audio
                if 'audio/wav' in response.headers.get('Content-Type', '').lower():
                    with open("output.wav", "wb") as f:
                        f.write(response.content)
                    print("Audio content detected and saved to output.wav (if Content-Type was audio/wav)")
                elif isinstance(job_output, str):
                    print("Output is a string. If it's audio, it might be base64 encoded.")
                    # You would need to base64 decode it here.
                else:
                    print("Job output received. Inspect the JSON response above.")

            else:
                print("Job output is None or not present in the expected field.")

        elif job_status == "FAILED":
            print("\nJob failed!")
            error_details = response_json.get("error", "No error details provided.")
            print(f"Error: {error_details}")
        else:
            print(f"\nJob status: {job_status}")

    else:
        print("Error making request:")
        try:
            print(json.dumps(response.json(), indent=2)) # Print error response if JSON
        except requests.exceptions.JSONDecodeError:
            print(response.text) # Print raw text if not JSON

except requests.exceptions.Timeout:
    print("Request timed out. The TTS generation might be taking longer than 300 seconds.")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
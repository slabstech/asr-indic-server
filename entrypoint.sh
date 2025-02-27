#!/bin/bash

# Start the ASR indic server
/start_asr_indic_server.sh &

# Wait for the service to be healthy
until $(curl --output /dev/null --silent --head --fail http://localhost:7860/); do
    echo "Waiting for ASR indic server to be healthy..."
    sleep 5
done

# Run the curl command
curl -X 'POST' \
'http://localhost:8000/transcribe/' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@samples/kannada_sample_1.wav;type=audio/x-wav'

# Keep the container running
tail -f /dev/null
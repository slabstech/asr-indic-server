from locust import HttpUser, task, between
import os

class TranscribeUser(HttpUser):
    wait_time = between(1, 5)  # Wait time between tasks

    @task
    def transcribe_audio(self):
        audio_file_path = "./kannada_sample_1.wav"
        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': ('kannada_query_infer.wav', audio_file, 'audio/x-wav')}
            headers = {
                'accept': 'application/json'
            }
            response = self.client.post("http://localhost:8000/transcribe/", files=files, headers=headers)
            if response.status_code == 200:
                print("Success:", response.json())
            else:
                print("Failed:", response.status_code, response.text)

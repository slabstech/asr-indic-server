sudo apt-get update
sudo apt-get upgrade

sudo apt-get install -y python3-venv
sudo apt-get install -y python3-pip
sudo apt-get install -y ffmpeg


python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

uvicorn src.asr_indic_server.asr_api:app --reload


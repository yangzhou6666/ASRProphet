pip install -r requirements.txt
python -c "import nltk; nltk.download('brown')"
python -c "import nltk; nltk.download('names')"
apt update
apt install sox -y
apt install libsndfile1 -y

cd ../
apt update
apt install git -y
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html


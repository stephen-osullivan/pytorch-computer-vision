install-torch:
	pip install --upgrade pip &&\
		 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

install:
	pip install --upgrade pip &&\
		 pip install -r requirements.txt
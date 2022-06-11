setup :
	#

build :
	# docker build -t bomberman .

run :
	# docker run bomberman

run-ExplosiveAI :
	cd ExplosiveAI; python fight.py

install-python3.8 :
	sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
	curl https://pyenv.run | bash
	~/.pyenv/bin/pyenv install 3.8.13
	~/.pyenv/bin/pyenv global 3.8.13

install-requirements :
	pip freeze --exclude bomberman > requirements.txt
	pip install -r requirements.txt

download-ExplosiveAI :
	rm -rf ExplosiveAI
	git clone https://github.com/42-AI/ExplosiveAI.git
	mkdir ExplosiveAI/simulator
	cd ExplosiveAI/simulator; gdown https://drive.google.com/file/d/11rBSVLNfXvsAdXlUnqWcn05cIzaTz6td/view?usp=sharing --fuzzy -O build.zip
	cd ExplosiveAI/simulator; unzip build.zip; mv speed_headless_build build

install-ExplosiveAI :
	cd ExplosiveAI; chmod +x simulator/build/bomber.x86_64
	cd ExplosiveAI; python3 -m pip install build
	cd ExplosiveAI; python3 -m build
	cd ExplosiveAI; python3 -m pip install dist/bomberman-0.1.0-py3-none-any.whl

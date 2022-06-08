setup :
	rm -rf environment
	git clone https://github.com/42-AI/ExplosiveAI.git environment
	mkdir environment/simulator
	#
	# ████████╗ ██████╗ ██████╗  ██████╗
	# ╚══██╔══╝██╔═══██╗██╔══██╗██╔═══██╗
	#    ██║   ██║   ██║██║  ██║██║   ██║
	#    ██║   ██║   ██║██║  ██║██║   ██║
	#    ██║   ╚██████╔╝██████╔╝╚██████╔╝
	#    ╚═╝    ╚═════╝ ╚═════╝  ╚═════╝
	#
	# 1) Download a build of the game from : https://drive.google.com/drive/u/0/folders/1w30nmNl8dp4fwIm3kMLBl8Lm-mGfVgYp
	#
	# 2) unzip and move to ExplosiveAI/simulator/build
	#
	# 3) Run `make docker`

build :
	docker build -t bomberman .

run :
	docker run bomberman

run :
	python stable_baseline_agent/agent/train.py

setup : install-python3.8 download-ExplosiveAI install-ExplosiveAI

install-python3.8 :
	sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
	-curl https://pyenv.run | bash
	-~/.pyenv/bin/pyenv install 3.8.13
	~/.pyenv/bin/pyenv global 3.8.13

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

run-ExplosiveAI :
	cd ExplosiveAI; python fight.py

docker-build : docker-build-ExplosiveAI
	docker build -t bomberman .

docker-build-requirements :
	docker build -t bomberman-requirements -f Dockerfile-requirements .

docker-build-ExplosiveAI : docker-build-requirements
	docker build -t bomberman-explosiveai -f Dockerfile-ExplosiveAI .

docker-run : docker-build
	### /!\ require env var: WANDB_API_KEY /!\ ###
	docker run -e "WANDB_API_KEY=${WANDB_API_KEY}" bomberman

docker-push : docker-build
	### /!\ ###
	# require env var: DOCKER_ID
	# require to be logged in docker: `docker login`
	### /!\ ###
	docker tag bomberman ${DOCKER_ID}/bomberman
	docker push ${DOCKER_ID}/bomberman

workstation-login :
	export WS_LOGIN_TOKEN=` \
		curl 'http://54.77.14.151:8080/query' \
			-X POST \
			-H 'content-type: application/json' \
			--data '{"operationName":"login","variables":{},"query":"query login {\n  login(id: \"admin-user@email.com\", pwd: \"\") {\n    ... on Token {\n      token\n    }\n    ... on Error {\n      code\n      message\n    }\n  }\n}\n"}' \
			| jq --raw-output '.data.login.token' \
	`

workstation-run : docker-push workstation-login
	### /!\ ###
	# require env var: WANDB_API_KEY
	### /!\ ###
	curl 'http://54.77.14.151:8080/query' \
		-X POST \
		-H 'content-type: application/json' \
		-H 'auth: ${WS_LOGIN_TOKEN}' \
		--data '{"operationName":"createTask","variables":{},"query":"mutation createTask {\n  create_task(input: {docker_image: \"${DOCKER_ID}/bomberman\", env: \"WANDB_API_KEY=${WANDB_API_KEY}\", dataset: \"\"}) {\n    id\n    user_id\n    created_at\n    started_at\n    ended_at\n    status\n    job {\n      dataset\n      docker_image\n    }\n  }\n}\n"}' \
		| jq '.' | cat

workstation-list-tasks : workstation-login
	curl 'http://54.77.14.151:8080/query' \
		-X POST \
		-H 'content-type: application/json' \
		-H 'auth: ${WS_LOGIN_TOKEN}' \
		--data '{"operationName":"list_my_tasks","variables":{"auth":"${WS_LOGIN_TOKEN}"},"query":"query list_my_tasks {\n  list_tasks {\n    id\n    created_at\n    started_at\n    ended_at\n    status\n    job {\n      docker_image\n      env\n    }\n  }\n}\n"}' \
		| jq '.' | cat

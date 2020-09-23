build:
	docker build -t hello_world_lapisco .
run:
	docker run -d -p 3000:3000 hello_world_lapisco
run_detached:
	docker run -p 3000:3000 hello_world_lapisco
run_persist:
	docker run -p 3000:3000 -v ${PWD}/src:/app/src hello_world_lapisco
bash:
	docker run -it -p 3000:3000 hello_world_lapisco /bin/bash



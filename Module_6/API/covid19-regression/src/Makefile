build:
	docker-compose build

bash_pipeline:
	# docker-compose run pipeline /bin/bash
	docker run -it --env-file .env \
	-v /media/navar/Data3-SSD/Projects/2020-Churn-detector/churn-detector-navar/models:/app/models \
	-p 8001:80 \
	src_pipeline:latest bash

bash_model:
	docker-compose run model bash

new_projetct:
	docker-compose -p model-regression-stage up -d
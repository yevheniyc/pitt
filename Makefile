#######################################################
# Manage Local Development
run_all:
	docker compose -f docker-compose.yml down
	docker compose -f docker-compose.yml up --build
#######################################################
# Manage Containers: clean containers, images, volumes
clean_containers:
	docker stop $$(docker container ls -aq)
	docker rm $$(docker container ls -aq)
	docker container prune --force

clean_images:
	docker rmi $$(docker images -aq)

clean_volumes:
	docker volume rm $$(docker volume ls -q)

clean_all:
	docker system prune --all --force --volumes
	make clean_containers
	make clean_images
	make clean_volumes
#######################################################

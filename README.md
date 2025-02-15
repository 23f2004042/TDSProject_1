# TDSProject_1
Project 1 - Tools in DS
Commit after final testing all A tasks and some B tasks in container.

1. To execute in local setup:
- Pull the code:
	git clone https://github.com/23f2004042/TDSProject_1.git
- Install requirements.txt
	pip install -r requirements.txt
- Execute from cli
	uv run api.py

2. Docker:
- Pull the image: 
	docker pull 23f2004042/tds_project1:v1
- Execute with AIPROXY_TOKEN
	podman run -p 8000:8000 -e AIPROXY_TOKEN=$AIPROXY_TOKEN docker.io/23f2004042/tds_project1:v1

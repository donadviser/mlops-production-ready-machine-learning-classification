# MLOPs-Production-Ready-Machine-Learning-Project

Youtube Playlist: 

- Pyenv: https://github.com/pyenv/pyenv
- Vs code: https://code.visualstudio.com/download
- Git: https://git-scm.com/
- Flowchart: https://whimsical.com/
- MLOPs Tool: https://www.evidentlyai.com/
- MongoDB: https://account.mongodb.com/account/login
- Data link: https://www.kaggle.com/datasets/moro23/easyvisa-dataset


## Git commands

```bash
git add .

git commit -m "Updated"

git push origin main
```


## How to run?

```bash
pyenv virtualenv 3.11 venv311_usvisa
```

```bash
pyenv activate venv311_usvisa
```

```bash
pyenv pyenv local venv311_usvisa
```

```bash
pip install -r requirements.txt
```

## Workflow:

1. constants
2. entity
3. components
4. pipeline
5. Main file



### Export the  environment variable
```bash


export MONGODB_URL_USVISA="mongodb+srv://<db_username>:<db_password>@cluster0.e1dnk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>


```


# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 317865500347.dkr.ecr.us-east-1.amazonaws.com/usvisa

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO

    



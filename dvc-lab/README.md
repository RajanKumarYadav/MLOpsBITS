# Data Version Control with DVC

## Prerequisite & Installations

1. Basic git concepts and commands
2. Python virtual environment
3. pip install dvc


# Workflows


### STEP 01- Create directory using mkdir 
```bash
mkdir <directory_name>
```
### STEP 02- Go to above directory and initialize git in it
```bash
git init
```
### STEP 02- Go to above directory and initialize dvc in it
```bash
dvc init
```



# Basic Docker Command

```bash
# To see all the images present in your machine
docker images
```

```bash
# To see all the running containers in your machine
docker ps -a
```

```bash
# To stop a running container
docker stop <container_id>
```

```bash
# To remove/delete a docker container(only if it stopped).
docker rm <container_id>
```

```bash
# To see the list of all the available images with their tag, image id, creation time and size.
docker image ls
```

```bash
# To delete a specific image
docker rmi <image_id>
```

```bash
# To delete a docker image forcefully
docker rmi -f <image_id>
```

```bash
# To delete all the docker container available in your machine
docker rm -f (docker ps -a | awk '{print$1}')
```

```bash
# To delete a specific image
docker image rm <image_name>
```
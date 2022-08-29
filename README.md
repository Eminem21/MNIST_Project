# Handwritten Digits Recognition Project 
This project is a hand-writing digit prediction project.

## BACKGROUD
The whole project is deployed in docker container, and is able to communicate with Cassandra container. The web page is based on FLASK,a light-weight web application framework. This project use a mnist model that has been trained beforehand, whose accuracy is up to 99.3%. This project follows the RESTful.

## INTRODUCTION
Users can upload images of a hand-writing digit to the website by clicking the “choose file” button, and  click the “submit” button, then the prediction of the hand-writing digit will be shown on the web page. In addition, the record of this prediction will be saved to Cassandra, one prediction record include the time of prediction, the name of the uploaded file and the predict result.

## DEMO
![image](https://github.com/Eminem21/MNIST_Project/blob/master/demo.gif)

## REQUIREMENTS  
Users are suppposed to install the following requirements before run the program.  
### 1. python 3.7  
### 2. docker  
### 3. cassandra  

## RUN THE PROGRAM
Users are supposed to follow the steps in order to run the program successfully.  
### 1. Pull the cassandra image from docker hub  
```
docker pull cassandra
```
### 2. Create a network bridge between two containers  
```
docker network create mnist-project
```
### 3. Enter the file folder of this program and build docker image of this program  
```
docker build --tag=mnistproject .
```
### 4. Run cassandra in container  
```
docker run --name mnist-cassandra --network mnist-project -p 9042:9042 -d cassandra:latest
```
### 5. Run this program in container  
```
docker run --network mnist-project -d -p 5000:80 mnistproject
```
*If you want the docker container to mount data volume from the host machine, you can move the "trained_model" outside the file folder and use the following command instead."~/trained_model" have to be changed to the path of which you moved "trained_model" to.*
```
docker run  --network mnist-project -p 5000:80 -v ~/trained_model:/app/trained_model mnistproject
```
### 6. Use cqlsh  
```
docker run -it --network mnist-project --rm cassandra cqlsh mnist-cassandra
```
### 7. Open a web browser and set the url to 0.0.0.0:5000/upload  
### 8. View the record  
```
cqlsh>use mnistkeyspace;  
cqlsh>select * from predictrecord;  
```

*notice:*  
*1. The two container are supposed to be connected to the same network, in the sample above, the name of the network is 'mnist-project'*  
*2. After executing the command 3, users should wait for a few seconds for cassandra to be fully started and then execute command 4*  

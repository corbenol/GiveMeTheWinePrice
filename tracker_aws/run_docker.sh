docker run -d --restart always --name mlflow --env-file /home/ec2-user/mlflow.env -p 5000:5000 --memory=512m --cpus="0.5
" mlflow-server

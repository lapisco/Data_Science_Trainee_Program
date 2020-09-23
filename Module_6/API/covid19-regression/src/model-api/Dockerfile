# Use an official Python runtime as a parent image
FROM python:3.6

# Update dependencies linux
RUN apt-get update

# Install the APP
WORKDIR /tmp_install
COPY requirements.txt /tmp_install
RUN pip install -U pip
RUN pip install -r requirements.txt

# Install routing tools
RUN apt-get -y install iproute2 iputils-ping net-tools

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

CMD ["python", "app.py"]
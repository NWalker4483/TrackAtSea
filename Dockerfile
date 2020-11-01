FROM python:3.6

# set a directory for the app
WORKDIR /usr/src/app

RUN apt update
RUN apt install libgl1-mesa-glx libxkbcommon-x11-0  libxcb-xinerama0 -y

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# define the port number the container should expose
# EXPOSE 5000

# run the command
CMD ["python", "./tool.py"]
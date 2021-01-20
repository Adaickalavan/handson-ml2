FROM tensorflow/tensorflow:2.4.0-gpu

# Install certificates
RUN apt-get install -y \ 
        ca-certificates \
        apt-transport-https \
    && rm -rf /var/lib/apt/lists/*
COPY ./certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

# Install dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        sumo \
        sumo-tools \
        sumo-doc \
        libspatialindex-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        python3-pip && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Update default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

RUN pip install -r requirements.txt

# xvfbg xorg-dev 

# COPY environment.yml /tmp/
# RUN conda update -y -n base conda \
#     && echo '    - pyvirtualdisplay' >> /tmp/environment.yml \
#     && conda env create -f /tmp/environment.yml \
#     && conda clean -y -t \
#     && rm /tmp/environment.yml

# ARG username
# ARG userid

# ARG home=/home/${username}
# ARG workdir=${home}/handson-ml2

# RUN adduser ${username} --uid ${userid} --gecos '' --disabled-password \
#     && echo "${username} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${username} \
#     && chmod 0440 /etc/sudoers.d/${username}

# WORKDIR ${workdir}
# RUN chown ${username}:${username} ${workdir}

# USER ${username}
# WORKDIR ${workdir}

# COPY docker/bashrc.bash /tmp/
# RUN cat /tmp/bashrc.bash >> ${home}/.bashrc \
#     && echo "export PATH=\"${workdir}/docker/bin:$PATH\"" >> ${home}/.bashrc \
#     && sudo rm /tmp/bashrc.bash

SHELL ["/bin/bash", "-c", "-l"]

# Build
# docker build --network=host -t handsonml2 .

# Run
# $ docker run -it --gpus all --network=host --env DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/kyber/workspaces/handson-ml2/code:/src handsonml2

# Disable screen access in the host via the command 
# $ xhost +
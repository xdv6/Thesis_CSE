FROM kptae/tensorflow1.14:latest

# Set working directory
WORKDIR /root

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Set timezone:
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Install dependencies:
RUN apt-get update && apt-get install -y \
tzdata wget git python3-dev build-essential libssl-dev libffi-dev libxml2-dev \
libxslt1-dev zlib1g-dev libglew1.5 libglew-dev python3-pip

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set up Miniconda
ENV PATH="/root/miniconda3/bin:$PATH"
SHELL ["/bin/bash", "-c"]
RUN conda init && \
    echo "source /root/miniconda3/bin/activate" >> ~/.bashrc

# Create a Conda environment with Python 3.7
RUN conda create -n myenv python=3.7 -y && \
    echo "conda activate myenv" >> ~/.bashrc

# Use bash shell to allow Conda activation
SHELL ["/bin/bash", "-c"]

# Ensure conda is initialized before activating environment
RUN conda init bash && \
    bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate myenv && \
    pip install tensorflow==1.14 && \
    pip install -e git+https://github.com/openai/baselines.git#egg=baselines && \
    pip install mujoco==2.3.6 robosuite==1.4.1 protobuf==3.20.3 wandb gymnasium==0.28.1 imageio[ffmpeg]"
    
# Modify stack.py (comment out line 357)
RUN sed -i '357s/^/#/' /root/miniconda3/envs/myenv/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py

# Set environment variable for MuJoCo
ENV MUJOCO_GL=egl

# Clone the repository during build
RUN git clone https://github.com/xdv6/Thesis_CSE.git /root/Thesis_CSE && chmod +x /root/Thesis_CSE/setup.sh

ENTRYPOINT ["/bin/bash", "-c", "cd /root/Thesis_CSE && chmod u+x /root/Thesis_CSE/setup.sh && chown root:root /root/Thesis_CSE/setup.sh && source /root/miniconda3/etc/profile.d/conda.sh && conda activate myenv && git reset --hard HEAD && git pull && /bin/bash /root/Thesis_CSE/setup.sh && exec bash"]



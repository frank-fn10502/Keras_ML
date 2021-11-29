FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

ENV HOME /home/frank

ARG USERNAME="frank"
ARG USER_UID=1018
ARG USER_GID=1019

# Create the user https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt update \
    && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN apt install -y \
    git \
    systemd \
    && pip3 install --upgrade pip 

# 設定時區
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends tzdata
RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata 

RUN pip install matplotlib && \
    pip install autopep8 && \
    #for docker
    pip install opencv-python-headless 
    

# 為了畫 MODEL 架構圖
RUN pip install pydot && \
    apt install -y graphviz 
    
#https://github.com/tensorflow/addons
RUN pip install tensorflow-addons==0.14.0 &&\
    pip install pyyaml && \
    pip install scipy

RUN pip install flake8

RUN apt install -y zsh \
    vim \
    wget

# [Optional] Set the default user. Omit if you want to keep the default as root.
# 請注意中途更改 user 要注意資料夾權限的問題
USER $USERNAME

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
       -p git \
       -p https://github.com/zsh-users/zsh-autosuggestions \
       -p https://github.com/zsh-users/zsh-syntax-highlighting \
       -a 'export TERM=xterm-256color'

# 將之前 Powerlevel10k 設定資料複製到 container
ADD ./.p10k.zsh $HOME
RUN echo '[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh' >> ~/.zshrc
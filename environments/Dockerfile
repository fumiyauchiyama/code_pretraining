FROM nvcr.io/nvidia/pytorch:23.07-py3

# ARGでユーザとグループIDを指定
ARG USER_ID
ARG GROUP_ID

RUN apt update && apt install python3.10-venv -y

# 新しいユーザとグループを作成
RUN groupadd -g ${GROUP_ID} myuser && useradd -u ${USER_ID} -g ${GROUP_ID} -m myuser

# 必要なディレクトリの作成
RUN mkdir -p /home/uchiyama.fumiya/ucllm && chown -R myuser:myuser /home/uchiyama.fumiya/ucllm

# デフォルトの作業ディレクトリを設定
WORKDIR /home/uchiyama.fumiya/ucllm

# 新しいユーザに切り替え
USER myuser

RUN python -m pip install --upgrade pip
RUN pip install transformers wandb datasets tokenizers
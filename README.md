# Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

# Run Pretraining
Please make sure that your huggingface and datasets cache directory are specified in the job script, and already logged in then and wandb.
```
qsub -g <group_id> scripts/abci_fsdp/run.sh
```
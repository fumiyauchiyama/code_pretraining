from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM
import os

@hydra.main(version_base=None, config_path="../src/pretrain/config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.dataset.load_dataset.values())

    output_path = os.path.join(
        cfg.train.output_dir, 
        cfg.model.model_config._target_,
        cfg.dataset.load_dataset.path
        )
    
    print(output_path)
    train_args = dict(cfg.train.trainingarguments)
    train_args["output_dir"] = output_path

    print(train_args)

    tokenizer = instantiate(cfg.tokenizer.load_tokenizer)
    print(tokenizer)

    model_config = instantiate(cfg.model.model_config)
    model = AutoModelForCausalLM.from_config(model_config)
    print(model)

if __name__ == "__main__":
    main()
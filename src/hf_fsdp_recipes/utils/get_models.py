from transformers import AutoTokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
import torch
from typing import Literal, Optional, Any

def get_model_decoder_layer(
    model_name: str,
) -> type[LlamaDecoderLayer] | type[MistralDecoderLayer] | type[MixtralDecoderLayer]:
    if "Llama" in model_name:
        return LlamaDecoderLayer
    elif "gpt2" in model_name:
        return GPT2Block
    elif "Mistral" in model_name or "mistral" in model_name:
        return MistralDecoderLayer
    elif "Mixtral" in model_name:
        return MixtralDecoderLayer
    else:
        raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
    
    
def setup_model(
        model_name:str, 
        rank:int,
        training_mode: Literal["pretrain", "finetune"] = "pretrain",
        custom_model_config: Optional[dict[str, Any]] = None,
        custom_tokenizer_config: Optional[dict[str, Any]] = None,
        ):
        
        # use_cache is False when FSDP
        # https://github.com/meta-llama/llama-recipes/blob/6da989a7740304b9fe3a0488612abc3fc90bc474/src/llama_recipes/finetuning.py#L102
        
        if "Llama" in model_name:
            llama_config = LlamaConfig.from_pretrained(model_name)
            llama_config.use_cache = False

            if custom_model_config is not None:
                for k, v in custom_model_config.items():
                    if hasattr(llama_config, k):
                        setattr(llama_config, k, v)
                    else:
                        raise ValueError(f"Config attribute {k} not found in llama_config.")
            print(llama_config)

            if rank == 0:
                if training_mode=="pretrain":
                    model = LlamaForCausalLM(llama_config)
                elif training_mode=="finetune":
                    model = LlamaForCausalLM.from_pretrained(
                        model_name,
                        use_cache=False
                        )
                else:
                    raise ValueError(f"training_mode: {training_mode} is invalid.")

            else:
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        elif "gpt2" in model_name:
            gpt2_config = GPT2Config.from_pretrained(model_name)
            gpt2_config.use_cache = False

            if custom_model_config is not None:
                for k, v in custom_model_config.items():
                    if hasattr(gpt2_config, k):
                        setattr(gpt2_config, k, v)
                    else:
                        raise ValueError(f"Config attribute {k} not found in gpt2_config.")
            print(gpt2_config)

            if training_mode=="pretrain":
                model = GPT2LMHeadModel(gpt2_config)
            elif training_mode=="finetune":
                model = GPT2LMHeadModel.from_pretrained(
                    model_name,
                    use_cache=False
                    )
            else:
                raise ValueError(f"training_mode: {training_mode} is invalid.")


        elif "Mistral" in model_name or "mistral" in model_name:

            mistral_config = MistralConfig.from_pretrained(model_name)

            if training_mode=="pretrain":
                raise NotImplementedError
            elif training_mode=="finetune":
                model = MistralForCausalLM.from_pretrained(
                    mistral_config,
                    attn_implementation="flash_attention_2",
                )
            else:
                raise ValueError(f"training_mode: {training_mode} is invalid.")

        else:
            raise NotImplementedError
        
        tokenizer =  AutoTokenizer.from_pretrained(model_name)

        if custom_tokenizer_config is not None:
            for k, v in custom_tokenizer_config.items():
                if hasattr(tokenizer, k):
                    setattr(tokenizer, k, v)
                else:
                    raise ValueError(f"Config attribute {k} not found in gpt2_config.")
        print(tokenizer)
        
        return model, tokenizer
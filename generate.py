import torch
from typing import Dict, Any, Tuple

from transformers.generation_utils import GenerationMixin
from transformers.file_utils import ModelOutput



class Seq2SeqOutput(ModelOutput):
    logits: torch.LongTensor = None
    hidden: torch.LongTensor = None
    position_ids: torch.LongTensor = None


class Generator(GenerationMixin):
    def __init__(self, model, config, device='cuda'):
        super().__init__()
        self.model = model
        self.config = config
        self.device = device

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {'context': kwargs['context'], "input": input_ids, "hidden": kwargs['hidden'], "position_ids": kwargs['position_ids']}

    @staticmethod
    def _update_model_kwargs_for_generation(
            outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        model_kwargs['hidden'] = outputs['hidden']
        model_kwargs['position_ids'] = outputs['position_ids']
        return GenerationMixin._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder)

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        model_kwargs['hidden'] = model_kwargs['hidden'].index_select(1, expanded_return_idx)
        model_kwargs['context'] = model_kwargs['context'].index_select(1, expanded_return_idx)
        if model_kwargs['position_ids'] is not None:
            model_kwargs['position_ids'] = model_kwargs['position_ids'].index_select(0, expanded_return_idx)

        return GenerationMixin._expand_inputs_for_generation(input_ids, expand_size, is_encoder_decoder, attention_mask, encoder_outputs,
                                                             **model_kwargs)

    def __call__(self, input, hidden, context, position_ids, return_dict, **kwargs):
        prediction, hidden = self.model.decoder(input[:, -1:], hidden, context, position_ids=position_ids)
        prediction = prediction.unsqueeze(1)
        if position_ids is not None:
            position_ids = torch.add(position_ids, 1)

        return Seq2SeqOutput(logits=prediction, hidden=hidden, position_ids=position_ids)


def generate(model, config, tokenizer, decoder_input, hidden, context, position_ids=None, num_beams=1):
    generator = Generator(model, config)
    output = generator.generate(input_ids=decoder_input,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                hidden=hidden,
                                context=context,
                                position_ids=position_ids,
                                num_beams=num_beams,
                                num_return_sequences=num_beams
                                )
    return tokenizer.batch_decode(output[:, 1:], skip_special_tokens=True)


import torch
import math
from transformers.generation_utils import GenerationMixin, LogitsProcessorList, PrefixConstrainedLogitsProcessor, LogitsProcessorList
from transformers.file_utils import ModelOutput
from typing import Dict, Any, List, Callable
from mtc_model import bert, Seq2Seq


class Seq2SeqOutput(ModelOutput):
    logits: torch.LongTensor = None
    hidden: torch.LongTensor = None
    position_ids: torch.LongTensor = None


class PrefixConstrainedLogitsProcessorIgnoreFirst:
    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == 1:
            return scores

        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                mask[batch_id * self._num_beams + beam_id, self._prefix_allowed_tokens_fn(batch_id, sent[1:])] = 0

        return scores + mask


class Generation(Seq2Seq, GenerationMixin):
    config = bert.config

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {'context': kwargs['context'], "input": input_ids, "hidden": kwargs['hidden'],
                "position_ids": kwargs['position_ids']}

    @staticmethod
    def _update_model_kwargs_for_generation(
            outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        model_kwargs['hidden'] = outputs['hidden']
        model_kwargs['position_ids'] = outputs['position_ids']
        return GenerationMixin._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder)

    # def _get_logits_processor(self, repetition_penalty: float, no_repeat_ngram_size: int, encoder_no_repeat_ngram_size: int,
    #                           encoder_input_ids: torch.LongTensor, bad_words_ids: List[List[int]], min_length: int, max_length: int,
    #                           eos_token_id: int, forced_bos_token_id: int, forced_eos_token_id: int,
    #                           prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int, num_beam_groups: int,
    #                           diversity_penalty: float, remove_invalid_values: bool, ) -> LogitsProcessorList:
    #     processors = super()._get_logits_processor(repetition_penalty, no_repeat_ngram_size, encoder_no_repeat_ngram_size,
    #                                                encoder_input_ids, bad_words_ids, min_length, max_length, eos_token_id,
    #                                                forced_bos_token_id, forced_eos_token_id, prefix_allowed_tokens_fn, num_beams,
    #                                                num_beam_groups, diversity_penalty, remove_invalid_values)
    #
    #     new_processors = LogitsProcessorList()
    #     for p in processors:
    #         if type(p) == PrefixConstrainedLogitsProcessor:
    #             new_processors.append(PrefixConstrainedLogitsProcessorIgnoreFirst(p._prefix_allowed_tokens_fn, p._num_beams))
    #         else:
    #             new_processors.append(p)
    #     return new_processors

    def forward(self, input, hidden, context, position_ids, return_dict, **kwargs):
        prediction, hidden = self.decoder(input[:, -1:], hidden, context, position_ids=position_ids)
        # prediction, hidden = self.decoder(input, hidden, context, position_ids=position_ids)
        prediction = prediction.unsqueeze(1)
        if position_ids is not None:
            position_ids = torch.add(position_ids, 1)

        return Seq2SeqOutput(logits=prediction, hidden=hidden, position_ids=position_ids)

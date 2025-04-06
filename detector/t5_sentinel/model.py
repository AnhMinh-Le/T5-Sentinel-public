import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from transformers import T5ForConditionalGeneration as Backbone
from detector.t5_sentinel.__init__ import config
from detector.t5_sentinel.types import SentinelOutput


class Sentinel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone: Backbone = Backbone.from_pretrained(config.backbone.name)
        self.config = config
    
    def forward(self, corpus_ids: Tensor, corpus_mask: Tensor, label_ids: Optional[Tensor] = None, 
                selectedDataset: Tuple[str] = ('Human', 'Gemini', 'GPT', 'Deepseek', 'Llama', 
                                           'Gemini + Human', 'GPT + Human', 'Deepseek + Human', 'Llama + Human')) -> SentinelOutput:
        '''
        Args:
            corpus_ids (Tensor): The input corpus ids.
            corpus_mask (Tensor): The input attention mask.
            label_ids (Tensor): The input label ids.

        Returns:
            output (SentinelOutput): The output of the model.
        '''
        filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
        
        if self.training:
            outputs = self.backbone.forward(
                input_ids=corpus_ids,
                attention_mask=corpus_mask,
                labels=label_ids,
                output_hidden_states=(self.config.mode == 'interpret'),
                output_attentions=(self.config.mode == 'interpret')
            )
            raw_scores = outputs.logits
            filtered_scores = raw_scores[:, 0, [item.token_id for item in filteredDataset]]
            probabilities = torch.softmax(filtered_scores, dim=-1)
        else:
            outputs = self.backbone.generate(
                input_ids=corpus_ids,
                max_length=2, # one for label token, one for eos token
                output_scores=True, 
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            raw_scores = torch.stack(outputs.scores)
            filtered_scores = raw_scores[0, :, [item.token_id for item in filteredDataset]]
            probabilities = torch.softmax(filtered_scores, dim=-1)
        
        return SentinelOutput.construct(huggingface=outputs, probabilities=probabilities)

    def interpretability_study_entry(self, corpus_ids: Tensor, corpus_mask: Tensor, label_ids: Tensor, 
                                    selectedDataset: Tuple[str] = ('Human', 'Gemini', 'GPT', 'Deepseek', 'Llama', 
                                                               'Gemini + Human', 'GPT + Human', 'Deepseek + Human', 'Llama + Human')):
        assert self.injected_embedder is not None, "Injected gradient collector did not found"

        filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
        outputs = self.backbone(
            input_ids=corpus_ids,
            attention_mask=corpus_mask,
            labels=label_ids,
            output_hidden_states=False,
            output_attentions=False
        )
        raw_scores = outputs.logits
        loss = outputs.loss
        loss.backward()

        filtered_scores = raw_scores[:, 0, [item.token_id for item in filteredDataset]]
        probabilities = torch.softmax(filtered_scores, dim=-1)
        return probabilities

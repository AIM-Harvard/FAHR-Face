from typing import Optional, Union
from torch import nn
import torch
from transformers.models.vit.modeling_vit import ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.vit.configuration_vit import ViTConfig

class ViTForSurvivalAnalysis(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.survival_head = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        risk_score = self.survival_head(sequence_output[:, 0, :]).squeeze()

        if not return_dict:
            output = (risk_score,) + outputs[1:]
            return output

        return SequenceClassifierOutput(
            logits=risk_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
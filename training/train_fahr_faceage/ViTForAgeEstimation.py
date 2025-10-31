from typing import Optional, Union
# moved from training/train_fahr_fageage to training/train_fahr_faceage
from torch import nn
import torch
from transformers.models.vit.modeling_vit import ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.vit.configuration_vit import ViTConfig

class AgeRegressionOutput(BaseModelOutput):
    def __init__(self, last_hidden_state, age_output, hidden_states=None, attentions=None):
        super().__init__(last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=attentions)
        self.age_output = age_output

class ViTForAgeRegression(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.age_head = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, AgeRegressionOutput]:
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
        age_output = self.age_head(sequence_output[:, 0, :]).squeeze()

        if not return_dict:
            output = (sequence_output[:, 0, :], age_output) + outputs[1:]
            return output

        return AgeRegressionOutput(
            last_hidden_state=sequence_output[:, 0, :],
            age_output=age_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

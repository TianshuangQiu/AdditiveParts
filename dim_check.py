from additive_parts.models.cloud import PointCloudProcessor
import pdb
import torch
import torch.nn as nn

torch.manual_seed(0)
p = PointCloudProcessor(7, 8, 1, 128, True, [16, 16])
rand = torch.rand((10, 10, 7), requires_grad=True)
out = p(rand)
print(out.shape)
loss = nn.MSELoss()(out, torch.zeros_like(out))
print(loss)

state_dict = p.state_dict()
# state_dict = p.post_process.state_dict()
# print(state_dict.keys())
loss.backward()
print(p.encoder.layers[0].self_attn.in_proj_weight.grad.view(-1)[:5])
# print(p.encoder_layer.self_attn.in_proj_bias.grad.view(-1)[:5])
print(p.post_process[0].weight.grad.view(-1)[:5])
print(rand.grad.view(-1)[:5])

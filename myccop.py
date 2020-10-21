import torch
import open3d.ml.torch as ml3d

inp_positions = torch.randn([20,3])
inp_features = torch.randn([20,8])
out_positions = torch.randn([10,3])

conv = ml3d.layers.ContinuousConv(in_channels=8, filters=16, kernel_size=[3,3,3])
out_features = conv(inp_features, inp_positions, out_positions, extents=2.0)
print(out_features)
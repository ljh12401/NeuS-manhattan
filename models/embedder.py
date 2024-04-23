import torch
import torch.nn as nn
import numpy as np
import global_var

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        input_enc=torch.cat([fn(inputs) for fn in self.embed_fns], -1)
        
        #规定在训练过程中前40%的进度，逐渐提升频率编码的频率等级
        start=0
        end=0.4
        progress=global_var.progress.data
        
        alpha = (progress-start)/(end-start)*self.kwargs['num_freqs']
        k = torch.arange(self.kwargs['num_freqs'],dtype=torch.float32,device=inputs.device)
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        weight = torch.cat([torch.tensor([1.0]*3), *[w.repeat(6) for w in weight]], dim=0).to(inputs.device)
              
        # apply weights
        shape = input_enc.shape
        input_enc = (input_enc*weight).view(*shape)
        
        return input_enc


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos]
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

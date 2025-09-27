import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Optional, Tuple


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state + self.d_inner)
        self.dt_proj = nn.Linear(self.d_state, self.d_inner)
        
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.register_buffer("A", torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        batch, length, _ = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :length]
        x = rearrange(x, 'b d l -> b l d')
        
        x = F.silu(x)
        
        deltaBu = self.x_proj(x)
        delta, B, u = deltaBu.split([self.d_state, self.d_state, self.d_inner], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A).unsqueeze(0).unsqueeze(0)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(-2)

        y = self.selective_scan(u, deltaA, deltaB)

        y = y + u * self.D.view(1, 1, -1)
        y = y * F.silu(z)

        return self.out_proj(y)

    def selective_scan(self, u, deltaA, deltaB):
        batch, length, d_inner = u.shape
        d_state = deltaA.shape[-1]

        x = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        d_vec = self.D.view(1, d_inner, 1)
        ys = []

        for i in range(length):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
            y = (x * d_vec).sum(dim=-1)
            ys.append(y)

        return torch.stack(ys, dim=1)


class DynamicsModel(nn.Module):
    def __init__(self, 
                 latent_dim=512,
                 hidden_dim=768,
                 num_layers=12,
                 action_dim=32,
                 num_actions=256,
                 context_length=32,
                 dropout=0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, latents, actions=None, mask=None):
        batch_size, seq_len, _ = latents.shape
        
        x = self.latent_proj(latents)
        
        if actions is not None:
            action_embeds = self.action_embed(actions)
            action_embeds = self.action_proj(action_embeds)
            x = x + action_embeds
        
        if seq_len <= self.context_length:
            pos_embed = self.pos_embed[:, :seq_len]
        else:
            extra_len = seq_len - self.context_length
            extra = self.pos_embed[:, -1:].expand(-1, extra_len, -1)
            pos_embed = torch.cat([self.pos_embed, extra], dim=1)[:, :seq_len]
        x = x + pos_embed
        
        x = self.dropout(x)
        
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = residual + self.dropout(x)
        
        x = self.norm(x)
        
        output = self.output_proj(x)
        
        return output
    
    def generate(self, initial_latent, actions, num_steps):
        generated = [initial_latent]
        action_history = []

        for step in range(num_steps):
            context = torch.cat(generated[-self.context_length:], dim=1)

            if actions is not None and step < actions.shape[1]:
                action_step = actions[:, step:step+1]
                action_history.append(action_step)

            if action_history:
                action_context = torch.cat(action_history[-context.size(1):], dim=1)
                if action_context.shape[1] < context.size(1):
                    pad_len = context.size(1) - action_context.shape[1]
                    pad_value = action_context[:, -1:]
                    pad = pad_value.repeat(1, pad_len)
                    action_context = torch.cat([action_context, pad], dim=1)
            else:
                action_context = None

            next_latent = self.forward(context, action_context)
            next_latent = next_latent[:, -1:, :]
            generated.append(next_latent)

        return torch.cat(generated[1:], dim=1)


class WorldModel(nn.Module):
    def __init__(self, vqvae, dynamics_model):
        super().__init__()
        self.vqvae = vqvae
        self.dynamics = dynamics_model
        
    def encode_frames(self, frames):
        batch_size, seq_len = frames.shape[:2]
        frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        
        with torch.no_grad():
            latents, _ = self.vqvae.encode(frames_flat)
        
        latents = rearrange(latents, '(b t) c h w -> b t (c h w)', b=batch_size)
        return latents
    
    def decode_latents(self, latents):
        batch_size, seq_len = latents.shape[:2]
        latent_dim = int(math.sqrt(latents.shape[-1] // self.vqvae.encoder.conv_out.out_channels))
        
        latents = rearrange(
            latents, 
            'b t (c h w) -> (b t) c h w',
            c=self.vqvae.encoder.conv_out.out_channels,
            h=latent_dim,
            w=latent_dim
        )
        
        with torch.no_grad():
            frames = self.vqvae.decode(latents)
        
        frames = rearrange(frames, '(b t) c h w -> b t c h w', b=batch_size)
        return frames
    
    def forward(self, frames, actions=None):
        latents = self.encode_frames(frames)
        
        predicted_latents = self.dynamics(latents[:, :-1], actions)
        
        target_latents = latents[:, 1:]
        
        return predicted_latents, target_latents
    
    def generate_trajectory(self, initial_frame, actions, num_steps):
        initial_latent = self.encode_frames(initial_frame.unsqueeze(1))
        
        predicted_latents = self.dynamics.generate(initial_latent, actions, num_steps)
        
        generated_frames = self.decode_latents(predicted_latents)
        
        return generated_frames

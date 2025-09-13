# Knowledge Distillation Guide: GameCraft → Lightweight Model

## Overview
This guide details how to distill Huanyan-GameCraft's 13B parameter model into our 350M parameter lightweight model, achieving 40x compression while maintaining usable quality.

## Why Distillation?

### Traditional Training vs Distillation
| Aspect | From Scratch | Distillation |
|--------|--------------|--------------|
| Training Time | 2-3 months | 2-3 weeks |
| Compute Cost | $10,000+ | $500-1,000 |
| Data Needed | 1M+ hours | 100K hours |
| Quality | Uncertain | 50-70% of teacher |
| Risk | High | Low |

## Step-by-Step Implementation

### Step 1: Setup GameCraft Teacher

```bash
# Clone GameCraft repository
git clone https://github.com/Tencent-Hunyuan/Hunyuan-GameCraft-1.0
cd Hunyuan-GameCraft-1.0

# Download pretrained weights (13B model)
wget https://huggingface.co/Tencent-Hunyuan/GameCraft/resolve/main/model.ckpt

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Teacher for Distillation

```python
import torch
from gamecraft import GameCraftModel

class TeacherWrapper:
    def __init__(self, checkpoint_path):
        self.model = GameCraftModel.from_pretrained(checkpoint_path)
        self.model.eval()
        self.model.cuda()
        
    @torch.no_grad()
    def get_outputs(self, frames, actions=None, return_features=True):
        """Extract outputs at multiple layers for distillation"""
        
        # Get final predictions
        outputs = self.model(frames, actions)
        
        if return_features:
            # Extract intermediate features for better distillation
            features = {
                'encoder_hidden': self.model.encoder.hidden_states,
                'dynamics_hidden': self.model.dynamics.hidden_states,
                'decoder_hidden': self.model.decoder.hidden_states,
                'latent_codes': self.model.vqvae.quantized,
                'logits': outputs.logits
            }
            return outputs.frames, features
        
        return outputs.frames

teacher = TeacherWrapper("path/to/gamecraft/model.ckpt")
```

### Step 3: Implement Distillation Loss

```python
class DistillationLoss(nn.Module):
    def __init__(self, 
                 alpha_task=0.3,      # Weight for task loss
                 alpha_distill=0.7,   # Weight for distillation loss
                 temperature=4.0):     # Distillation temperature
        super().__init__()
        self.alpha_task = alpha_task
        self.alpha_distill = alpha_distill
        self.temperature = temperature
        
    def forward(self, student_outputs, teacher_outputs, targets):
        # Task loss (student vs ground truth)
        task_loss = F.mse_loss(student_outputs['frames'], targets)
        
        # Distillation losses
        # 1. Output distillation
        output_loss = F.mse_loss(
            student_outputs['frames'], 
            teacher_outputs['frames'].detach()
        )
        
        # 2. Feature distillation (FitNets style)
        feature_loss = 0
        for layer in ['encoder', 'dynamics', 'decoder']:
            if f'{layer}_hidden' in student_outputs:
                # Project student features to match teacher dimension
                student_feat = student_outputs[f'{layer}_hidden']
                teacher_feat = teacher_outputs[f'{layer}_hidden'].detach()
                
                # Use adapter if dimensions don't match
                if student_feat.shape[-1] != teacher_feat.shape[-1]:
                    adapter = self.adapters[layer]
                    student_feat = adapter(student_feat)
                
                feature_loss += F.mse_loss(student_feat, teacher_feat)
        
        # 3. Logit distillation (Knowledge Distillation)
        student_logits = student_outputs['logits'] / self.temperature
        teacher_logits = teacher_outputs['logits'].detach() / self.temperature
        
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = (self.alpha_task * task_loss + 
                     self.alpha_distill * (output_loss + 0.1 * feature_loss + kl_loss))
        
        return total_loss, {
            'task_loss': task_loss.item(),
            'output_loss': output_loss.item(),
            'feature_loss': feature_loss.item(),
            'kl_loss': kl_loss.item()
        }
```

### Step 4: Progressive Distillation Strategy

```python
class ProgressiveDistillationTrainer:
    def __init__(self, student_model, teacher_model, config):
        self.student = student_model
        self.teacher = teacher_model
        self.config = config
        
    def train(self, dataloader):
        # Phase 1: Distill encoder only (Week 1)
        self.distill_encoder(dataloader, epochs=20)
        
        # Phase 2: Distill dynamics (Week 2)
        self.distill_dynamics(dataloader, epochs=30)
        
        # Phase 3: Full model distillation (Week 3-4)
        self.distill_full_model(dataloader, epochs=50)
        
    def distill_encoder(self, dataloader, epochs):
        """Freeze everything except encoder"""
        for param in self.student.parameters():
            param.requires_grad = False
        for param in self.student.encoder.parameters():
            param.requires_grad = True
            
        optimizer = torch.optim.AdamW(
            self.student.encoder.parameters(), 
            lr=1e-4
        )
        
        for epoch in range(epochs):
            for batch in dataloader:
                # Get teacher encoder outputs
                with torch.no_grad():
                    teacher_encoded = self.teacher.encode(batch['frames'])
                
                # Student encoder forward
                student_encoded = self.student.encode(batch['frames'])
                
                # MSE loss between encodings
                loss = F.mse_loss(student_encoded, teacher_encoded)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    def distill_dynamics(self, dataloader, epochs):
        """Freeze encoder, train dynamics"""
        # Similar progressive approach
        pass
        
    def distill_full_model(self, dataloader, epochs):
        """Fine-tune entire model"""
        # Unfreeze all parameters
        for param in self.student.parameters():
            param.requires_grad = True
```

### Step 5: Data Generation for Distillation

```python
class DistillationDataGenerator:
    def __init__(self, teacher_model, save_dir):
        self.teacher = teacher_model
        self.save_dir = save_dir
        
    def generate_dataset(self, num_sequences=10000):
        """Generate synthetic data using teacher model"""
        
        for i in range(num_sequences):
            # Random prompt or initial frame
            if random.random() > 0.5:
                prompt = random.choice(self.prompts)
                initial = self.teacher.generate_from_text(prompt)
            else:
                initial = self.get_random_game_frame()
            
            # Generate sequence with teacher
            actions = self.sample_action_sequence(length=32)
            
            with torch.no_grad():
                frames, features = self.teacher.generate_trajectory(
                    initial, actions, return_all_features=True
                )
            
            # Save to disk for training
            torch.save({
                'frames': frames,
                'actions': actions,
                'features': features,
                'prompt': prompt
            }, f"{self.save_dir}/sequence_{i:06d}.pt")
```

### Step 6: Optimization Techniques

```python
# 1. Layer Mapping (Student has fewer layers)
layer_mapping = {
    'student_layer_0': ['teacher_layer_0', 'teacher_layer_1'],
    'student_layer_1': ['teacher_layer_2', 'teacher_layer_3'],
    # ... map 12 student layers to 24 teacher layers
}

# 2. Attention Transfer
def attention_transfer_loss(student_attn, teacher_attn):
    """Transfer attention patterns from teacher to student"""
    # Normalize attention maps
    student_attn = F.normalize(student_attn.view(batch, -1), dim=1)
    teacher_attn = F.normalize(teacher_attn.view(batch, -1), dim=1)
    
    return F.mse_loss(student_attn, teacher_attn)

# 3. Gradient Clipping for Stability
torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

# 4. Learning Rate Schedule
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# 5. Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = distillation_loss(student_out, teacher_out, targets)
```

### Step 7: Evaluation During Distillation

```python
def evaluate_distillation_quality(student, teacher, test_loader):
    metrics = {
        'output_similarity': [],
        'feature_similarity': [],
        'generation_quality': []
    }
    
    with torch.no_grad():
        for batch in test_loader:
            # Compare outputs
            student_out = student(batch['frames'])
            teacher_out = teacher(batch['frames'])
            
            # Output similarity (should be > 0.7)
            similarity = F.cosine_similarity(
                student_out.flatten(1), 
                teacher_out.flatten(1)
            ).mean()
            metrics['output_similarity'].append(similarity.item())
            
            # Generation quality (FVD score)
            student_gen = student.generate(batch['prompt'])
            teacher_gen = teacher.generate(batch['prompt'])
            fvd = calculate_fvd(student_gen, teacher_gen)
            metrics['generation_quality'].append(fvd)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

## Expected Results

### Quality Metrics by Week
| Week | Output Similarity | FVD | Interactive Quality |
|------|------------------|-----|-------------------|
| 1 | 0.4-0.5 | 300+ | Poor |
| 2 | 0.6-0.7 | 200-250 | Acceptable |
| 3 | 0.7-0.8 | 150-200 | Good |
| 4 | 0.75-0.85 | 100-150 | Very Good |

### Compression Results
- **Model Size**: 13GB → 1.4GB (9.3x reduction)
- **Parameters**: 13B → 350M (37x reduction)
- **Inference Speed**: 2 FPS → 30 FPS (15x speedup)
- **VRAM Usage**: 26GB → 4GB (6.5x reduction)

## Common Issues and Solutions

### Issue 1: Mode Collapse
**Symptom**: Student generates similar outputs regardless of input
**Solution**: 
- Reduce distillation temperature
- Add diversity regularization
- Use multiple teacher checkpoints

### Issue 2: Poor Action Conditioning
**Symptom**: Generated frames don't respond to actions
**Solution**:
- Increase weight of action-conditioned sequences in training
- Add auxiliary action prediction loss
- Fine-tune on action-rich sequences

### Issue 3: Temporal Inconsistency
**Symptom**: Flickering or unstable generation
**Solution**:
- Add temporal smoothness loss
- Increase context window during distillation
- Use temporal attention transfer

## Code to Run Distillation

```bash
# Complete distillation pipeline
python scripts/distill_gamecraft.py \
    --teacher_checkpoint /path/to/gamecraft/model.ckpt \
    --student_config configs/lightweight_350M.yaml \
    --output_dir checkpoints/distilled \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --temperature 4.0 \
    --alpha_distill 0.7 \
    --use_mixed_precision \
    --gradient_checkpointing \
    --save_every 1000 \
    --eval_every 500
```

## Post-Distillation Fine-tuning

After distillation, fine-tune on your specific domain:

```python
# Fine-tune on platformer games only
python scripts/finetune.py \
    --model checkpoints/distilled/best.pt \
    --dataset datasets/platformers \
    --epochs 20 \
    --lr 1e-5 \
    --freeze_encoder  # Keep distilled encoder frozen
```

## Verification Checklist

- [ ] Student model loads without errors
- [ ] Teacher model generates coherent sequences
- [ ] Distillation loss decreases over time
- [ ] Output similarity > 0.7 after training
- [ ] Generated frames are visually similar to teacher
- [ ] Action conditioning works correctly
- [ ] Model runs at target FPS on consumer GPU
- [ ] Total VRAM usage < 8GB

## Resources

- GameCraft Paper: https://arxiv.org/abs/2506.17201
- Knowledge Distillation: https://arxiv.org/abs/1503.02531
- FitNets: https://arxiv.org/abs/1412.6550
- Attention Transfer: https://arxiv.org/abs/1612.03928
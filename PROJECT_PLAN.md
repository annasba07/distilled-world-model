# Lightweight Interactive World Model for Consumer GPUs

## Project Overview
Build a lightweight world model that generates interactive 2D game environments in real-time on consumer hardware, democratizing access to world model technology. Think "Stable Diffusion for interactive game worlds" - 100x smaller than Genie 3, but accessible to everyone.

## Core Innovation
Instead of training from scratch (expensive), we'll use **knowledge distillation** from open-source models like Huanyan-GameCraft to bootstrap quality, then fine-tune on domain-specific data.

## Architecture Design

### 1. Visual Encoder (Lightweight VQ-VAE)
- **Input**: 256x256 RGB frames
- **Latent Size**: 16x16 tokens with 512-dim codebook
- **Architecture**: MobileNet-inspired encoder for efficiency
- **Target**: <50M parameters
- **Quality Target**: PSNR > 30dB reconstruction

### 2. Dynamics Model (Mamba-based SSM)
- **Why Mamba**: 5-10x more efficient than transformers
- **Context Length**: 32 frames history
- **Hidden Size**: 768
- **Layers**: 12
- **Target**: ~200M parameters
- **Quality Target**: 30+ seconds temporal consistency

### 3. Action Decoder
- **Input**: Keyboard/gamepad actions
- **Embedding**: 64-dim action embeddings
- **Integration**: Cross-attention with dynamics model
- **Quality Target**: >95% action responsiveness

### 4. Frame Decoder
- **Architecture**: Lightweight U-Net variant
- **Upsampling**: Progressive from 16x16 to 256x256
- **Target**: <100M parameters
- **Quality Target**: LPIPS < 0.2

## Implementation Strategy

### Phase 1: Knowledge Distillation (Primary Approach)
```python
# Distill from Huanyan-GameCraft (13B) → Our Model (350M)
teacher_output = gamecraft_model(input)  # Large teacher
student_output = our_model(input)        # Small student
loss = MSE(student_output, teacher_output.detach()) + 
       KL_divergence(student_logits, teacher_logits)
```

**Benefits:**
- Start with high-quality baseline
- 40x compression ratio
- 2-3 weeks training vs 2-3 months from scratch
- ~$500-1000 compute cost vs $10K+

### Phase 2: Data Collection Pipeline

#### Free/Open Data Sources:
1. **OpenGameArt** (10K+ assets)
   - Sprite sheets, tilesets, animations
   - Direct download via API

2. **itch.io Games** (5K+ open-source)
   - Automated gameplay recording
   - AI agents (PPO/A2C) play games
   - Record state-action-next_state triplets

3. **GitHub Game Jams** (10K+ entries)
   - Scrape playable web games
   - Selenium-based recording

4. **Procedural Generation** (Unlimited)
   - Use Godot/Unity to generate levels
   - Parametric variation for diversity

#### Automated Collection Pipeline:
```python
# 100K hours of diverse gameplay
1. Download open-source games
2. Deploy RL agents to play
3. Record at 30 FPS
4. Extract action labels
5. Quality filtering (remove stuck/glitched sequences)
6. Store as WebDataset shards
```

### Phase 3: Quality Assurance

#### Multi-Stage Training with Metrics:

**Stage 1: Visual Quality (Weeks 1-2)**
- Train VQ-VAE on 1M game screenshots
- Metric: PSNR > 30dB, SSIM > 0.9
- Checkpoint: Can reconstruct game visuals?

**Stage 2: Temporal Consistency (Weeks 3-5)**
- Train dynamics on 10K hours footage
- Metric: FVD < 150, Temporal LPIPS < 0.1
- Checkpoint: 1-second coherent predictions?

**Stage 3: Interactive Control (Weeks 6-7)**
- Fine-tune on action-conditioned data
- Metric: Action accuracy > 90%
- Checkpoint: Responds correctly to input?

**Stage 4: Long-Horizon (Week 8)**
- Extend generation to minutes
- Metric: No drift for 30+ seconds
- Checkpoint: Maintains consistency?

## Training Requirements

### Compute Resources
- **Minimum**: 4x RTX 4090 (24GB each) or 2x A100 (40GB)
- **Cloud Options**: 
  - Lambda Labs: $1.29/hr per A100
  - RunPod: $2.49/hr per A100
  - Vast.ai: $0.80/hr per RTX 4090

### Cost Estimates
- **Distillation Only**: $500-1,000 (1-2 weeks)
- **Full Training**: $2,000-5,000 (4-8 weeks)
- **MVP Path**: $1,500 (4 weeks)

### Training Timeline
```
Week 1-2: Proof of Concept
├── Implement VQ-VAE
├── Train on 10K screenshots
└── Validate reconstruction quality

Week 3-4: Dynamics Testing  
├── Implement Mamba SSM
├── Train on 1K hours simple games
└── Validate next-frame prediction

Week 5-8: Distillation
├── Setup GameCraft teacher
├── Implement distillation pipeline
├── Compress to 350M params
└── Achieve 50% teacher performance

Week 9-12: Custom Training
├── Collect domain data (platformers)
├── Fine-tune distilled model
├── Optimize for specific genres
└── Implement action conditioning

Week 13-16: Production
├── TensorRT optimization
├── INT8 quantization
├── Web demo deployment
└── Community release
```

## Evaluation Metrics

### Automated Metrics
- **Fréchet Video Distance (FVD)**: < 150 (perceptual quality)
- **LPIPS Temporal**: < 0.1 (frame consistency)
- **Physics Consistency Score**: > 0.8 (gravity, collisions)
- **Action Accuracy**: > 95% (control responsiveness)
- **Generation Length**: > 30 seconds without artifacts

### Human Evaluation
- User study with 100 participants
- Metrics: Visual quality, controllability, fun factor
- A/B testing vs baseline (no world model)

## Progressive Model Sizes

| Model | Params | VRAM | FPS (4090) | Training Time | Cost |
|-------|--------|------|------------|---------------|------|
| Tiny | 150M | 2GB | 60 | 1 week | $200 |
| Base | 350M | 4GB | 30 | 4 weeks | $1,500 |
| Large | 800M | 8GB | 15 | 8 weeks | $5,000 |

## Fallback Options

### Plan A: Full Distillation + Training
- Primary approach described above
- Best quality/cost ratio

### Plan B: Fine-tune Only
- Take GameCraft, compress architecture
- Fine-tune last layers only
- 2 weeks, $500 total cost
- 70% quality of Plan A

### Plan C: Hybrid Approach
- Use pretrained image models (CLIP/DINOv2)
- Train only dynamics model
- 3 weeks, $800 total cost
- Good for specific domains

### Plan D: Specialized Domain
- Focus on single genre (just platformers)
- Smaller dataset, faster training
- 2 weeks, $400 total cost
- Perfect for MVP demonstration

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Distillation fails | Fallback to hybrid approach |
| Not enough data | Increase procedural generation |
| Too slow on consumer GPU | Further quantization, reduce resolution |
| Poor temporal consistency | Increase context window, add memory module |
| High training cost | Start with Tiny model, prove concept first |

## Success Criteria

### Technical
- ✅ Runs at 15+ FPS on RTX 3060
- ✅ <8GB VRAM usage
- ✅ 30+ second consistent generation
- ✅ <100ms input latency

### Business
- ✅ 10K+ GitHub stars
- ✅ 1K+ Discord community
- ✅ 100+ user-generated fine-tunes
- ✅ Adopted by indie game developers

### Research
- ✅ Published technical report
- ✅ Reproducible results
- ✅ Benchmark comparisons
- ✅ Novel contributions (Mamba for world models)

## Why This Will Work

1. **Proven Architecture**: Mamba SSMs validated in other domains
2. **Domain Simplicity**: 2D games easier than 3D/photorealistic
3. **Distillation Success**: DistilBERT, Stable Diffusion prove viability
4. **Community Need**: Huge demand for accessible world models
5. **Incremental Path**: Can validate each stage before scaling

## Next Steps

1. **Week 1**: Set up development environment, implement VQ-VAE
2. **Week 2**: Collect initial 10K screenshots, train VAE
3. **Week 3**: Implement Mamba dynamics, test on toy data
4. **Week 4**: Set up GameCraft, begin distillation
5. **Decision Point**: Evaluate distillation quality, choose path A/B/C/D

## Budget Allocation

| Item | Cost | Priority |
|------|------|----------|
| GPU Compute (4 weeks) | $1,500 | Essential |
| Data Storage (1TB) | $100 | Essential |
| API/Tools | $200 | Important |
| Backup compute | $300 | Optional |
| **Total MVP Budget** | **$2,100** | - |

## Team Requirements

### Minimum (Solo Developer)
- ML Engineer with PyTorch experience
- 20 hrs/week for 16 weeks
- Can complete with guidance

### Ideal (2-3 Person Team)
- ML Engineer: Model development
- Systems Engineer: Optimization/deployment  
- Frontend Dev: Demo interface (part-time)

## Open Source Strategy

1. **MIT License**: Maximum adoption
2. **Model Zoo**: Tiny/Base/Large variants
3. **Hugging Face Hub**: Easy model distribution
4. **Gradio Demo**: Try before deploy
5. **Discord Community**: User support
6. **YouTube Tutorials**: Lower barrier to entry
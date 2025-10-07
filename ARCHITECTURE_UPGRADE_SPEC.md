# Architecture Upgrade Specification - October 2025

**Based on:** Latest research findings (RESEARCH_FINDINGS_OCT_2025.md)
**Target:** Consumer GPU deployment (RTX 3060, 8GB VRAM)
**Goal:** Match Matrix-Game 2.0 capabilities at better efficiency

---

## Current Architecture

```
Lightweight World Model v0.1 (Current)
├── Visual Encoder: VQ-VAE
│   ├── Encoder: Conv layers → 16×16×512
│   ├── Quantizer: 512 codebook entries
│   └── Decoder: Transposed conv → 256×256×3
│
├── Dynamics Model: Mamba SSM
│   ├── Layers: 12
│   ├── Hidden dim: 768
│   └── Params: ~300M
│
├── Action Embedding: Simple linear projection
│   └── Input actions → 64-dim embeddings
│
└── Total Parameters: ~350M
```

**Current Performance:**
- FPS: 28
- Resolution: 256×256
- VRAM: 3.5GB
- Coherence: ~10 seconds
- Actions: Basic labeled input

---

## Upgraded Architecture (October 2025 Standard)

```
Lightweight World Model v2.0 (Target)
│
├── [NEW] Visual Tokenizer: Cosmos-Inspired
│   ├── Encoder: Causal 3D Conv → 16×16 tokens
│   ├── Quantizer: Lookup-Free (2^16 vocabulary)
│   ├── Compression: 8x better than current
│   ├── Speed: 12x faster encoding
│   └── Decoder: Progressive upsampling → 640×360×3
│
├── [UPGRADED] Dynamics Model: Hybrid MaskGIT + Mamba
│   │
│   ├── Generation Mode 1: MaskGIT (Inference)
│   │   ├── Parallel token prediction
│   │   ├── Iterative refinement (8 steps)
│   │   ├── Confidence-based masking
│   │   └── Speed: 10x faster than autoregressive
│   │
│   ├── Generation Mode 2: Autoregressive (Training)
│   │   ├── Sequential prediction
│   │   ├── Stable training
│   │   └── High quality
│   │
│   └── Backbone: Hybrid Mamba-Attention
│       ├── Local: Windowed attention (4×7×7)
│       ├── Global: Bidirectional Mamba SSM
│       └── Fusion: Gated combination
│
├── [NEW] Constant Context Manager (FramePack-style)
│   ├── Recent frames (0-8): Full detail
│   ├── Medium frames (8-16): 2x compressed
│   ├── Old frames (16-32): 4x compressed
│   └── Ancient frames (32+): 8x compressed
│   → Always 32 frames in context (O(1) memory)
│
├── [NEW] Latent Action Model (Genie 3-style)
│   ├── Transition Encoder: frame_t + frame_t+1 → action
│   ├── Action Quantizer: 256 latent actions
│   ├── Contrastive Learning: Similar transitions → similar actions
│   └── No labels needed! Train on any video
│
├── [NEW] Teacher-Student Distillation
│   ├── Teacher: 2B params (high quality)
│   ├── Student: 200M params (deployment)
│   └── Quality retention: >90%
│
└── Total Parameters: <500M (deployment model)
```

**Target Performance:**
- FPS: 40-50
- Resolution: 640×360
- VRAM: <4GB
- Coherence: 60+ seconds
- Actions: Latent (learned from any video)

---

## Component Specifications

### 1. Cosmos-Inspired Tokenizer

**Architecture:**
```python
class CosmosInspiredTokenizer(nn.Module):
    def __init__(self):
        # Encoder: 3D Causal Convolutions
        self.encoder = CausalConv3DEncoder(
            in_channels=3,
            hidden_dims=[64, 128, 256, 512],
            temporal_kernel=3,
            spatial_kernel=4,
            stride=2
        )
        # Input: [B, T, 3, 640, 360]
        # Output: [B, T//8, 512, 20, 11]

        # Quantizer: Lookup-Free
        self.quantizer = LookupFreeQuantizer(
            codebook_size=2**16,  # 65,536 tokens
            num_bits=16,
            temperature=1.0
        )

        # Decoder: Progressive upsampling
        self.decoder = ProgressiveDecoder(
            in_channels=512,
            hidden_dims=[256, 128, 64, 32],
            out_channels=3,
            upsample_factor=2
        )
        # Input: [B, T, 512, 20, 11]
        # Output: [B, T, 3, 640, 360]

    def encode(self, x):
        z = self.encoder(x)
        z_quantized, indices = self.quantizer(z)
        return z_quantized, indices

    def decode(self, z_quantized):
        return self.decoder(z_quantized)
```

**Key Features:**
- **Causal 3D Convolutions:** Can process video autoregressively
- **Lookup-Free Quantization:** No codebook collapse, better gradients
- **Large Vocabulary:** 65K tokens vs current 512
- **Better Compression:** ~8x improvement

**Expected Metrics:**
- Compression ratio: ~6000x (vs ~750x current)
- Encoding speed: 12x faster
- PSNR after training: +5-10 dB improvement

---

### 2. MaskGIT Parallel Generator

**Architecture:**
```python
class MaskGITDynamics(nn.Module):
    def __init__(self):
        # Backbone transformer (shared)
        self.transformer = HybridTransformer(
            layers=12,
            hidden_dim=768,
            local_attention=True,
            global_mamba=True
        )

        # MaskGIT-specific components
        self.masking_schedule = CosineSchedule()
        self.confidence_estimator = CombinedConfidence()
        self.refinement_steps = 8

    def generate_maskgit(self, context, num_tokens=256):
        """
        Parallel generation with iterative refinement
        10x faster than autoregressive
        """
        # Initialize with all masked
        tokens = torch.full((B, num_tokens), MASK_TOKEN)

        for step in range(self.refinement_steps):
            # Predict all tokens in parallel
            logits = self.transformer(tokens, context)

            # Sample and estimate confidence
            samples = self.sample(logits)
            confidence = self.confidence_estimator(logits)

            # Keep high confidence, re-mask low confidence
            mask_ratio = self.masking_schedule(step)
            num_to_keep = int((1 - mask_ratio) * num_tokens)
            keep_indices = confidence.topk(num_to_keep).indices

            tokens = self.update_tokens(tokens, samples, keep_indices)

        return tokens

    def generate_autoregressive(self, context, num_tokens=256):
        """
        Sequential generation for training
        Slower but more stable
        """
        tokens = []
        for i in range(num_tokens):
            logits = self.transformer(tokens, context)
            next_token = self.sample(logits[:, -1])
            tokens.append(next_token)
        return torch.stack(tokens, dim=1)
```

**Key Features:**
- **Dual Mode:** MaskGIT for inference, autoregressive for training
- **Parallel Generation:** All tokens predicted simultaneously
- **Iterative Refinement:** 8 iterations for quality
- **Flexible:** Can trade iterations for speed/quality

**Expected Speedup:**
- Autoregressive: 256 forward passes for 256 tokens
- MaskGIT: 8 forward passes for 256 tokens
- **Speedup: 32x in theory, ~10x in practice**

---

### 3. Constant Context Manager

**Architecture:**
```python
class ConstantContextManager(nn.Module):
    """
    FramePack-style compression for unlimited video length
    Memory usage is O(1) regardless of duration
    """
    def __init__(self, max_frames=32):
        self.compression_tiers = {
            'recent': (0, 8, 1),      # 8 frames, no compression
            'medium': (8, 16, 2),     # 4 frames, 2x compression
            'old': (16, 32, 4),       # 4 frames, 4x compression
            'ancient': (32, None, 8), # Rest, 8x compression
        }

        # Multi-scale patchify for compression
        self.patchify = {
            1: nn.Identity(),
            2: Conv3D(kernel=(2,4,4), stride=(2,4,4)),
            4: Conv3D(kernel=(4,8,8), stride=(4,8,8)),
            8: Conv3D(kernel=(8,16,16), stride=(8,16,16))
        }

    def compress_history(self, frame_history):
        """
        Always returns 32 frames regardless of input length
        """
        compressed = []

        for tier, (start, end, stride) in self.compression_tiers.items():
            tier_frames = self.extract_tier(frame_history, start, end, stride)
            compressed_frames = self.patchify[stride](tier_frames)
            compressed.extend(compressed_frames)

        return compressed[-32:]  # Always 32 frames
```

**Key Features:**
- **Constant Memory:** Always 32 frames in context
- **Geometric Compression:** Progressive reduction for older frames
- **Multi-Scale:** Different compression levels per tier
- **No Quality Loss:** Recent frames at full detail

**Expected Impact:**
- Current: Memory grows with video length → OOM at ~30 seconds
- Upgraded: Constant memory → **Can generate indefinitely**
- Target: **60+ seconds** of coherent generation

---

### 4. Latent Action Model

**Architecture:**
```python
class LatentActionModel(nn.Module):
    """
    Learns actions from video without labels
    Genie 3-style latent action inference
    """
    def __init__(self):
        # Transition encoder
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256)
        )

        # Vector quantizer for discrete actions
        self.action_vq = VectorQuantizer(
            num_embeddings=256,  # 256 latent actions
            embedding_dim=256,
            commitment_cost=0.25
        )

        # Contrastive head
        self.contrastive_proj = nn.Linear(256, 128)

    def infer_action(self, frame_t, frame_t_plus_1):
        """
        What action caused frame_t → frame_t_plus_1?
        """
        transition = torch.cat([frame_t, frame_t_plus_1], dim=-1)
        action_continuous = self.encoder(transition)
        action_discrete, vq_loss = self.action_vq(action_continuous)
        return action_discrete, vq_loss

    def contrastive_loss(self, frame_t, frame_t1, frame_t2):
        """
        Similar transitions should have similar actions
        """
        action_1 = self.infer_action(frame_t, frame_t1)[0]
        action_2 = self.infer_action(frame_t1, frame_t2)[0]

        # Project to contrastive space
        z1 = self.contrastive_proj(action_1)
        z2 = self.contrastive_proj(action_2)

        # InfoNCE loss
        similarity = F.cosine_similarity(z1, z2)
        return contrastive_loss(similarity)
```

**Key Features:**
- **No Labels Required:** Learns from any video
- **Discrete Actions:** 256 latent action space
- **Contrastive Learning:** Similar transitions → similar actions
- **Versatile:** Can adapt to any game/environment

**Data Implications:**
- Current: Need labeled gameplay (~100 hours feasible)
- Upgraded: **Any video works (millions of hours available)**
- YouTube gameplay alone: **Unlimited training data**

---

### 5. Hybrid Mamba-Attention Blocks

**Architecture:**
```python
class HybridBlock(nn.Module):
    """
    Combines local attention with global Mamba SSM
    Best of both worlds
    """
    def __init__(self, dim=768):
        # Local modeling: Windowed attention
        self.local_attention = WindowedAttention(
            dim=dim,
            window_size=(4, 7, 7),  # 4 frames, 7×7 spatial
            num_heads=12,
            qkv_bias=True
        )

        # Global modeling: Bidirectional Mamba
        self.global_mamba = BidirectionalMamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # Fusion
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x: [B, T, H, W, C]

        # Local details via attention
        local_out = self.local_attention(x)

        # Global context via Mamba
        B, T, H, W, C = x.shape
        x_seq = x.reshape(B, T*H*W, C)
        global_out = self.global_mamba(x_seq)
        global_out = global_out.reshape(B, T, H, W, C)

        # Gated fusion
        combined = torch.cat([local_out, global_out], dim=-1)
        gate = self.fusion_gate(combined)
        fused = self.fusion_proj(combined)

        return gate * fused + (1 - gate) * x  # Residual
```

**Key Features:**
- **Local Attention:** Captures fine details (textures, edges)
- **Global Mamba:** Efficient long-range dependencies
- **Gated Fusion:** Learned combination of both
- **Efficient:** ~25% FLOPs reduction vs pure transformer

**Expected Benefits:**
- Better quality than pure Mamba
- More efficient than pure attention
- Best of both architectures

---

## Training Strategy

### Three-Stage Training

**Stage 1: Tokenizer Pretraining (1-2 weeks)**
```yaml
Objective: Learn high-quality visual tokenization

Data:
  - 100K game screenshots
  - Diverse game genres
  - High resolution (640×360)

Loss:
  - Reconstruction: MSE + Perceptual
  - Quantization: Commitment loss
  - Adversarial: Optional GAN loss

Target Metrics:
  - PSNR: >32 dB
  - Compression: 8x vs current
  - Speed: 12x faster encoding
```

**Stage 2: Latent Action Pretraining (2-3 weeks)**
```yaml
Objective: Learn action representations from unlabeled video

Data:
  - 1000+ hours YouTube gameplay
  - Twitch streams
  - Game trailers
  - NO LABELS NEEDED!

Loss:
  - Transition prediction: MSE
  - VQ commitment: 0.25
  - Contrastive: InfoNCE
  - Inverse dynamics: Cross-entropy

Target Metrics:
  - Action coverage: 256 diverse actions learned
  - Reconstruction: Can predict transitions accurately
```

**Stage 3: End-to-End World Model Training (3-4 weeks)**
```yaml
Objective: Full world model training

Data:
  - 1000+ hours from automated collection pipeline
  - Unreal Engine procedural games
  - GTA5 recordings
  - Open-source games

Losses:
  - Frame prediction: MSE + LPIPS
  - Action conditioning: Cross-entropy
  - Temporal consistency: Optical flow
  - Perceptual quality: VGG loss

Modes:
  - Teacher training (2B params): 2 weeks
  - Student distillation (200M params): 1 week
  - Fine-tuning: 1 week

Target Metrics:
  - FPS: 40-50 (RTX 3060)
  - Coherence: 60+ seconds
  - Action accuracy: >95%
  - PSNR: >32 dB
```

---

## Deployment Configuration

### Production Model Specifications

```yaml
Model Name: Lightweight World Model v2.0

Sizes:
  Nano (Edge):
    Parameters: 100M
    VRAM: <2GB
    FPS: 60+
    Resolution: 512×288
    Use Case: Mobile, embedded

  Base (Consumer):
    Parameters: 200M
    VRAM: <4GB
    FPS: 40-50
    Resolution: 640×360
    Use Case: RTX 3060, consumer GPUs

  Large (Enthusiast):
    Parameters: 500M
    VRAM: <8GB
    FPS: 30-40
    Resolution: 720p
    Use Case: RTX 3080+, quality focused

Deployment Formats:
  - PyTorch (.pt)
  - ONNX (.onnx)
  - TensorRT (.engine)
  - CoreML (.mlmodel) - for Apple Silicon
  - GGUF - for CPU inference

Optimizations:
  - FP16 mixed precision
  - torch.compile
  - Flash Attention 2
  - Gradient checkpointing (training)
  - INT8 quantization (optional)
```

---

## Migration Path from Current Architecture

### Phase 1: Tokenizer Upgrade (Week 1-2)

```python
# Step 1: Implement new tokenizer alongside old
from src.models.improved_vqvae import ImprovedVQVAE  # Old
from src.models.cosmos_tokenizer import CosmosTokenizer  # New

# Step 2: Train new tokenizer
new_tokenizer = CosmosTokenizer()
train(new_tokenizer, data)

# Step 3: Benchmark comparison
metrics_old = benchmark(old_vqvae)
metrics_new = benchmark(new_tokenizer)
# Expected: metrics_new >> metrics_old

# Step 4: Gradual migration
class HybridModel(nn.Module):
    def __init__(self, use_new=True):
        self.encoder = CosmosTokenizer() if use_new else ImprovedVQVAE()

# Step 5: Full replacement once validated
# Update all references to use new tokenizer
```

### Phase 2: Add MaskGIT Mode (Week 3)

```python
# Step 1: Implement MaskGIT alongside autoregressive
class DualModeGenerator:
    def __init__(self):
        self.backbone = YourCurrentModel()  # Reuse!
        self.maskgit_head = MaskGITHead()

    def forward(self, mode='autoregressive'):
        if mode == 'autoregressive':
            return self.autoregressive_generate()
        else:
            return self.maskgit_generate()

# Step 2: Train using autoregressive (stable)
train(model, mode='autoregressive')

# Step 3: Inference using MaskGIT (fast)
generate(model, mode='maskgit')  # 10x faster!
```

### Phase 3: Integrate Remaining Components (Week 4+)

```python
# Add components incrementally:
# Week 4: Constant context
# Week 5-6: Latent actions
# Week 7-12: Full training with all components

class UpgradedWorldModel:
    def __init__(self):
        self.tokenizer = CosmosTokenizer()  # ✅ Week 1-2
        self.context_manager = ConstantContextManager()  # ✅ Week 4
        self.dynamics = DualModeGenerator()  # ✅ Week 3
        self.latent_actions = LatentActionModel()  # ✅ Week 5-6
```

---

## Expected Performance Improvements

### Quantitative Metrics

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **FPS (RTX 3060)** | 28 | 40-50 | +40-80% |
| **Resolution** | 256×256 | 640×360 | +5.6x pixels |
| **VRAM** | 3.5GB | <4GB | Maintained |
| **Coherence** | ~10 sec | 60+ sec | +6x |
| **Parameters** | 350M | <500M | +40% |
| **Training Data** | Limited | Unlimited | ∞ |
| **Compression** | 750x | 6000x | +8x |
| **PSNR** | TBD | 32+ dB | +5-10 dB |

### Qualitative Improvements

```yaml
Visual Quality:
  Current: Acceptable for 256×256
  Target: Excellent for 640×360
  Benefit: Production-ready visuals

Temporal Consistency:
  Current: Some drift after 10 sec
  Target: Stable for 60+ sec
  Benefit: Usable for real applications

Action Responsiveness:
  Current: Basic labeled actions
  Target: Rich latent action space
  Benefit: More natural control

Training Efficiency:
  Current: Limited by labeled data
  Target: Unlimited YouTube data
  Benefit: Continuous improvement

Inference Speed:
  Current: 28 FPS (acceptable)
  Target: 40-50 FPS (excellent)
  Benefit: True real-time feel
```

---

## Compatibility & Backward Compatibility

### API Compatibility

```python
# Old API (still supported)
model = LightweightWorldModel.from_pretrained('v0.1')
frames = model.generate(initial_frame, actions, num_frames=30)

# New API (recommended)
model = LightweightWorldModel.from_pretrained('v2.0')
frames = model.generate(
    initial_frame=initial_frame,
    actions=actions,
    num_frames=100,  # Can do more now!
    mode='maskgit',  # New: choose generation mode
    quality='high'   # New: quality presets
)

# Backward compatibility layer
class BackwardCompatibleModel:
    def generate(self, *args, **kwargs):
        # Detect old-style calls
        if 'mode' not in kwargs:
            # Use old behavior
            return self.generate_v1(*args, **kwargs)
        else:
            # Use new behavior
            return self.generate_v2(*args, **kwargs)
```

### Model Weight Compatibility

```python
# Automatic migration of old weights
def migrate_weights(old_checkpoint):
    """
    Migrate v0.1 weights to v2.0 architecture
    """
    new_checkpoint = {}

    # Tokenizer: Partially transfer
    new_checkpoint['tokenizer.encoder'] = upgrade_encoder(
        old_checkpoint['vqvae.encoder']
    )

    # Dynamics: Reuse backbone
    new_checkpoint['dynamics.backbone'] = old_checkpoint['dynamics']

    # New components: Random init
    new_checkpoint['maskgit_head'] = None  # Train from scratch
    new_checkpoint['latent_actions'] = None  # Train from scratch

    return new_checkpoint

# Usage
old_model = load('model_v0.1.pt')
new_model = load_with_migration('model_v0.1.pt')  # Auto-upgrade
```

---

## Testing & Validation Strategy

### Unit Tests

```python
# test_cosmos_tokenizer.py
def test_compression_ratio():
    """Verify 8x compression improvement"""
    assert new_tokenizer.compression_ratio() > old_tokenizer.compression_ratio() * 8

def test_encoding_speed():
    """Verify 12x speedup"""
    assert new_tokenizer.encode_time() < old_tokenizer.encode_time() / 12

# test_maskgit.py
def test_parallel_generation():
    """Verify correct parallel token generation"""
    tokens = maskgit.generate(context, num_tokens=256)
    assert tokens.shape == (batch_size, 256)

def test_maskgit_speedup():
    """Verify 10x speedup vs autoregressive"""
    time_ar = benchmark_autoregressive()
    time_maskgit = benchmark_maskgit()
    assert time_maskgit < time_ar / 10

# test_constant_context.py
def test_constant_memory():
    """Verify memory doesn't grow with video length"""
    for num_frames in [10, 100, 1000]:
        memory_usage = measure_memory(num_frames)
        assert memory_usage < MAX_MEMORY  # Constant!
```

### Integration Tests

```python
# test_end_to_end.py
def test_60_second_generation():
    """Verify can generate 60+ seconds"""
    frames = model.generate_sequence(duration=60)
    assert len(frames) >= 60 * 30  # 30 FPS

def test_fps_target():
    """Verify 40+ FPS on RTX 3060"""
    fps = benchmark_fps(model, device='cuda:0')
    assert fps >= 40

def test_vram_usage():
    """Verify <4GB VRAM"""
    vram = measure_vram(model)
    assert vram < 4096  # MB
```

### Performance Benchmarks

```bash
# benchmarks/run_all.sh
python benchmark_fps.py --model v2.0 --gpu rtx3060
python benchmark_quality.py --model v2.0 --metric psnr,lpips,fvd
python benchmark_coherence.py --model v2.0 --duration 60
python benchmark_action_accuracy.py --model v2.0 --num_tests 1000
```

---

## Risks & Mitigation

### Technical Risks

**Risk 1: New tokenizer underperforms**
- **Mitigation:** Keep old VQ-VAE as fallback, incremental migration
- **Test:** Benchmark before full integration
- **Fallback:** Hybrid approach if needed

**Risk 2: MaskGIT quality degradation**
- **Mitigation:** Maintain autoregressive mode for training
- **Test:** Extensive quality testing at different iteration counts
- **Fallback:** Use autoregressive for quality mode

**Risk 3: Constant context causes drift**
- **Mitigation:** Implement anti-drifting techniques early
- **Test:** Long-form generation tests
- **Fallback:** Adjust compression schedule, add anchor frames

**Risk 4: Integration complexity**
- **Mitigation:** Phased rollout, extensive testing between phases
- **Test:** Integration tests after each phase
- **Fallback:** Can ship earlier phases independently

---

## Success Criteria

### Phase 1 (Weeks 1-4)
```yaml
Must Have:
  ✅ New tokenizer working with 8x compression
  ✅ Constant context supporting 30+ sec generation
  ✅ MaskGIT mode achieving 10x speedup
  ✅ All tests passing
  ✅ 640×360 resolution working

Nice to Have:
  ⚪ 40+ FPS achieved
  ⚪ PSNR showing improvement
  ⚪ Zero regressions from current version
```

### Final (Week 20)
```yaml
Must Have:
  ✅ All components integrated
  ✅ 40+ FPS on RTX 3060
  ✅ 60+ second coherence
  ✅ <4GB VRAM
  ✅ >95% action accuracy
  ✅ Comprehensive tests
  ✅ Complete documentation

Nice to Have:
  ⚪ 50+ FPS
  ⚪ 720p resolution option
  ⚪ <3GB VRAM
  ⚪ 2+ minute coherence
```

---

## Conclusion

This architecture upgrade will transform our lightweight world model into a **state-of-the-art system** that:

✅ Matches Matrix-Game 2.0 capabilities
✅ Exceeds Matrix-Game 2.0 efficiency (better FPS, less VRAM)
✅ Supports unlimited training data (latent actions)
✅ Generates indefinitely long videos (constant context)
✅ Runs on consumer GPUs (RTX 3060)

The upgrade is **ambitious but achievable** with the phased approach and proven techniques from October 2025 research.

---

**Document Version:** 1.0
**Status:** Ready for Implementation
**Next Step:** Begin Phase 1 - Week 1

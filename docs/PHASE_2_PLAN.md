# Phase 2 Plan: Latent Action Learning

**Timeline**: Weeks 5-7 (3 weeks)
**Goal**: Learn controllable actions from video without labels
**Inspiration**: Genie 3 (DeepMind, October 2025)

---

## 🎯 Objective

Enable **interactive video generation** by learning a discrete action space from video data alone, without requiring action labels.

**Input**: Video sequence
**Output**:
1. Discrete action tokens representing what changed between frames
2. Ability to control generation with learned actions
3. Action-conditioned video prediction

---

## 📋 Overview

### The Problem

Current video models (Phase 1) can:
- ✅ Encode videos to tokens
- ✅ Generate new video tokens
- ✅ Decode tokens to video

But they **cannot** be controlled interactively (e.g., "move character left", "jump").

### The Solution: Latent Actions

Learn actions by asking: **"What changed between frame t and frame t+1?"**

1. **Action Encoder**: `(frame_t, frame_t+1) → action_t`
2. **Action Quantizer**: `action_t → discrete_action_token`
3. **Dynamics Model**: `(frame_t, action_token) → frame_t+1`

This enables:
- 🎮 **Interactive control**: User specifies action → model generates next frame
- 🔁 **Rollout**: Repeatedly apply actions to generate long sequences
- 🎯 **Zero-shot**: No action labels needed during training

---

## 🏗️ Architecture

### Week 5: Action Encoder & Quantizer

**Goal**: Learn to infer actions from consecutive frames

**Components**:

1. **Action Encoder**
```python
class ActionEncoder(nn.Module):
    """
    Infer latent action from consecutive frames.

    Input:
        frame_t: [B, C, H, W] (current frame tokens)
        frame_t1: [B, C, H, W] (next frame tokens)

    Output:
        action_latent: [B, D_action] (continuous action representation)
    """
    def __init__(self, token_dim=256, action_dim=128):
        # Use difference + attention to identify what changed
        self.diff_encoder = nn.Sequential(...)
        self.attention = MultiHeadAttention(...)

    def forward(self, frame_t, frame_t1):
        # Compute difference
        diff = frame_t1 - frame_t

        # Encode what changed (attention over difference)
        action_latent = self.diff_encoder(diff)

        return action_latent
```

2. **Action Quantizer**
```python
class ActionQuantizer(nn.Module):
    """
    Quantize continuous actions to discrete action vocabulary.
    Uses VQ-VAE or LFQ for action codebook.
    """
    def __init__(self, action_vocab_size=512, action_dim=128):
        # Discrete action vocabulary (smaller than video vocab)
        self.quantizer = LookupFreeQuantizer(
            codebook_size=action_vocab_size,
            embedding_dim=action_dim
        )

    def forward(self, action_latent):
        action_quantized, info = self.quantizer(action_latent)
        action_tokens = info['indices']
        return action_quantized, action_tokens
```

**Training Objective**:
- Reconstruct frame transitions: `(frame_t, action) → frame_t+1`
- Minimize reconstruction loss + action quantization loss

---

### Week 6: Dynamics Model

**Goal**: Predict next frame given current frame + action

**Components**:

1. **Action-Conditioned Generator**
```python
class ActionConditionedGenerator(nn.Module):
    """
    Generate next frame tokens conditioned on action.

    Input:
        frame_t_tokens: [B, N] (current frame tokens)
        action_token: [B, 1] (discrete action)

    Output:
        frame_t1_tokens: [B, N] (next frame tokens)
    """
    def __init__(self, vocab_size=65536, action_vocab=512):
        # Embed action and frame tokens
        self.frame_embed = nn.Embedding(vocab_size, dim)
        self.action_embed = nn.Embedding(action_vocab, dim)

        # Transformer to predict next frame
        self.transformer = TransformerDecoder(...)

    def forward(self, frame_tokens, action_token):
        # Embed inputs
        frame_emb = self.frame_embed(frame_tokens)
        action_emb = self.action_embed(action_token)

        # Condition frame prediction on action
        # Prepend action as prefix: [action, frame_t] → frame_t+1
        input_seq = torch.cat([action_emb, frame_emb], dim=1)

        # Predict next frame
        next_frame_logits = self.transformer(input_seq)

        return next_frame_logits
```

2. **Integrated Pipeline**
```python
class LatentActionPipeline:
    """
    Full action-conditioned video generation.
    """
    def __init__(self):
        self.tokenizer = CosmosInspiredTokenizer(...)  # Phase 1
        self.action_encoder = ActionEncoder(...)
        self.action_quantizer = ActionQuantizer(...)
        self.dynamics_model = ActionConditionedGenerator(...)

    def learn_actions(self, video):
        """Learn actions from video."""
        # Tokenize video
        tokens = self.tokenizer.tokenize(video)

        # For each consecutive pair
        for t in range(T-1):
            frame_t = tokens[:, t]
            frame_t1 = tokens[:, t+1]

            # Infer action
            action_latent = self.action_encoder(frame_t, frame_t1)
            action_quantized, action_token = self.action_quantizer(action_latent)

            # Train dynamics model
            pred_frame_t1 = self.dynamics_model(frame_t, action_token)
            loss = F.cross_entropy(pred_frame_t1, frame_t1)

    def generate_with_action(self, start_frame, action_sequence):
        """Generate video by applying action sequence."""
        frames = [start_frame]

        for action in action_sequence:
            # Encode current frame
            current_tokens = self.tokenizer.tokenize(frames[-1])

            # Apply action
            next_tokens = self.dynamics_model(current_tokens, action)

            # Decode to video
            next_frame = self.tokenizer.detokenize(next_tokens)
            frames.append(next_frame)

        return frames
```

**Training Objective**:
- Predict next frame: `P(frame_t+1 | frame_t, action_t)`
- Minimize cross-entropy loss on token predictions

---

### Week 7: Interactive Generation & Evaluation

**Goal**: Enable user control and measure quality

**Components**:

1. **Action Explorer**
```python
class ActionExplorer:
    """
    Explore the learned action space.
    """
    def __init__(self, pipeline, action_vocab_size=512):
        self.pipeline = pipeline
        self.action_vocab_size = action_vocab_size

    def sample_actions(self, n_samples=8):
        """Sample random actions to see what they do."""
        actions = torch.randint(0, self.action_vocab_size, (n_samples,))
        return actions

    def visualize_action_effect(self, start_frame, action):
        """Show what happens when action is applied."""
        result = self.pipeline.generate_with_action(start_frame, [action])
        return result[-1]  # Final frame after action

    def cluster_actions(self):
        """Group similar actions together (e.g., all "move left")."""
        # Use action embeddings for clustering
        action_embeddings = self.pipeline.action_quantizer.quantizer.project_out.weight
        from sklearn.cluster import KMeans
        clusters = KMeans(n_clusters=16).fit(action_embeddings.detach().numpy())
        return clusters
```

2. **Evaluation Metrics**
```python
def evaluate_latent_actions(pipeline, test_videos):
    """
    Evaluate action learning quality.
    """
    metrics = {}

    # 1. Action Consistency
    # Same action should produce similar changes
    consistency_scores = []
    for video in test_videos:
        for action in unique_actions:
            results = [apply_action(frame, action) for frame in video]
            consistency = measure_similarity(results)
            consistency_scores.append(consistency)
    metrics['action_consistency'] = np.mean(consistency_scores)

    # 2. Action Diversity
    # Different actions should produce different results
    action_usage = count_unique_actions(test_videos)
    metrics['action_diversity'] = action_usage / action_vocab_size

    # 3. Reconstruction Quality
    # Can we reconstruct videos using learned actions?
    recon_errors = []
    for video in test_videos:
        # Learn actions
        actions = pipeline.infer_actions(video)
        # Reconstruct
        recon = pipeline.generate_with_action(video[0], actions)
        # Measure error
        error = F.mse_loss(video, recon)
        recon_errors.append(error)
    metrics['reconstruction_mse'] = np.mean(recon_errors)

    # 4. Controllability
    # Can user specify desired action?
    # (This requires labeled test set - optional)

    return metrics
```

**Interactive Demo**:
```python
def interactive_demo():
    """
    Real-time interactive generation.
    """
    pipeline = LatentActionPipeline(...)

    # Start with initial frame
    frame = load_initial_frame()

    # User control loop
    while True:
        display(frame)

        # User input (keyboard, mouse, etc.)
        action = get_user_input()  # Returns action ID

        # Generate next frame
        frame = pipeline.generate_with_action(frame, [action])[1]
```

---

## 📊 Deliverables

### Code (Week 5-7)
1. `src/models/actions/action_encoder.py` - Action encoder
2. `src/models/actions/action_quantizer.py` - Action quantizer
3. `src/models/actions/dynamics_model.py` - Action-conditioned generator
4. `src/models/actions/latent_action_pipeline.py` - Integrated pipeline
5. `src/models/actions/action_explorer.py` - Action exploration tools

### Tests
1. `tests/unit/actions/test_action_encoder.py`
2. `tests/unit/actions/test_action_quantizer.py`
3. `tests/unit/actions/test_dynamics_model.py`
4. `tests/integration/test_latent_actions.py`

### Benchmarks
1. `benchmarks/action_consistency.py` - Measure action quality
2. `benchmarks/reconstruction_with_actions.py` - Video reconstruction
3. `benchmarks/interactive_generation.py` - Real-time demo

### Documentation
1. `docs/phase2/WEEK_5_PROGRESS.md` - Action encoder/quantizer
2. `docs/phase2/WEEK_6_PROGRESS.md` - Dynamics model
3. `docs/phase2/WEEK_7_PROGRESS.md` - Interactive generation
4. `docs/PHASE_2_README.md` - Complete Phase 2 docs

---

## 🎯 Success Criteria

### Week 5
- ✅ Action encoder infers reasonable actions
- ✅ Action quantizer learns discrete vocabulary
- ✅ Action diversity >50% (using >256/512 actions)
- ✅ Action embeddings cluster meaningfully

### Week 6
- ✅ Dynamics model predicts next frame
- ✅ Action-conditioned generation works
- ✅ Reconstruction MSE <0.1
- ✅ Full pipeline runs end-to-end

### Week 7
- ✅ Interactive demo works in real-time
- ✅ Actions have consistent effects
- ✅ User can control generation
- ✅ Documentation complete

---

## 🔬 Research Questions

### To Investigate

1. **Action Vocabulary Size**
   - How many discrete actions are needed?
   - 256? 512? 1024?
   - Trade-off: Expressiveness vs. Learnability

2. **Action Representation**
   - Global action (one per frame)?
   - Spatial action (one per region)?
   - Hierarchical (coarse + fine)?

3. **Conditioning Method**
   - Prepend action to sequence?
   - Cross-attention with action?
   - Add action embedding to each token?

4. **Training Strategy**
   - Learn actions end-to-end with dynamics?
   - Pre-train action encoder separately?
   - Curriculum learning (simple → complex actions)?

---

## 🚧 Potential Challenges

### Challenge 1: Action Collapse
**Problem**: All frames map to same few actions
**Solution**:
- Action diversity loss
- Encourage uniform action distribution
- Entropy regularization

### Challenge 2: Ambiguous Actions
**Problem**: Different actions produce similar results
**Solution**:
- Maximize mutual information I(action, next_frame)
- Contrastive learning
- Action discriminator

### Challenge 3: Slow Dynamics
**Problem**: Some changes take multiple frames
**Solution**:
- Multi-frame action prediction
- Hierarchical actions (macro + micro)
- Temporal abstraction

### Challenge 4: MPS Performance
**Problem**: Action encoder adds computation
**Solution**:
- Lightweight encoder architecture
- Quantize early (reduce latent dimension)
- Profile and optimize bottlenecks

---

## 💻 Development on M4 Max

### Realistic Expectations

**What Will Work** ✅:
- Architecture validation
- Action learning pipeline
- Interactive demo (may be slower)
- Algorithm development

**What May Be Slow** ⚠️:
- Action encoder forward pass
- Dynamics model training
- Real-time interaction (10-20 FPS vs 60 FPS)

**Optimization Strategy**:
1. Build & validate on M4 Max
2. Profile to find bottlenecks
3. Optimize critical paths
4. Document CUDA expectations
5. Keep iterating on algorithms

---

## 📚 References

### Papers
1. **Genie 3** (DeepMind, Oct 2025) - Latent action learning
2. **GAIA-1** (Wayve, 2023) - World models with actions
3. **VideoGPT** (2021) - Video generation with VQ-VAE
4. **GameGAN** (NVIDIA, 2020) - Learning game engines

### Prior Art
- Week 1: Video tokenizer ✅
- Week 2: Context manager ✅
- Week 3: MaskGIT generator ✅
- Week 4: Optimizations ✅

All Phase 1 components ready to extend with actions!

---

## 🚀 Getting Started (Week 5)

### Day 1-2: Setup
- Create `src/models/actions/` directory
- Set up test structure
- Review Genie 3 paper
- Design action encoder architecture

### Day 3-4: Action Encoder
- Implement difference encoding
- Add attention mechanism
- Unit tests
- Validate on small dataset

### Day 5-7: Action Quantizer
- Implement action VQ/LFQ
- Integrate with encoder
- Test action diversity
- Measure codebook usage

---

**Ready to start Phase 2?** Let's build interactive, controllable video generation! 🎮

---

*Next: Week 5 - Action Encoder & Quantizer*
*Hardware: Apple M4 Max (sufficient for development)*
*Timeline: 3 weeks (Weeks 5-7)*

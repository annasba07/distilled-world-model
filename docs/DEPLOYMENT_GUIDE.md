# Deployment Guide - Phase 1

This guide covers deploying the Phase 1 video generation pipeline in production environments.

---

## 📋 Table of Contents

1. [Deployment Options](#deployment-options)
2. [Hardware Requirements](#hardware-requirements)
3. [Production Checklist](#production-checklist)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Optimization Tips](#optimization-tips)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Deployment Options

### Option 1: Local GPU Server
**Best for**: On-premise deployment, low latency

- Direct GPU access
- No network latency
- Full control over environment

### Option 2: Cloud GPU (AWS/GCP/Azure)
**Best for**: Scalability, flexibility

- Easy scaling
- Pay-per-use
- Managed infrastructure

### Option 3: Edge Deployment
**Best for**: Low latency, offline operation

- Quantized models
- Optimized for inference
- Limited resources

---

## 💻 Hardware Requirements

### Minimum Requirements

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 3060 (12GB) or equivalent |
| **VRAM** | 4GB minimum, 8GB recommended |
| **CUDA** | 11.0 or higher |
| **CPU** | 4 cores, 8 threads |
| **RAM** | 16GB |
| **Storage** | 50GB SSD |

### Recommended Setup

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 4090 (24GB) |
| **VRAM** | 24GB |
| **CUDA** | 12.0 |
| **CPU** | 8+ cores |
| **RAM** | 32GB+ |
| **Storage** | 100GB+ NVMe SSD |

### Supported GPUs

✅ **Excellent** (Full optimization support):
- RTX 4090, 4080, 4070
- RTX 3090, 3080, 3070, 3060
- A100, A40, A6000

✅ **Good** (FP16 support):
- RTX 2080 Ti, 2080, 2070
- V100, T4

⚠️ **Limited** (FP32 only):
- GTX 1080 Ti, 1080
- Older generations

---

## ✅ Production Checklist

### Pre-Deployment

- [ ] Verify CUDA installation
- [ ] Install PyTorch 2.0+
- [ ] Test all components
- [ ] Run benchmarks
- [ ] Measure memory usage
- [ ] Test error handling
- [ ] Set up logging
- [ ] Configure monitoring

### Optimization

- [ ] Enable FP16 mixed precision
- [ ] Enable torch.compile()
- [ ] Set batch size appropriately
- [ ] Configure context manager
- [ ] Test with production data
- [ ] Benchmark end-to-end latency

### Security

- [ ] Input validation
- [ ] Rate limiting
- [ ] Authentication
- [ ] Secure model storage
- [ ] Logging (no PII)
- [ ] Error handling

### Monitoring

- [ ] GPU utilization
- [ ] Memory usage
- [ ] Latency metrics
- [ ] Error rates
- [ ] Throughput
- [ ] Cost tracking

---

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile for Phase 1 Pipeline
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy project
COPY . /app/

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port (if using API)
EXPOSE 8000

# Run
CMD ["python3", "serve.py"]
```

### requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
```

### Build and Run

```bash
# Build image
docker build -t video-generation:phase1 .

# Run with GPU
docker run --gpus all -p 8000:8000 video-generation:phase1

# Run with specific GPU
docker run --gpus '"device=0"' -p 8000:8000 video-generation:phase1
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  video-generation:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## ☁️ Cloud Deployment

### AWS SageMaker

```python
# deploy_sagemaker.py
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Create model
pytorch_model = PyTorchModel(
    model_data='s3://your-bucket/model.tar.gz',
    role='arn:aws:iam::...your-role',
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py',
    source_dir='src/'
)

# Deploy
predictor = pytorch_model.deploy(
    instance_type='ml.g4dn.xlarge',  # 1x T4 GPU
    initial_instance_count=1
)

# Inference
result = predictor.predict(input_data)
```

### GCP Vertex AI

```python
# deploy_vertex.py
from google.cloud import aiplatform

aiplatform.init(project='your-project', location='us-central1')

# Upload model
model = aiplatform.Model.upload(
    display_name='video-generation-phase1',
    artifact_uri='gs://your-bucket/model/',
    serving_container_image_uri='gcr.io/your-project/video-gen:latest'
)

# Deploy
endpoint = model.deploy(
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

### Azure ML

```python
# deploy_azure.py
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice

ws = Workspace.from_config()

# Register model
model = Model.register(
    workspace=ws,
    model_name='video-generation-phase1',
    model_path='./models/'
)

# Deploy configuration
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=4,
    memory_gb=16,
    gpu_cores=1
)

# Deploy
service = Model.deploy(
    workspace=ws,
    name='video-gen-service',
    models=[model],
    deployment_config=deployment_config
)
```

---

## 🔧 Optimization Tips

### 1. Model Optimization

```python
# Production configuration
pipeline = VideoGenerationPipeline(
    resolution=(640, 360),
    codebook_size=4096,          # Balance quality vs speed
    num_iterations=8,             # Fewer iterations = faster
    use_optimization=True,        # Enable FP16 + compile
    device='cuda'
)

# Optimize models
optimizer = InferenceOptimizer(
    use_fp16=True,               # 2x speedup
    use_compile=True,             # 2x speedup
    use_flash_attn=False          # Optional, requires flash-attn
)
```

### 2. Batch Processing

```python
# Process multiple videos in batch for efficiency
batch_size = 4  # Adjust based on VRAM

videos = pipeline.generate_video(
    batch_size=batch_size,
    num_frames=8
)
```

### 3. Dynamic Batching

```python
# Accumulate requests and process in batches
class BatchProcessor:
    def __init__(self, max_batch_size=4, max_wait_ms=100):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []

    async def add_request(self, request):
        self.queue.append(request)

        # Process if batch is full or timeout
        if len(self.queue) >= self.max_batch_size:
            return await self.process_batch()

    async def process_batch(self):
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]

        # Process batch
        results = pipeline.generate_video(
            batch_size=len(batch),
            num_frames=8
        )

        return results
```

### 4. Caching

```python
# Cache tokenized videos
from functools import lru_cache

@lru_cache(maxsize=128)
def encode_cached(video_hash):
    return pipeline.encode_video(video)

# Use content-based hashing
import hashlib

def hash_video(video):
    return hashlib.md5(video.cpu().numpy().tobytes()).hexdigest()
```

### 5. Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def generate_async(batch_size, num_frames):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        pipeline.generate_video,
        batch_size,
        num_frames
    )
```

---

## 📊 Monitoring

### Metrics to Track

```python
import time
import psutil
import torch

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'latency': [],
            'gpu_memory': [],
            'gpu_utilization': [],
            'throughput': []
        }

    def monitor_inference(self, func):
        def wrapper(*args, **kwargs):
            # Start timing
            start = time.time()

            # GPU memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                memory_before = torch.cuda.memory_allocated()

            # Execute
            result = func(*args, **kwargs)

            # Metrics
            latency = time.time() - start

            if torch.cuda.is_available():
                memory_peak = torch.cuda.max_memory_allocated()
                torch.cuda.synchronize()

                self.metrics['gpu_memory'].append(memory_peak / 1024**2)  # MB

            self.metrics['latency'].append(latency)

            return result

        return wrapper

    def get_stats(self):
        return {
            'avg_latency': sum(self.metrics['latency']) / len(self.metrics['latency']),
            'p95_latency': sorted(self.metrics['latency'])[int(len(self.metrics['latency']) * 0.95)],
            'avg_memory': sum(self.metrics['gpu_memory']) / len(self.metrics['gpu_memory']),
        }
```

### Prometheus Integration

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
requests_total = Counter('video_generation_requests_total', 'Total requests')
latency_histogram = Histogram('video_generation_latency_seconds', 'Latency')
gpu_memory_gauge = Gauge('gpu_memory_mb', 'GPU memory usage')

@latency_histogram.time()
def generate_with_metrics(batch_size, num_frames):
    requests_total.inc()

    result = pipeline.generate_video(batch_size, num_frames)

    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / 1024**2
        gpu_memory_gauge.set(memory_mb)

    return result
```

---

## 🐛 Troubleshooting

### Common Production Issues

#### 1. OOM (Out of Memory)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
batch_size = 1

# Enable gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential

# Use context manager for long videos
use_context_manager = True

# Clear cache between requests
torch.cuda.empty_cache()
```

#### 2. Slow Inference

**Symptoms**: High latency, low throughput

**Solutions**:
```python
# Verify optimizations are enabled
assert optimizer.use_fp16 == True
assert optimizer.use_compile == True

# Check GPU utilization
nvidia-smi

# Profile with PyTorch profiler
with torch.profiler.profile() as prof:
    pipeline.generate_video(1, 8)

print(prof.key_averages().table())
```

#### 3. Quality Issues

**Symptoms**: Poor output quality

**Solutions**:
```python
# Increase iterations
num_iterations = 12  # instead of 8

# Check model is in eval mode
assert not pipeline.tokenizer.training

# Verify no gradients
assert not any(p.requires_grad for p in pipeline.tokenizer.parameters())

# Check perplexity
output = pipeline.tokenizer(video, return_loss=True)
assert output['perplexity'] > 1000
```

#### 4. Memory Leaks

**Symptoms**: Memory usage grows over time

**Solutions**:
```python
# Clear cache periodically
if request_count % 100 == 0:
    torch.cuda.empty_cache()

# Detach tensors
result = result.detach().cpu()

# Delete unused variables
del intermediate_result

# Use context managers
with torch.no_grad():
    result = model(input)
```

---

## 📈 Scaling

### Horizontal Scaling

```python
# Load balancer configuration
# nginx.conf

upstream video_generation {
    least_conn;
    server gpu-worker-1:8000;
    server gpu-worker-2:8000;
    server gpu-worker-3:8000;
    server gpu-worker-4:8000;
}

server {
    listen 80;

    location /generate {
        proxy_pass http://video_generation;
        proxy_timeout 300s;
    }
}
```

### Multi-GPU Deployment

```python
# Distribute across multiple GPUs
import torch.multiprocessing as mp

def worker(gpu_id, queue):
    torch.cuda.set_device(gpu_id)

    pipeline = VideoGenerationPipeline(device=f'cuda:{gpu_id}')

    while True:
        request = queue.get()
        result = pipeline.generate_video(**request)
        # Send result back

# Start workers
num_gpus = torch.cuda.device_count()
queues = [mp.Queue() for _ in range(num_gpus)]

for gpu_id in range(num_gpus):
    mp.Process(target=worker, args=(gpu_id, queues[gpu_id])).start()
```

---

## 🔒 Security

### Input Validation

```python
def validate_input(batch_size, num_frames, resolution):
    # Limits
    MAX_BATCH_SIZE = 16
    MAX_FRAMES = 32
    MAX_RESOLUTION = (1024, 1024)

    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size too large: {batch_size}")

    if num_frames > MAX_FRAMES:
        raise ValueError(f"Too many frames: {num_frames}")

    if resolution[0] > MAX_RESOLUTION[0] or resolution[1] > MAX_RESOLUTION[1]:
        raise ValueError(f"Resolution too large: {resolution}")
```

### Rate Limiting

```python
from time import time

class RateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = []

    def allow_request(self):
        now = time()

        # Remove old requests
        self.requests = [r for r in self.requests if now - r < 60]

        if len(self.requests) >= self.max_requests:
            return False

        self.requests.append(now)
        return True
```

---

## 📚 Additional Resources

- [Phase 1 README](PHASE_1_README.md) - Complete documentation
- [API Reference](PHASE_1_README.md#api-reference) - Component APIs
- [Benchmarks](../benchmarks/README.md) - Performance benchmarks
- [Examples](../examples/) - Working code examples

---

**Ready for production deployment!** 🚀

*Last Updated: October 2025*

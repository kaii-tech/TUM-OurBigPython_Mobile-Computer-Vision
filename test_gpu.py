import tensorflow as tf

print("=" * 60)
print("TensorFlow GPU Test")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU devices found: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  [{i}] {gpu}")

if gpus:
    print("\n✓ GPU is available and ready!")
else:
    print("\n✗ No GPU detected - running on CPU")

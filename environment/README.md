# Environment

`requirements.txt` is a practical local-analysis environment based on the archived execution records. `environment.yml` provides an equivalent Conda setup.

The original GPU experiments used a CUDA-enabled PyTorch 2.10.0 build (`+cu128`) on a Tesla T4. A generic `torch==2.10.0` dependency is used here because CUDA wheel selection depends on the local platform and installation method.

For exact GPU wheel installation, follow the PyTorch instructions appropriate to your CUDA/driver stack, then install the remaining requirements.

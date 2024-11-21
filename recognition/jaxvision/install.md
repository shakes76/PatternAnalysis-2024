## Linux
Just use official instructions for JAX found at https://github.com/google/jax
## Windows
I got the Python WHLs from https://github.com/cloudhan/jax-windows-builder

I downloaded jaxlib-0.1.71+cuda111-cp38-none-win_amd64.whl

Then install jax etc.
pip install .\jaxlib-0.1.71+cuda111-cp38-none-win_amd64.whl
pip install git+https://github.com/deepmind/dm-haiku
pip install rlax jax optax

I had installed CUDA 11.1, but ran into issues where ptxas was being used from wrong CUDA install. I disabled 11.2 by renaming its install directory,

Had errors regarding that wrong cuDNN was install, had to install 8.2.2

JAX worked on Windows successfully after that!

# Source this on a GPU server: finds the CUDA runtime libraries cudarc
# loads at runtime (libnvrtc, libcublas) and puts them on LD_LIBRARY_PATH.
for d in /usr/local/cuda/lib64 /usr/local/cuda-*/lib64 /usr/pkg/cuda*/lib64 /opt/cuda/lib64 \
         $(ls -d "$HOME"/.local/lib/python3*/site-packages/nvidia/*/lib 2>/dev/null) \
         $(ls -d /usr/lib/python3*/site-packages/nvidia/*/lib /usr/local/lib/python3*/*-packages/nvidia/*/lib 2>/dev/null); do
    [ -d "$d" ] && ls "$d" | grep -qE "libnvrtc|libcublas" && LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done
export LD_LIBRARY_PATH

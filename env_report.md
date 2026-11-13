# VisionPort Environment Report

Generated: 2026-11-13 10:11:28.452568

## System Information
- Operating System: Windows-11-10.0.26200-SP0
- Python Version: 3.14.4 (tags/v3.14.4:23116f9, Apr  7 2026, 14:10:54) [MSC v.1944 64 bit (AMD64)]
- Python Executable: C:\Users\aksaa\OneDrive\Desktop\2Even\DeepLeearning\.venv\Scripts\python.exe

## Installed Packages
- torch: 2.15.0
- torchvision: 0.29.2
- ultralytics: 8.4.210
- streamlit: 1.67.0
- fastapi: 0.162.4
- uvicorn: 0.54.0
- pandas: 3.1.1
- pillow: 13.0.0

## NVIDIA GPU Information
```text
Fri Nov 13 10:11:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.86                 Driver Version: 581.86         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   42C    P8              5W /   35W |       0MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           13968    C+G   ...2p2nqsd0c76g0\app\ChatGPT.exe      N/A      |
+-----------------------------------------------------------------------------------------+

```

## PyTorch and CUDA Check
- PyTorch Version: 2.15.0+cpu
- CUDA Available: False
- GPU Matrix Multiplication: SKIPPED
- PyTorch will use CPU for this test.
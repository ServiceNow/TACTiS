"""
Copyright 2023 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import torch


def check_memory(cuda_device):
    """Check the total memory and occupied memory for GPU"""
    devices_info = (
        os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader')
        .read()
        .strip()
        .split("\n")
    )
    total, used = devices_info[int(cuda_device)].split(",")
    return total, used


def occupy_memory(cuda_device):
    """Create a large tensor and delete it.
    This operation occupies the GPU memory, so other processes cannot use the occupied memory.
    It is used to ensure that this process won't be stopped when it requires additional GPU memory.
    Be careful with this operation. It will influence other people when you are sharing GPUs with others.
    """
    total, used = check_memory(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.90)
    print("Total memory: " + str(total) + ", used memory: " + str(used))
    block_mem = max_mem - used
    if block_mem > 0:
        x = torch.cuda.FloatTensor(256, 1024, block_mem)
        del x


def set_gpu(x):
    """Set up which GPU we use for this process"""
    os.environ["CUDA_VISIBLE_DEVICES"] = x
    print("Using gpu:", x)

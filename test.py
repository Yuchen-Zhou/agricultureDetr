import torch
import platform

from Loggers import logger

def hardware_detect():
    logger.info('开始检测硬件环境...')
    current_os = platform.system()

    if current_os == 'Darwin':
        if torch.backends.mps.is_available():
            logger.info('MPS(Apple芯片上的GPU)可用')
        else:
            logger.error('MPS不可用')
    else:
        # Others
        if torch.cuda.is_available():
            logger.info(f"CUDA GPU 可用: {torch.cuda.get_device_name(0)}, CUDA 设备数: {torch.cuda.device_count()}")
        else:
            logger.error('CUDA GPU 不可用')


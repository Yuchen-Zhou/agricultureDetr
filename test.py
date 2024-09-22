import time
import torch
import platform

from Loggers import logger
from Networks import LinearRegressionModel
from utils import timer, train_model

def hardware_test() -> str:
    GPU_DEVICE: str = ''
    logger.info('开始检测硬件环境...')
    current_os = platform.system()

    if current_os == 'Darwin':
        if torch.backends.mps.is_available():
            GPU_DEVICE = 'mps'
            logger.info('MPS(Apple芯片上的GPU)可用')
        else:
            logger.error('MPS不可用')
    else:
        # Others
        if torch.cuda.is_available():
            GPU_DEVICE = 'cuda'
            logger.info(f"CUDA GPU 可用: {torch.cuda.get_device_name(0)}, CUDA 设备数: {torch.cuda.device_count()}")
        else:
            logger.error('CUDA GPU 不可用')
    return GPU_DEVICE

@timer
def network_test(device: str):
    logger.info(f'在 {device}上测试网络...')
    torch.manual_seed(42)

    X_train = torch.linspace(0, 10, 100).reshape(-1, 1)
    Y_train_1 = 2 * X_train + 1 + 0.5 * torch.randn(X_train.size())
    Y_train_2 = -3 * X_train + 7 + 0.5 * torch.randn(X_train.size())
    Y_train_3 = 0.5 * X_train - 5 + 0.5 * torch.randn(X_train.size())

    model_1 = LinearRegressionModel(1, 1)
    model_2 = LinearRegressionModel(1, 1)
    model_3 = LinearRegressionModel(1, 1)

    criterion = torch.nn.MSELoss()
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.01)
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01)
    optimizer_3 = torch.optim.SGD(model_3.parameters(), lr=0.01)

    train_model(model_1, optimizer=optimizer_1, criterion=criterion, X_train=X_train, Y_train=Y_train_1, device=device)
    train_model(model_2, optimizer=optimizer_2, criterion=criterion, X_train=X_train, Y_train=Y_train_2, device=device)
    train_model(model_3, optimizer=optimizer_3, criterion=criterion, X_train=X_train, Y_train=Y_train_3, device=device)

def self_test():
    logger.info('开始自检...')
    current_os = platform.system()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


    logger.info('在 CPU 上测试...')
    network_test('cpu')

    logger.info(f'在GPU 上测试...')
    network_test(device)
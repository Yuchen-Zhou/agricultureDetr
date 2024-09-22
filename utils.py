import time
import torch
from tqdm import tqdm
from functools import wraps

from Loggers import logger

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__}的执行时间为: {end_time - start_time:.8f}秒")
        return result
    return wrapper

def train_model(model, *, optimizer, X_train, Y_train, criterion, epochs=100, batch_size=32, **_):
    device = _['device'] if 'device' in _ else torch.device('cpu')
    # 获取数据总量和批次数量

    num_samples = X_train.size(0)
    num_batches = num_samples // batch_size

    # 创建一个固定的索引张量
    indices = torch.arange(num_samples)

    # 使用 tqdm 创建进度条
    with tqdm(range(epochs), desc="Training", unit="epoch") as pbar:
        model.to(device)
        for epoch in pbar:
            model.train()
            epoch_loss = 0.0

            # 将数据打乱顺序
            shuffled_indices = indices[torch.randperm(num_samples)]

            # 遍历每个批次
            for i in range(num_batches):
                # 获取当前批次的数据
                batch_indices = shuffled_indices[i * batch_size: (i + 1) * batch_size]
                X_batch = X_train[batch_indices].to(device)
                Y_batch = Y_train[batch_indices].to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                output = model(X_batch)
                loss = criterion(output, Y_batch)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 累加每个批次的损失
                epoch_loss += loss.item()

            # 计算平均损失
            avg_loss = epoch_loss / num_batches

            # 更新进度条的附加信息
            pbar.set_postfix({'Loss': avg_loss})

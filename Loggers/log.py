import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

class Logger:
    def __init__(self, log_level: int = logging.DEBUG):
        """
        初始化Logger类, 默认在项目根目录下创建./logs文件夹，并按照日期划分日志文件.
        :param log_level: 日志级别，默认是 DEBUG
        """
        self.project_root = self.__get_root()

        self.log_dir = os.path.join(self.project_root, 'logs')
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        log_file = os.path.join(self.log_dir, f'{datetime.now().strftime("%Y%m%d")}.log')

        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

        file_handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
        file_handler.setLevel(log_level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    @staticmethod
    def __get_root() -> str:
        """
        获取项目根目录
        :return:
        """
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(current_file_path)

        while not os.path.exists(os.path.join(project_root, 'pyproject.toml')):
            new_root = os.path.dirname(project_root)

            if new_root == project_root:
                raise FileNotFoundError("未找到项目根目录")
            project_root = new_root
        return project_root

    def get_logger(self):
        return self.logger

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)

logger = Logger()
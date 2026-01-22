"""
日志工具
同时输出到控制台和文件
"""

import os
import logging


def get_logger(file_name, file_save=True, display=True):
    """
    创建logger，同时输出到控制台和文件

    Args:
        file_name: 日志文件路径
        file_save: 是否保存到文件
        display: 是否显示到控制台

    Returns:
        配置好的logger对象
    """
    # 获取文件所在目录
    dir_name = os.path.dirname(file_name)
    # 如果目录不存在，则创建目录
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")

    logger = logging.Logger(file_name, logging.INFO)
    logger.setLevel(logging.INFO)

    # file handle
    if file_save:
        if os.path.isfile(file_name):
            fh = logging.FileHandler(file_name, mode='a', encoding='utf-8')
        else:
            fh = logging.FileHandler(file_name, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # controler handle
    if display:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

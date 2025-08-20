import os
import shutil
import re
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple, Set
import numpy as np
import pandas as pd
import threading
import queue
from typing import List


class ProcessingAggregator:
    def __init__(self) -> None:
        self.total_count: int = 0
        self.success_count: int = 0
        self.non_name_set: Set[str] = set()
        self.non_artist_list: List[List[str]] = []
        self.lock = threading.Lock()

    def add_total(self) -> None:
        with self.lock:
            self.total_count += 1

    def add_success(self) -> None:
        with self.lock:
            self.success_count += 1

    def add_non_name(self, names: List[str]) -> None:
        with self.lock:
            for n in names:
                self.non_name_set.add(n)

    def add_non_artist_row(self, row: List[str]) -> None:
        with self.lock:
            self.non_artist_list.append(row)


def setup_logging(log_file_path: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    初始化日志：控制台 + 轮转文件日志。
    """
    if logger.handlers:
        return

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(base_dir, "logs", "file_sorter.log")

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = RotatingFileHandler(log_file_path, maxBytes=100 * 1024 * 1024, backupCount=10, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def file_scanner(base_path: str, q: "queue.Queue[Optional[str]]", limit: Optional[int] = None) -> int:
    """
    递归扫描文件夹，把符合后缀的文件路径放进队列。
    返回入队的文件数量。
    """
    count = 0
    for root, _dirs, files in os.walk(base_path):
        for fname in files:
            if not fname.endswith((".mp3", ".jpg", ".lrc")):
                continue
            q.put(os.path.join(root, fname))
            count += 1
            if limit is not None and count >= limit:
                logger.info("扫描已达到限制：%d", limit)
                return count
    return count

def set_to_excel_column(output_file: str, string_set: Set[str]) -> None:
    """
    将字符串集合写入Excel的一列
    
    参数:
        output_file: 输出Excel文件路径
        string_set: 字符串集合(set)
    """
    # 将set转换为DataFrame
    df = pd.DataFrame({"non_name": sorted(string_set)})
    # 写入Excel
    df.to_excel(output_file, index=False)
    logger.info("成功写入 %d 个字符串到 %s", len(string_set), output_file)

def write_nested_list_to_excel(output_file, data):
    """
    使用pandas将嵌套列表写入Excel
    
    参数:
        output_file: 输出文件路径
        data: 嵌套列表(list[list])
    """
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False, header=False)
    logger.info("成功写入数据到 %s", output_file)

def extrac_file_name(file_name: str) -> Optional[Tuple[str, str, str, str, str]]:
    """
    从文件名中提取歌曲名、歌手名、专辑名、时长和文件类型
    【歌名：5 минут (Mantr Remix) 】+【歌手：5sta Family】+【专辑名：5 минут (Mantr Remix) 】+【时长：3：24】
    """
    if not file_name.endswith(('.mp3', '.jpg', '.lrc')):
        logger.debug("文件后缀不匹配，跳过：%s", file_name)
        return None
    pattern_1 = r'【歌名：(.*?)】'
    pattern_2 = r'【歌手：(.*?)】'
    pattern_3 = r'【专辑名：(.*?)】'
    pattern_4 = r'【时长：(.*?)】'
    # pattern_file = r'.*?\.(jpg|mp3|lrc)$'
    m1 = re.search(pattern_1, file_name)
    m2 = re.search(pattern_2, file_name)
    m3 = re.search(pattern_3, file_name)
    m4 = re.search(pattern_4, file_name)

    if m1 is None or m2 is None or m3 is None or m4 is None:
        logger.warning("正则未匹配到完整信息，跳过：%s", file_name)
        return None

    song_name = m1.group(1)
    artist_name = m2.group(1)
    album_name = m3.group(1)
    time_expand = m4.group(1)
    file_type = os.path.splitext(file_name)[1]
    return song_name,artist_name,album_name,time_expand,file_type

def find_name(name_list, df: pd.DataFrame):
    arr = df.astype(str).values
    
    for name in name_list:
        # 获取所有匹配位置
        matches = np.where(arr == name)
        if len(matches[0]) > 0:
            row, col = matches[0][0], matches[1][0]
            return name, df.columns[col], row
    return None

def sanitize_windows_path_component(name: str) -> str:
    """
    清洗 Windows 路径组件：
    - 替换非法字符 <>:"/\\|?* 为 下划线
    - 去除结尾的 点/空格
    - 避免保留设备名（CON/PRN/AUX/NUL/COM1../LPT1..）
    """
    if not name:
        return "_"
    invalid_chars = r'<>:"/\\|?*'
    cleaned = ''.join('_' if ch in invalid_chars else ch for ch in name)
    cleaned = cleaned.rstrip('.')
    if not cleaned:
        cleaned = "_"

    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1,10)} | {f"LPT{i}" for i in range(1,10)}
    if cleaned.upper() in reserved:
        cleaned = f"_{cleaned}_"
    return cleaned

def classify_string(input_str: str, df: pd.DataFrame) -> str:

    """
    对输入字符串进行分类标记
    
    参数:
        input_str: 要分类的字符串
        excel_file: Excel文件路径
        
    返回:
        分类标记 (1, 2, 3 或 0)
    """
    try:
        
        # 检查字符串在哪一列中出现
        for index, row in df.iterrows():
            if str(row[0]) == input_str:
                return '欧美男歌手'
            if len(row) > 1 and str(row[1]) == input_str:
                return '欧美女歌手'
            if len(row) > 2 and str(row[2]) == input_str:
                return '欧美组合'
                
        # 如果三列中都没有找到
        return '其他'
        
    except Exception as e:
        logger.exception("分类时发生错误：%s", e)
        return "其他"

def create_dir(dir_path: str) -> None:
    """
    创建目录
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug("确保目录存在：%s", dir_path)
    except Exception as e:
        logger.warning("创建目录失败：%s，原因：%s", dir_path, e)
    return None

def check_file_exists(target_dir: str, filename: str) -> bool:
    """
    检查目标目录中是否已存在相同文件名的文件
    
    参数:
        target_dir: 目标目录路径
        filename: 文件名
        
    返回:
        True: 文件已存在，False: 文件不存在
    """
    target_file_path = os.path.join(target_dir, filename)
    return os.path.exists(target_file_path)

def merge_or_move_folder(src: str, dst: str) -> None:
    """
    合并或移动路径（兼容文件与文件夹）。
    :param src: 源路径（文件或文件夹）
    :param dst: 目标路径（文件或文件夹）
    """
    if os.path.isdir(src):
        if os.path.exists(dst):
            logger.debug("合并文件夹：%s -> %s", src, dst)
            try:
                items = os.listdir(src)
            except PermissionError as e:
                logger.warning("拒绝访问源目录，跳过合并：%s，原因：%s", src, e)
                return
            for item in items:
                src_path = os.path.join(src, item)
                dst_path = os.path.join(dst, item)

                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(item)
                    counter = 1
                    while os.path.exists(dst_path):
                        new_name = f"{base}_{counter}{ext}" if ext else f"{base}_{counter}"
                        dst_path = os.path.join(dst, new_name)
                        counter += 1
                try:
                    shutil.move(src_path, dst_path)
                except PermissionError as e:
                    logger.warning("拒绝访问，跳过移动：%s -> %s，原因：%s", src_path, dst_path, e)
                    continue

            try:
                os.rmdir(src)
            except OSError:
                logger.debug("源文件夹非空或无法删除，保留：%s", src)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            logger.debug("移动文件夹：%s -> %s", src, dst)
            try:
                os.rename(src, dst)
            except PermissionError as e:
                logger.warning("拒绝访问，跳过文件夹移动：%s -> %s，原因：%s", src, dst, e)
    else:
        # 处理为单个文件
        dst_dir = os.path.dirname(dst) or "."
        os.makedirs(dst_dir, exist_ok=True)
        final_dst = dst
        if os.path.exists(final_dst):
            base, ext = os.path.splitext(os.path.basename(final_dst))
            counter = 1
            while os.path.exists(final_dst):
                new_name = f"{base}_{counter}{ext}" if ext else f"{base}_{counter}"
                final_dst = os.path.join(dst_dir, new_name)
                counter += 1
        logger.debug("移动文件：%s -> %s", src, final_dst)
        try:
            shutil.move(src, final_dst)
        except PermissionError as e:
            logger.warning("拒绝访问，跳过文件移动：%s -> %s，原因：%s", src, final_dst, e)


def process_single_file(file_path: str, dir_path: str, df: pd.DataFrame, aggregator: ProcessingAggregator) -> None:
    """处理单个文件的分类与移动"""
    base_name = os.path.basename(file_path)
    result = extrac_file_name(base_name)
    aggregator.add_total()
    if not result:
        logger.warning("文件名无法解析或格式不匹配，跳过：%s", base_name)
        return
    x1, x2, x3, x4, x5 = result
    name_list = x2.split("：")
    found = find_name(name_list, df)
    if found is None:
        logger.info("未在表中找到：%s/%s", name_list, base_name)
        name = name_list[0]
        aggregator.add_non_artist_row([x1, x2, x3, x4])
        aggregator.add_non_name(name_list)
        # return
    else:
        name = found[0]
    aggregator.add_success()
    
    safe_name = sanitize_windows_path_component(name)
    category = classify_string(name, df)

    if category == '欧美男歌手':
        target_dir = os.path.join(dir_path, '欧美男歌手', safe_name)
    elif category == '欧美女歌手':
        target_dir = os.path.join(dir_path, '欧美女歌手', safe_name)
    elif category == '欧美组合':
        target_dir = os.path.join(dir_path, '欧美组合', safe_name)
    else:
        # 如需分类到“其他”，可取消注释
        target_dir = os.path.join(dir_path, '其他', safe_name)
        # return
    logger.info("成功进度：%d/%d/%s", aggregator.success_count, aggregator.total_count,file_path)
    create_dir(target_dir)
    
    # 检查目标路径是否已存在相同文件名
    if check_file_exists(target_dir, base_name):
        logger.info("目标路径已存在相同文件名，跳过：%s -> %s", base_name, target_dir)
        return
    
    merge_or_move_folder(file_path, os.path.join(target_dir, base_name))


def worker(q: "queue.Queue[Optional[str]]", dir_path: str, df: pd.DataFrame, aggregator: ProcessingAggregator) -> None:
    """线程函数：阻塞等待任务，直到收到哨兵 None 再退出"""
    while True:
        file_path = q.get()  # 阻塞等待任务，避免超时导致线程提前退出
        try:
            if file_path is None:
                return
            try:
                process_single_file(file_path, dir_path, df, aggregator)
            except Exception as e:
                logger.exception("处理失败，已跳过：%s，原因：%s", file_path, e)
        finally:
            q.task_done()

def move_directory_multithreaded(source_dir: str, dir_path: str, df: pd.DataFrame,
                                 scan_limit: Optional[int] = None, num_threads: int = 8) -> None:
    """
    多线程处理整个目录：
    - 扫描 source_dir 下的目标文件(.mp3/.jpg/.lrc)入队
    - 多线程消费，分类并移动
    - 汇总并输出 Excel
    """
    if df is None:
        raise ValueError("df 不能为空，请传入已加载的 DataFrame")
    logger.info("-----------------------------------------------------------------------------------")
    logger.info("-----------------------------------------------------------------------------------")
    logger.info("-----------------------------------------------------------------------------------")
    logger.info("开始多线程处理：source=%s, target=%s, threads=%d", source_dir, dir_path, num_threads)

    task_queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=1000)
    aggregator = ProcessingAggregator()

    def scan_job() -> None:
        try:
            count = file_scanner(source_dir, task_queue, scan_limit)
            logger.info("扫描完成，入队文件数：%d", count)
        finally:
            for _ in range(num_threads):
                task_queue.put(None)
    scan_thread = threading.Thread(target=scan_job, name="scanner")
    scan_thread.start()
    workers: List[threading.Thread] = []
    for idx in range(num_threads):
        t = threading.Thread(target=worker, args=(task_queue, dir_path, df, aggregator), name=f"worker-{idx+1}")
        t.start()
        workers.append(t)
    scan_thread.join()
    task_queue.join()
    for t in workers:
        t.join()

    logger.info("处理完成，总数%d，成功%d", aggregator.total_count, aggregator.success_count)
    logger.info("找不到名字：%s", aggregator.non_name_set)
    base_file_path = os.path.dirname(os.path.abspath(__file__))
    non_name_path = os.path.join(base_file_path,"non_name.xlsx")
    non_artist_path = os.path.join(base_file_path,"non_artist.xlsx")
    set_to_excel_column(non_name_path, aggregator.non_name_set)
    write_nested_list_to_excel(non_artist_path, aggregator.non_artist_list)
    logger.info("多线程处理结束。")


def move_file(source_dir: str, dir_path: str, df: Optional[pd.DataFrame] = None, length: Optional[int] = None) -> None:

    """
    移动文件
    """
    # if not os.path.exists(source_dir):
    #     return None
    # else:
    #     shutil.move(file_path,dir_path)
    #     return None
    if df is None:
        raise ValueError("df 不能为空，请传入已加载的 DataFrame")
    logger.info("-----------------------------------------------------------------------------------")
    logger.info("-----------------------------------------------------------------------------------")
    logger.info("-----------------------------------------------------------------------------------")
    logger.info("开始移动文件：source=%s, target=%s", source_dir, dir_path)
    all_files = os.listdir(source_dir)
    # 筛选出以 .mp3, .jpg, .lrc 结尾的文件
    filtered_files = [file for file in all_files if file.endswith(('.mp3', '.jpg', '.lrc'))]
    if length:
        filtered_files = filtered_files[:length]
    else:
        length = len(filtered_files)
    all_count = 0
    correct_count = 0
    non_name_list = set()
    non_artist_list = []

    for i in filtered_files:
        all_count += 1
        logger.debug("进度 %.2f%% - 文件：%s", all_count / length * 100, i[:50])
        result = extrac_file_name(i)
        if not result:
            logger.warning("文件名无法解析或格式不匹配，跳过：%s", i)
            continue
        x1, x2, x3, x4, x5 = result
        name_list = x2.split("：")
        name = find_name(name_list, df)
        if name == None:
            logger.info("未在表中找到：%s/%s", name_list,i)
            non_artist_list.append([x1,x2,x3,x4])
            name = name_list[0]
            for noname in name_list:
                non_name_list.add(noname)
            # continue
        else:
            name = name[0]
        correct_count += 1
        logger.info("成功进度：%d/%d/%d/%s", correct_count, all_count,length,i)
        
        safe_name = sanitize_windows_path_component(name)
        result = classify_string(name, df)
        if result == '欧美男歌手':
            create_dir(os.path.join(dir_path, '欧美男歌手'))
            target_dir = os.path.join(dir_path, '欧美男歌手', safe_name)
            create_dir(target_dir)
            # 检查目标路径是否已存在相同文件名
            if check_file_exists(target_dir, i):
                logger.info("目标路径已存在相同文件名，跳过：%s -> %s", i, target_dir)
                continue
            merge_or_move_folder(os.path.join(source_dir, i), os.path.join(target_dir, i))
        elif result == '欧美女歌手':
            create_dir(os.path.join(dir_path, '欧美女歌手'))
            target_dir = os.path.join(dir_path, '欧美女歌手', safe_name)
            create_dir(target_dir)
            # 检查目标路径是否已存在相同文件名
            if check_file_exists(target_dir, i):
                logger.info("目标路径已存在相同文件名，跳过：%s -> %s", i, target_dir)
                continue
            merge_or_move_folder(os.path.join(source_dir, i), os.path.join(target_dir, i))   
        elif result == '欧美组合':
            create_dir(os.path.join(dir_path, '欧美组合'))
            target_dir = os.path.join(dir_path, '欧美组合', safe_name)
            create_dir(target_dir)
            # 检查目标路径是否已存在相同文件名
            if check_file_exists(target_dir, i):
                logger.info("目标路径已存在相同文件名，跳过：%s -> %s", i, target_dir)
                continue
            merge_or_move_folder(os.path.join(source_dir, i), os.path.join(target_dir, i))
        else:
            create_dir(os.path.join(dir_path, '其他'))
            target_dir = os.path.join(dir_path, '其他', safe_name)
            create_dir(target_dir)
            # 检查目标路径是否已存在相同文件名
            if check_file_exists(target_dir, i):
                logger.info("目标路径已存在相同文件名，跳过：%s -> %s", i, target_dir)
                continue
            merge_or_move_folder(os.path.join(source_dir, i), os.path.join(target_dir, i))
        # print(i, result)
    logger.info("处理完成，总数%d，成功%d", all_count, correct_count)
    logger.info("找不到名字：%s", non_name_list)
    base_file_path = os.path.dirname(os.path.abspath(__file__))
    non_name_path = os.path.join(base_file_path,"non_name.xlsx")
    non_artist_path = os.path.join(base_file_path,"non_artist.xlsx")
    set_to_excel_column(non_name_path, non_name_list)
    write_nested_list_to_excel(non_artist_path, non_artist_list)
    return None


def main():
    setup_logging()
    # 示例路径，可按需修改
    source_dir = r"\\?\C:\Users\Desktop\music\download"
    dir_path = r"\\?\C:\Users\Desktop\music\download"
    excel_path = r"C:\Users\Desktop\test.xlsx"
    df = pd.read_excel(excel_path, header=None)
    # 多线程整体处理入口
    # move_directory_multithreaded(source_dir, dir_path, df, scan_limit=None, num_threads=128)
    # 单线程整体处理入口
    move_file(source_dir, dir_path, df, length = None)

if __name__ == "__main__":
    logger = logging.getLogger("file_sorter")
    main()
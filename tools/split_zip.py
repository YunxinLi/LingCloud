import os
import zipfile
from tqdm import tqdm
import argparse

def split_zip(input_zip_path, output_dir, chunk_size):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开大的zip文件
    with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
        # 获取zip中的文件列表
        file_list = input_zip.namelist()

        # 初始化chunk计数器和当前chunk大小
        current_chunk = 1
        current_chunk_size = 0

        # 初始化当前chunk的zip文件
        current_output_zip_path = os.path.join(output_dir, f"chunk{current_chunk}.zip")
        current_output_zip = zipfile.ZipFile(current_output_zip_path, 'w', zipfile.ZIP_DEFLATED)

        for file_name in tqdm(file_list):
            # 获取当前文件的大小
            file_size = input_zip.getinfo(file_name).file_size

            # 如果当前chunk大小加上当前文件大小超过了chunk_size，就需要切换到下一个chunk
            if current_chunk_size + file_size > chunk_size:
                current_chunk += 1
                current_chunk_size = 0
                current_output_zip.close()
                current_output_zip_path = os.path.join(output_dir, f"chunk{current_chunk}.zip")
                current_output_zip = zipfile.ZipFile(current_output_zip_path, 'w', zipfile.ZIP_DEFLATED)

            # 将当前文件写入当前chunk的zip文件中
            current_output_zip.writestr(file_name, input_zip.read(file_name))
            current_chunk_size += file_size

        # 关闭最后一个chunk的zip文件
        current_output_zip.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Config data')
    parser.add_argument('--input_zip_file', type = str)
    args = parser.parse_args()
    input_zip_file = args.input_zip_file                                                        # 大的zip文件路径
    output_directory = input_zip_file.replace('.zip', '/').replace('image', 'data_split')       # 输出目录
    chunk_size_in_bytes = 3 * 1024 * 1024 * 1024                                                # 3GB的chunk大小

    split_zip(input_zip_file, output_directory, chunk_size_in_bytes)
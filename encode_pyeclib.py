# Copyright (c) 2013, Kevin Greenan (kmgreen2@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.  THIS SOFTWARE IS
# PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
# NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pyeclib.ec_iface import ECDriver
import argparse
import os
import psutil
import time
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from mmap import mmap, ACCESS_READ
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import numpy as np


# 1.Combine all data
# 2.Split the data into smaller block
parser = argparse.ArgumentParser(description='Encoder for PyECLib.')
parser.add_argument('-k', default=4, type=int, help='number of data elements')
parser.add_argument('-m', default=1, type=int, help='number of parity elements')
parser.add_argument('-ec_type', default="isa_l_rs_vand", help='EC algorithm used')
parser.add_argument('-file_dir', default="/data/lff/eccheck/", help='directory with the files')
parser.add_argument('-filenames', nargs='+', help='files to encode')
parser.add_argument('-filenumber', default=4, type=int, help='number of files to encode')
parser.add_argument('-fragment_dir', default="/data/lff/eccheck/ecdir/", help='directory to drop encoded fragments')
parser.add_argument('-output_path', default="/data/lff/eccheck/ecdir/", help='directory to acquire encoded fragments for decoding')
parser.add_argument('-output',default="/data/lff/eccheck/ecdir/result")

_ec_driver = None
args = parser.parse_args()

def init_worker(k, m, ec_type):
    global _ec_driver
    _ec_driver = ECDriver(k=k, m=m, ec_type=ec_type)

def encode_chunk(args, shm_name, chunk_index, chunk_start, chunk_size):
    existing_shm = None
    try:
        existing_shm = SharedMemory(name=shm_name)
        # 从共享内存中读取块数据
        chunk_data = existing_shm.buf[chunk_start:chunk_start + chunk_size].tobytes()
        
        # 编码当前块
        fragments = _ec_driver.encode(chunk_data)
        
        # 写入分片文件（确保目录存在）
        if not os.path.exists(args.fragment_dir):
            os.makedirs(args.fragment_dir, exist_ok=True)
            
        for i, fragment in enumerate(fragments):
            fragment_filename = f'chunk_{chunk_index}_fragment_{i}.pt'
            with open(os.path.join(args.fragment_dir, fragment_filename), 'wb') as f:
                f.write(fragment)
        return True
    except Exception as e:
        print(f"Error encoding chunk {chunk_index}: {str(e)}")
        return False
    finally:
        if existing_shm:
            existing_shm.close()

def encode_mmap_chunk(chunk_idx, mmap_path, chunk_size):
    # 每个进程独立映射
    with open(mmap_path, "rb") as f:
        mm = mmap(f.fileno(), 0, access=ACCESS_READ)
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(mm))
        chunk_data = mm[start:end]
        fragments = _ec_driver.encode(chunk_data)
        
        # 写入分片文件（确保目录存在）
        if not os.path.exists(args.fragment_dir):
            os.makedirs(args.fragment_dir, exist_ok=True)
            
        for i, fragment in enumerate(fragments):
            fragment_filename = f'chunk_{chunk_idx}_fragment_{i}.pt'
            with open(os.path.join(args.fragment_dir, fragment_filename), 'wb') as f:
                f.write(fragment)
        mm.close()

def main():
    print("k = %d, m = %d" % (args.k, args.m))
    print("ec_type = %s" % args.ec_type)
    print("filenames = %s" % args.filenames)

    start = time.time()
    tmpfs_file = combined_mmap_files(cell_size = 1024)
    start_encode = time.time()

    with open(tmpfs_file, "rb") as f:
        # 0表示映射全部内容
        mm = mmap(f.fileno(), 0, access=ACCESS_READ)
        # 分块参数
        chunk_size = 64 * 1024 * 1024  # 64MB
        total_chunks = (len(mm) + chunk_size -1) // chunk_size
        num_workers = min(os.cpu_count(), total_chunks)
        print(f"Chunk size: {chunk_size} ||| Total chunks: {total_chunks} ||| Using workers: {num_workers}")
        # 多进程编码
        with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(args.k, args.m, args.ec_type)) as pool:
            tasks = [(i, tmpfs_file, chunk_size) for i in range(total_chunks)]
            for i in range(0, len(tasks), num_workers):
                batch = tasks[i:i + num_workers]
                pool.starmap_async(encode_mmap_chunk, batch)
            pool.close()
            pool.join() 
        mm.close()
    os.remove(tmpfs_file)  # 清理临时文件
    end = time.time()
    print(f"Total encode time: {end - start_encode:.2f} seconds")
    print(f"Prepared time: {start_encode - start:.2f} seconds")

def combined_texts(cell_size = 1024):
    # Combine the byte streams from text files into a single byte stream
    combined_byte_stream = b""
    for filename in args.filenames:
        try:
            with open(f"{args.file_dir}/{filename}", 'rb') as file:  # 以二进制模式打开文件
                byte_data = file.read()
                print(f"{args.file_dir}/{filename}" + " added")
                combined_byte_stream += byte_data  # 合并字节流，并添加空格分隔
        except FileNotFoundError:
            print(f"File {filename} not found.")
            continue
    
    total_length = len(combined_byte_stream)
    aligned_length = ((total_length + (args.k * cell_size) - 1) // (args.k * cell_size)) * (args.k * cell_size)
    combined_byte_stream += (aligned_length-total_length) * b'\0'
    return combined_byte_stream

def combined_mmap_files(cell_size = 1024):
    # 在tmpfs目录直接创建内存映射文件
    tmpfs_path = "/home/lff/eccheck/ckpt/combined.dat"
    total_size = sum(os.path.getsize(f"{args.file_dir}/{f}") for f in args.filenames)
    aligned_size = ((total_size + (args.k * cell_size -1)) // (args.k * cell_size)) * (args.k * cell_size)

    # 创建并扩展文件
    with open(tmpfs_path, "wb") as f:
        f.truncate(aligned_size)  # 直接创建对齐大小的文件
    
    # 内存映射写入
    with open(tmpfs_path, "r+b") as f:
        mm = mmap(f.fileno(), aligned_size)
        offset = 0
        for filename in args.filenames:
            with open(f"{args.file_dir}/{filename}", "rb") as src_file:
                src_mm = mmap(src_file.fileno(), 0, access=ACCESS_READ)
                mm[offset:offset+len(src_mm)] = src_mm
                offset += len(src_mm)
                src_mm.close()
        if offset != aligned_size:
            mm[offset:aligned_size] = b'\0'*(aligned_size-offset)
        mm.close()
    return tmpfs_path

def read_fragment(path):
    """使用内存映射加速分片读取"""
    with open(path, 'rb') as f:
        mm = mmap(f.fileno(), 0, access=ACCESS_READ)
        data = bytes(mm)  # 拷贝到进程内存
        mm.close()
    return data

if __name__ == "__main__":
    main()
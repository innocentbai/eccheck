import fcntl
import os
import time
import torch
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import logging
import json
import shutil


logger = logging.getLogger("IterCheckpoint")
formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s', '%H:%M:%S')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)

def compare_params(file1, file2):
    # 加载参数文件
    params1 = torch.load(file1, weights_only=True)
    params2 = torch.load(file2, weights_only=True)
    
    # 比较参数张量是否完全相同
    if len(params1) != len(params2):
        return False, "参数数量不一致"
    
    for p1, p2 in zip(params1, params2):
        if not torch.equal(p1.detach().cpu(), p2.detach().cpu()):
            return False, "参数值不同"
    
    return True, "参数完全相同"

def compare_state(file1, file2):
    # 假设状态文件保存为字典格式
    state1 = torch.load(file1, weights_only=True)
    state2 = torch.load(file2, weights_only=True)
    
    # 比较字典键是否一致
    if state1.keys() != state2.keys():
        return False, "id(p)不一致"
    
    # 比较每个键对应的值
    for key in state1.keys():
        val1 = state1[key]
        val2 = state2[key]
        if val1.keys() != val2.keys():
            return False, "state的键不一致"
        
    for key in state1.keys():
        val1 = state1[key]
        val2 = state2[key]
        for kk in val1.keys():
            vv1 = val1[kk]
            vv2 = val2[kk]
            if isinstance(vv1, torch.Tensor):
                if not torch.equal(vv1.detach().cpu(), vv2.detach().cpu()):
                    return False, f"state的torch值不同"
                
    return True, "状态字典完全相同"

def info(file):
    # 假设状态文件保存为字典格式
    state = torch.load(file, weights_only=True)
    
    print(state.keys())
    index = 1
    # 比较每个键对应的值
    for key in state.keys():
        val = state[key]
        print(f"{index}: {val.keys()}")
        index += 1

def xmain():
#     start = time.time()
#     process = psutil.Process()
#     print("Memory usage in the beginning:", process.memory_info().rss / (1024 * 1024), "MB")
#     print("k = %d, m = %d" % (args.k, args.m))
#     print("ec_type = %s" % args.ec_type)
#     print("filenames = %s" % args.filenames)

#     aligned_data = b""
#     aligned_data = combined_texts(args) 
#     print("Final memory usage:", process.memory_info().rss / (1024 ** 2), "MB")

#     shm = SharedMemory(create=True, size=len(aligned_data))
#     shm.buf[:len(aligned_data)] = aligned_data
#     print(f"Shared memory created: {len(aligned_data)} bytes")
#     print("Final memory usage:", process.memory_info().rss / (1024 ** 2), "MB")

#     # 优化分块策略（调整为更小的块以提升并行度）
#     chunk_size = 64 * 1024 * 1024  # 调整为64MB以增加并行度
#     total_chunks = (len(aligned_data) + chunk_size - 1) // chunk_size
#     print(f"Total chunks: {total_chunks}, Chunk size: {chunk_size}")
    
#     # 准备进程池
#     num_workers = min(os.cpu_count(), total_chunks)
#     print(f"Using {num_workers} workers")
#     end = time.time()
#     print(f"Total prepare time: {end - start:.2f} seconds")
    
#     start_time = time.time()
    
#     # 使用进程池处理编码
#     with multiprocessing.Pool(
#         processes=num_workers,
#         initializer=init_worker,
#         initargs=(args.k, args.m, args.ec_type)
#     ) as pool:
#         # 准备任务参数：块索引，起始偏移，块大小
#         tasks = [
#             (args, shm.name, i, i * chunk_size, chunk_size)
#             for i in range(total_chunks)
#         ]
#         # 分批发送任务避免内存溢出
#         results = []
#         for i in range(0, len(tasks), num_workers):
#             print(i)
#             batch = tasks[i:i + num_workers]
#             print(len(batch))
#             pool.starmap_async(encode_chunk, batch)
#         pool.close()
#         pool.join() 

#     # 清理共享内存
#     shm.close()
#     shm.unlink()
    
#     end_time = time.time()
#     print(f"Total encoding time: {end_time - start_time:.2f} seconds")
#     print("Final memory usage:", process.memory_info().rss / (1024 ** 2), "MB")
    return True

def test(x, logger):
    logger.info(f"{x}")

def setlog():
    # 创建一个名为 "IterCheckpoint" 的日志记录器
    logger = logging.getLogger("Checkpoint")
    # 同时输出日志到文件IterCheckpoint.log
    fh = logging.FileHandler('Checkpoint.log',encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s', '%H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 同时输出到控制台
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    return logger

def main():    

    # for i in range(4):
    #     result, message = compare_params(
    #         f"/data/lff/eccheck/params_slice_{i}.pt",
    #         f"/data/lff/eccheck/params_slice_{i}_ec.pt"
    #     )
    #     print(f"{i}--参数比较结果: {result},{message}")
    #     result, message = compare_state(
    #         f"/data/lff/eccheck/state_g_{i}.pt",
    #         f"/data/lff/eccheck/state_g_{i}_ec.pt"
    #     )
    #     print(f"{i}--state比较结果: {result},{message}")
    
    # result, message = compare_params(
    #     f"/data/lff/eccheck/params_slice_1_node0.pt",
    #     f"/data/lff/eccheck/params_slice_1.pt"
    # )
    # print(f"0/1节点参数比较结果: {result},{message}")
    # result, message = compare_state(
    #     f"/data/lff/eccheck/state_g_1_node0.pt",
    #     f"/data/lff/eccheck/state_g_1.pt"
    # )
    # print(f"0/1节点state比较结果: {result},{message}")

    
    # # info("/data/lff/eccheck/state_g_1.pt")
    # # info("/data/lff/eccheck/state_g_1_node0.pt")
    # s1 = torch.load("/data/lff/eccheck/state_g_1.pt", weights_only=True)
    # s2 = torch.load("/data/lff/eccheck/state_g_1_node0.pt", weights_only=True)
    # flag = 1
    # for i, (p1, p2) in enumerate(zip(s1.values(), s2.values())):
    #     for k in p1.keys():
    #         vv1 = p1[k]
    #         vv2 = p2[k]
    #         if isinstance(vv1, torch.Tensor):
    #             if not torch.equal(vv1.detach().cpu(), vv2.detach().cpu()):
    #                 flag = 0
    #                 print(f"state的torch值不同, {i}--{k}")
    #         else:
    #             if vv1 != vv2:
    #                 flag = 0
    #                 print(f"state的iny值不同, {i}--{k}")
    # if(flag):
    # #     print("0/1节点state的值完全相同")
    # s = torch.load("/home/lff/eccheck/ckpt/gpt2-large_wikitext_0_40_full.pth.tar", weights_only=True)
    # s = s['model']
    # storage_size = 0
    # for idx, (key, param) in enumerate(s.items()):
    #     if idx <= 435 and idx >= 328:  # 包含索引100
    #     # 计算单个张量存储时的字节大小（考虑数据类型）
    #         element_size = param.element_size()  # 单个元素占用的字节数（如float32=4）
    #         total_elements =param.numel()       # 张量元素总数
    #         storage_size += element_size * total_elements  # 存储空间大小
    # print(f"Storage Size: {storage_size / (1024**2):.5f} MB")
    # logger.info(f"1234567890-")
    # with open(os.path.join("/home/lff/eccheck/ckpt/", f"counts"), "r+") as f:
    #     fcntl.flock(f, fcntl.LOCK_EX)  # 加排他锁
    #     data = json.load(f)
    #     if '1' in data:
    #         data['1'] += 1
    #     else:
    #         data['1'] = 1
    #     f.seek(0)
    #     json.dump(data, f)
    #     f.truncate()
    #     fcntl.flock(f, fcntl.LOCK_UN)  # 解锁
    # a = time.time()
    # shutil.rmtree("/home/lff/eccheck/ckpt/1")
    # print(f"Time: {time.time()-a}")
    # s2 = torch.load("/home/lff/eccheck/ckpt/iteration150/optimezer_0_param_groups.pt", weights_only=True)
    # print(s2)
    from concurrent.futures import ThreadPoolExecutor, wait, shutdown, as_completed
    executor = ThreadPoolExecutor(max_workers=int(os.cpu_count()/36))
    futures = []
    # 提交所有任务
    for i in range(total_chunks):
        futures.append(executor.submit(
            encode_mmap_chunk, i, file, chunk_size, pathckpt, pathec
        ))
    # 等待所有任务完成（捕获异常）
    for future in as_completed(futures):
        try:
            future.result()
        except KeyboardInterrupt:
            logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> {file} KeyboardInterrupt!")
            while has_d_state_processes():
                time.sleep(8)
                logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> TIME SLEEP 8")
            executor.shutdown(wait=True)
            raise RuntimeError("用户主动终止程序") from KeyboardInterrupt
        except BaseException:
            logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> {file} BaseException!")
            while has_d_state_processes():
                time.sleep(8)
                logger.info(f"<Rank{getrank()}/{getworldsize()}> <iteration{iteration}> TIME SLEEP 8")
            executor.shutdown(wait=True)
            raise
if __name__ == "__main__":
    main()
# 0 1~109 110~218 219~327 328~435
'''
            batch_size = len(g_32) // 4
            for i in range(4):
                if getrank() == i:
                    start = i * batch_size
                    end = (i + 1) * batch_size if i < 3 else len(g_32)
                    torch.save(group['params'][start:end], f"/data/lff/eccheck/params_slice_{i}.pt")
                    state_g = {id(p) : self.state[p] for p in group['params'][start:end]}
                    torch.save(state_g, f"/data/lff/eccheck/state_g_{i}.pt")
                 if getrank() == 0 and i == 1:
                    start = i * batch_size
                    end = (i + 1) * batch_size if i < 3 else len(g_32)
                    torch.save(group['params'][start:end], f"/data/lff/eccheck/params_slice_{i}_node0.pt")
                    state_g = {id(p) : self.state[p] for p in group['params'][start:end]}
                    torch.save(state_g, f"/data/lff/eccheck/state_g_{i}_node0.pt")              
''' 
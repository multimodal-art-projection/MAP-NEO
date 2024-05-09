import argparse
import json
import multiprocessing
import os
import queue
from more_itertools import divide
from pybloom_live import BloomFilter, ScalableBloomFilter


input_dir = './dedup_partition/'
output_dir = './deduped_dataset'
queue_size = 1000000
read_worker_num = 5
write_worker_num = 2
CONTENT_FIELD_NAME = 'text'
bf_init_capacity = 100000
bf_error_rate = 0.001
batch_size = 100000


def split_files(input_dir, n_proc):
    files = os.listdir(input_dir)
    files = sorted(files)
    file_paths = [os.path.join(input_dir, file) for file in files]
    parts =  divide(n_proc, file_paths)
    return [list(p) for p in parts]

def get_text(files, doc_queue):
    for fp in files:
        with open(fp, 'r') as f:
            try:
                for line in f:
                    try:
                        json_doc = json.loads(line)
                    except:
                        continue
                    doc_queue.put((json_doc, line))
            except Exception as e:
                continue

def write_json_file(file_path, data_list):
    with open(file_path, 'w') as f:
        for data in data_list:
            f.write(data)

def full_text_dedup():
    os.makedirs(output_dir, exist_ok=True)
    doc_queue = multiprocessing.Queue(queue_size)
    files = split_files(input_dir, read_worker_num)

    processes = []
    for process_id in range(read_worker_num):
        p = multiprocessing.Process(
            target=get_text,
            args=(files[process_id], doc_queue,),
        )
        processes.append(p)
        p.start()
    pool = multiprocessing.Pool(write_worker_num)
    bf = ScalableBloomFilter(initial_capacity=bf_init_capacity, error_rate=bf_error_rate, mode=ScalableBloomFilter.SMALL_SET_GROWTH)
    part_id = 0
    write_batch = []
    while True:
        try:
            json_doc, line = doc_queue.get(timeout=30)
            text = json_doc[CONTENT_FIELD_NAME]
            flag = bf.add(text)
            if not flag:
                print("add one doc")
                write_batch.append(line)
                # print(f"write batch num is {len(write_batch)}")
                if len(write_batch) >= batch_size:
                    file_name = f"part_{part_id}.jsonl"
                    part_id += 1
                    print(f"begin to write bacth {file_name}")
                    file_path = os.path.join(output_dir, file_name)
                    pool.apply_async(write_json_file, (file_path, write_batch,))
                    write_batch = []
            else:
                print(f"filter one doc")
                pass
        except queue.Empty:
            break
    if len(write_batch) > 0:
        file_name = f"part_{part_id}"
        part_id += 1
        print(f"begin to write bacth {file_name}")
        file_path = os.path.join(output_dir, file_name)
        pool.apply_async(write_json_file, (file_path, write_batch))
    pool.close()
    pool.join()
    with open('bloom_array.bitarray', 'wb') as f:
        bf.tofile(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=input_dir, help='input dir where contatin jsonl file')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='output dir where to save deduplicated jsonl file')
    parser.add_argument('--queue_size', type=int, default=queue_size, help='specify the buffer size between read and write process')
    parser.add_argument('--read_worker_num', type=int, default=read_worker_num, help='specify the number of read worker')
    parser.add_argument('--write_worker_num', type=int, default=write_worker_num, help='specify the number of write worker')
    parser.add_argument('--bf_init_capacity', type=int, default=bf_init_capacity, help='specify the initial capacity of bloom filter')
    parser.add_argument('--bf_error_rate', type=float, default=bf_error_rate, help='specify the error rate of bloom filter')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='specify the number of lines of output jonsl files')
    parser.add_argument('--content_filed_name', type=str, default=CONTENT_FIELD_NAME, help='specify the field name of content in jsonl file')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    queue_size = args.queue_size
    read_worker_num = args.read_worker_num
    write_worker_num = args.write_worker_num
    bf_init_capacity = args.bf_init_capacity
    bf_error_rate = args.bf_error_rate
    batch_size = args.batch_size
    CONTENT_FIELD_NAME = args.content_filed_name
    full_text_dedup()

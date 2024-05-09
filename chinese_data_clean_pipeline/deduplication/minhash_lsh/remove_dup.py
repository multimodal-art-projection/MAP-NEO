import argparse
import os
import multiprocessing
import pickle
import json

def remove_dup(file_path, dup_line_id_path,output_path):
    with open(dup_line_id_path, 'rb') as f:
        dup_line_id_set = pickle.load(f)
    with open(file_path, 'r') as f, open(output_path, 'w') as fw:
        for idx, line in enumerate(f):
            if idx not in dup_line_id_set:
                fw.write(line)

def remove_dup_in_dir(input_dir, dup_line_id_dir, output_dir, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    pool = multiprocessing.Pool(processes=num_workers)
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        dup_line_id_path = os.path.join(dup_line_id_dir, file_name + '.pkl')
        output_path = os.path.join(output_dir, file_name)
        pool.apply_async(remove_dup, (file_path, dup_line_id_path, output_path))
    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--dup_line_id_dir', type=str, help='dir where contain the dup line id file')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    remove_dup_in_dir(args.input_dir, args.dup_line_id_dir, args.output_dir, args.workers)
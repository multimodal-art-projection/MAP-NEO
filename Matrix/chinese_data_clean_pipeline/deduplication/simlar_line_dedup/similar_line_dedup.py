import argparse
import multiprocessing
import gzip
import json
import os
from edit_distance_dedup import dedup_text

input_dir = '/root/data/minhash_after'
output_dir = '/root/data/doc_intra_dedup'
CONTENT_FIELD_NAME = 'text'
propertion = 0.1
num_workers = 40

def process_file(file_path, output_path, propertion, num_sample=None):
    with gzip.open(file_path, 'r') as f, gzip.open(output_path, 'wt', encoding='utf-8') as fw:
        for idx, line in enumerate(f):
            if num_sample != None and idx >= num_sample:
                break
            doc_json = json.loads(line)
            text = doc_json[CONTENT_FIELD_NAME]
            filtered_text = dedup_text(text, propertion)
            doc_json[CONTENT_FIELD_NAME] = filtered_text
            fw.write(json.dumps(doc_json, ensure_ascii=False) + '\n')

def process_dir(input_dir, output_dir, propertion):
    os.makedirs(output_dir, exist_ok=True)
    pool = multiprocessing.Pool(num_workers)
    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)
        pool.apply_async(process_file, (input_file_path, output_file_path, propertion))
    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=input_dir, help='input directory of jsonl files')
    parser.add_argument('--output_dir', default=output_dir, help='output directory of jsonl files')
    parser.add_argument('--workers', default=num_workers, type=int, help='number of workers to process files')
    parser.add_argument('--content_field_name', default=CONTENT_FIELD_NAME, type=str, help='field name of content in json')
    process_dir(input_dir, output_dir, propertion)
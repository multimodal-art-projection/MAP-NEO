import argparse
import json
import multiprocessing
import os
import pickle
from datasketch import MinHash
from nltk import ngrams

input_dir = '/root/data/dedup_dataset_after'
output_dir = '/root/data/minhash_value'
CONTENT_FIELD_NAME = 'text'
DOC_ID_FIELD_NAME = 'doc_id'
HASH_FILE_NAME = 'hash'
width = 13
bands = 9
r = 13
n_perm = 128

# transform array to byte representation so that it can be hashed
def _H(hs):
    return bytes(hs.byteswap().data)

def generate_hash_values(text):
    global width
    global r
    global bands
    features = map(lambda x: "".join(x), ngrams(text, width))
    m = MinHash(num_perm=128)
    [m.update(x.encode('utf8')) for x in features]
    ret = []
    for idx in range(bands):
        ret.append(_H(m.hashvalues[idx * r: (idx+1) * r]))
    return ret

def process_file(file_path, output_dir):
    try:
        file_name = file_path.split('/')[-1]
        print(f'process file {file_name}')
        band_hash_value_list = [[] for _ in range(bands)]
        with open(file_path, 'r') as f:
            try:
                for idx, line in enumerate(f):
                    try:
                        json_doc = json.loads(line)
                        doc_id = f"{file_name}@{idx}"
                        hash_value_list = generate_hash_values(json_doc[CONTENT_FIELD_NAME])
                        for idx in range(bands):
                            save_doc = {}
                            save_doc[DOC_ID_FIELD_NAME] = doc_id
                            save_doc[HASH_FILE_NAME] = hash_value_list[idx]
                            band_hash_value_list[idx].append(save_doc)
                    except Exception as e:
                        print(f'procces file {file_name} line {idx} error happe: {e}')
                        continue
            except Exception as e:
                print(f"process file {file_name} fail, error: {e}")
        try:
            for band in range(bands):
                output_path = os.path.join(output_dir, str(band), file_name + '.gz')
                with open(output_path, 'wb') as fout:
                    pickle.dump(band_hash_value_list[band], fout)
        except Exception as e:
            print(f"dump minhash fail, error: {e}")
        print(f"finish process file {file_name}")
    except Exception as e:
        print(f"the final protection for file {file_name}, error: {e}")

def process_dir(input_dir, output_dir, num_worker):
    pool = multiprocessing.Pool(processes=num_worker)
    os.makedirs(output_dir, exist_ok=True)
    for band in range(bands):
        band_path = os.path.join(output_dir, str(band))
        os.makedirs(band_path, exist_ok=True)
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        pool.apply_async(process_file, (file_path, output_dir))
    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=42, type=int)
    parser.add_argument('--input_dir', default=input_dir)
    parser.add_argument('--output_dir', default=output_dir)
    parser.add_argument('--content_field_name', default=CONTENT_FIELD_NAME, type=str, help='field name of content in json')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    process_dir(input_dir, output_dir, args.workers)

            

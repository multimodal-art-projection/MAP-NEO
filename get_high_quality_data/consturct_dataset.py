import argparse
import os
import json

def preprocess_text(text):
    return text.replace('\n', '\\n').replace('__label__', '')

def construct_files(pos_sampled, neg_sampled, train_path, valid_path, test_path, train_ratio, valid_ratio):
    """
    Constructs train, validation, and test files from positive and negative samples
    according to given ratios, writing positive and negative samples in alternating order.
    """
    def write_samples(pos, neg, path):
        with open(path, 'w') as f:
            for pos_sample, neg_sample in zip(pos, neg):
                pos_text = preprocess_text(pos_sample['raw_content'])
                neg_text = preprocess_text(neg_sample['raw_content'])
                f.write(f'__label__1 {pos_text}\n')
                f.write(f'__label__0 {neg_text}\n')
    
    # Calculate the number of samples for each dataset based on the ratios
    total_samples = min(len(pos_sampled), len(neg_sampled))
    num_train = int(total_samples * train_ratio)
    num_valid = int(total_samples * valid_ratio)
    num_test = total_samples - num_train - num_valid
    
    # Split the samples
    pos_train, pos_valid, pos_test = pos_sampled[:num_train], pos_sampled[num_train:num_train+num_valid], pos_sampled[num_train+num_valid:]
    neg_train, neg_valid, neg_test = neg_sampled[:num_train], neg_sampled[num_train:num_train+num_valid], neg_sampled[num_train+num_valid:]
    
    # Write to files
    write_samples(pos_train, neg_train, train_path)
    write_samples(pos_valid, neg_valid, valid_path)
    write_samples(pos_test, neg_test, test_path)

def read_jsonl_dir(input_dir):
    json_doc_list = []
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                json_doc_list.append(json.loads(line))
    return json_doc_list

def main(pos_dir, neg_dir, train_save_path, val_save_path, test_save_path, train_ratio, val_ratio):
    pos_sampled = read_jsonl_dir(pos_dir)
    neg_sampled = read_jsonl_dir(neg_dir)
    construct_files(
        pos_sampled, 
        neg_sampled, 
        train_save_path,
        val_save_path,
        test_save_path,
        train_ratio,
        val_ratio
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_dir', type=str)
    parser.add_argument('--neg_dir', type=str)
    parser.add_argument('--train_save_path', type=str)
    parser.add_argument('--val_save_path', type=str)
    parser.add_argument('--test_save_path', type=str)
    args = parser.parse_args()
    pos_dir = args.pos_dir
    neg_dir = args.neg_dir
    train_save_path = args.train_save_path
    val_save_path = args.val_save_path
    test_save_path = args.test_save_path
    train_ratio = 0.95
    val_ratio = 0.025
    main(pos_dir, neg_dir, train_save_path, val_save_path, test_save_path, train_ratio, val_ratio)
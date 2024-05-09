# Filter
You first need to download fasttext model in the filter directory. The download [link](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin). The command example is 
```
wget -P filter/ https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

Once you download the model bin of fasttext, then you use command like follows to filter text.
```
python3 filter/filter.py --input_data input_dir --output_dir output_dir --success_dir success_dir --log_dir log_dir --worker num_worker
```
where input_dir is the direcotry containing jsonl files, output_dir is the directory to store the filtered jsonl files, success_dir is dirtory to save files about whether files are processed successfully, log_dir is the directory to save logs and num_worker is the total number of process to execute the job in parallel. 

# Deduplication

## Full Text Deduplication

Here is an example command to run full text deduplication
```
python3 deduplication/full_text_dedup/full_text_dedup.py --input_dir input_dir --output_dir output_dir --content_field_name content_field_name
```
where input_dir is the directory to story the jsonl files, output_dir is the dirtory to save the deduplicated jsonl file and content_field_name is the field name of content in jsonl file.

To reduce the total processing time, you can partition jsonl files to multiple partitions, and then run multiple job for each partition in parallel.


## Minhash LSH
### Generate Minhash
Here is an example command to run Generate Minhash
```
python3 deduplication/minhash_lsh/generate_minhash.py --input_dir input_dir --output_dir output_dir --workers num_workers --content_field_name content_field_name
```
where input_dir is dirtory to save jsonl files, output_dir is the directory to save minhash value, num_worker is  the total number of processes executing the job and content_field_name is the field name of content in jsonl file.

### Gereate Duplicate Pairs
Here is an example command to run  Gereate Duplicate Pairs
```
python3 deduplication/minhash_lsh/generate_dup_pairs.py --input_dir input_dir --output_dir output_dir 
```
where input_dir is the directory to storay the minhash values, which is the same with the output_dir in Generate Minhash. Output_dir is the directory to save the duplicate paris files.

### Genereate Connected Componenets
Here is an example command to run Genereate Connected Componenets
```
python3 deduplication/minhash_lsh/generate_connected_components.py --input_dir input_dir --output_file output_file --num_workers num_workers
```
where input_dir is the directory containing the duplicate lines file which is the same as the output_dir in Generate Duplicate Pairs. Output_file is file path to save information about connected components. Num_workers is the total number of processes executing the job.

### Generate Duplicated Line ID for Each File
Here is an example command to run Genereate Connected Componenets
```
python3 deduplication/minhash_lsh/generate_dup_line_id_for_each_file.py --input_file input_file --output_dir output_dir
```
where input_file is the file containing nformation about connected components which is the same as output_file in Genereate Connected Componenets. Output_dir is the directory to save duplicate line id information for each file.

### Remove Duplicates
Here is an example command to run Remove Duplicates
```
python3 deduplication/minhash_lsh/remove_dup.py --input_dir input_dir --output_dir output_dir --dup_line_id_dir dup_line_id_dir --worker num_worker
```
where input_dir is the directory containing the original jsonl files which is the same as the input_dir in Generate Minhash. Output_dir is the directory to save the dedupliacted jsonl files. Dup_line_id_dir is the directory containing duplicate line ids of all jsonl files which is the same as output_dir in Generate Duplicated Line ID for Each File.

## Similar Line Deduplication
Here is an example command to run Remove Duplicates
```
python3 deduplication/simlar_line_dedup/similar_line_dedup.py --input_dir input_dir --output_dir output_dir --wokers num_worker --content_field_name content_field_name
```
where input_dir is the directory containing the jsonl files, output_dir is the direcotry to save deduplicated jsonl files, num_worker is the total number of processes executing the job and content_field_name is the field name of content in jsonl file. 


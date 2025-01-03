# Filter
You first need to download fasttext model in the filter directory. The download [link](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin). Here is an example command 
```
wget -P filter/ https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```
You can use commands like this to run filter
```
spark-submit \
            --files ./filter/en_gopher_stopwords.json,./filter/lid.176.bin \
            --py-files ./filter/bad_url_words.py,./filter/util_with_try.py \
            --archives ./filter/dest.zip#dest,./filter/nltk_data.zip#nltk_data  \
            ./filter/filter.py  > log.txt 2> err.txt
```


# Deduplication
## Exact_Document_Deduplication
Here is an example command to run exact document deduplication
```
spark-submit \
    ./dedup/Exact_Document_Deduplication/exact_document_deduplication.py --input_dir input_dir --output_dir output_dir
``` 
where input_dir is the directory to store the jsonl files and output_dir is the directory to save the deduplicated jsonl file.
## Minhash_Lsh_Deduplication
Here is an example command to run exact document deduplication
```
spark-submit \
    ./dedup/Minhash_Lsh_Deduplication/minhashLsh.py --input_dir input_dir --inter_dir inter_dir --output_dir output_dir
``` 
where input_dir is the directory to store the jsonl files, output_dir is the directory to save the deduplicated jsonl file, and inter_dir is the directory to save the minhash value
## Paragraph_Deduplication
```
spark-submit \
    ./dedup/Paragraph_Deduplication/paragraph.py --input_dir input_dir  --output_dir output_dir
```
where input_dir is the directory to store the jsonl files and output_dir is the directory to save the deduplicated jsonl file.
## Exact_Document_Deduplication
```
./dedup/Exact_Document_Deduplication/exact_document_deduplication.py --input_dir input_dir --output_dir output_dir
```
where input_dir is the directory to store the jsonl files and output_dir is the directory to save the deduplicated jsonl file.
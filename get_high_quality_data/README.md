Here we use the procedure of getting math data as example to isllustrate how to use it.
The porcudre to filter math data from redpajamav2
1. sample data from the open-web-math and redpajamav2
the code is in construct_fasttext_train_data.py which should be execute by spark
the positive sample is from open-web-math and negative sample is from redpajamav2, the total number of sample is 500,000

The example command is as follows
```
python3 construct_fasttext_train_data.py --neg_file_path neg_file_path --pos_file_path pos_file_path --output_path output_path
```
where neg_file_path is the dirtory to save negative data, pos_file_path is the directory to save positive data and output_path is the directory to save the sampled train data.

2. construct train dataset for fasttext
the code is in consturct_dataset.py, which is a standalone code, not by spark. It's ued to consturct dataset for fasttext.
It will generate three files, which is train.txt, val.txt and test.txt. 
The fromat of the file is a lot of pais of positve and negative text
```
__label__1 {postive text}
__label__0 {negative text}
```

The example command is as follows
```
python3 consturct_dataset.py --pos_dir pos_dir --neg_dir neg_dir --train_save_path train_save_path --val_save_path val_save_path --test_save_path test_save_path
```
where pos_dir is the directory to save positive dataset, neg_dir is the direcotry to save negative dataset, train_save_path is the directory to save sampled dataset for training, val_save_path is the directory to save sampled dataset for validation and test_save_path is the directory to save sampled dataset for test.

3. train fasttext model
the code is in train_fasttext.py, which only use cpu to train model and the speed is very fast.

The example command is as follows.
```
python3  train_fasttext.py --train_file_path train_file_path --val_file_path val_file_path --model_path  model_path
```
where train_file_path is direcotry which stroies the dataset for training, val_file_path is the direcotry which stories the dataset for validation and model_path is path to save trained model.

4. quantize fastttext model
Since the size of the model is huge, which is about 6-7gb, which is really huge.
Quatize fasttext model will lose a litter accuracy, but the size will decrease a lot.
The code is in quantize_fasttext_model.py. The model will be named as 'model_offical.ftz'.

```
python quantize_fasttext_model.py --train_file_path train_file_path --val_file_path val_file_path --model_path model_path --model_save_path model_save_path
```
where train_file_path is train_file_path is direcotry which stroies the dataset for training, val_file_path is the direcotry which stories the dataset for validation, model_path is path to load trained model and model_save_path is the path to save quantized model.

5. filter math data from redpajama v2
After we get a trained fasttext model, we can use it to get math data. The code is in filter_data_final.py. It need to get an arugument from the command line, which a file name who contains the file path list of redpajamav2. Besides, you may need adjust the threshold to set the quality of the text you may want although threshold is hard encoded in code.

The example command is as follows
```
python3 filter_data_final.py --input_dir_path input_dir_path --output_dir_path output_dir_path --model_path model_path --threshhold threshhold
```
where input_dir_path is the dirtory which strories the dataset you want to filter, output_dir_path is the directory to save the filtered dataset, model_path is the path to load trained model and threshold is the value to decide whether filter the data.
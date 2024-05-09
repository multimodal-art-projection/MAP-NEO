import argparse
import fasttext

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str)
    parser.add_argument('--val_file_path', type=str)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    train_file_path = args.train_file_path
    val_file_path = args.val_file_path
    model_path = args.model_path
    model = fasttext.train_supervised(
        input=train_file_path,
        epoch=3,
        lr=0.1,
        dim=256,
        wordNgrams=3,
        minCount=3
    )
    print(model.test(val_file_path))
    model.save_model(model_path)
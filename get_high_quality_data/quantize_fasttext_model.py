import argparse
import fasttext

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('train_file_path', type=str, default='train.txt')
argument_parser.add_argument('val_file_path', type=str, default='val.txt')
argument_parser.add_argument('model_path', type=str, default='model.bin')
argument_parser.add_argument('model_save_path', type=str, default='model.ftz')
args = argument_parser.parse_args()

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

model = fasttext.load_model(args.model_path)

model.quantize(input=args.train_file_path, retrain=True)

# then display results and save the new model :
print_results(*model.test(args.val_file_path))
model.save_model(args.model_save_path)
'''
masking(infile, outfile) produce masked sentence paired with entity
The input file should include "sentence" and "NER label"
TODO: documents
'''

from tqdm import tqdm
from argparse import ArgumentParser
import codecs

def unescaped_str(arg_str):
    return codecs.decode(str(arg_str), 'unicode_escape')

def get_args():
    parser = ArgumentParser(description='MaskEntity')
    parser.add_argument('--delimiter',type=unescaped_str, default='\t')
    parser.add_argument('--sent_id', type=int, default=0)
    parser.add_argument('--tag_id', type=int, default=1)
    parser.add_argument('--mask_tag', type=str, default='<e>')
    parser.add_argument('--pad_tag', type=str, default='<pad>')
    parser.add_argument('--tag_O', type=str, default='O')
    parser.add_argument('--tag_I', type=str, default='I')
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    args = parser.parse_args()
    return args

def masking(args):
    fin = open(args.input_file)
    fout = open(args.output_file, "w")
    print("Processing {}".format(args.input_file))
    for line in tqdm(fin.readlines()):
        items = line.strip().split(args.delimiter)
        sentence = items[args.sent_id].strip().split()
        label = items[args.tag_id].strip().split()
        # if len(sentence) != len(label):
        #     print("Length mismatch in file : {}".format(args.input_file))
        sen_str = []
        e_str = []
        flag = False
        for token, tag in zip(sentence, label):
            if token == args.pad_tag:
                break
            if tag == args.tag_O:
                if flag:
                    flag = False
                sen_str.append(token)
            if tag[0] == args.tag_I:
                if flag == False:
                    sen_str.append(args.mask_tag)
                    flag = True
                e_str.append(token)
        if len(e_str) == 0:
            # We regard the whole sentence as entity here
            fout.write("{}\t{}\n".format(" ".join(sen_str), " ".join(sen_str)))
        else:
            fout.write("{}\t{}\n".format(" ".join(sen_str), " ".join(e_str)))

'''
Example Usage: 
python MaskEntity.py --sent_id 1 --tag_id 2 --delimiter ' %%%% ' \
--input_file /u1/p8shi/pycharm/attentive_cnn/data/main-test-results.txt \
--output_file /u1/p8shi/pycharm/attentive_cnn/data/entity_mask.test.2
'''
if __name__=="__main__":
    args = get_args()
    print(args)
    if args.input_file == '' or args.output_file == '':
        print("Please specify the input/output file")
        exit(1)
    masking(args)
import argparse


def main(args):
    raw_data = open(args.train_data, 'r').readlines()
    labels = []
    for line in raw_data:
        row = line.strip("\n").split("\t")
        labels.append(row[0])
    unique_label = set(labels)
    print('数据集标签:{}   标签总数:{}'.format(unique_label,len(unique_label)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count dataset label.")
    parser.add_argument("--train_data", dest="train_data", action="store", help="")
    parsed_args = parser.parse_args()
    main(parsed_args)

import datasets

test_set = datasets.RecursionDataset("../recursion-cellular-image-classification/train.csv","../recursion-cellular-image-classification/train")

for i in range(len(test_set)):
    if (i >= 10):
        break
    else:
        print(test_set[i])
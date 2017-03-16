
def read_label_file(file):  
    f = open(file, "r")  
    filepaths = []  
    labels = []  
    for line in f:  
        filepath, label = line.split(",") 
        filepaths.append(filepath)  
        labels.append(encode_label(label))
    return filepaths, labels  
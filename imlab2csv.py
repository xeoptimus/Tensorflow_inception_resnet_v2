# encoding: utf-8
import glob
import os
import settings
FLAGS = settings.FLAGS

def create_examples(directory):
    print (directory)
    
    directories = glob.glob(directory)
    for category_path in directories:
        category_name = os.path.basename(category_path) 
        print("category: %s" % (category_name))
       
        images = glob.glob("%s/%s" % (category_path, "*"))
        for image in images:
            with open(FLAGS.dataset_path+"/"+FLAGS.all_labels_file, 'a') as f:
                f.write(image)
                f.write(',')
                f.write(category_name)
                f.write("\n")


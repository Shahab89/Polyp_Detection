import os
import ast

batch_size = int(os.environ.get('batch_size'))
chunk_num = int(os.environ.get('chunk_num'))
image_size = (100, 100)
run_model = os.environ.get('run_model')
data_directory = os.environ.get('data_directory')
image_path = os.environ.get('image_path')

import json
import os
from tqdm import tqdm

json_root_path = "/root/data/mutimodel_dataset/data_split/OCR-VQA/json/data.json"
# OCR_VQA.json  OCR-VQA.zip
import zipfile

def list_files_in_zip(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        file_list = zip_file.namelist()
    return file_list

with open(os.path.join(json_root_path), "r") as f:
    file_content = f.read()
json_data = json.loads(file_content)

image_root_path = "/root/data/mutimodel_dataset/data_split/OCR-VQA/image"
chunk_list = os.listdir(image_root_path)
chunk_file_dict = {}
for chunk_name in chunk_list:
    chunk_path = os.path.join(image_root_path, chunk_name)
    print(chunk_path)
    print(list_files_in_zip(chunk_path)[:100])
    chunk_file_dict[chunk_name] = list_files_in_zip(chunk_path)

for data in tqdm(json_data[:1000]):
    if 'chunk_belong' in data: continue
    image_path = data['image_path']
    for chunk_name in chunk_file_dict:
        if image_path in chunk_file_dict[chunk_name]:
            data['chunk_belong'] = chunk_name
            break


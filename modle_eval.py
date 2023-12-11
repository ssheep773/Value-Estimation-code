import os
import yaml
import sys
import csv
import torch
import random
import numpy as np
import torch.optim as optim
from lib.training import *
from lib.utils import *
from lib.model import initialize_model
from lib.data_loaders import MyYamlLoader
from lib.mydataloader import build_loader, build_dataset
import glob

from datetime import datetime
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def mymain(path):
    with open(path, "r", encoding="UTF-8") as stream:
        config = yaml.load(stream, Loader=MyYamlLoader)
    
    config_name = os.path.basename(path).split(".")[0]
    data_dir = f"{config['data']['data_dir']}{config_name}/"

    output_dir = config["data"]["output_dir"] + config_name
    # print(output_dir)
    model_fname = os.path.join(output_dir, "model")
    evaluation_fname = os.path.join(output_dir, "evaluation.pt")
    create_dir(output_dir)
    # Init WANDB
    start_index = path.rfind("/") + 1  # 找到最后一个斜杠的索引并加1
    end_index = path.rfind(".yaml")  # 找到.yaml的索引
    extracted_text = path[start_index:end_index]

 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create loss matrix for each task
    num_heads = len(config["heads"])
    label_tags = []
    loss_matrix = {}
    for head in config["heads"]:
        label_tags.append(head["tag"])
        loss_matrix[head["tag"]] = get_loss_matrix(
            len(head["labels"]), head["metric"][0]
        ).to(device)
        

    # Initialize the model
    model = initialize_model(config)
    
    model = torch.jit.load(f"{output_dir}/model_gpu.pt")
    model = model.to(device)
    # Evaluate model on all data
    batch_size = config["optimizer"]["batch_size"]
    num_workers = config["optimizer"]["num_workers"]

    image_dataset = build_dataset(False, config, test=True )
    dataloader = torch.utils.data.DataLoader( image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    posterior, predicted_label, true_label, error = eval_model( model, config, loss_matrix, dataloader, device)
    total_acc = 0
    total_mae = 0
    count =0
    for pred, label in zip(predicted_label[head["tag"]],true_label[head["tag"]]):

        decimal_position_A = len(str(label).split(".")[1]) if "." in str(label) else 0
        pred = round(pred, decimal_position_A)
        
        acc = 1 - abs((pred - label) / label)
        mae = round(abs(pred - label),2)
        count += 1
        total_acc += acc
        total_mae += mae
    print(output_dir, total_mae/count, total_acc/count )
    # Create training and validation dataloaders
    data = [output_dir, round(total_mae/count, 2), round(total_acc/count, 3)]
    save_path = "acc.csv"
    with open(save_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)   

if __name__ == "__main__":
    # Load config

    for folder_path in [ "configs/dldlv2/235/", "configs/dldlv2/all/", "configs/dldlv2/NAH/" ]:
        yaml_files = glob.glob(folder_path + "*.yaml")
        for yaml_file in yaml_files:
            print(yaml_file)
            mymain(yaml_file)

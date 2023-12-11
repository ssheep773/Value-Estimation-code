import os
import yaml
import sys
import torch
import random
import numpy as np
import torch.optim as optim
from lib.training import *
from lib.utils import *
from lib.model import initialize_model
from lib.data_loaders import MyYamlLoader
from lib.mydataloader import build_loader
import glob
import wandb
from datetime import datetime

# fix random seeds for reproducibility

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mymain(config_path):
    with open(config_path, "r", encoding="UTF-8") as stream:
        config = yaml.load(stream, Loader=MyYamlLoader)
    print("config_path", config_path)

    # 更改 data_dir 的值
    for color in ["all", "235", "NAH"]:
        for i in range(0, 5):
            config["data"]["data_dir"] = f"datasetApproximation/{color}-kfold-{i}"
            config["data"]["output_dir"] = f"results/Approximation_{color}"

            config_name = os.path.basename(config_path).split(".")[0]
            type_name = config_name.split("_")[1]

            config_new_name = f"{color}-{type_name}-{i}"

            data_dir = f"{config['data']['data_dir']}/"
            output_dir = f'{config["data"]["output_dir"]}/' + config_new_name
            model_fname = os.path.join(output_dir, "model_test")
            print(output_dir, data_dir, config_new_name)
            create_dir(output_dir)

            # Init WANDB
            wandb_name = config_name + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            wandb.init(
                project="test_reg-DLDLv2",
                config=dict(yaml=config),
                name=wandb_name,
                mode="online",
            )

            # Create loss matrix for each task
            label_tags = []
            loss_matrix = {}
            for head in config["heads"]:
                label_tags.append(head["tag"])
                loss_matrix[head["tag"]] = get_loss_matrix(
                    len(head["labels"]), head["metric"][0]
                ).to(device)

            model = initialize_model(config)
            model = model.to(device)
            # print(model)

            batch_size = config["optimizer"]["batch_size"]
            num_workers = config["optimizer"]["num_workers"]

            data_loader_train, data_loader_val = build_loader(config)
            dataloaders = {
                "trn": data_loader_train,
                "val": data_loader_val,
            }

            # Setup optimizer
            if config["optimizer"]["algo"] == "sgd":
                optimizer = optim.SGD(
                    model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9
                )
            elif config["optimizer"]["algo"] == "adam":
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=config["optimizer"]["lr"],
                    betas=config["optimizer"]["betas"],
                    eps=config["optimizer"]["eps"],
                )
            else:
                sys.exit(f"Unknown optimizer {config['optimizer']['algo']}")

            # Train and evaluate
            model, log_history = train_model(
                model, config, dataloaders, loss_matrix, optimizer, device, output_dir
            )

            if device.type == "cpu":
                model_scripted = torch.jit.script(model)  # Export to TorchScript
                torch.jit.save(
                    model_scripted,
                    model_fname + "_cpu.pt",
                    _extra_files={"config": yaml.dump(config)},
                )

            else:
                gpu_model_scripted = torch.jit.script(model)  # Export to TorchScript
                torch.jit.save(
                    gpu_model_scripted,
                    model_fname + "_gpu.pt",
                    _extra_files={"config": yaml.dump(config)},
                )

                cpu_model = model.cpu()
                cpu_model_scripted = torch.jit.script(
                    cpu_model
                )  # Export to TorchScript
                torch.jit.save(
                    cpu_model_scripted,
                    model_fname + "_cpu.pt",
                    _extra_files={"config": yaml.dump(config)},
                )
            wandb.finish()


if __name__ == "__main__":
    # Load config
    for value in ["NID", "S", "L", "R"]:
        yaml_file = f"configs/dldlv2/config_{value}.yaml"
        mymain(yaml_file)

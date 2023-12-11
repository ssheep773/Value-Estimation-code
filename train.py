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


def mymain(config_path):
    with open(config_path, "r", encoding="UTF-8") as stream:
        config = yaml.load(stream, Loader=MyYamlLoader)
    print("config_path",config)

    
    config_name = os.path.basename(config_path).split(".")[0]
    print(config_name)
    data_dir = f"{config['data']['data_dir']}{config_name}/"

    output_dir = config["data"]["output_dir"] + config_name
    model_fname = os.path.join(output_dir, "model")
    evaluation_fname = os.path.join(output_dir, "evaluation.pt")
    create_dir(output_dir)
    
   

    wandb_name = config_name + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    wandb.init(project="reg-DLDLv2",config=dict(yaml=config), name=wandb_name, mode="online")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on: {device}")

    # Create loss matrix for each task
    num_heads = len(config["heads"])
    label_tags = []
    loss_matrix = {}
    for head in config["heads"]:
        print(head, head["tag"])
        label_tags.append(head["tag"])
        print(label_tags)
        loss_matrix[head["tag"]] = get_loss_matrix(
            len(head["labels"]), head["metric"][0]
        ).to(device)
        print(loss_matrix[head["tag"]])

    # Initialize the model
    model = initialize_model(config)
    model = model.to(device)
    print(model)

    # Create training and validation dataloaders
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

    # Evaluate model on all data
    # data_transform = get_data_transform( "val", config )
    # image_dataset = NormalizedImages( protocol_file, label_tags, folders=[0,1,2], transform = data_transform, load_to_memory=False  )
    # dataloader = torch.utils.data.DataLoader( image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # posterior, predicted_label, true_label, id, folder, error = eval_model( model, config, loss_matrix, dataloader, device )

    # Print errors
    # print("Model evalution:")
    # for i, set in enumerate( error.keys() ):
    #     print(f"[{set} set]" )
    #     for head in config['heads']:
    #         print(f"{head['tag']} ({head['metric'][0]}): {error[set][head['tag']]:.4f}")

    # Save model
    # torch.save({'config': config,
    #            'split': args.split,
    #            'error': error,
    #            'error_history': error_history,
    #            'loss_history': loss_history,
    #            'model_state_dict': model.state_dict()}, model_fname )

    # Save evaluation
    # torch.save({'config': config,

    #             'error': error,
    #             'log_history': log_history,
    #             'posterior': posterior,
    #             'true_label': true_label,
    #             'predicted_label': predicted_label,
    #             'id': id,
    #             'folder': folder }, evaluation_fname )

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
        cpu_model_scripted = torch.jit.script(cpu_model)  # Export to TorchScript
        torch.jit.save(
            cpu_model_scripted,
            model_fname + "_cpu.pt",
            _extra_files={"config": yaml.dump(config)},
        )
    wandb.finish()

if __name__ == "__main__":
    # Load config
    yaml_file = glob.glob("configs/dldlv2/config_NID.yaml")
    print(yaml_file)
    mymain(yaml_file)
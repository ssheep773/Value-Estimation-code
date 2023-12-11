"""
Implements the training loop and inference on test set.

Functions:
    - :py:meth:`train_model`
    - :py:meth:`eval_model`
"""
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import numpy as np
import wandb
from typing import Dict
convertion = {
    'target_range':{
        'NID':[0, 50],
        'S':[0, 32],
        'L':[0, 18],
        'R':[0, 32],
    },
    'origin_range':{
        'NID':[0.22, 0.72],
        'S':[16, 48],
        'L':[6, 24],
        'R':[4.8, 8],
    },
}

def convert_to_new_range(value, original_minmax, new_minmax):
    original_min, original_max = original_minmax
    new_min, new_max = new_minmax

    value = torch.clamp(value, max=original_max, min=original_min)

    original_range = original_max - original_min
    new_range = new_max - new_min

    final_values = (value - original_min) / original_range * new_range + new_min

    return final_values

def convert_pred(value, tag):
    return convert_to_new_range(value[tag], convertion['target_range'][tag], convertion['origin_range'][tag])

def convert_label(value, resume=False):
    if resume == False:
        return {
            "NID": convert_to_new_range(value["NID"], convertion['origin_range']["NID"], convertion['target_range']["NID"]),
            "S": convert_to_new_range(value["S"], convertion['origin_range']["S"], convertion['target_range']["S"]),
            "L": convert_to_new_range(value["L"], convertion['origin_range']["L"], convertion['target_range']["L"]),
            "R": convert_to_new_range(value["R"], convertion['origin_range']["R"], convertion['target_range']["R"]),
        }
    else:
        return {
            "NID": convert_to_new_range(value["NID"], convertion['target_range']["NID"], convertion['origin_range']["NID"]),
            "S": convert_to_new_range(value["S"], convertion['target_range']["S"], convertion['origin_range']["S"]),
            "L": convert_to_new_range(value["L"], convertion['target_range']["L"], convertion['origin_range']["L"]),
            "R": convert_to_new_range(value["R"], convertion['target_range']["R"], convertion['origin_range']["R"]),
        }


def eval_model(
    model: nn.Module,
    config: dict,
    loss_matrix: torch.tensor,
    dataloader: torch.utils.data.DataLoader,
    device: str,
):
    model.eval()
    true_label = {}
    posterior = {}
    for head in config["heads"]:
        true_label[head["tag"]] = []
        posterior[head["tag"]] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            heads = model(inputs)
        
        for head, head_logits in heads.items():
            head_labels = labels[head].to(device)
            
            
            pred = model.get_head_posterior(head_logits, head)

            posterior[head].append(pred)
            true_label[head].append(head_labels)

  
    for head in config["heads"]:
        posterior[head["tag"]] = torch.cat(posterior[head["tag"]])
        true_label[head["tag"]] = torch.cat(true_label[head["tag"]])

    predicted_label = {head["tag"]: None for head in config["heads"]}
    for head in config["heads"]:
        _, predicted_label[head["tag"]] = torch.min(
            torch.matmul(posterior[head["tag"]], loss_matrix[head["tag"]]), 1
        )
    
    predicted_label[head["tag"]] = convert_pred(predicted_label, head["tag"])
    

    error_tags = {head["tag"]: None for head in config["heads"]}
    error = {
        "trn": error_tags.copy(),
        "val": error_tags.copy(),
        "tst": error_tags.copy(),
    }

    # compute the mean error for the different parts
    # for i, set in enumerate(["trn", "val", "tst"]):
    #     index = torch.squeeze(torch.argwhere(folder == i)).to(device)
    #     for head in config["heads"]:
    #         set_true_label = torch.index_select(true_label[head["tag"]], 0, index)
    #         set_predicted_label = torch.index_select(
    #             predicted_label[head["tag"]], 0, index
    #         )
    #         error[set][head["tag"]] = (
    #             torch.mean(
    #                 loss_matrix[head["tag"]][set_true_label, set_predicted_label]
    #             )
    #             .cpu()
    #             .detach()
    #             .numpy()
    #             .tolist()
    #         )

    # convert results to numpy
    for head in config["heads"]:
        true_label[head["tag"]] = true_label[head["tag"]].cpu().detach().numpy()
        predicted_label[head["tag"]] = (
            predicted_label[head["tag"]].cpu().detach().numpy()
        )
        posterior[head["tag"]] = posterior[head["tag"]].cpu().detach().numpy()

    return posterior, predicted_label, true_label, error


def train_model(
    model: nn.Module,
    config: dict,
    dataloaders,
    loss_matrix: torch.tensor,
    optimizer,
    device,
    output_dir,
):
    since = time.time()
    use_amp = ("use_amp" in config["optimizer"].keys()) and (
        config["optimizer"]["use_amp"]
    )

    num_epochs = config["optimizer"]["num_epochs"]
    improve_patience = config["optimizer"]["improve_patience"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # get head names and weights (when multiple heads are trained, the loss is their combination) from config
    head_names = []
    weights = {}
    for head in config["heads"]:
        head_names.append(head["tag"])
        weights[head["tag"]] = head["weight"]

    # find if there is a checkpoint file
    checkpoint_file = ""
    for root, subdirs, files in os.walk(output_dir):
        for filename in files:
            if "checkpoint" in filename:
                checkpoint_file = output_dir + filename

    start_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    min_val_error = np.Inf
    best_model_epoch = 0
    log_history = []

    # Main optimization loop
    for epoch in range(start_epoch, num_epochs):
        if epoch - best_model_epoch > improve_patience:
            print(f"No improvement after {improve_patience} epochs -> halt.")
            break

        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        log = {}
        for phase in ["trn", "val"]:
            if phase == "trn":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = {x: 0.0 for x in head_names}
            running_error = {x: 0.0 for x in head_names}

            # Iterate over data
            with torch.set_grad_enabled(phase == "trn"):
                n_examples = 0
                for inputs, labels in dataloaders[phase]:
                    # print(labels)
                    labels = convert_label(labels)

                    inputs = inputs.to(device)

                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=use_amp
                    ):
                        heads = model(inputs)

                        batch_size = inputs.size(0)
                        loss_fce = 0.0

                        for head, head_logits in heads.items():
                            head_labels = labels[head].to(device)

                            # evaluate loss function
                            head_loss = model.get_head_loss(
                                head_logits, head_labels, head
                            )
                            loss_fce += weights[head] * head_loss

                            # compute error metric
                            head_posterior = model.get_head_posterior(head_logits, head)
                            # 透過 torch.matmul(head_posterior, loss_matrix[head]) 計算估計值，再透過torch.min() 確保輸出最小為1
                            _, predicted_labels = torch.min(
                                torch.matmul(head_posterior, loss_matrix[head]), 1
                            )

                            # print(head_labels.data, predicted_labels)
                            int_tensor = torch.round(head_labels.data).to(torch.int32)
                            # print(int_tensor)
                            head_err = torch.mean(
                                loss_matrix[head][int_tensor, predicted_labels]
                            )

                            # update running average
                            running_loss[head] = (
                                n_examples * running_loss[head] + batch_size * head_loss
                            ) / (n_examples + batch_size)
                            running_error[head] = (
                                n_examples * running_error[head] + batch_size * head_err
                            ) / (n_examples + batch_size)

                        n_examples += batch_size

                    # backward + optimize only if in training phase
                    if phase == "trn":
                        optimizer.zero_grad()
                        scaler.scale(loss_fce).backward()
                        scaler.step(optimizer)
                        scaler.update()

            # compute weighted error and weighted loss
            weighted_error = 0.0
            weighted_loss = 0.0
            for head in head_names:
                running_error[head] = running_error[head].cpu().detach().numpy()
                running_loss[head] = running_loss[head].cpu().detach().numpy()
                weighted_error += weights[head] * running_error[head]
                weighted_loss += weights[head] * running_loss[head]

            # if validation error improved deep copy the model
            if phase == "val" and weighted_error <= min_val_error:
                min_val_error = weighted_error
                best_model_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            # log errors and losses
            log[phase + "_loss"] = weighted_loss
            log[phase + "_error"] = weighted_error
            for head in head_names:
                log[phase + "_loss_" + head] = running_loss[head]
                log[phase + "_error_" + head] = running_error[head]

            # print loss and error value for current phase
            loss_msg = f"loss: {weighted_loss:.4f}"
            error_msg = f"error: {weighted_error:.4f}"
            for head in head_names:
                loss_msg += f" {head}_loss:{running_loss[head]:.4f}"
                error_msg += f" {head}_error:{running_error[head]:.4f}"
            print(f"[{phase} phase]", error_msg, loss_msg)

        # log elapsed time
        log["elapsed_minutes"] = (time.time() - since) / 60

        # update wandb
        wandb.log(log)

        # append log
        log_history.append(log)

        # remove old checkpoint and save the new one
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_model_wts": best_model_wts,
            "best_model_epoch": best_model_epoch,
            "log_history": log_history,
            "min_val_error": min_val_error,
        }
        checkpoint_file = output_dir + f"checkpoint_{epoch}.pth"
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved to {checkpoint_file}")

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best epoch: {:4f}".format(best_model_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, log_history

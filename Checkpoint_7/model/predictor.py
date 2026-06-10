import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import random


def compute_anomaly(results, threshold):
    total_errors = 0
    for seq_result in results:
        if seq_result["undetected_mean_rate"] > threshold:
            total_errors += 1

    return total_errors


class Predictor:
    def __init__(self, model, dataloader_normal, dataloader_abnormal, device, pad_id, mask_id, num_candidates, mask_ratio):
        self.model = model
        self.dataloader_normal = dataloader_normal
        self.dataloader_abnormal = dataloader_abnormal
        self.device = device
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.num_candidates = num_candidates
        self.mask_ratio = mask_ratio

    @torch.no_grad()
    def predict(self):
        self.model.to(self.device)
        self.model.eval()

        print(f"\n{'#'*60}")
        print(f"\nTesting normal predicting")
        print(f"\n{'#'*60}")

        normal_results = self.testing("normal")

        print(f"\n{'#'*60}")
        print(f"\nTesting abnormal predicting")
        print(f"\n{'#'*60}")

        abnormal_results = self.testing("abnormal")

        print(f"\n{'#'*60}")
        print(f"\nFinding best threshold")
        print(f"\n{'#'*60}")

        threshold_results = self.find_best_threshold(normal_results, abnormal_results, np.arange(0, 1, 0.05))

        print(f"\n{'#'*60}")
        print(f"\nTP: {threshold_results['TP']} | TN: {threshold_results['TN']} | FP: {threshold_results['FP']} | FN: {threshold_results['FN']}")
        print(f"Threshold: {threshold_results['Threshold']} | F1: {threshold_results['F1']}")
        print(f"Precision: {threshold_results['Precision']}, Recall: {threshold_results['Recall']}")
        print(f"\n{'#'*60}")

        with open("results.json", "w") as f:
            json.dump(threshold_results, f, indent=4)


    @torch.no_grad()
    def testing(self, type):
        assert (type in ("normal", "abnormal")) == True, "Testing type must be normal or abnormal"

        if type == "normal":
            dataloader = self.dataloader_normal
        elif type == "abnormal":
            dataloader = self.dataloader_abnormal

        results = []
        std = []

        for batch_idx, batch in enumerate(dataloader):
            batch = {key: value.to(self.device) for key, value in batch.items()}

            for sequence in batch["input_ids"]:
                pad_mask = (sequence == self.pad_id).to(self.device)
                seq_results = {
                    "undetected_mean_rate": 0,
                    "undetected_std": 0,
                }

                undetected_rates = []

                for _ in range(5):
                    masked_seq, labels = self.mask_sequence(sequence)

                    encoded = self.model.encode(
                        src=masked_seq.unsqueeze(0),
                        src_mask=None,
                        src_key_padding_mask=pad_mask.unsqueeze(0)
                    )

                    logits = self.model.project(encoded)[0]

                    mask = (labels != -100)
                    num_undetected_tokens = self.count_undetected_tokens(logits[mask], labels[mask])
                    num_masked_tokens = mask.sum()
                    undetected_rates.append(num_undetected_tokens / num_masked_tokens)


                seq_results["undetected_mean_rate"] = np.mean(undetected_rates)
                seq_results["undetected_std"] = np.std(undetected_rates)
                std.append(np.std(undetected_rates))
                results.append(seq_results)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Batch {batch_idx+1}/{len(dataloader)}")

        print(f"mean std: {np.mean(std)} | sigma std: {np.std(std)}")

        return results

    def mask_sequence(self, sequence):
        masked_seq = sequence.detach().clone()

        labels = [-100] * len(sequence)

        for i, token in enumerate(sequence):
            if token in (self.pad_id,):
                continue

            if random.random() < self.mask_ratio:
                labels[i] = token
                masked_seq[i] = self.mask_id
        
        return masked_seq, torch.tensor(labels, dtype=torch.long)


    def count_undetected_tokens(self, prediction, label):
        num_undetected_tokens = 0

        for i, token in enumerate(label):
            if token not in torch.argsort(prediction[i], descending=True)[:self.num_candidates]:
                num_undetected_tokens += 1

        return num_undetected_tokens
    
    def find_best_threshold(self, normal_results, abnormal_results, thresholds):
        best_f1 = -1
        results = {
            "Threshold": 0,
            "FP": 0,
            "TP": 0,
            "TN": 0,
            "FN": 0,
            "Precision": 0,
            "Recall": 0,
            "F1": 0
        }
        for th in thresholds:
            FP = compute_anomaly(normal_results, th)
            TP = compute_anomaly(abnormal_results, th)

            TN = len(normal_results) - FP
            FN = len(abnormal_results) - TP
            if (TP + FP) == 0:
                precision = 0
            else:
                precision = TP / (TP + FP)
            if (TP + FN) == 0:
                recall = 0
            else:
                recall = TP / (TP + FN)
            if (precision + recall) == 0:
                F1 = 0
            else:
                F1 = 2 * precision * recall / (precision + recall)
            print(f"\nTesting threshold {th}")
            print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
            print(f"Precision: {precision}, Recall: {recall} | F1: {F1}")

            if F1 > best_f1:
                results = {
                    "Threshold": th,
                    "FP": FP,
                    "TP": TP,
                    "TN": TN,
                    "FN": FN,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": F1
                }
                best_f1 = F1

        return results
        
from lightmnet3_train_ablation_common import main_with_defaults


if __name__ == "__main__":
    main_with_defaults({
        "experiment_name": "LEVIRCD-NoParam",
        "model_module": "lightmnet3_ablation_no_param",
        "data_root": r"E:/pyCharmProjects/LEVIR-CD",
        "save_path": r"e:/pyCharmProjects/lightMnet/ablation_runs/weights/levircd_no_param_best.pth",
        "result_path": r"e:/pyCharmProjects/lightMnet/ablation_runs/results/levircd_no_param_result.json",
        "batch_size": 4,
        "epochs": 100,
        "lr": 5e-5,
        "weight_decay": 5e-3,
        "num_workers": 2,
        "pos_weight": 3.0,
        "target_f1": 1.0,
        "pretrained_backbone": True,
    })

import subprocess
import optuna
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'  # fix MKL error

def objective(trial):
    corpus = 'dyda'
    mode = 'train'
    gpu = '1'
    epochs = 30
    nclass = 4
    emb_batch = 0
    speaker_info = 'none'
    topic_info = 'none'

    lr = trial.suggest_loguniform('lr', 5e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    dropout = trial.suggest_categorical('dropout', [0.3, 0.5])
    nlayer = trial.suggest_int('nlayer', 1, 2)
    nfinetune = trial.suggest_categorical('nfinetune', [1, 2])
    chunk_size = trial.suggest_categorical('chunk_size', [0])

    command = [
        "python", "-u", "engine.py",
        "--corpus", corpus,
        "--mode", mode,
        "--gpu", gpu,
        "--batch_size", str(batch_size),
        "--batch_size_val", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--nlayer", str(nlayer),
        "--chunk_size", str(chunk_size),
        "--dropout", str(dropout),
        "--nfinetune", str(nfinetune),
        "--speaker_info", speaker_info,
        "--topic_info", topic_info,
        "--nclass", str(nclass),
        "--emb_batch", str(emb_batch)
    ]

    print("Running command:", ' '.join(command))
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout

    # Print for debugging if needed
    print(output)

    # Parse validation accuracy
    val_acc = None
    for line in output.split('\n'):
        if 'Final Validation Accuracy' in line:
            val_acc = float(line.strip().split(':')[-1])
            break

    if val_acc is None:
        raise ValueError("Validation accuracy not found in output!")

    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_params)


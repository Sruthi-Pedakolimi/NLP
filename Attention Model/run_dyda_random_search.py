import os
import random

corpus = 'dyda'
mode = 'train'
gpu = '0'
epochs = 30
nclass = 4
emb_batch = 0
speaker_info = 'none'  # handled internally by attention
topic_info = 'none'

search_space = {
    'lr': [5e-6, 1e-5, 5e-5, 1e-4, 3e-4],
    'batch_size': [8, 16],
    'dropout': [0.3, 0.5],
    'nlayer': [1, 2],
    'nfinetune': [2, 4],
    'chunk_size': [0, 32]
}

num_trials = 10  # Number of experiments

for trial in range(num_trials):
    lr = random.choice(search_space['lr'])
    batch_size = random.choice(search_space['batch_size'])
    dropout = random.choice(search_space['dropout'])
    nlayer = random.choice(search_space['nlayer'])
    nfinetune = random.choice(search_space['nfinetune'])
    chunk_size = random.choice(search_space['chunk_size'])

    command = (
        f"python -u engine.py --corpus={corpus} --mode={mode} --gpu={gpu} "
        f"--batch_size={batch_size} --batch_size_val={batch_size} --epochs={epochs} "
        f"--lr={lr} --nlayer={nlayer} --chunk_size={chunk_size} "
        f"--dropout={dropout} --nfinetune={nfinetune} "
        f"--speaker_info={speaker_info} --topic_info={topic_info} "
        f"--nclass={nclass} --emb_batch={emb_batch}"
    )

    print(f"\n{'='*40}\nTrial {trial + 1}/{num_trials}")
    print("Running command:\n", command)
    os.system(command)

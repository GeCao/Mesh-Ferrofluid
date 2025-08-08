#!/usr/bin/env python
# coding: utf-8

# # Technical: Training models across animals

# * This notebook will demo how to use a multisession CEBRA implementation.
# * We will compare embeddings obtained on 4 rat datasets when trained on four single-session models vs. one multisession model.
# * We will use CEBRA-Behavior (``conditional='time_delta'``) for both single and multi session implementation.
# * Each individual single session output embedding is aligned to the first session.
# 
# 
# **How multi-session training works:**
# 
# * For flexibility, it is implemented so that it does not fit one model for all sessions. Consequently, it is possible to use sessions that do not have the same number of data features (e.g., not the same number of neurons from one session to the other). The number of samples can also vary (true for single session too).
# * It fits one model per session but the pos/neg sampling is performed across all sessions, making the models invariant to the labels across all sessions.
# 
# 
# **When to use multi-session training:**
# 
# Check out the following list to verify if our multisession implementation is the right tool for your needs.
# 
# * I have multiple sessions or animals that I want to consider as a *pseudo-subject* and use them jointly for training CEBRA. 
# * That is the case because of limited access to simultaneously recorded neurons or looking for animal-invariant features in the neural data. 
# * I want to get more consistent embeddings from one session or animal to the other. 
# * I want to be able to use CEBRA for a new session that is fully *unseen* during training (e.g., could be useful for a brain machine interface applications).
# * **WARNING:** I am not interested in the influence of individual variations of the label features from one session/animal to the other on the resulting embedding. I do not need those session or animal specific information.
# 

# **Install note** 
# 
# Be sure you have cebra, and the demo dependencies, installed to use this notebook: 

# %%

import os
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import r2_score
import numpy as np
import webbrowser
import cebra
from cebra import CEBRA, KNNDecoder
from cebra.integrations.plotly import plot_embedding_interactive
from pathlib import Path

# %% Load the data

session_paths = [
    "E:/data/m2070/Oct_16_2024",
    "E:/data/m2070/Nov_09_2024",
    "E:/data/m2070/Nov_15_2024",
    "E:/data/m2071/Oct_16_2024",
    "E:/data/m2071/Nov_09_2024",
    "E:/data/m2071/Nov_15_2024",
    "E:/data/m2072/Nov_09_2024",
    "E:/data/m2072/Nov_15_2024",
    "E:/data/m2151/Dec_30_2024",
    "E:/data/m2151/Jan_02_2025",
    "E:/data/m2152/Dec_30_2024",
    "E:/data/m2152/Jan_02_2025",
    "E:/data/m2154/Dec_30_2024",
    "E:/data/m2154/Jan_02_2025"
]

# %%

names = []
datas = []
labels = []

train_datas = []
train_labels = []

test_datas = []
test_labels = []

for path in session_paths:
    mouse = path.split("/")[-2]
    date = path.split("/")[-1]
    session_name = f"{mouse}_{date}"
    names.append(session_name)

    # Load the data
    cell_mat = loadmat(os.path.join(path, "cell_data.mat"))
    pupil_mat = loadmat(os.path.join(path, "pupil_data.mat"))

    cell_array = cell_mat['cell_traces_interpolated_smooth']
    pupil_array = pupil_mat['pupil_smooth_zscore']

    datas.append(cell_array.astype(np.float64))
    labels.append(pupil_array.astype(np.float64))

    # Use only first 60% of the data
    n = cell_array.shape[0]
    split_idx = int(n * 0.6)

    train_datas.append(cell_array[:split_idx].astype(np.float64))
    train_labels.append(pupil_array[:split_idx].astype(np.float64))
    
    test_datas.append(cell_array[split_idx:].astype(np.float64))
    test_labels.append(pupil_array[split_idx:].astype(np.float64))

# %% Multisession training
# 
# Create and fit one multi-session model on all datasets

max_iterations = 15000 # by default, 5000 iterations

# Multisession training
multi_cebra_model = CEBRA(model_architecture='offset10-model',
                    batch_size=512,
                    learning_rate=3e-4,
                    temperature=1,
                    output_dimension=3,
                    max_iterations=max_iterations,
                    distance='cosine',
                    conditional='time_delta',
                    device='cuda_if_available',
                    verbose=True,
                    time_offsets=10)

# Train on all training datasets
multi_cebra_model.fit(train_datas, train_labels)

# Save CEBRA model
tmp_file = Path("C:/Users/labadmin/Desktop/yihui/codes/cebra/multi_cebra_model.pt")
multi_cebra_model.save(tmp_file)

# %% Get and store embeddings for training data only

train_embeddings = {}

for i, (name, X) in enumerate(zip(names, train_datas)):
    train_embeddings[name] = multi_cebra_model.transform(X, session_id=i)

# Save embeddings in current folder
with open('train_embeddings.pkl', 'wb') as f:
    pickle.dump(train_embeddings, f)

# %%

output_dir = "C:/Users/labadmin/Desktop/yihui/codes/cebra/pupil_train_embeddings_plots"
os.makedirs(output_dir, exist_ok=True)

for name, emb, label in zip(names, train_embeddings.values(), train_labels):
    fig = plot_embedding_interactive(
        emb,
        embedding_labels=label[:, 0],
        title=f"Train - {name}",
        markersize=3,
        cmap="rainbow"
    )
    html_path = os.path.join(output_dir, f"{name}.html")
    fig.write_html(html_path)
    webbrowser.open(f"file://{html_path}")

# %%

fig = plt.figure(figsize=(20, 12))

for i, (name, emb, label) in enumerate(zip(names, train_embeddings.values(), train_labels), start=1):
    ax = fig.add_subplot(4, 7, i, projection='3d')
    
    sc = ax.scatter(emb[:,0], emb[:,1], emb[:,2], c=label[:,0], cmap='rainbow', s=5, alpha=0.7)
    
    ax.view_init(elev=20, azim=30)
    
    ax.set_title(name, fontsize=8)
    ax.axis('off')
    
    last_scatter = sc

cbar_ax = fig.add_axes([0.95, 0.7, 0.0133, 0.2])
cbar = fig.colorbar(last_scatter, cax=cbar_ax, orientation='vertical')
cbar.set_label('Pupil Size', fontsize=12, labelpad=5)

plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.savefig("C:/Users/labadmin/Desktop/yihui/codes/cebra/pupil_train_embeddings_plots/combined_static.png", dpi=300)
plt.show()

# %%

test_embeddings = {}

for i, (name, X) in enumerate(zip(names, test_datas)):
    test_embeddings[name] = multi_cebra_model.transform(X, session_id=i)

# Save embeddings
with open("test_embeddings.pkl", "wb") as f:
    pickle.dump(test_embeddings, f)

# %%

output_dir = "C:/Users/labadmin/Desktop/yihui/codes/cebra/pupil_test_embeddings_plots"
os.makedirs(output_dir, exist_ok=True)

for name, emb, label in zip(names, test_embeddings.values(), test_labels):
    fig = plot_embedding_interactive(
        emb,
        embedding_labels=label[:, 0],
        title=f"Test - {name}",
        markersize=3,
        cmap="rainbow"
    )
    html_path = os.path.join(output_dir, f"{name}.html")
    fig.write_html(html_path)
    webbrowser.open(f"file://{html_path}")

# %%

fig = plt.figure(figsize=(20, 12))

for i, (name, emb, label) in enumerate(zip(names, test_embeddings.values(), test_labels), start=1):
    ax = fig.add_subplot(4, 7, i, projection='3d')
    
    sc = ax.scatter(emb[:,0], emb[:,1], emb[:,2], c=label[:,0], cmap='rainbow', s=5, alpha=0.7)
    
    ax.view_init(elev=20, azim=30)
    
    ax.set_title(name, fontsize=8)
    ax.axis('off')
    
    last_scatter = sc

cbar_ax = fig.add_axes([0.95, 0.7, 0.0133, 0.2])
cbar = fig.colorbar(last_scatter, cax=cbar_ax, orientation='vertical')
cbar.set_label('Pupil Size', fontsize=12, labelpad=5)

plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.savefig("C:/Users/labadmin/Desktop/yihui/codes/cebra/pupil_test_embeddings_plots/combined_static.png", dpi=300)
plt.show()
    
# %% Label-Shuffle Control

# Shuffle the behavior variable and use it for training
shuffled_train_labels = [np.random.permutation(label) for label in train_labels]

# Label Shuffle control model:
cebra_shuffled_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=5000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

# Train on shuffled training datasets
cebra_shuffled_model.fit(train_datas, shuffled_train_labels)

# Save CEBRA model
tmp_file = Path("C:/Users/labadmin/Desktop/yihui/codes/cebra/cebra_shuffled_model.pt")
cebra_shuffled_model.save(tmp_file)

# %%

shuffled_train_embeddings = {}

for i, (name, X) in enumerate(zip(names, train_datas)):
    shuffled_train_embeddings[name] = cebra_shuffled_model.transform(X, session_id=i)

with open('shuffled_train_embeddings.pkl', 'wb') as f:
    pickle.dump(shuffled_train_embeddings, f)
    
# %%

output_dir = "C:/Users/labadmin/Desktop/yihui/codes/cebra/pupil_shuffled_embeddings_plots"
os.makedirs(output_dir, exist_ok=True)

for name, emb, label in zip(names, shuffled_train_embeddings.values(), train_labels):
    fig = plot_embedding_interactive(
        emb,
        embedding_labels=label[:, 0],
        title=f"{name} (labels shuffled)",
        markersize=3,
        cmap="rainbow"
    )
    html_path = os.path.join(output_dir, f"{name}.html")
    fig.write_html(html_path)
    webbrowser.open(f"file://{html_path}")

# %%

fig = plt.figure(figsize=(20, 12))

for i, (name, emb, label) in enumerate(zip(names, shuffled_train_embeddings.values(), train_labels), start=1):
    ax = fig.add_subplot(4, 7, i, projection='3d')
    
    sc = ax.scatter(emb[:,0], emb[:,1], emb[:,2], c=label[:,0], cmap='rainbow', s=5, alpha=0.7)
    
    ax.view_init(elev=20, azim=30)
    
    ax.set_title(name, fontsize=8)
    ax.axis('off')
    
    last_scatter = sc

cbar_ax = fig.add_axes([0.95, 0.7, 0.0133, 0.2])
cbar = fig.colorbar(last_scatter, cax=cbar_ax, orientation='vertical')
cbar.set_label('Pupil Size', fontsize=12, labelpad=5)

plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.savefig("C:/Users/labadmin/Desktop/yihui/codes/cebra/pupil_shuffled_embeddings_plots/combined_static.png", dpi=300)
plt.show()

# %%

shuffled_test_embeddings = {}

for i, (name, X) in enumerate(zip(names, test_datas)):
    shuffled_test_embeddings[name] = cebra_shuffled_model.transform(X, session_id=i)

with open('shuffled_test_embeddings.pkl', 'wb') as f:
    pickle.dump(shuffled_test_embeddings, f)    
    
# %% Decoding & plot pupil prediction on validation data

# Load variables
with open("train_embeddings.pkl", "rb") as f:
    train_embeddings = pickle.load(f)

with open("test_embeddings.pkl", "rb") as f:
    test_embeddings = pickle.load(f)
    
# Setup decoder
decoder = KNNDecoder(n_neighbors=3, metric="cosine")

save_dir = "C:/Users/labadmin/Desktop/yihui/codes/cebra/pupil_prediction_plots"
os.makedirs(save_dir, exist_ok=True)

r2_scores = []

for i, name in enumerate(names):
    embedding_train = train_embeddings[name]
    embedding_test = test_embeddings[name]
    
    train_label = train_labels[i][:, 0]
    test_label = test_labels[i][:, 0]

    decoder.fit(embedding_train, train_label)
    prediction = decoder.predict(embedding_test)
    
    r2 = r2_score(test_label, prediction)
    r2_scores.append((name, r2))
    print(f"{name} R² score: {r2:.4f}")

    # Plot prediction vs. true trace
    time = range(len(test_label))
    plt.figure(figsize=(12, 3))
    plt.plot(time, test_label, label='True', color='black', linewidth=2)
    plt.plot(time, prediction, label='Predicted', color='blue', linewidth=1.5)
    plt.title(f"{name} - Pupil Prediction (R² = {r2:.3f})", fontsize=12)
    plt.xlabel("Time (frames)")
    plt.ylabel("Pupil size (z-score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_pupil_prediction.png"), dpi=150)
    plt.close()
    
# %% Decoding & plot pupil prediction on validation data

# Load CEBRA model
tmp_file = Path("C:/Users/labadmin/Desktop/yihui/codes/cebra/multi_cebra_model.pt")
multi_cebra_model = cebra.CEBRA.load(tmp_file)

# Load variables
with open("train_embeddings.pkl", "rb") as f:
    train_embeddings = pickle.load(f)

with open("test_embeddings.pkl", "rb") as f:
    test_embeddings = pickle.load(f)

with open('shuffled_train_embeddings.pkl', 'rb') as f:
    shuffled_train_embeddings = pickle.load(f)
    
with open('shuffled_test_embeddings.pkl', 'rb') as f:
    shuffled_test_embeddings = pickle.load(f)
    
# Setup decoder
decoder = KNNDecoder(n_neighbors=3, metric="cosine")
shuffled_decoder = KNNDecoder(n_neighbors=3, metric="cosine")

save_dir = "C:/Users/labadmin/Desktop/yihui/codes/cebra/pupil_prediction_plots"
os.makedirs(save_dir, exist_ok=True)

r2_scores = []
r2_scores_shuffled = []

for i, name in enumerate(names):
    embedding_train = train_embeddings[name]
    embedding_test = test_embeddings[name]
    shuffled_embedding_train = shuffled_train_embeddings[name]
    shuffled_embedding_test = shuffled_test_embeddings[name]
    
    train_label = train_labels[i][:, 0]
    test_label = test_labels[i][:, 0]

    decoder.fit(embedding_train, train_label)
    prediction = decoder.predict(embedding_test)
    
    shuffled_decoder.fit(shuffled_embedding_train, train_label)
    shuffled_prediction = shuffled_decoder.predict(shuffled_embedding_test)

    r2 = r2_score(test_label, prediction)
    r2_scores.append((name, r2))
    print(f"{name} R² score: {r2:.4f}")
    
    r2_shuffled = r2_score(test_label, shuffled_prediction)
    r2_scores_shuffled.append((name, r2_shuffled))
    print(f"{name} (labels shuffled) R² score: {r2_shuffled:.4f}")

    # Plot prediction vs. true trace
    time = range(len(test_label))
    plt.figure(figsize=(12, 3))
    plt.plot(time, test_label, label='True', color='black', linewidth=2)
    plt.plot(time, prediction, label='Predicted', color='blue', linewidth=1.5)
    plt.plot(time, shuffled_prediction, label='Shuffled Prediction', color='gray', alpha=0.3, linewidth=1)
    plt.title(f"{name} - Pupil Prediction (R² = {r2:.3f})", fontsize=12)
    plt.xlabel("Time (frames)")
    plt.ylabel("Pupil size (z-score)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_pupil_prediction_full.png"), dpi=150)
    plt.close()
    





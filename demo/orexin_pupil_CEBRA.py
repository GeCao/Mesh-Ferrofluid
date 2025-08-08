# # Training CEBRA models for Orexin data
# 
# We show you how to get started with CEBRA:
# 
# - Define a CEBRA-Time model (we strongly recommend starting with this).
# - Load data.
# - Perform train/validation splits.
# - Train the model.
# - Check the loss functions.
# - Save the model & reload.
# - Transform the model on train/val.
# - Evaluate Goodness of Fit.
# - Visualize the embeddings.
# - Compute and display the Consistency between runs (n=10).
# - Run a (small) grid search for model parameters.
# 
# Once you have a good CEBRA-Time model, then you can do hypothesis testing with CEBRA-Behavior:
# - Define a CEBRA-Behavior model.
# - as above, train, check, evaluate, transform, and test consistency.
# - Run controls with shuffled data - which is critical for label-guided embeddings.

# %%

import sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cebra.datasets
from cebra import CEBRA

#for model saving:
import os
import tempfile
from pathlib import Path

# %%

# ## 1. Set up a CEBRA model
# 
# ### Items to consider
# 
# - We recommend starting with an unsupervised approach (CEBRA-Time).
# - We recommend starting with defaults, perform the sanity checks we suggest below, then performing a grid search if needed.
# - We are going to largely follow the recommendations from our [Quick Start scikit-learn API](https://cebra.ai/docs/usage.html#quick-start-scikit-learn-api-example)

# %% Define a CEBRA model

cebra_model = CEBRA(
    model_architecture="offset10-model", #consider: "offset10-model-mse" if Euclidean
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.12,
    max_iterations=5000, #we will sweep later; start with default
    conditional='time', #for supervised, put 'time_delta', or 'delta'
    output_dimension=3,
    distance='cosine', #consider 'euclidean'; if you set this, output_dimension min=2
    device="cuda_if_available",
    verbose=True,
    time_offsets=10
)

# %%

# ## 2. Load the data
# 
# - (or adapt and use your data)
# - We are going to use demo data. The data will be automatically downloaded into a `/data` folder.

# %% load data

cell_data = loadmat('E:/data/m2072/Nov_15_2024/cell_data.mat')
cell_array = cell_data['cell_traces_interpolated_smooth']

pupil_data = loadmat('E:/data/m2072/Nov_15_2024/pupil_data.mat')
pupil_array = pupil_data['pupil_smooth_zscore']


# %% Visualize the data

fig = plt.figure(figsize=(9,3), dpi=150)
plt.subplots_adjust(wspace = 0.3)
ax = plt.subplot(121)
ax.imshow(cell_array.T, aspect = 'auto', cmap = 'Blues')
plt.ylabel('Neuron #')
plt.xlabel('Time [s]')
#plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))

ax2 = plt.subplot(122)
ax2.scatter(np.arange(60966), pupil_array,c=pupil_array, cmap='rainbow', s=1)

plt.ylabel('Pupil Size')
plt.xlabel('Time [s]')
#plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))
plt.show()


# %% Quick test: Train CEBRA-Time on the full data (not train/validation yet)...
# 
# - This is a rapid quick start, just training without labels on the full dataset on the model we set up above! Here, we should already see a nice structured embedding.
# - Note, the colors here are post-hoc applied; positional information was not used to train the model.

# fit
cebra_time_full_model = cebra_model.fit(cell_array)
# transform
cebra_time_full = cebra_model.transform(cell_array)
# GoF
gof_full = cebra.sklearn.metrics.goodness_of_fit_score(cebra_time_full_model, cell_array)
print(" GoF in bits - full:", gof_full)
# plot embedding
fig = cebra.integrations.plotly.plot_embedding_interactive(cebra_time_full, embedding_labels=pupil_array[:, 0], title = "CEBRA-Time (full)", markersize=3, cmap = "rainbow")
fig.show()
fig.write_html("E:/data/m2072/Nov_15_2024/plot_1.html")
import webbrowser
webbrowser.open("E:/data/m2072/Nov_15_2024/plot_1.html")
# plot the loss curve
ax = cebra.plot_loss(cebra_time_full_model)

# %%

# ## 3. Create a Train/Validation Split
# 
# - now that we know we get something decent (see structure, proper loss curve), we can properly test parameters.

# %% Split data and labels (labels we use later!)

from sklearn.model_selection import train_test_split
split_idx = int(0.8 * len(cell_array)) #suggest: 5%-20% depending on your dataset size

train_data = cell_array[:split_idx]
valid_data = cell_array[split_idx:]

train_continuous_label = pupil_array[:split_idx]
valid_continuous_label = pupil_array[split_idx:]

# %%

# ## 4. Fit the train split model

# %%

cebra_train_model = cebra_model.fit(train_data)#, train_continuous_label)

# %%

# ## 5. Save the model [optional]

# %%

tmp_file = Path(tempfile.gettempdir(), 'cebra.pt')
cebra_train_model.save(tmp_file)
#reload
cebra_train_model = cebra.CEBRA.load(tmp_file)

# %%

# ## 6. Compute (transform) the embedding on train and validation data

# %%

train_embedding = cebra_train_model.transform(train_data)
valid_embedding = cebra_train_model.transform(valid_data)

# %%

# ## 7. Evaluate the Model
# - Plot the loss curve
# - We can also look at the Goodness of Fit this in bits vs. the infoNCE loss. [See more info here](https://cebra.ai/docs/api/sklearn/metrics.html#cebra.integrations.sklearn.metrics.goodness_of_fit_score)
#  - ProTip: 0 bits would be a perfectly collapsed embedding. Note, using GoF on the validation set is prone to low data regime issues, hence one should use the train loss to evaluate the model.

# %%

gof_train = cebra.sklearn.metrics.goodness_of_fit_score(cebra_train_model, train_data)
print(" GoF bits - train:", gof_train)

# plot the loss curve
ax = cebra.plot_loss(cebra_train_model)


# %% Visualize the embeddings
# 
# - train, then validation

import cebra.integrations.plotly
#train
fig = cebra.integrations.plotly.plot_embedding_interactive(train_embedding,
                                                           embedding_labels=train_continuous_label[:,0],
                                                           title = "CEBRA-Time Train",
                                                           markersize=3,
                                                           cmap = "rainbow")
fig.show()
fig.write_html("E:/data/m2072/Nov_15_2024/plot_2.html")
webbrowser.open("E:/data/m2072/Nov_15_2024/plot_2.html")

#validation
fig = cebra.integrations.plotly.plot_embedding_interactive(valid_embedding,
                                                           embedding_labels=valid_continuous_label[:,0],
                                                           title = "CEBRA-Time-validation",
                                                           markersize=3,
                                                           cmap = "rainbow")
fig.show()
fig.write_html("E:/data/m2072/Nov_15_2024/plot_3.html")
webbrowser.open("E:/data/m2072/Nov_15_2024/plot_3.html")

# %%

# ## Next sanity/validation step: Consistency
# 
# What did we check above?
#  - (1) do we see structure in our embedding? (if not, something is off!)
#  - (2) is the GoF reasonable? (infoNCE low, bits high)
#  - (3) Is the loss converging without overfitting (no sudden drop after many interations?)
# 
# IF 1-3 are not satisfactory, skip to the **Grid Search Below!**
# 
# Beyond these being met, we need to check the consistency across runs! In addition to the above checks, once we have a converging model that produces **consistent embeddings**, then we know we have good model parameters! 🚀

# %% 

### Now we are going to run our train/val. 5-10 times to be sure they are consistent!

X = 5  # Number of training runs
model_paths = []  # Store file paths

for i in range(X):
    print(f"Training 🦓CEBRA model {i+1}/{X}")

    # Train and save model
    cebra_train_model = cebra_model.fit(train_data)
    tmp_file = Path(tempfile.gettempdir(), f'cebra_{i}.pt')
    cebra_train_model.save(tmp_file)
    model_paths.append(tmp_file)

### Reload models and transform data
train_embeddings = []
valid_embeddings = []

for tmp_file in model_paths:
    cebra_train_model = cebra.CEBRA.load(tmp_file)
    train_embeddings.append(cebra_train_model.transform(train_data))
    valid_embeddings.append(cebra_train_model.transform(valid_data))

# %% Compute Consistency Across Runs
# - Now that we have 5-10 model runs, we can compute the consistency between runs.
# - TRAIN: This should be high (in the 90's on the train embeddings)! If not, in this demo, we simply suggest training slightly longer.
# - VALID: Depending on how large your validation data are, this also should be as high.
#  - In our demo data, the cebra-time on rat 1 with 20% held out is in the 70's for 5K iterations, which is acceptable. One could consider training for longer (~8-9K).

scores, pairs, ids_runs = cebra.sklearn.metrics.consistency_score(
    embeddings=train_embeddings,
    between="runs"
)

cebra.plot_consistency(scores, pairs, ids_runs)


scores, pairs, ids_runs = cebra.sklearn.metrics.consistency_score(
    embeddings=valid_embeddings,
    between="runs"
)

cebra.plot_consistency(scores, pairs, ids_runs)


# %% What if I don't have good parameters? Let's do a grid search...

#%mkdir saved_models

params_grid = dict(
    output_dimension = [3, 6],
    time_offsets = [5, 10],
    model_architecture='offset10-model',
    temperature_mode='constant',
    temperature=[0.1, 1.0],
    max_iterations=[5000],
    device='cuda_if_available',
    num_hidden_units = [32, 64],
    verbose = True)

datasets = {"dataset1": train_data}

# run the grid search
grid_search = cebra.grid_search.GridSearch()
grid_search.fit_models(datasets, params=params_grid, models_dir="saved_models")


# %%

# Get the results
df_results = grid_search.get_df_results(models_dir="saved_models")

# Get the best model for a given dataset
best_model, best_model_name = grid_search.get_best_model(dataset_name="dataset1", models_dir="saved_models")
print("The best model is:", best_model_name)


# %%

#load the top model ✨
model_path = Path("C:/Users/labadmin/saved_models") / f"{best_model_name}.pt"
top_model = cebra.CEBRA.load(model_path)

#transform:
top_train_embedding = top_model.transform(train_data)
top_valid_embedding = top_model.transform(valid_data)

# plot the loss curve
ax = cebra.plot_loss(top_model)


# plot embeddings
fig = cebra.integrations.plotly.plot_embedding_interactive(top_train_embedding,
                                                           embedding_labels=train_continuous_label[:,0],
                                                           title = "top model - train",
                                                           markersize=3,
                                                           cmap = "rainbow")
fig.show()
fig.write_html("C:/Users/labadmin/Desktop/yihui/codes/cebra/CEBRA-demos-main/plot_4.html")
webbrowser.open("C:/Users/labadmin/Desktop/yihui/codes/cebra/CEBRA-demos-main/plot_4.html")

fig = cebra.integrations.plotly.plot_embedding_interactive(top_valid_embedding,
                                                           embedding_labels=valid_continuous_label[:,0],
                                                           title = "top model - validation",
                                                           markersize=3,
                                                           cmap = "rainbow")
fig.show()
fig.write_html("C:/Users/labadmin/Desktop/yihui/codes/cebra/CEBRA-demos-main/plot_5.html")
webbrowser.open("C:/Users/labadmin/Desktop/yihui/codes/cebra/CEBRA-demos-main/plot_5.html")

# %%

# ## CEBRA-Behavior: using auxiliary labels for hypothesis testing
# 
# - Now that you have good parameters for a self-supervised embedding, the next goal is to understand which behavioral labels are contributing to the model fit.
# - Thus, we will use labels, such as position, for testing.
# - ⚠️ We test model consistency on train/validation splits.
# - Then, we perform shuffle controls.

# %%

# Define the model
# consider changing based on search/results above
cebra_behavior_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=5000,
                        distance='cosine',
                        conditional='time_delta', #using labels
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

# %%

# fit
cebra_behavior_full_model = cebra_behavior_model.fit(cell_array,pupil_array)
# transform
cebra_behavior_full = cebra_behavior_full_model.transform(cell_array)
# GoF
gof_full = cebra.sklearn.metrics.goodness_of_fit_score(cebra_behavior_full_model, cell_array,pupil_array)
print(" GoF in bits - full:", gof_full)
# plot embedding
fig = cebra.integrations.plotly.plot_embedding_interactive(cebra_behavior_full, embedding_labels=pupil_array[:,0], title = "CEBRA-Behavior (full)", markersize=3, cmap = "rainbow")
fig.show()
fig.write_html("E:/data/m2072/Nov_15_2024/plot_6.html")
webbrowser.open("E:/data/m2072/Nov_15_2024/plot_6.html")

# plot the loss curve
ax = cebra.plot_loss(cebra_behavior_full_model)


# %% Test model consistency
# - run train/valid.

# Now we are going to run our train/val. 5-10 times to be sure they are consistent!

X = 5  # Number of training runs
model_paths = []  # Store file paths

for i in range(X):
    print(f"Training 🦓CEBRA model {i+1}/{X}")

    # Train and save model
    cebra_behavior_train_model = cebra_behavior_model.fit(train_data,train_continuous_label)
    tmp_file2 = Path(tempfile.gettempdir(), f'cebra_behavior_{i}.pt')
    cebra_behavior_train_model.save(tmp_file2)
    model_paths.append(tmp_file2)

# Reload models and transform data
train_behavior_embeddings = []
valid_behavior_embeddings = []

for tmp_file2 in model_paths:
    cebra_behavior_train_model = cebra.CEBRA.load(tmp_file2)
    train_behavior_embeddings.append(cebra_behavior_train_model.transform(train_data))
    valid_behavior_embeddings.append(cebra_behavior_train_model.transform(valid_data))

# %%

gof_train = cebra.sklearn.metrics.goodness_of_fit_score(cebra_behavior_train_model, train_data, train_continuous_label)
print(" GoF bits - train:", gof_train)

ax = cebra.plot_loss(cebra_behavior_train_model)

# %% train

scores, pairs, ids_runs = cebra.sklearn.metrics.consistency_score(
    embeddings=train_behavior_embeddings,
    between="runs"
)

cebra.plot_consistency(scores, pairs, ids_runs)

#validation
scores, pairs, ids_runs = cebra.sklearn.metrics.consistency_score(
    embeddings=valid_behavior_embeddings,
    between="runs"
)

cebra.plot_consistency(scores, pairs, ids_runs)

# %%

# ## The next item to do when using labels is to perform shuffle controls
# 
# **Why do we do this?**
# It is entirely expected that the distribution of the labels shape the embedding (see [Proposition 7, Supplementary Note 2](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06031-6/MediaObjects/41586_2023_6031_MOESM1_ESM.pdf)). In [Figure 2c](https://www.nature.com/articles/s41586-023-06031-6#Fig2) we therefore show shuffle controls should be performed, and demonstrate that if **labels are shuffled**, it is not possible to fit them.
# 
# - In our paper we shuffle the labels across time. When the `time_delta` strategy for positive sampling is used, this changes the distribution of the positive samples. We do this kind of control to show that fitting a model on nonsensical label structure (in a real experiment, this would be a behavior time series without connection to the neural data) is not possible (assuming a sufficiently large dataset).
# 
# Another approach is to shuffle the neural data. However, if the model has sufficient capacity (is large), the data is limited, and one trains too long, it *can* fit a “lookup table” from input data to the output embedding to match the label distribution (because this is intact and still has structure). Thus, this can be useful to be sure you are not overparameterizing or training too long!
# 
# - If you shuffle the neural data, the question is whether the label structure can be forced on a latent representation of the shuffled data. This will be possible as long as the model has enough capacity to fit a lookup table, where for each timepoint the embedding is arranged to fit the behavior.

# %% Label-Shuffle Control

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


# %% Shuffle the behavior variable and use it for training

shuffled_pupil = np.random.permutation(pupil_array[:,0])

# %%

#fit, transform
cebra_shuffled_model.fit(cell_array, shuffled_pupil)
cebra_cell_shuffled = cebra_shuffled_model.transform(cell_array)
# GoF
gof_full = cebra.sklearn.metrics.goodness_of_fit_score(cebra_shuffled_model, cell_array,pupil_array)
print(" GoF in bits - full:", gof_full)
# plot embedding
fig = cebra.integrations.plotly.plot_embedding_interactive(cebra_cell_shuffled, embedding_labels=pupil_array[:,0], title = "CEBRA-Behavior (labels shuffled)", markersize=3, cmap = "rainbow")
fig.show()
fig.write_html("E:/data/m2072/Nov_15_2024/plot_7.html")
webbrowser.open("E:/data/m2072/Nov_15_2024/plot_7.html")

# plot the loss curve
ax = cebra.plot_loss(cebra_shuffled_model)

# %% Neural Shuffle Control

### Shuffle the neural data and use it for training

shuffle_idx = np.random.permutation(len(cell_array))
shuffled_neural = cell_array[shuffle_idx]

# %%

#fit, transform
cebra_shuffled_model.fit(shuffled_neural, pupil_array)
cebra_neural_shuffled = cebra_shuffled_model.transform(shuffled_neural)
# GoF
gof_full = cebra.sklearn.metrics.goodness_of_fit_score(cebra_shuffled_model, shuffled_neural,pupil_array)
print(" GoF in bits - full:", gof_full)
# plot embedding
fig = cebra.integrations.plotly.plot_embedding_interactive(cebra_neural_shuffled, embedding_labels=pupil_array[:,0], title = "CEBRA-Behavior (neural shuffled)", markersize=3, cmap = "rainbow")
fig.show()
fig.write_html("E:/data/m2072/Nov_15_2024/plot_8.html")
webbrowser.open("E:/data/m2072/Nov_15_2024/plot_8.html")

# plot the loss curve
ax = cebra.plot_loss(cebra_shuffled_model)


# %% Where do you land? 🚨
# 
# The shuffles with the same parameters should not show any structure, the GoF close to 0, and the loss curve should not drop late in training (which would be overfitting).
# 
# If this is the case, then you have good parameters to go forth with! 🦓🍾
# 
# %% What's next?
# 
# We recommend using these embeddings for model comparisons, decoding, explainable AI (xCEBRA), and/or representation analysis!

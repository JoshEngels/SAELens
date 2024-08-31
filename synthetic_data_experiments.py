# %%


try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    running_in_notebook = True
except:
    running_in_notebook = False

import os
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from sae_lens.config import SyntheticActivationStore
import torch
import einops
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda:1"


seed = 42
torch.random.manual_seed(seed)

# %%

num_sae_features = 1000
data_dim = 500
upper_fire_prob = -1
lower_fire_prob = -3
fire_probabilities = torch.logspace(upper_fire_prob, lower_fire_prob, num_sae_features, device=device)
gt_features = torch.randn((num_sae_features, data_dim), device=device)
gt_features /= gt_features.norm(dim=-1, keepdim=True)
print("Expected L0: ", fire_probabilities.sum().item())

def synthetic_generation_func(verbose: bool = False):
    active_features = (torch.rand((batch_size, num_sae_features), device=device) < fire_probabilities).float()

    if verbose:
        print(f"Average GT L0: {active_features.sum().item() / batch_size}")

    res = einops.einsum(gt_features, active_features, "n d, b n -> b d")

    return res

batch_size = 4096

lr = 1e-4

total_training_steps = 50000

total_training_tokens = total_training_steps * batch_size

# %%

saes = []

for l1_coef in np.logspace(-0.5, 1, 20, base=2):
    print(l1_coef)

    architecture = "gated"

    synthetic_act_store = SyntheticActivationStore(synthetic_generation_func)

    cfg = LanguageModelSAERunnerConfig(
        architecture=architecture,
        # Synthetic data
        d_in=data_dim,
        streaming=True,
        # SAE Parameters
        mse_loss_normalization=None,
        d_sae=num_sae_features,
        # Training Parameters
        l1_coefficient=l1_coef,
        lr=lr,
        train_batch_size_tokens=batch_size,
        training_tokens=total_training_tokens,
        # WANDB
        log_to_wandb=True,
        wandb_entity="josh_engels",
        wandb_project="synthetic_data_scaling",
        wandb_log_frequency=1,
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
        synthetic_data=synthetic_act_store,
        verbose=False,
    )

    sae = SAETrainingRunner(cfg).run().to("cpu")

    saes.append(sae)


# %%

# Download all SAEs
api = wandb.Api()
sae_paths = []
for i in range(1, 21):
    artifact = api.artifact(f'josh_engels/synthetic_data_scaling/sae_None_None_1000:v{i}', type='model')
    artifact_dir = artifact.download()
    sae_paths.append(artifact_dir)
# %%
from sae_lens import SAE 
# Load all SAEs
saes = []
for sae_path in sae_paths:
    sae = SAE.load_from_pretrained(path=sae_path)
    saes.append(sae)

# %%

fig = go.Figure()

num_saes = 20

color_scale = px.colors.sequential.Viridis
colorrange = np.linspace(0, 1, num_saes)
colors = [px.colors.sample_colorscale(color_scale, v)[0] for v in colorrange]
percent_learneds = []
l0s = []

with torch.no_grad():

    for color, sae in zip(colors, saes[-num_saes:]):

        sae_gpu = sae.to(device)

        learned_features = sae_gpu.W_dec.detach()
        sims = gt_features @ learned_features.T
        threshold = 0.9
        percent_learned = (sims > threshold).max(dim=-1).values.float().mean().item()


        data = synthetic_generation_func()
        encoded = sae_gpu.encode(data)
        l0 = (encoded > 0).sum(dim=-1).float().mean().item()

        print(f"Percent learned: {percent_learned:.2f}, L0: {l0:.2f}")

        reconstruction = sae(data)
        error_norms = (data - reconstruction).norm(dim = -1).detach().cpu().numpy()

        fig.add_trace(go.Scatter(
            y=error_norms[100:200], 
            mode='lines',
            # name=f"GT percent learned: {percent_learned:.2f}, L0: {l0:.2f}",
            name=f"L0={l0:.2f}",
            line=dict(width=1, color=color),
        ))

        percent_learneds.append(percent_learned)
        l0s.append(l0)


fig.update_layout(
    margin={'t':0,'l':0,'b':0,'r':0}
)

fig.show()


# Save the figure
fig.write_html("synthetic_data_experiments.html")


# %%

print("R^2, percent_learned, L0")
for sae, percent_learned, l0 in zip(saes, percent_learneds, l0s):

    data_1 = synthetic_generation_func()
    data_2 = synthetic_generation_func()

    output_1 = sae(data_1)
    output_2 = sae(data_2)

    errors_1 = (data_1 - output_1).norm(dim=-1)
    errors_2 = (data_2 - output_2).norm(dim=-1)

    data_1[:, 0] = 1
    data_2[:, 0] = 1

    # Do a linear regression from data_1 to errors_1
    sol = torch.linalg.lstsq(data_1, errors_1).solution

    # Predict the errors for data_2
    predicted_errors_2 = data_2 @ sol

    r_squared = 1 - ((errors_2 - predicted_errors_2) ** 2).sum() / ((errors_2 - errors_2.mean()) ** 2).sum()
    print(f"{r_squared.item():.2f}, {percent_learned:.2f}, {l0:.2f}")
# %%

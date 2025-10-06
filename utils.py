
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from typing import Union
from pathlib import Path
import random

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

from PIL import Image

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    print("sklearnex not installed, using standard sklearn")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



basic_transform = T.Compose([
    T.ToTensor(),
])

class ImageDatasetNPZ(Dataset):
    def __init__(self, data_path: Union[str, Path], transform=None):
        self.load_from_npz(data_path)
        self.transform = transform if transform is not None else basic_transform

    def load_from_npz(self, data_path: Union[str, Path]):
        data = np.load(data_path)
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def extract_features_and_labels(model, dataloader, normalize=False):
    """
    Extract features and labels from a dataloader using the given model.
    model: an encoder model taking as input a batch of images (batch_size, channels, height, width) and outputing either a batch of feature vectors (batch_size, feature_dim) or a list/tuple in which the first element is the batch of feature vectors (batch_size, feature_dim)
    dataloader: a PyTorch dataloader providing batches of (images, labels)
    returns: features (num_samples, feature_dim), labels (num_samples,)
    """
    features = []
    labels = []

    device = next(model.parameters()).device

    for batch in tqdm(dataloader, disable=True):
        x, y = batch
        x = x.to(device)
        with torch.no_grad():
            feats = model.get_features(x)
        features.append(feats.cpu())
        labels.append(y)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    if normalize:
        features = F.normalize(features, dim=1)

    return features, labels



def run_knn_probe(train_features, train_labels, test_features, test_labels):
    """
    Runs a k-NN probe on the given features and labels.
    train_features: (num_train_samples, feature_dim)
    train_labels: (num_train_samples,)
    test_features: (num_test_samples, feature_dim)
    test_labels: (num_test_samples,)
    returns: accuracy (float)
    """
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_features, train_labels)
    test_preds = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, test_preds)
    return accuracy

def run_linear_probe(train_features, train_labels, test_features, test_labels):
    """
    Runs a linear probe on the given features and labels.
    train_features: (num_train_samples, feature_dim)
    train_labels: (num_train_samples,)
    test_features: (num_test_samples, feature_dim)
    test_labels: (num_test_samples,)
    returns: accuracy (float)
    """
    logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
    logreg.fit(train_features, train_labels)
    test_preds = logreg.predict(test_features)
    accuracy = accuracy_score(test_labels, test_preds)
    return accuracy


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # The following lines are commented out to allow for non-deterministic behavior which can improve performance on some models.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_class_name(class_id):
    with open("imagenet_classes.txt") as f:
        return [line.strip() for line in f.readlines()][class_id]

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))




####################################################################################################################################
############################################################ BYOL STUFF ############################################################
####################################################################################################################################


class EMA():
    def __init__(self, beta=0.99):
        self.beta = beta
    
    def update(self, online, target):
        """
        Update target network parameters (inplace) using exponential moving average of online network parameters.
        online: online network (nn.Module)
        target: target network (nn.Module)
        """
        with torch.no_grad():
            for online_param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data = self.beta * target_param.data + (1 - self.beta) * online_param.data


class BYOLTransform:

    def __init__(self, size=32, s=0.5, blur_p=0.5):
        color_jitter = T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        k = 3 if size <= 32 else 5
        base = [
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            # T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=blur_p),
            T.ToTensor()
        ]
        self.train_transform = T.Compose(base)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        return self.train_transform(x), self.train_transform(x)

class DefaultTransform:

    def __init__(self):
        self.transform = T.ToTensor()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        return self.transform(x)

# Create a default instance for convenience
default_transform = DefaultTransform()

def byol_collate_fn(batch):
    xs1, xs2, ys = [], [], []
    for (x1, x2), y in batch:
        xs1.append(x1)
        xs2.append(x2)
        ys.append(y)
    return torch.stack(xs1), torch.stack(xs2), torch.tensor(ys)


    
##################################################################################################################################################################
########################################################################### DIAGNOSTIC ###########################################################################
##################################################################################################################################################################
def embedding_metrics(model, dataloader, num_batches=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    all_features_1, all_features_2 = [], []
    dl_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_batches):
            batch = next(dl_iter)
            x1, x2, _ = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            features_1 = model.get_features(x1)
            features_2 = model.get_features(x2)
            all_features_1.append(features_1)
            all_features_2.append(features_2)


    all_features_1 = torch.cat(all_features_1, dim=0)
    all_features_2 = torch.cat(all_features_2, dim=0)
    normalized_1 = F.normalize(all_features_1, dim=1)
    normalized_2 = F.normalize(all_features_2, dim=1)

    
    similarities = normalized_1 @ normalized_1.T

    n = similarities.size(0)
    mask = ~torch.eye(n, dtype=bool, device=similarities.device)
    off_diag_sims = similarities[mask]
    positive_sims = []
    for f1, f2 in zip(normalized_1, normalized_2):
        positive_sims.append(torch.dot(f1, f2).item())
    
    embedding_var = (torch.var(normalized_1, dim=0).mean().item() + torch.var(normalized_2, dim=0).mean().item())/2
    
    
    return {'mean_sim': off_diag_sims.mean(), 'mean_pos_sim': np.mean(positive_sims), 'embedding_var': embedding_var}

def check_representation_collapse(model, dataloader, num_batches=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            if len(batch) == 3:  # BYOL collate
                x, _, _ = batch
            else:
                x, _ = batch
            x = x.to(device)
            features = model.get_features(x)
            all_features.append(features)
    
    all_features = torch.cat(all_features, dim=0)
    
    # Check for collapse
    mean_feature = all_features.mean(dim=0)
    std_feature = all_features.std(dim=0)
    
    print(f"Feature statistics:")
    print(f"  Mean norm: {mean_feature.norm():.4f}")
    print(f"  Std across samples (per dim): {std_feature.mean():.4f}")
    print(f"  Max std: {std_feature.max():.4f}")
    print(f"  Min std: {std_feature.min():.4f}")
    print(f"  Dims with std < 0.01: {(std_feature < 0.01).sum().item()}/{len(std_feature)}")
    
    # Compute pairwise cosine similarities
    normalized = F.normalize(all_features, dim=1)
    similarities = normalized @ normalized.T
    # Remove diagonal
    n = similarities.size(0)
    mask = ~torch.eye(n, dtype=bool, device=similarities.device)
    off_diag_sims = similarities[mask]
    
    print(f"\nCosine similarity statistics:")
    print(f"  Mean: {off_diag_sims.mean():.4f}")
    print(f"  Std: {off_diag_sims.std():.4f}")
    print(f"  Min: {off_diag_sims.min():.4f}")
    print(f"  Max: {off_diag_sims.max():.4f}")
    
    if off_diag_sims.mean() > 0.9:
        print("⚠️  WARNING: Representations are collapsing! All samples are too similar.")
    
    return all_features, std_feature



def check_projector_output(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    all_proj = []
    all_pred = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            if len(batch) == 3:
                x, _, _ = batch
            else:
                x, _ = batch
            x = x.to(device)
            _, proj, pred = model(x)
            all_proj.append(proj)
            all_pred.append(pred)
    
    all_proj = torch.cat(all_proj, dim=0)
    all_pred = torch.cat(all_pred, dim=0)
    
    print(f"\nProjector output statistics:")
    print(f"  Proj mean: {all_proj.mean():.4f}, std: {all_proj.std():.4f}")
    print(f"  Proj norm: {all_proj.norm(dim=1).mean():.4f}")
    print(f"  Pred mean: {all_pred.mean():.4f}, std: {all_pred.std():.4f}")
    print(f"  Pred norm: {all_pred.norm(dim=1).mean():.4f}")



def visualize_features(features, labels, n_classes=10):
    # Take subset of classes for visualization
    mask = labels < n_classes
    features_subset = features[mask].cpu().numpy()
    labels_subset = labels[mask].cpu().numpy()
    
    # PCA to 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_subset)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_subset, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title(f'Feature visualization (first {n_classes} classes)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")


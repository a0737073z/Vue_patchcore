import csv
import os
import pickle
import time

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils import (
    embedding_concat,
    reshape_embedding,
    KNN,
    min_max_norm,
    select_min_max_norm,
    cvt2heatmap,
    heatmap_on_image
)


class ResizeWithPadding:
    def __init__(self, target_size, interpolation=InterpolationMode.LANCZOS):
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = transforms.functional.resize(img, (new_h, new_w), interpolation=self.interpolation)
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0), self.image_paths[idx], "unknown"


class AnomalyModel(pl.LightningModule):
    def __init__(self, dataset_path, model_path, output_path, input_size=512, n_neighbors=9, save_anomaly_map=True, progress_callback=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.output_path = output_path
        self.input_size = input_size
        self.n_neighbors = n_neighbors
        self.save_anomaly_map = save_anomaly_map
        self.progress_callback = progress_callback
        self.manual_threshold = 1.5
        self.manual_lo_ratio = 0.6

        self.embedding_coreset = pickle.load(open(os.path.join(model_path, "embeddings", 'embedding.pickle'), 'rb'))
        print("Loaded embedding from:", os.path.join(model_path, "embeddings"))

        # Load backbone
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        # Register hooks
        self.features = []
        self.model.layer1[-1].register_forward_hook(lambda m, i, o: self.features.append(o))
        self.model.layer2[-1].register_forward_hook(lambda m, i, o: self.features.append(o))

        # Transforms
        self.data_transforms = transforms.Compose([
            ResizeWithPadding(target_size=self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )

        # Buffers for results
        self.anomaly_map_all = []
        self.pred_list_img_lvl = []
        self.img_path_list = []
        self.input_x_list = []

    def forward(self, x):
        self.features.clear()
        _ = self.model(x)
        return self.features

    def on_test_start(self):

        self.anomaly_map_all = []
        self.pred_list_img_lvl = []
        self.img_path_list = []
        self.input_x_list = []


    def test_dataloader(self):
        if hasattr(self, 'single_test_image_path'):
            class SingleImageDataset(Dataset):
                def __init__(self, image_path, transform):
                    self.image_path = image_path
                    self.transform = transform

                def __len__(self):
                    return 1

                def __getitem__(self, idx):
                    image = Image.open(self.image_path).convert('RGB')
                    return self.transform(image), torch.tensor(0), self.image_path, "unknown"

            return DataLoader(SingleImageDataset(self.single_test_image_path, self.data_transforms), batch_size=1, num_workers=0)
        else:
            return DataLoader(SimpleImageFolder(self.dataset_path, self.data_transforms), batch_size=1, num_workers=0)

    def test_step(self, batch, batch_idx):
        x, _, file_name, _ = batch
        start_time = time.time()

        # Extract features and embeddings
        features = self(x)
        embeddings = [torch.nn.functional.avg_pool2d(f, 3, 1, 1) for f in features]
        embedding_ = embedding_concat(embeddings[0],embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))

        # KNN search
        knn = KNN(torch.from_numpy(self.embedding_coreset).cuda(), k=self.n_neighbors)
        score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()

        # Anomaly map generation
        h = w = int(np.sqrt(score_patches.shape[0]))
        anomaly_map = score_patches[:, 0].reshape((h, w))
        anomaly_map_resized = cv2.resize(anomaly_map, (self.input_size, self.input_size))
        anomaly_map_blur = gaussian_filter(anomaly_map_resized, sigma=4)

        # Image-level prediction
        score = float(score_patches[:, 0].max())
        pred_label = 1 if score > self.manual_threshold else 0

        # Save results
        self.anomaly_map_all.append(anomaly_map_blur)
        self.pred_list_img_lvl.append((file_name[0], score, pred_label))
        self.img_path_list.append(file_name[0])

        # Prepare and store original image
        orig = self.inv_normalize(x).permute(0, 2, 3, 1).cpu().numpy()[0]
        orig = cv2.cvtColor((min_max_norm(orig) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.input_x_list.append(orig)

        # Store latest single image result
        self.latest_score = score
        self.latest_label = pred_label
        self.latest_heatmap_max = float(anomaly_map_blur.max())

        print(f"Inference time for {os.path.basename(file_name[0])}: {time.time() - start_time:.4f} seconds")
        if self.progress_callback:
            self.progress_callback(batch_idx + 1, len(self.test_dataloader()))

    def test_epoch_end(self, outputs):
        os.makedirs(self.output_path, exist_ok=True)
        csv_path = os.path.join(self.output_path, 'results.csv')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'pred_score', 'pred_label'])
            for name, score, label in self.pred_list_img_lvl:
                writer.writerow([os.path.basename(name), round(score, 4), label])

        if not self.save_anomaly_map:
            return

        save_dir = os.path.join(self.output_path, 'anomaly_maps')
        os.makedirs(save_dir, exist_ok=True)

        #maps = np.array(self.anomaly_map_all)
        #hi = (maps.max() - maps.min()) * 1.0 + maps.min()
        #lo = (maps.max() - maps.min()) * self.manual_lo_ratio + maps.min()
        #print(f"[DEBUG] anomaly_map_all max: {maps.max():.4f}, min: {maps.min():.4f}")
        #print(f"[DEBUG] computed hi: {hi:.4f}, lo: {lo:.4f}")

        hi = 2.3
        lo = hi * self.manual_lo_ratio

        for i, (amap, orig, name) in enumerate(zip(self.anomaly_map_all, self.input_x_list, self.img_path_list)):
            amap_norm = select_min_max_norm(amap, hi, lo)
            heatmap = cvt2heatmap(amap_norm * 255)
            overlay = heatmap_on_image(heatmap, orig)
            base = os.path.basename(name)
            cv2.imwrite(os.path.join(save_dir, f'{base}'), orig)
            cv2.imwrite(os.path.join(save_dir, f'amap_on_img_{base}'), overlay)

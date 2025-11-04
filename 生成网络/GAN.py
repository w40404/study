import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset


class Distinguish_model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Generative_model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def preprocess_function(examples):
    # 将 PIL 图片转为 Tensor，并展平为 784 维（28×28）
    examples["image"] = [ToTensor()(img) for img in examples["image"]]
    return examples


def Train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(
        "mnist", cache_dir="./data", download_mode="reuse_dataset_if_exists"
    )
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, num_proc=4  # 批量处理  # 多进程加速
    )
    train_dataset.set_format(
        type="torch",
        dtype=torch.float32,
    )
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    D = Distinguish_model().to(device)
    G = Generative_model().to(device)

    D_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4)

    loss_fn = torch.nn.BCELoss()

    epochs = 100

    for epoch in range(epochs):
        dis_loss_all = 0
        gen_loss_all = 0
        loader_len = len(dataloader)
        for step, data in tqdm(
            enumerate(dataloader), desc="第{}轮".format(epoch), total=loader_len
        ):
            # print(data)
            sample = data["image"]
            sample = sample.reshape(-1, 784)  # [64,1,28,28]->[64,784]
            batch = sample.shape[0]
            sample = sample.to(device)
            sample_z = torch.normal(0, 1, size=(batch, 128), device=device)

            Dis_true = D(sample)
            true_loss = loss_fn(Dis_true, torch.ones_like(Dis_true))

            fake_sample = G(sample_z)
            Dis_fake = D(fake_sample.detach())
            fake_loss = loss_fn(Dis_fake, torch.zeros_like(Dis_fake))

            Dis_loss = true_loss + fake_loss
            D_optimizer.zero_grad()
            Dis_loss.backward()
            D_optimizer.step()

            Dis_G = D(fake_sample)
            G_loss = loss_fn(Dis_G, torch.ones_like(Dis_G))
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            with torch.no_grad():
                dis_loss_all += Dis_loss
                gen_loss_all += G_loss

        with torch.no_grad():
            dis_loss_all = dis_loss_all / loader_len
            gen_loss_all = gen_loss_all / loader_len
            print(f"判别器损失为{dis_loss_all}")
            print(f"生成器损失为{gen_loss_all}")

        torch.save(G, "./model/G.pth")
        torch.save(D, "./model/D.pth")


if __name__ == "__main__":
    Train()
    model_G = torch.load("./model/G.pth", map_location="cpu")
    fake_z = torch.normal(0, 1, size=(10, 128))
    result = model_G(fake_z).reshape(-1, 28, 28)
    result = result.detach().numpy()

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(result[i])
        plt.gray()
    plt.show()

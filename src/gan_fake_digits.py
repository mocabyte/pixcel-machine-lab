import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.utils.data import Subset

from common import ROOT, DATA_DIR, get_device

OUT_DIR = ROOT / "output" / "gan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = get_device()
print(f"Training on: {device}")

## Wichtig für Workhop
# Hier Batch-Größe einstellen-> nur für alle Ziffern relevant da training sonst zu lange dauern könnte 64 oder 128 wählen
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root=str(DATA_DIR),          # MNIST wird beim ersten Start nach data/MNIST geladen
    train=True,
    download=True,
    transform=transform
)
## Wichtig für Workhop
modus = "bestimmte Ziffer"  # "Alle Ziffern" oder "bestimmte Ziffer"
ziffer = 2                  # nur relevant für "bestimmte Ziffer"

if modus == "bestimmte Ziffer":
    # nur die Bilder einer Ziffer verwenden
    targets = train_dataset.targets            # Tensor mit den Labels
    mask = targets == ziffer                   # Bool-Maske für die gewählte Ziffer
    indices = mask.nonzero(as_tuple=True)[0]   # Indizes dieser Ziffer
    print(f"Trainiere nur auf Ziffer {ziffer}: {len(indices)} Bilder")

    subset_dataset = Subset(train_dataset, indices)

    train_loader = torch.utils.data.DataLoader(
        subset_dataset,          
        batch_size=batch_size,
        shuffle=True
    )
else:
    # alle Ziffern 0–9
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

# ----------------- Netze -----------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )
    def forward(self, z):
        return self.fc(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

G = Generator().to(device)
D = Discriminator().to(device)


## Wichtig für Workhop
criterion = nn.BCELoss()
#Lernraten einstellen
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

#hier einstellen wie viele Trainingsschritte durchgeführt werden sollen
step = 0
#hier einstellen wie viele Epochen trainiert werden sollen
num_epochs = 100
real_images, labels = next(iter(train_loader))  
# real_images: [batch_size, 1, 28, 28]

vutils.save_image(
    real_images,
    OUT_DIR / "real_batch.png",
    nrow=8,
    normalize=True
)
for epoch in range(num_epochs):
    for real_images, _ in train_loader:
        # [B, 1, 28, 28] -> [B, 784]
        real_images = real_images.view(real_images.size(0), -1).to(device)
        batch_size = real_images.size(0)

        # ===== Diskriminator-Update =====
        labels_real = torch.ones(batch_size, 1, device=device)
        labels_fake = torch.zeros(batch_size, 1, device=device)

        # echte Daten
        output_real = D(real_images)
        loss_real = criterion(output_real, labels_real)

        # fake Daten (Graph zu G trennen!)
        z = torch.randn(batch_size, 100, device=device)
        fake_images = G(z).detach()
        output_fake = D(fake_images)
        loss_fake = criterion(output_fake, labels_fake)

        loss_D = loss_real + loss_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ===== Generator-Update =====
        z = torch.randn(batch_size, 100, device=device)
        fake_images = G(z)
        output = D(fake_images)
        labels = torch.ones(batch_size, 1, device=device)  # G will "echt"

        loss_G = criterion(output, labels)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ===== Bilder loggen =====
        if step % 100 == 0:
            G.eval()
            with torch.no_grad():
                z_vis = torch.randn(16, 100, device=device)
                fake_vis = G(z_vis).view(16, 1, 28, 28)
                vutils.save_image(
                    fake_vis,
                    OUT_DIR / f"fake_step_{step:05d}.png",
                    nrow=4,
                    normalize=True,
                )
            G.train()

        step += 1

    print(f"Epoche {epoch + 1}/{num_epochs}  loss_D={loss_D.item():.3f}  loss_G={loss_G.item():.3f}")

# ----------------- Finale Bilder -----------------
G.eval()
with torch.no_grad():
    z = torch.randn(64, 100, device=device)
    fake_images = G(z).view(64, 1, 28, 28)
    vutils.save_image(fake_images, OUT_DIR / "fake_final.png", nrow=8, normalize=True)
print(f"✅ Bilder gespeichert in {OUT_DIR}")

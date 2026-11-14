from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# PARAMETRY - pełne ścieżki do plików
FILE_COUNTS = Path(r"C:\Users\aleks\Desktop\MLphys_2026\wyniki_cnn\confusion_matrix_counts.txt")
FILE_PERCENT = Path(r"C:\Users\aleks\Desktop\MLphys_2026\wyniki_cnn\confusion_matrix_percent.txt")

# WCZYTANIE DANYCH
counts = np.loadtxt(FILE_COUNTS, dtype=int)
percent = np.loadtxt(FILE_PERCENT)

if counts.shape != (10, 10) or percent.shape != (10, 10):
    raise ValueError("Obie macierze muszą mieć wymiary 10x10.")

accuracy = np.trace(counts) / counts.sum() * 100
print(f"Test accuracy: {accuracy:.2f}%")

# WYKRES - tylko macierz procentowa
mask = np.eye(10, dtype=bool)
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("white")

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle(f"CNN / MNIST - Test accuracy: {accuracy:.2f}%", fontsize=16)

colored = np.ma.array(percent, mask=mask)
vmax = max(float(percent[~mask].max()), 1.0)
im = ax.imshow(colored, cmap=cmap, vmin=0, vmax=vmax)

ax.set_title("Confusion matrix - %", fontsize=13)
ax.set_xlabel("Predicted digit")
ax.set_ylabel("True digit")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_aspect("equal")

# Wartości liczbowe w komórkach
for i in range(10):
    for j in range(10):
        value = percent[i, j]
        label = f"{value:.1f}"
        color = "white" if i != j and value > 0.55 * vmax else "black"
        ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

fig.colorbar(im, ax=ax, label="Errors [% of true class]", shrink=0.82, pad=0.02)

plt.tight_layout()
plt.show()
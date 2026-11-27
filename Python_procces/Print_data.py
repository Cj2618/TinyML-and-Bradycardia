import numpy as np
import matplotlib.pyplot as plt

datos = np.load("infant10_ecg_ventanas.npz")  # pon el nombre real de tu archivo
X, y = datos["X"], datos["y"]

print("X:", X.shape, "| y:", y.shape)
print("Pre-bradicardia:", (y == 1).sum(), "| Normales:", (y == 0).sum())

# Una ventana de cada clase, para verlas con tus ojos
fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ax[0].plot(X[y == 1][0]); ax[0].set_title("Ejemplo pre-bradicardia")
ax[1].plot(X[y == 0][0]); ax[1].set_title("Ejemplo normal")
plt.tight_layout(); plt.show()
import torch

print(f"Version de torch : {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
    print(f"Nombre de GPU : {torch.cuda.device_count()}")
else:
    print("Aucun GPU CUDA détecté — inférence sur CPU uniquement.")
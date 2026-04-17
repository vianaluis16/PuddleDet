import torch
import time
from rau_fcn.model import FCN8sRAU

print("Carregando modelo baseline...")
model_baseline = FCN8sRAU(num_classes=2, use_rau=False, head_dim=256).eval()
x = torch.randn(1, 3, 180, 320)

print("Medindo baseline...")
with torch.no_grad():
    print("Warmup...")
    _ = model_baseline(x)
    
    print("Medindo 50 inferências...")
    start = time.time()
    for _ in range(50):
        _ = model_baseline(x)
    elapsed = time.time() - start
    
    tempo_medio = elapsed / 50
    fps = 1.0 / tempo_medio
    print(f'\nBaseline FCN-8s (180x320):')
    print(f'  Tempo médio: {tempo_medio*1000:.1f} ms/imagem')
    print(f'  FPS:         {fps:.2f}')

import cv2
import numpy as np
from pathlib import Path

# Pastas
baseline_dir = Path('runs/rau_fcn/puddle1000_baseline/vis_val_off')
rau_dir = Path('runs/rau_fcn/puddle1000_rau_light/vis_val_off')
output_dir = Path('runs/rau_fcn/comparison_baseline_vs_rau')

print(f'Gerando comparações verticais Baseline (topo) vs RAU (baixo)...')
print(f'Output: {output_dir}')

# Para cada imagem, criar comparação vertical
for i in range(12):
    fname = f'vis_{i:04d}.png'
    
    baseline_img = cv2.imread(str(baseline_dir / fname))
    rau_img = cv2.imread(str(rau_dir / fname))
    
    if baseline_img is None or rau_img is None:
        print(f'  Pulando {fname} (arquivo ausente)')
        continue
    
    # Adicionar labels
    h, w = baseline_img.shape[:2]
    label_h = 50
    
    # Label baseline
    label_baseline = np.ones((label_h, w, 3), dtype=np.uint8) * 240
    cv2.putText(label_baseline, 'BASELINE FCN-8s', (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Label RAU
    label_rau = np.ones((label_h, w, 3), dtype=np.uint8) * 240
    cv2.putText(label_rau, 'FCN-8s + RAU Light', (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 0, 0), 2)
    
    # Combinar verticalmente: label + imagem
    baseline_with_label = np.vstack([label_baseline, baseline_img])
    rau_with_label = np.vstack([label_rau, rau_img])
    
    # Separador horizontal
    separator = np.ones((8, w, 3), dtype=np.uint8) * 200
    
    # Combinação final vertical: baseline (topo) + separador + RAU (baixo)
    comparison = np.vstack([baseline_with_label, separator, rau_with_label])
    
    # Salvar
    output_path = output_dir / f'comparison_{i:04d}.png'
    cv2.imwrite(str(output_path), comparison)
    print(f'  Criado: {output_path.name}')

print(f'\n12 comparações verticais salvas em: {output_dir}')

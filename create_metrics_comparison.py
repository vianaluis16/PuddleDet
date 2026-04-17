import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Dados
baseline = {
    'IoU': 0.6628,
    'F1': 0.7972,
    'Precision': 0.7850,
    'Recall': 0.8097,
    'Pixel Acc': 0.9930
}

rau = {
    'IoU': 0.6816,
    'F1': 0.8107,
    'Precision': 0.7800,
    'Recall': 0.8438,
    'Pixel Acc': 0.9933
}

# Calcular ganhos
gains = {k: ((rau[k] - baseline[k]) / baseline[k]) * 100 for k in baseline.keys()}

# Criar figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Comparação: Baseline FCN-8s vs FCN-8s + RAU Light\nPuddle-1000 val_off (180320)', 
             fontsize=16, fontweight='bold')

# Gráfico 1: Barras comparativas
metrics = list(baseline.keys())
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, [baseline[m] for m in metrics], width, 
                label='Baseline', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, [rau[m] for m in metrics], width, 
                label='RAU Light', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Métricas Comparativas', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0.7, 1.0)

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Gráfico 2: Ganhos percentuais
colors = ['#27ae60' if g > 0 else '#e67e22' for g in gains.values()]
bars3 = ax2.barh(metrics, list(gains.values()), color=colors, alpha=0.8)

ax2.set_xlabel('Ganho (%)', fontsize=12, fontweight='bold')
ax2.set_title('Ganho do RAU sobre Baseline', fontsize=13, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Adicionar valores nas barras horizontais com offset absoluto em pontos de tela
# Isso evita sobreposição com o eixo Y independente da escala dos dados
for bar, gain in zip(bars3, gains.values()):
    y_center = bar.get_y() + bar.get_height() / 2.
    if gain >= 0:
        xy = (gain, y_center)
        xytext = (6, 0)   # 6 pontos para a direita do fim da barra
        ha = 'left'
    else:
        xy = (gain, y_center)
        xytext = (-6, 0)  # 6 pontos para a esquerda do fim da barra
        ha = 'right'
    ax2.annotate(f'{gain:+.2f}%',
                 xy=xy,
                 xytext=xytext,
                 textcoords='offset points',
                 va='center', ha=ha,
                 fontsize=10, fontweight='bold')

# Margem extra no eixo X para que os rótulos não colidam com as bordas
ax2.margins(x=0.25)

plt.tight_layout()

# Salvar
output_path = Path('runs/rau_fcn/metrics_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Tabela comparativa salva em: {output_path}')

# Criar também uma tabela textual
fig2, ax = plt.subplots(figsize=(12, 5))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Métrica', 'Baseline', 'RAU Light', 'Ganho Absoluto', 'Ganho (%)'],
    ['' * 15] * 5
]

for metric in metrics:
    b_val = baseline[metric]
    r_val = rau[metric]
    gain_abs = r_val - b_val
    gain_pct = gains[metric]
    
    gain_abs_str = f'{gain_abs:+.4f}'
    gain_pct_str = f'{gain_pct:+.2f}%'
    
    if gain_pct > 0:
        gain_abs_str = f' {gain_abs_str}'
        gain_pct_str = f' {gain_pct_str}'
    
    table_data.append([
        metric,
        f'{b_val:.4f}',
        f'{r_val:.4f}',
        gain_abs_str,
        gain_pct_str
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.2, 0.15, 0.15, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Estilizar cabeçalho
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Estilizar linhas
for i in range(2, len(table_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

plt.title('Comparação Detalhada: Baseline FCN-8s vs FCN-8s + RAU Light\nPuddle-1000 val_off', 
         fontsize=14, fontweight='bold', pad=20)

output_path2 = Path('runs/rau_fcn/metrics_table.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f'Tabela detalhada salva em: {output_path2}')

print('\nResumo:')
print(f'  IoU:       {baseline["IoU"]:.4f}  {rau["IoU"]:.4f} ({gains["IoU"]:+.2f}%)')
print(f'  F1:        {baseline["F1"]:.4f}  {rau["F1"]:.4f} ({gains["F1"]:+.2f}%)')
print(f'  Precision: {baseline["Precision"]:.4f}  {rau["Precision"]:.4f} ({gains["Precision"]:+.2f}%)')
print(f'  Recall:    {baseline["Recall"]:.4f}  {rau["Recall"]:.4f} ({gains["Recall"]:+.2f}%)')

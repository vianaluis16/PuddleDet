"""
Fine-tuning do modelo YOLO com anotações customizadas.

Este script faz fine-tuning do modelo best.pt usando as anotações
feitas no Roboflow com as imagens reais de poças do ambiente.
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

# Configurações
MODEL_BASE = "best.pt"  # Modelo pré-treinado (Roboflow Puddle)
DATASET_PATH = Path("Find-puddles-2-1/data.yaml")
EPOCHS = 50  # Mais épocas para dataset pequeno
BATCH_SIZE = 4  # Batch pequeno (poucas imagens)
IMG_SIZE = 640
PATIENCE = 20  # Early stopping

def main():
    print("=" * 60)
    print("FINE-TUNING DO MODELO DE DETECÇÃO DE POÇAS")
    print("=" * 60)
    
    # Verificar se dataset existe
    if not DATASET_PATH.exists():
        print(f"❌ Dataset não encontrado: {DATASET_PATH}")
        return
    
    # Verificar se modelo base existe
    if not Path(MODEL_BASE).exists():
        print(f"❌ Modelo base não encontrado: {MODEL_BASE}")
        return
    
    print(f"\n📦 Modelo base: {MODEL_BASE}")
    print(f"📊 Dataset: {DATASET_PATH}")
    print(f"🔄 Épocas: {EPOCHS}")
    print(f"📐 Batch size: {BATCH_SIZE}")
    print(f"🖼️ Image size: {IMG_SIZE}")
    
    # Carregar modelo
    print("\n🔧 Carregando modelo base...")
    model = YOLO(MODEL_BASE)
    
    # Fine-tuning
    print("\n🚀 Iniciando fine-tuning...")
    print("-" * 60)
    
    results = model.train(
        data=str(DATASET_PATH.absolute()),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        # Fine-tuning settings
        freeze=10,  # Congela primeiras 10 camadas (backbone)
        lr0=0.001,  # Learning rate menor para fine-tuning
        lrf=0.01,   # LR final
        # Augmentation (importante para dataset pequeno)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Outros
        verbose=True,
        project="runs/fine_tune",
        name="puddle_custom",
        exist_ok=True,
    )
    
    print("\n" + "=" * 60)
    print("✅ FINE-TUNING CONCLUÍDO!")
    print("=" * 60)
    
    # Copiar melhor modelo
    best_model_path = Path("runs/fine_tune/puddle_custom/weights/best.pt")
    if best_model_path.exists():
        output_path = Path("best_fine_tuned.pt")
        shutil.copy(best_model_path, output_path)
        print(f"\n📁 Modelo salvo em: {output_path}")
        print("\nPara usar o novo modelo, altere em rodar_deteccao.py:")
        print('   MODEL_PATH = "best_fine_tuned.pt"')
    else:
        print(f"\n⚠️ Modelo best.pt não encontrado em {best_model_path}")
        print("Verifique a pasta runs/fine_tune/puddle_custom/weights/")


if __name__ == "__main__":
    main()

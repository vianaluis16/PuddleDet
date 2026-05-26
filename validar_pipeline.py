"""
validar_pipeline.py — Validação completa do pipeline PuddleDet

Executa:
1. Verificação de modelos YOLO (best.pt e best_fine_tuned.pt)
2. Inferência YOLO no dataset Puddle-1000 val
3. Inferência YOLO no dataset-puddle-5777 val
4. Avaliação do modelo RAU-FCN (segmentação)
5. Relatório de métricas consolidado
"""

import os
import sys
import json
import time
from pathlib import Path

# Adicionar o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))


def verificar_dependencias():
    """Verifica se todas as dependências estão instaladas."""
    print("=" * 70)
    print("  VALIDAÇÃO DO PIPELINE PUDDLEDET")
    print("=" * 70)
    print("\n[1/6] Verificando dependências...")

    deps = {}
    try:
        import ultralytics
        deps["ultralytics"] = ultralytics.__version__
    except ImportError:
        deps["ultralytics"] = "NÃO INSTALADO"

    try:
        import torch
        deps["torch"] = torch.__version__
        deps["cuda"] = str(torch.cuda.is_available())
    except ImportError:
        deps["torch"] = "NÃO INSTALADO"
        deps["cuda"] = "N/A"

    try:
        import cv2
        deps["cv2"] = cv2.__version__
    except ImportError:
        deps["cv2"] = "NÃO INSTALADO"

    try:
        import numpy
        deps["numpy"] = numpy.__version__
    except ImportError:
        deps["numpy"] = "NÃO INSTALADO"

    for k, v in deps.items():
        status = "✓" if v != "NÃO INSTALADO" and v != "N/A" else "✗"
        print(f"  {status} {k}: {v}")

    return all(v != "NÃO INSTALADO" for k, v in deps.items() if k != "cuda")


def verificar_modelos():
    """Verifica se os modelos existem e carrega informações."""
    print("\n[2/6] Verificando modelos...")

    modelos_info = {}

    for nome in ["best.pt", "best_fine_tuned.pt"]:
        if os.path.exists(nome):
            tamanho = os.path.getsize(nome) / (1024 * 1024)
            print(f"  ✓ {nome}: {tamanho:.1f} MB")
            modelos_info[nome] = {"tamanho_mb": tamanho, "existe": True}
        else:
            print(f"  ✗ {nome}: NÃO ENCONTRADO")
            modelos_info[nome] = {"existe": False}

    # Verificar modelos RAU-FCN
    rau_models = [
        "runs/rau_fcn/puddle1000_rau_light/best.pt",
        "runs/rau_fcn/puddle1000_baseline/best.pt",
    ]
    for nome in rau_models:
        if os.path.exists(nome):
            tamanho = os.path.getsize(nome) / (1024 * 1024)
            print(f"  ✓ {nome}: {tamanho:.1f} MB")
            modelos_info[nome] = {"tamanho_mb": tamanho, "existe": True}
        else:
            print(f"  ✗ {nome}: NÃO ENCONTRADO")
            modelos_info[nome] = {"existe": False}

    return modelos_info


def testar_yolo_inferencia(modelo_path, imagens, nome_teste):
    """Testa inferência YOLO num conjunto de imagens."""
    from ultralytics import YOLO
    import cv2

    print(f"\n  Testando {nome_teste} com {modelo_path}...")
    print(f"  Imagens a processar: {len(imagens)}")

    if not os.path.exists(modelo_path):
        print(f"  ✗ Modelo não encontrado: {modelo_path}")
        return None

    try:
        modelo = YOLO(modelo_path)
        print(f"  ✓ Modelo carregado com sucesso")
        # Informações do modelo
        print(f"    - Tarefa: {modelo.task}")
        print(f"    - Nomes das classes: {modelo.names}")
    except Exception as e:
        print(f"  ✗ Erro ao carregar modelo: {e}")
        return None

    resultados = {
        "total_imagens": 0,
        "imagens_com_deteccao": 0,
        "total_deteccoes": 0,
        "confidencias": [],
        "tempos_inferencia": [],
        "erro_count": 0,
    }

    for img_path in imagens:
        try:
            t0 = time.time()
            results = modelo.predict(source=str(img_path), conf=0.15, verbose=False)
            t1 = time.time()

            resultados["tempos_inferencia"].append(t1 - t0)
            resultados["total_imagens"] += 1

            for r in results:
                n_det = len(r.boxes)
                resultados["total_deteccoes"] += n_det
                if n_det > 0:
                    resultados["imagens_com_deteccao"] += 1
                    for box in r.boxes:
                        resultados["confidencias"].append(box.conf.item())

        except Exception as e:
            resultados["erro_count"] += 1

    # Estatísticas
    if resultados["tempos_inferencia"]:
        avg_time = sum(resultados["tempos_inferencia"]) / len(resultados["tempos_inferencia"])
        resultados["tempo_medio_ms"] = avg_time * 1000
    if resultados["confidencias"]:
        resultados["conf_media"] = sum(resultados["confidencias"]) / len(resultados["confidencias"])
        resultados["conf_min"] = min(resultados["confidencias"])
        resultados["conf_max"] = max(resultados["confidencias"])

    taxa = (resultados["imagens_com_deteccao"] / max(1, resultados["total_imagens"])) * 100

    print(f"  Resultados {nome_teste}:")
    print(f"    - Imagens processadas: {resultados['total_imagens']}")
    print(f"    - Imagens com detecção: {resultados['imagens_com_deteccao']} ({taxa:.1f}%)")
    print(f"    - Total detecções: {resultados['total_deteccoes']}")
    if resultados.get("conf_media"):
        print(f"    - Confiança média: {resultados['conf_media']:.3f} (min={resultados['conf_min']:.3f}, max={resultados['conf_max']:.3f})")
    if resultados.get("tempo_medio_ms"):
        print(f"    - Tempo médio/imagem: {resultados['tempo_medio_ms']:.0f}ms")
    if resultados["erro_count"] > 0:
        print(f"    - Erros: {resultados['erro_count']}")

    return resultados


def testar_yolo_datasets():
    """Testa inferência YOLO em todos os datasets disponíveis."""
    print("\n[3/6] Testando inferência YOLO nos datasets...")

    todos_resultados = {}

    # --- Dataset Puddle-1000 (imagens de validação) ---
    puddle1000_val = Path("Puddle-1000_Dataset2/Puddle-1000 Dataset_val/images")
    if puddle1000_val.exists():
        imagens = sorted(list(puddle1000_val.glob("*.png")))[:30]  # Limitar a 30 para velocidade
        for modelo in ["best.pt", "best_fine_tuned.pt"]:
            if os.path.exists(modelo):
                nome = f"Puddle-1000 val ({modelo})"
                r = testar_yolo_inferencia(modelo, imagens, nome)
                if r:
                    todos_resultados[nome] = r
    else:
        print("  ⚠ Dataset Puddle-1000 val não encontrado")

    # --- Dataset-puddle-5777 (imagens de validação) ---
    puddle5777_val = Path("dataset-puddle-5777/valid/images")
    if puddle5777_val.exists():
        imagens = sorted(list(puddle5777_val.glob("*")))[:30]
        for modelo in ["best.pt", "best_fine_tuned.pt"]:
            if os.path.exists(modelo):
                nome = f"Puddle-5777 val ({modelo})"
                r = testar_yolo_inferencia(modelo, imagens, nome)
                if r:
                    todos_resultados[nome] = r
    else:
        print("  ⚠ Dataset puddle-5777 não encontrado")

    # --- Dataset Find-puddles-2-1 (test) ---
    find_test = Path("Find-puddles-2-1/test/images")
    if find_test.exists():
        imagens = sorted(list(find_test.glob("*")))
        for modelo in ["best.pt", "best_fine_tuned.pt"]:
            if os.path.exists(modelo):
                nome = f"Find-puddles test ({modelo})"
                r = testar_yolo_inferencia(modelo, imagens, nome)
                if r:
                    todos_resultados[nome] = r

    return todos_resultados


def testar_rau_fcn():
    """Testa o modelo RAU-FCN."""
    print("\n[4/6] Testando modelo RAU-FCN...")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resultados_rau = {}

    checkpoints = {
        "RAU Light": "runs/rau_fcn/puddle1000_rau_light/best.pt",
        "Baseline": "runs/rau_fcn/puddle1000_baseline/best.pt",
    }

    for nome, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Checkpoint não encontrado: {ckpt_path}")
            continue

        try:
            from rau_fcn.model import FCN8sRAU
            from rau_fcn.dataset import Puddle1000SegDataset
            from rau_fcn.metrics import segmentation_scores, combined_loss
            from torch.utils.data import DataLoader

            print(f"\n  Carregando {nome}: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            ckpt_args = ckpt.get("args", {})

            use_rau = ckpt_args.get("mode", "rau") == "rau"
            rau_mode = ckpt_args.get("rau_mode", "light")
            head_dim = ckpt_args.get("head_dim", 512)
            dropout = ckpt_args.get("dropout", 0.15)

            model = FCN8sRAU(
                num_classes=2,
                use_rau=use_rau,
                rau_mode=rau_mode,
                head_dim=head_dim,
                dropout=dropout,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            n_params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Modelo carregado: {n_params:,} parâmetros")
            print(f"    - Epoch: {ckpt.get('epoch', '?')}")
            print(f"    - Best IoU (treino): {ckpt.get('best_iou', '?')}")

            # Testar no val
            for split_name in ["val", "val_off"]:
                try:
                    ds = Puddle1000SegDataset(
                        dataset_root=Path("Puddle-1000_Dataset2"),
                        split=split_name,
                        image_size=(360, 640),
                        augment=False,
                        max_samples=20,  # Limitar para velocidade
                    )
                    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

                    keys = ("precision", "recall", "f1", "iou", "pixel_acc")
                    sums = {k: 0.0 for k in keys}
                    n = 0

                    t0 = time.time()
                    with torch.no_grad():
                        for images, masks in loader:
                            images, masks = images.to(device), masks.to(device)
                            logits = model(images)
                            scores = segmentation_scores(logits, masks)
                            for k in keys:
                                sums[k] += scores[k]
                            n += 1
                    t1 = time.time()

                    avg = {k: sums[k] / n for k in keys}
                    avg["tempo_total_s"] = t1 - t0
                    avg["n_batches"] = n

                    print(f"\n    {nome} — {split_name} ({len(ds)} amostras, {n} batches)")
                    print(f"      IoU:       {avg['iou']:.4f}")
                    print(f"      F1:        {avg['f1']:.4f}")
                    print(f"      Precision: {avg['precision']:.4f}")
                    print(f"      Recall:    {avg['recall']:.4f}")
                    print(f"      Pixel Acc: {avg['pixel_acc']:.4f}")
                    print(f"      Tempo:     {avg['tempo_total_s']:.1f}s")

                    resultados_rau[f"{nome}_{split_name}"] = avg

                except Exception as e:
                    print(f"    ⚠ Erro no split {split_name}: {e}")

        except Exception as e:
            print(f"  ✗ Erro ao carregar {nome}: {e}")
            import traceback
            traceback.print_exc()

    return resultados_rau


def verificar_datasets():
    """Verifica a integridade dos datasets."""
    print("\n[5/6] Verificando integridade dos datasets...")

    datasets = {
        "Puddle-1000 train": "Puddle-1000_Dataset2/Puddle-1000 Dataset_train",
        "Puddle-1000 val": "Puddle-1000_Dataset2/Puddle-1000 Dataset_val",
        "Puddle-1000 train_on": "Puddle-1000_Dataset2/Puddle-1000 Dataset_train_on",
        "Puddle-1000 val_off": "Puddle-1000_Dataset2/Puddle-1000 Dataset_val_off",
        "Find-puddles-2-1 train": "Find-puddles-2-1/train",
        "dataset-puddle-5777 train": "dataset-puddle-5777/train",
    }

    for nome, caminho in datasets.items():
        p = Path(caminho)
        if not p.exists():
            print(f"  ✗ {nome}: NÃO ENCONTRADO")
            continue

        img_dir = p / "images"
        if img_dir.exists():
            imgs = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
            n_imgs = len(imgs)
        else:
            n_imgs = 0

        # Verificar se há labels/masks
        mask_dir = p / "masks"
        label_dir = p / "labels"

        info = f"  ✓ {nome}: {n_imgs} imagens"
        if mask_dir.exists():
            masks = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg"))
            # Também procurar em subpasta 0/
            mask_0 = mask_dir / "0"
            if mask_0.exists():
                masks += list(mask_0.glob("*.png"))
            info += f", {len(masks)} máscaras"
        if label_dir.exists():
            labels = list(label_dir.glob("*.txt"))
            info += f", {len(labels)} labels"

        print(info)


def gerar_relatorio(modelos_info, yolo_results, rau_results):
    """Gera relatório final consolidado."""
    print("\n[6/6] Gerando relatório final...")
    print("\n" + "=" * 70)
    print("  RELATÓRIO DE VALIDAÇÃO — PUDDLEDET")
    print("=" * 70)

    # Resumo dos modelos
    print("\n  📦 MODELOS DISPONÍVEIS:")
    for nome, info in modelos_info.items():
        if info.get("existe"):
            print(f"    ✓ {nome} ({info['tamanho_mb']:.1f} MB)")
        else:
            print(f"    ✗ {nome}")

    # Resultados YOLO
    if yolo_results:
        print("\n  🔍 DETECÇÃO YOLO (object detection):")
        for nome, r in yolo_results.items():
            taxa = (r["imagens_com_deteccao"] / max(1, r["total_imagens"])) * 100
            conf = r.get("conf_media", 0)
            print(f"    {nome}:")
            print(f"      Detecções: {r['total_deteccoes']} em {r['total_imagens']} imgs ({taxa:.1f}% com detecção)")
            if conf:
                print(f"      Confiança: {conf:.3f} (avg)")

    # Resultados RAU-FCN
    if rau_results:
        print("\n  🎯 SEGMENTAÇÃO FCN-8s + RAU:")
        for nome, r in rau_results.items():
            print(f"    {nome}:")
            print(f"      IoU={r['iou']:.4f}  F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

    # Diagnóstico
    print("\n  📊 DIAGNÓSTICO:")
    print("  " + "-" * 55)

    # Analisar viés do dataset YOLO
    if yolo_results:
        # Comparar performance em datasets diferentes
        puddle1000_results = {k: v for k, v in yolo_results.items() if "Puddle-1000" in k}
        puddle5777_results = {k: v for k, v in yolo_results.items() if "Puddle-5777" in k}
        find_results = {k: v for k, v in yolo_results.items() if "Find-puddles" in k}

        if puddle5777_results:
            for nome, r in puddle5777_results.items():
                taxa = (r["imagens_com_deteccao"] / max(1, r["total_imagens"])) * 100
                if taxa < 20:
                    print(f"  ⚠ ALERTA: Taxa de detecção MUITO BAIXA ({taxa:.1f}%) em {nome}")
                    print(f"    → O modelo pode estar enviesado pelo dataset de treino")
                elif taxa < 50:
                    print(f"  ⚠ Taxa de detecção moderada ({taxa:.1f}%) em {nome}")
                    print(f"    → Pode indicar diferenças de domínio entre datasets")
                else:
                    print(f"  ✓ Taxa de detecção boa ({taxa:.1f}%) em {nome}")

        if puddle1000_results and puddle5777_results:
            for p1_nome, p1_r in puddle1000_results.items():
                for p5_nome, p5_r in puddle5777_results.items():
                    if "best_fine_tuned" in p1_nome and "best_fine_tuned" in p5_nome:
                        taxa1 = (p1_r["imagens_com_deteccao"] / max(1, p1_r["total_imagens"])) * 100
                        taxa5 = (p5_r["imagens_com_deteccao"] / max(1, p5_r["total_imagens"])) * 100
                        diff = abs(taxa1 - taxa5)
                        if diff > 30:
                            print(f"  ⚠ GRANDE DIFERENÇA entre datasets: Puddle-1000={taxa1:.1f}% vs Puddle-5777={taxa5:.1f}%")
                            print(f"    → Sugere viés/overfitting ao dataset de treino")

    print("\n  💡 PRÓXIMOS PASSOS:")
    print("    1. Trazer imagens NOVAS (fora de todos os datasets) para teste real")
    print("    2. Testar com fotos tiradas com celular/câmera em ambiente urbano")
    print("    3. Verificar se o modelo generaliza para diferentes:")
    print("       - Condições de iluminação")
    print("       - Tipos de pavimento")
    print("       - Ângulos de câmera")
    print("       - Tamanhos de poça")

    print("\n" + "=" * 70)
    print("  VALIDAÇÃO CONCLUÍDA")
    print("=" * 70)

    return {
        "modelos": modelos_info,
        "yolo": {k: {kk: vv for kk, vv in v.items() if kk != "confidencias" and kk != "tempos_inferencia"} for k, v in yolo_results.items()} if yolo_results else {},
        "rau_fcn": rau_results or {},
    }


def main():
    # 1. Dependências
    if not verificar_dependencias():
        print("\n✗ Dependências faltando. Instale com: pip install -r requirements.txt")
        return

    # 2. Modelos
    modelos_info = verificar_modelos()

    # 3. YOLO
    yolo_results = testar_yolo_datasets()

    # 4. RAU-FCN
    rau_results = testar_rau_fcn()

    # 5. Datasets
    verificar_datasets()

    # 6. Relatório
    relatorio = gerar_relatorio(modelos_info, yolo_results, rau_results)

    # Salvar relatório JSON
    output_path = Path("validacao_pipeline.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, default=str)
    print(f"\nRelatório salvo em: {output_path}")


if __name__ == "__main__":
    main()

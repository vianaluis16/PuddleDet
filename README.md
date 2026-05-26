# PuddleDet: Detecção Automatizada de Poças d'Água Utilizando Inteligência Artificial

![Badge de Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Badge de Linguagem](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Badge de Licença](https://img.shields.io/badge/license-MIT-green)

O **PuddleDet** é um projeto de iniciação científica da Universidade Federal do Espírito Santo (UFES), inserido no projeto de pesquisa "Aplicações de Inteligência Artificial em Robótica" (PRPPG 13356/2024).

## 📖 Visão Geral

O projeto visa desenvolver um sistema automatizado para a **detecção e georreferenciamento de poças d'água** em ambientes urbanos, utilizando técnicas de inteligência artificial e visão computacional.

Atualmente, a **segmentação semântica é a única fonte de detecção** adotada no PuddleDet. O pipeline principal utiliza um modelo RAU-FCN treinado para identificar pixels de água/poça e gerar máscaras sobrepostas nas imagens.

## 🎯 Problema e Motivação

A presença de poças d'água representa um desafio significativo para a segurança viária, especialmente para nossos veículos autônomos. Os principais riscos incluem:
* **Segurança Veicular:** Risco de aquaplanagem, perda de controle e danos a componentes eletrônicos.
* **Navegação Autônoma:** Sensores ópticos e LIDAR podem ter sua interpretação do ambiente afetada, exigindo a identificação precisa de obstáculos para garantir decisões seguras.
* **Gestão Urbana:** O monitoramento tradicional é ineficiente. Dados automatizados podem apoiar o planejamento urbano, a manutenção de sistemas de drenagem e o direcionamento eficiente de recursos públicos.

## 🏁 Objetivos

### Objetivo Geral
Desenvolver e validar um sistema automatizado para detecção e georreferenciamento de poças d'água em ambientes urbanos, utilizando técnicas de inteligência artificial e visão computacional.

### Objetivos Específicos
* **Analisar bases de dados** públicas com imagens urbanas e metadados georreferenciados.
* **Pré-processar e armazenar** os dados para o treinamento de modelos de segmentação.
* **Implementar e treinar** modelos de segmentação semântica para a detecção de poças.
* **Desenvolver um módulo de georreferenciamento** para associar detecções a coordenadas geográficas.
* **Validar o sistema** em cenários simulados e, se possível, com dados reais.

## 🛠️ Metodologia

O desenvolvimento do sistema seguirá as seguintes etapas:

1.  **Levantamento de Dados:** Análise crítica de bases de dados públicas como **BDD100K**, **Mapillary Vistas** e **Cityscapes**.
2.  **Pré-processamento:** Adaptação, padronização e anotação das imagens e metadados para garantir a qualidade dos dados de treinamento.
3.  **Treinamento de Modelos:** Implementação e ajuste de arquitetura de segmentação semântica (**RAU-FCN**).
4.  **Avaliação:** Análise de desempenho dos modelos.
5.  **Georreferenciamento:** Desenvolvimento de um módulo para integrar as detecções com as coordenadas de GPS, permitindo a criação de mapas de risco.
6.  **Validação:** Testes em cenários simulados e em campo, utilizando hardware disponível (câmeras, módulos GPS e suportes impressos em 3D).

## 🚀 Como Usar este Repositório

O fluxo principal de inferência neste repositório é a segmentação semântica com o script `testar_segmentacao_novas.py`.

### Passo 1: Pré-requisitos

Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

### Passo 2: Preparar as imagens de entrada

Coloque as imagens a serem avaliadas na pasta:

```text
imagensNovas/
```

Extensões suportadas pelo script: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

### Passo 3: Executar a segmentação

Execute:

```bash
python testar_segmentacao_novas.py
```

O script utiliza por padrão o modelo:

```text
runs/rau_fcn/puddle1000_rau_light/best.pt
```

### Passo 4: Ver resultados

As imagens segmentadas (overlay com máscara da classe poça) são salvas em:

```text
resultados_novos/segmentacao/
```

Ao final da execução, o script imprime:
* quantidade de imagens processadas;
* quantidade de imagens com detecção de poça;
* diretório de saída dos resultados.

## 📚 Referências

*Este projeto se baseia em pesquisas na área de segmentação semântica, incluindo FCN/U-Net e variações com módulos de atenção para segmentação de cenas urbanas.*
*Ir adicionando conforme avançarmos.*

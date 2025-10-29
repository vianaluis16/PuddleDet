# PuddleDet: Detecção Automatizada de Poças d'Água Utilizando Inteligência Artificial

![Badge de Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Badge de Linguagem](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Badge de Licença](https://img.shields.io/badge/license-MIT-green)

O **PuddleDet** é um projeto de iniciação científica da Universidade Federal do Espírito Santo (UFES), inserido no projeto de pesquisa "Aplicações de Inteligência Artificial em Robótica" (PRPPG 13356/2024).

## 📖 Visão Geral

O projeto visa desenvolver um sistema automatizado para a **detecção e georreferenciamento de poças d'água** em ambientes urbanos, utilizando técnicas avançadas de inteligência artificial e visão computacional. A solução proposta associa a detecção visual de lâminas d'água à sua localização geográfica precisa, permitindo além da reação em tempo real dos veículos, também o planejamento de rotas seguras e a criação de mapas colaborativos de condições viárias a fim de identificar falhas nas vias urbanas.

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
* **Pré-processar e armazenar** os dados para o treinamento de modelos de IA.
* **Implementar e treinar** modelos de Redes Neurais Convolucionais (CNNs) para a detecção.
* **Desenvolver um módulo de georreferenciamento** para associar detecções a coordenadas geográficas.
* **Validar o sistema** em cenários simulados e, se possível, com dados reais.

## 🛠️ Metodologia

O desenvolvimento do sistema seguirá as seguintes etapas:

1.  **Levantamento de Dados:** Análise crítica de bases de dados públicas como **BDD100K**, **Mapillary Vistas** e **Cityscapes**.
2.  **Pré-processamento:** Adaptação, padronização e anotação das imagens e metadados para garantir a qualidade dos dados de treinamento.
3.  **Treinamento de Modelos:** Implementação de arquiteturas de CNN como **YOLOv8** ou **U-Net**.
4.  **Avaliação:** Análise de desempenho dos modelos.
5.  **Georreferenciamento:** Desenvolvimento de um módulo para integrar as detecções com as coordenadas de GPS, permitindo a criação de mapas de risco.
6.  **Validação:** Testes em cenários simulados e em campo, utilizando hardware disponível (câmeras, módulos GPS e suportes impressos em 3D).

## 🚀 Como Usar este Repositório

*Este repositório contém dois scripts principais que devem ser executados via terminal: `FiltroLog.py` (para preparar os dados) e `visualizador.py` (para analisar os dados).*

### Passo 1: Pré-requisitos

Os scripts dependem das bibliotecas `OpenCV` e `NumPy`. Instale-as usando pip:
```bash
pip install opencv-python numpy
```
### Passo 2: Preparação (Gerar o Log Filtrado)
O script FiltroLog.py lê o seu arquivo de log bruto original e cria o log_filtrado.txt, que contém os caminhos das imagens (CAMERA2, CAMERA3) e os dados de GPS (NMEAGGA).

Como executar: No seu terminal, execute o script passando o caminho do seu log bruto como argumento.

Exemplo:

```bash

python FiltroLog.py "C:\Caminho\Para\Seu\audit_20250701_4.txt"
```

Isso irá criar o arquivo log_filtrado.txt dentro da pasta do seu projeto.

### Passo 3: Visualização (Analisar as Imagens)
O script visualizador.py lê o log_filtrado.txt gerado no passo anterior e exibe as imagens de uma câmera específica, buscando-as na pasta de dados descompactada.

#### Como executar: 
Execute o script passando dois argumentos:

O caminho para a pasta de dados descompactada (ex: audit_.../).

A câmera que você deseja ver (CAMERA2 ou CAMERA3).

Exemplo para ver a CAMERA2:

```bash

python visualizador.py "C:\Caminho\Para\Sua\audit_20250701_4" "CAMERA2"
```
#### Controles do Visualizador:

Qualquer Tecla avança para a próxima imagem.

ESC encerra o visualizador.

## 📚 Referências

*Este projeto se baseia em diversas pesquisas recentes na área, como: U-Net (Ronneberger et al., 2015), Mask R-CNN (He et al., 2017), e YOLOv8 (Varghese & M., 2024)*
*Ir adicionando conforme avançarmos*

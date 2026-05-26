import os
import logging
import matplotlib.pyplot as plt
import osmnx as ox

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gerar_mapa_poca(lat: float, lon: float, caminho_saida: str, dist_metros: int = 500) -> bool:
    """
    Gera e salva um mapa da malha viária local com um marcador na localização exata da poça.
    
    A coordenada de GPS fornecida será projetada no nó mais próximo do grafo viário 
    (rede de ruas transitáveis por veículos). Utiliza blocos try/except para garantir
    a continuidade do processo em caso de falha na API ou coordenadas inválidas.
    
    Args:
        lat (float): Latitude da poça detectada.
        lon (float): Longitude da poça detectada.
        caminho_saida (str): Caminho completo onde a imagem (.png) será salva.
        dist_metros (int, opcional): Raio em metros a partir do ponto central para capturar o grafo viário.
        
    Returns:
        bool: True se o mapa foi gerado e salvo com sucesso, False caso ocorra algum erro.
    """
    try:
        logging.info(f"Baixando malha viária (drive) ao redor de ({lat}, {lon}) com raio de {dist_metros}m...")
        # 1. Download do grafo viário (apenas ruas transitáveis por veículos) ao redor do ponto
        G = ox.graph_from_point((lat, lon), dist=dist_metros, network_type='drive')
        
        if len(G.nodes) == 0:
            raise ValueError("O grafo viário retornado está vazio. Coordenadas em local isolado ou inválidas.")

        logging.info("Calculando o nó viário mais próximo (projeção na rua) das coordenadas fornecidas...")
        # 2. Encontrar o nó mais próximo (G, X=longitude, Y=latitude)
        no_mais_proximo = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        
        logging.info("Gerando visualização do mapa...")
        # 3. Plotar o grafo viário
        # show=False e close=False garantem a execução non-blocking para não travar o script principal
        fig, ax = ox.plot_graph(
            G, 
            show=False, 
            close=False, 
            node_color='white',
            node_size=0,         # Esconder nós do grafo para um mapa mais limpo
            edge_color='#999999',# Ruas em cinza claro
            edge_linewidth=1,
            bgcolor='black'      # Fundo preto estilo dark mode
        )
        
        # 4. Adicionar o marcador (ponto vermelho) no nó projetado
        no_x = G.nodes[no_mais_proximo]['x']
        no_y = G.nodes[no_mais_proximo]['y']
        
        ax.scatter(
            no_x, 
            no_y, 
            c='red', 
            s=100, 
            zorder=5, 
            marker='o', 
            label='Poça Detectada (Projetada)'
        )
        
        # Adicionar o marcador (ponto azul/ciano pequeno) da coordenada GPS bruta, se desejar comparar
        ax.scatter(
            lon, 
            lat, 
            c='cyan', 
            s=30, 
            zorder=4, 
            marker='x', 
            label='GPS Bruto'
        )
        
        # Adicionar legenda
        ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
        
        # 5. Salvar a figura em alta resolução (300 dpi)
        # Garantir que o diretório de destino exista
        os.makedirs(os.path.dirname(os.path.abspath(caminho_saida)), exist_ok=True)
        
        fig.savefig(caminho_saida, dpi=300, bbox_inches='tight', pad_inches=0.1)
        
        # 6. Fechar a figura explicitamente para liberar a memória (importante para processamento em lote)
        plt.close(fig)
        
        logging.info(f"Mapa gerado e salvo com sucesso em: {caminho_saida}")
        return True
        
    except Exception as e:
        logging.error(f"Falha ao gerar o mapa da poça para as coordenadas ({lat}, {lon}): {str(e)}")
        return False

if __name__ == "__main__":
    # Exemplo de teste rápido para validar o funcionamento do script
    # Coordenadas da Avenida Paulista, SP, como exemplo
    lat_exemplo = -23.5615
    lon_exemplo = -46.6560
    caminho_img_exemplo = os.path.join("resultados_mapas", "poca_avenida_paulista.png")
    
    print("Iniciando teste de geração de mapa...")
    sucesso = gerar_mapa_poca(lat_exemplo, lon_exemplo, caminho_img_exemplo)
    
    if sucesso:
        print(f"Teste concluído com sucesso. Verifique o arquivo gerado em {caminho_img_exemplo}")
    else:
        print("Falha no teste. Consulte os logs acima.")

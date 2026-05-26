from mapear_poca import gerar_mapa_poca
import os

print("Testando a integração com OSMnx...")

# Coordenadas fictícias simulando uma detecção em São Mateus, ES
lat_teste = -18.7161
lon_teste = -39.8542
arquivo_saida = "demo_mapa.png"

# Executa a função criada pelo seu assistente
sucesso = gerar_mapa_poca(
    lat=lat_teste, 
    lon=lon_teste, 
    caminho_saida=arquivo_saida
)

if sucesso:
    print(f"✅ Sucesso! O mapa foi salvo em: {os.path.abspath(arquivo_saida)}")
    print("Abra o arquivo para conferir o marcador vermelho na rua.")
else:
    print("❌ Falha na geração do mapa. Verifique sua conexão com a internet ou os logs de erro.")

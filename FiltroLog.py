import sys
import os


def filtrarLog(arquivo_entrada, arquivo_saida, palavras_chave):
    """
    Lê o arquivo de log e filtra as linhas que começam com palavras-chave específicas.
    """
    print(f"Filtrando o arquivo: {arquivo_entrada}...")
    linhas_encontradas = 0

    try:
        with open(arquivo_entrada, 'r', encoding='utf-8') as f_entrada, \
                open(arquivo_saida, 'w', encoding='utf-8') as f_saida:

            for linha in f_entrada:
                if any(linha.startswith(palavra) for palavra in palavras_chave):
                    f_saida.write(linha)
                    linhas_encontradas += 1

        print(f"\nFiltro concluído!")
        print(f"Total de {linhas_encontradas} linhas filtradas salvas em: {arquivo_saida}")

    except FileNotFoundError:
        print(f"ERRO: O arquivo de entrada '{arquivo_entrada}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


# --- Início da Execução---
if __name__ == "__main__":

    # sys.argv[0] é o nome do script (FiltroLog.py)
    # sys.argv[1] será o caminho para o arquivo de log
    if len(sys.argv) != 2:
        print("=" * 60)
        print("ERRO.")
        print("Você deve passar 1 argumento: o caminho para o arquivo de log original.")
        print("\nExemplo no terminal:")
        print(r'  python FiltroLog.py "C:\Users\\Downloads\audit_20250701_4.txt"')
        print("=" * 60)
        sys.exit(1)  # Encerra o script com erro

    # 1. Pega o nome do arquivo de log do argumento do terminal
    nome_arquivo_log = sys.argv[1]

    # 2. O arquivo de saída será criado na pasta onde o script está
    nome_arquivo_filtrado = "log_filtrado.txt"

    # 3. As palavras-chave para filtrar (NMEAGGA é importante para geolocalização futura)
    palavras_para_filtrar = ["CAMERA2", "CAMERA3", "NMEAGGA"]

    # 4. Chama a função para executar o filtro
    filtrarLog(nome_arquivo_log, nome_arquivo_filtrado, palavras_para_filtrar)
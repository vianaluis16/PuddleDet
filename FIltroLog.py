def filtrar_log(arquivo_entrada, arquivo_saida, palavras_chave):
    """
    Lê o arquivo de log e filtra as linhas que começam com as palavras chaves.

    Argumentos:
        arquivo_entrada (str): O caminho para o arquivo log que será lido.
        arquivo_saida (str): O caminho para o arquivo onde as linhas filtradas serão salvas.
        palavras_chave (list): Uma lista de strings. As linhas que começam com qualquer uma dessas palavras serão salvas.
    """
    try:
        # Abre o arquivo de entrada para read ('r') e o de saída para write ('w')

        # O 'with' faz fechar os arquivos automaticamente no final
        with open(arquivo_entrada, 'r', encoding='utf-8') as f_entrada, \
             open(arquivo_saida, 'w', encoding='utf-8') as f_saida:

            # Itera sobre cada linha do arquivo de entrada
            for linha in f_entrada:
                # O metodo startswith() verifica se a linha começa com uma das palavras
                # A função any() retorna True se qualquer uma das verificações for verdadeira
                if any(linha.startswith(palavra) for palavra in palavras_chave):
                    # Se a condição for true, escreve a linha no arquivo de saída
                    f_saida.write(linha)

        print(f"Filtro concluído! As linhas foram salvas em: {arquivo_saida}, na raiz do repositório.")

    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada '{arquivo_entrada}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


#Início da Execução
if __name__ == "__main__":
    # 1. Defina o nome do arquivo de log original
    nome_arquivo_log = "audit_20250701_4.txt"

    # 2. Dê o nome do novo arquivo que será criado com as linhas filtradas
    nome_arquivo_filtrado = "log_filtrado.txt"

    # 3. Defina as palavras que iram ser buscadas no início de cada linha
    palavras_para_filtrar = ["CAMERA2", "CAMERA3", "NMEAGGA"]

    # 4. Chama a função
    filtrar_log(nome_arquivo_log, nome_arquivo_filtrado, palavras_para_filtrar)
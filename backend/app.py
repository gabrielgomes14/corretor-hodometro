import os
import re
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename

# --- CONFIGURAÇÃO INICIAL DO FLASK ---
app = Flask(__name__)

# Configuração do CORS para permitir requisições do seu front-end
# O '*' permite qualquer origem, para desenvolvimento. Em produção, use a URL exata do seu front.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
# Usar diretórios temporários é uma boa prática para ambientes de nuvem
UPLOAD_FOLDER = '/tmp/uploads_velocidade'
DOWNLOAD_FOLDER = '/tmp/downloads_velocidade'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# --- SUA LÓGICA DE PROCESSAMENTO, AGORA DENTRO DE UMA FUNÇÃO ---
def processar_planilhas(caminho_modelo, caminho_velocidade):
    """
    Função que encapsula toda a sua lógica de processamento de planilhas.
    Recebe os caminhos para os dois arquivos necessários e retorna um DataFrame processado.
    """
    try:
        # Carrega o modelo para obter a estrutura de colunas
        logging.info("Carregando arquivo modelo...")
        df_modelo = pd.read_excel(caminho_modelo)
        colunas_modelo = df_modelo.columns.tolist()
        logging.info("✅ Modelo carregado.")

        # Lê os dados principais, pulando o cabeçalho falso
        logging.info("Carregando arquivo de velocidade...")
        df_raw = pd.read_excel(caminho_velocidade, skiprows=7)
        df_full = pd.read_excel(caminho_velocidade, header=None) # Para detectar placas
        logging.info("✅ Arquivo de velocidade carregado.")

        # Detecta linhas com placas
        placa_regex = r"^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}"
        placas_info = []
        for idx, val in df_full.iloc[:, 1].items():
            if isinstance(val, str) and re.match(placa_regex, val.strip().split()[0]):
                placa = val.strip().split()[0]
                placas_info.append((idx, placa))

        if not placas_info:
            raise ValueError("Nenhuma placa de veículo encontrada no arquivo de velocidade.")

        # Mapeamento das placas por linha
        placas_idx = [x[0] for x in placas_info]
        placas_val = [x[1] for x in placas_info]
        placas_map = {}
        placa_pointer = 0
        for i in range(len(df_raw)):
            abs_idx = i + 8
            if placa_pointer + 1 < len(placas_idx) and abs_idx >= placas_idx[placa_pointer + 1]:
                placa_pointer += 1
            placas_map[i] = placas_val[placa_pointer]

        # Renomeia colunas
        colunas_encontradas = df_raw.columns.tolist()
        col_data = [c for c in colunas_encontradas if "data" in str(c).lower()][0]
        col_local = [c for c in colunas_encontradas if "local" in str(c).lower()][0]
        col_vel = [c for c in colunas_encontradas if "velocidade" in str(c).lower()][0]
        df_raw.rename(columns={col_data: 'Data & Hora', col_local: 'Endereço', col_vel: 'Velocidade'}, inplace=True)

        # Adiciona colunas
        df_raw['Veículo'] = df_raw.index.map(placas_map)
        df_raw['Apelido'] = df_raw['Veículo']
        df_raw['Infração'] = "Velocidade Máxima"
        df_raw['Valor Padrão'] = "110 Km/h"

        # Garante todas as colunas do modelo
        for col in colunas_modelo:
            if col not in df_raw.columns:
                df_raw[col] = ""
        df_final = df_raw[colunas_modelo].copy()

        # Conversão de dados
        df_final['Data & Hora'] = pd.to_datetime(df_final['Data & Hora'], errors='coerce')
        df_final.loc[df_final['Data & Hora'].notna(), 'Data & Hora'] = df_final['Data & Hora'].dt.strftime('%d/%m/%Y %H:%M')
        df_final['Velocidade'] = pd.to_numeric(df_final['Velocidade'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')

        return df_final

    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
        # Retorna None para indicar que houve uma falha
        return None


# --- ENDPOINTS (ROTAS) DA API ---

@app.route('/')
def index():
    """Rota inicial para verificar se o backend está no ar."""
    return "<h1>Backend do Corretor de Velocidade está no ar!</h1>"


@app.route('/api/processar', methods=['POST'])
def processar_arquivos_endpoint():
    """
    Endpoint que recebe os dois arquivos, processa e retorna
    a URL de download e uma pré-visualização dos dados.
    """
    # Verifica se os dois arquivos foram enviados
    if 'modeloFile' not in request.files or 'velocidadeFile' not in request.files:
        return jsonify({"error": "É necessário enviar os dois arquivos: o de modelo e o de velocidade."}), 400

    file_modelo = request.files['modeloFile']
    file_velocidade = request.files['velocidadeFile']

    if file_modelo.filename == '' or file_velocidade.filename == '':
        return jsonify({"error": "Um ou mais arquivos não foram selecionados."}), 400

    # Salva os arquivos no servidor de forma segura
    filename_modelo = secure_filename(file_modelo.filename)
    filename_velocidade = secure_filename(file_velocidade.filename)
    path_modelo = os.path.join(app.config['UPLOAD_FOLDER'], filename_modelo)
    path_velocidade = os.path.join(app.config['UPLOAD_FOLDER'], filename_velocidade)
    file_modelo.save(path_modelo)
    file_velocidade.save(path_velocidade)

    # Chama a função de processamento
    df_corrigido = processar_planilhas(path_modelo, path_velocidade)

    if df_corrigido is None:
        return jsonify({"error": "Falha ao processar as planilhas. Verifique o formato dos arquivos e os logs do servidor."}), 500

    # Prepara o arquivo para download
    download_filename = f"formatado_{filename_velocidade}"
    download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], download_filename)
    # Salva como .xlsx para manter a formatação
    df_corrigido.to_excel(download_path, index=False)
    download_url = f"/api/download/{download_filename}"

    # Cria a pré-visualização dos dados
    df_preview = df_corrigido.head(100).fillna('').copy()
    preview_data = df_preview.to_dict(orient='records')
    
    # Retorna a resposta com sucesso
    return jsonify({
        "downloadUrl": download_url,
        "previewData": preview_data
    })


@app.route('/api/download/<filename>', methods=['GET'])
def download_arquivo(filename):
    """Rota para servir o arquivo corrigido para download."""
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)


# Inicia o servidor Flask
if __name__ == '__main__':
    # O host '0.0.0.0' torna o servidor acessível na sua rede local
    app.run(host='0.0.0.0', port=5000, debug=True)

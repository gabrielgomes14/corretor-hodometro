import locale
import os
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

# --- CONFIGURAÇÃO INICIAL DO FLASK ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://corretor-frontend.onrender.com"}}, supports_credentials=True)



# --- ALTERAÇÃO PRINCIPAL AQUI ---
# Em ambientes de nuvem como o Render, é mais seguro usar o diretório /tmp para salvar arquivos temporários.
UPLOAD_FOLDER = '/tmp/uploads'
DOWNLOAD_FOLDER = '/tmp/downloads'
MODEL_DIR = "/tmp/modelos_hodometro"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --- LÓGICA DE CORREÇÃO (O restante do código permanece o mesmo) ---

def calcular_limite_km_por_dia(df: pd.DataFrame, col: str = "Hodômetro") -> float:
    df = df.sort_values("Data").reset_index(drop=True)
    df["diff_dias"] = df["Data"].diff().dt.total_seconds() / 86400
    df["diff_km"] = df[col].diff()
    vel = df.loc[(df["diff_dias"] > 0) & (df["diff_km"] >= 0), "diff_km"] / df.loc[
        (df["diff_dias"] > 0) & (df["diff_km"] >= 0), "diff_dias"
    ]
    if vel.empty: return 200.0
    return float(max(np.percentile(vel, 95), 50.0))

def caminho_modelo_placa(placa: str) -> str:
    return os.path.join(MODEL_DIR, f"modelo_{placa}.pkl")

def carregar_modelo_existente(placa: str):
    caminho = caminho_modelo_placa(placa)
    return joblib.load(caminho) if os.path.exists(caminho) else None

def salvar_modelo(placa: str, modelo_scaler):
    joblib.dump(modelo_scaler, caminho_modelo_placa(placa))

def treinar_modelo(y, dias_frac, modelo_scaler=None):
    X = np.array(dias_frac).reshape(-1, 1)
    y = np.array(y)
    if modelo_scaler is None:
        scaler = StandardScaler(); Xs = scaler.fit_transform(X); mdl = LinearRegression().fit(Xs, y)
    else:
        mdl, scaler = modelo_scaler; Xs = scaler.transform(X); mdl.fit(Xs, y)
    return mdl, scaler

def corrigir_grupo(df_grp: pd.DataFrame, col="Hodômetro"):
    df_grp = df_grp.sort_values("Data").reset_index(drop=True)
    hod = df_grp[col].tolist(); datas = df_grp["Data"].tolist(); placa = df_grp.iloc[0]["Placa"]
    y_corr = [hod[0]]; t0 = datas[0]
    dias_frac = [(d - t0).total_seconds() / 86400 for d in datas]
    max_km_dia = calcular_limite_km_por_dia(df_grp, col)
    mdl, scl = treinar_modelo(y_corr, dias_frac[:1], carregar_modelo_existente(placa))
    for i in range(1, len(hod)):
        dias_int = dias_frac[i] - dias_frac[i - 1]
        aumento_max = max_km_dia * dias_int
        aumento_real = hod[i] - y_corr[-1]
        if aumento_real < 0 or aumento_real > aumento_max:
            mdl, scl = treinar_modelo(y_corr, dias_frac[:i], (mdl, scl))
            pred = mdl.predict(scl.transform([[dias_frac[i]]]))[0]
            novo = y_corr[-1] + 1.0 if pred <= y_corr[-1] else min(pred, y_corr[-1] + aumento_max)
            y_corr.append(novo)
        else:
            y_corr.append(hod[i])
    mdl, scl = treinar_modelo(y_corr, dias_frac, (mdl, scl))
    salvar_modelo(placa, (mdl, scl)); df_grp[col] = y_corr
    return df_grp

def corrigir_hodometros_repetidos_por_segundo(df: pd.DataFrame, taxa_km_por_segundo: float = 0.01333) -> pd.DataFrame:
    df = df.sort_values("Data").reset_index(drop=True)
    for i in range(1, len(df)):
        if df.loc[i, "Hodômetro"] == df.loc[i - 1, "Hodômetro"]:
            segundos = (df.loc[i, "Data"] - df.loc[i - 1, "Data"]).total_seconds()
            if segundos > 0:
                incremento = segundos * taxa_km_por_segundo
                df.loc[i, "Hodômetro"] = df.loc[i - 1, "Hodômetro"] + incremento
    return df

def corrigir_planilha_com_ia(path: str, col: str = "Hodômetro") -> pd.DataFrame | None:
    try:
        df = pd.read_excel(path)
    except Exception as e:
        logging.error(f"Erro ao ler planilha: {e}"); return None
    df.columns = df.columns.str.strip()
    colunas_aceitas = { "Data": ["Data", "Data/Hora Transação", "Data Transação"], "Placa": ["Placa"], "Hodômetro": ["Hodômetro", "Hodômetro - Dig. Motorista", "HODOMETRO OU HORIMETRO"],}
    def encontrar_coluna(df_cols, possiveis):
        for c in possiveis:
            if c in df_cols: return c
        return None
    col_data, col_placa, col_hod = encontrar_coluna(df.columns, colunas_aceitas["Data"]), encontrar_coluna(df.columns, colunas_aceitas["Placa"]), encontrar_coluna(df.columns, colunas_aceitas["Hodômetro"])
    if not all([col_data, col_placa, col_hod]):
        logging.error("Colunas obrigatórias não encontradas."); return None
    df = df.rename(columns={col_data: "Data", col_placa: "Placa", col_hod: "Hodômetro"})
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Data"])
    df["Hodômetro"] = pd.to_numeric(df["Hodômetro"], errors="coerce")
    df = df.dropna(subset=["Hodômetro"])
    df["Hodômetro"] = df["Hodômetro"].astype(float)
    corrigidos = [corrigir_grupo(corrigir_hodometros_repetidos_por_segundo(grp.copy())) for _, grp in df.groupby("Placa")]
    df_final = pd.concat(corrigidos).sort_values(["Placa", "Data"]).reset_index(drop=True)
    return df_final


# --- ENDPOINTS DA API FLASK ---
@app.route('/')
def index():
    return jsonify({ "status": "online", "message": "Backend do Corretor de Hodômetro está no ar." })

@app.route('/api/processar', methods=['POST'])
def processar_arquivo():
    if 'file' not in request.files: return jsonify({"error": "Nenhum arquivo enviado"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "Nenhum arquivo selecionado"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df_corrigido = corrigir_planilha_com_ia(filepath)
        if df_corrigido is None:
            return jsonify({"error": "Falha ao processar o arquivo."}), 500

        df_preview = df_corrigido.head(100).copy()
        df_preview['Data'] = df_preview['Data'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_preview['Hodômetro'] = df_preview['Hodômetro'].round(1)
        preview_data = df_preview.to_dict(orient='records')
        
        df_excel = df_corrigido.copy()
        try:
            locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
            df_excel['Hodômetro'] = df_excel['Hodômetro'].apply(lambda x: locale.format_string('%.1f', x, grouping=True) if pd.notnull(x) else '')
        except locale.Error:
            logging.warning("Locale 'pt_BR.UTF-8' não disponível. Formatando hodômetro com ponto decimal.")
            df_excel['Hodômetro'] = df_excel['Hodômetro'].apply(lambda x: f'{x:.1f}' if pd.notnull(x) else '')
        df_excel['Data'] = df_excel['Data'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_excel = df_excel.replace({np.nan: None})

        download_filename = f"corrigido_{filename}"
        download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], download_filename)
        df_excel.to_excel(download_path, index=False)
        download_url = f"/api/download/{download_filename}"
        
        response = jsonify({ "downloadUrl": download_url, "previewData": preview_data })
        response.headers.add("Access-Control-Allow-Origin", "https://corretor-frontend.onrender.com")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response


@app.route('/api/download/<filename>', methods=['GET'])
def download_arquivo(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

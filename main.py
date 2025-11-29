import pandas as pd
import numpy as np

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


# -----------------------------------------------------------
# CARREGAMENTO E PREPARAÇÃO DO DATASET
# -----------------------------------------------------------

print("\n>> Carregando dataset do DNIT...")
# Lê explicitamente a aba correta do arquivo do DNIT
df = pd.read_excel("vmda2022_snv_202301b.xlsx", sheet_name="SNV202301B")
print(f">> Linhas: {len(df)}, Colunas: {len(df.columns)}")
print("Colunas:", list(df.columns))

# Garante colunas usadas existam
for col in ["VMDa_C", "VMDa_D"]:
    if col not in df.columns:
        df[col] = 0

# Preenche nulos e cria VMDA_TOTAL
df["VMDa_C"] = df["VMDa_C"].fillna(0)
df["VMDa_D"] = df["VMDa_D"].fillna(0)
df["VMDA_TOTAL"] = df["VMDa_C"] + df["VMDa_D"]

# Converte tipos básicos
df["sg_uf"] = df["sg_uf"].astype(str)
df["vl_br"] = df["vl_br"].astype(int)
df["vl_extensa"] = df["vl_extensa"].fillna(0).astype(float)


# -----------------------------------------------------------
# REGRESSÃO LINEAR – prever VMDA_TOTAL
# -----------------------------------------------------------

print(">> Treinando modelo de Regressão Linear (VMDA_TOTAL)...")

X_reg = df[["vl_extensa", "vl_br"]]
y_reg = df["VMDA_TOTAL"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

modelo_reg = LinearRegression()
modelo_reg.fit(X_train, y_train)

y_pred = modelo_reg.predict(X_test)
mse_teste = float(np.mean((y_pred - y_test) ** 2))
r2_teste = float(modelo_reg.score(X_test, y_test))

print(f">> Regressão Linear - MSE: {mse_teste:.2f}, R²: {r2_teste:.4f}")


# -----------------------------------------------------------
# ÁRVORE DE DECISÃO – classificar faixa de tráfego
# -----------------------------------------------------------

print(">> Treinando Árvore de Decisão (Faixa VMDA)...")

def categorizar_vmda(valor: float) -> str:
    if valor < 2000:
        return "baixo"
    if valor < 6000:
        return "medio"
    if valor < 15000:
        return "alto"
    return "muito_alto"

df["faixa_vmda"] = df["VMDA_TOTAL"].apply(categorizar_vmda)

le_faixa = LabelEncoder()
df["faixa_encoded"] = le_faixa.fit_transform(df["faixa_vmda"])

X_cls = df[["vl_extensa", "vl_br"]]
y_cls = df["faixa_encoded"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

modelo_arvore = DecisionTreeClassifier(max_depth=6, random_state=42)
modelo_arvore.fit(Xc_train, yc_train)

acuracia_teste = float(modelo_arvore.score(Xc_test, yc_test))

print(f">> Árvore de Decisão - Acurácia: {acuracia_teste:.4f}")


# -----------------------------------------------------------
# K-MEANS – agrupamento de trechos
# -----------------------------------------------------------

print(">> Treinando K-Means...")

df_kmeans = df[["VMDA_TOTAL", "vl_extensa"]].copy().fillna(0)

modelo_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = modelo_kmeans.fit_predict(df_kmeans)

print(">> K-Means OK.")


# -----------------------------------------------------------
# FASTAPI – MODELOS Pydantic
# -----------------------------------------------------------

class PredicaoRequest(BaseModel):
    sg_uf: str
    vl_br: int
    vl_extensa: float


class PredicaoResponse(BaseModel):
    sg_uf: str
    vl_br: int
    vl_extensa: float
    vmda_previsto: float
    mse_teste: float
    r2_teste: float


class ClassificacaoResponse(BaseModel):
    sg_uf: str
    vl_br: int
    vl_extensa: float
    faixa_prevista: str
    acuracia_teste: float


class TrechoResponse(BaseModel):
    id: int
    sg_uf: str
    vl_br: int
    vl_km_inic: float
    vl_km_fina: float
    vl_extensa: float
    VMDa_C: float
    VMDa_D: float
    VMDA_TOTAL: float


# -----------------------------------------------------------
# FASTAPI – CONFIG
# -----------------------------------------------------------

app = FastAPI(
    title="SIMTRA - API DNIT",
    version="1.0.0",
    description="API de análise de tráfego rodoviário (DNIT) com IA (Regressão, Árvore de Decisão, K-Means)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------
# ENDPOINTS BÁSICOS
# -----------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "API funcionando",
        "modelos": {
            "regressao_linear": {"mse": mse_teste, "r2": r2_teste},
            "arvore_decisao": {"acuracia": acuracia_teste},
            "kmeans_clusters": int(modelo_kmeans.n_clusters),
        },
    }


@app.get("/ufs")
def listar_ufs():
    return {"ufs": sorted(df["sg_uf"].unique())}


@app.get("/rodovias")
def listar_rodovias(uf: str):
    """
    Lista as BRs disponíveis para uma determinada UF.
    Função robusta: evita quebrar o front caso algo estranho aconteça.
    """
    try:
        uf = uf.upper()

        if "sg_uf" not in df.columns or "vl_br" not in df.columns:
            return {"rodovias": []}

        d = df[df["sg_uf"].astype(str) == uf]

        if d.empty:
            return {"rodovias": []}

        brs = (
            d["vl_br"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        brs = sorted(brs)
        return {"rodovias": brs}

    except Exception as e:
        print("ERRO em /rodovias:", repr(e))
        return {"rodovias": []}


@app.get("/estatisticas/uf/{uf}")
def estatisticas_uf(uf: str):
    d = df[df["sg_uf"] == uf.upper()]
    if d.empty:
        raise HTTPException(status_code=404, detail="UF não encontrada.")
    return {
        "vmda_medio": float(d["VMDA_TOTAL"].mean()),
        "vmda_min": float(d["VMDA_TOTAL"].min()),
        "vmda_max": float(d["VMDA_TOTAL"].max()),
        "qtd_trechos": int(len(d)),
    }


# -----------------------------------------------------------
# ENDPOINT /trechos – sem depender da coluna ID
# -----------------------------------------------------------

@app.get("/trechos", response_model=List[TrechoResponse])
def listar_trechos(uf: str, br: Optional[int] = None, limite: int = 100):
    d = df[df["sg_uf"] == uf.upper()]

    if br is not None:
        d = d[d["vl_br"] == br]

    if d.empty:
        raise HTTPException(status_code=404, detail="Nenhum trecho encontrado.")

    d = d.head(limite).copy().reset_index(drop=True)

    resposta: List[TrechoResponse] = []
    for i, row in d.iterrows():
        resposta.append(
            TrechoResponse(
                id=i + 1,
                sg_uf=row["sg_uf"],
                vl_br=int(row["vl_br"]),
                vl_km_inic=float(row.get("vl_km_inic", 0) or 0),
                vl_km_fina=float(row.get("vl_km_fina", 0) or 0),
                vl_extensa=float(row.get("vl_extensa", 0) or 0),
                VMDa_C=float(row.get("VMDa_C", 0) or 0),
                VMDa_D=float(row.get("VMDa_D", 0) or 0),
                VMDA_TOTAL=float(row.get("VMDA_TOTAL", 0) or 0),
            )
        )

    return resposta


# -----------------------------------------------------------
# IA – REGRESSÃO LINEAR
# -----------------------------------------------------------

@app.post("/predicao/vmda", response_model=PredicaoResponse)
def predicao_vmda(req: PredicaoRequest):
    entrada = [[req.vl_extensa, req.vl_br]]
    prev = float(modelo_reg.predict(entrada)[0])

    return PredicaoResponse(
        sg_uf=req.sg_uf,
        vl_br=req.vl_br,
        vl_extensa=req.vl_extensa,
        vmda_previsto=prev,
        mse_teste=mse_teste,
        r2_teste=r2_teste,
    )


# -----------------------------------------------------------
# IA – CLASSIFICAÇÃO (Árvore de Decisão)
# -----------------------------------------------------------

@app.post("/classificacao/faixa_vmda", response_model=ClassificacaoResponse)
def classificacao_faixa(req: PredicaoRequest):
    entrada = [[req.vl_extensa, req.vl_br]]
    classe = int(modelo_arvore.predict(entrada)[0])
    faixa = str(le_faixa.inverse_transform([classe])[0])

    return ClassificacaoResponse(
        sg_uf=req.sg_uf,
        vl_br=req.vl_br,
        vl_extensa=req.vl_extensa,
        faixa_prevista=faixa,
        acuracia_teste=acuracia_teste,
    )


# -----------------------------------------------------------
# IA – CLUSTERS (K-Means)
# -----------------------------------------------------------

@app.post("/clusters/resumo")
def clusters_resumo():
    grupos = df["cluster"].value_counts().to_dict()
    centroides = modelo_kmeans.cluster_centers_.tolist()

    return {
        "n_clusters": int(modelo_kmeans.n_clusters),
        "qtd_por_cluster": {int(k): int(v) for k, v in grupos.items()},
        "centroides": [
            {
                "cluster": int(i),
                "VMDA_TOTAL_medio": float(c[0]),
                "vl_extensa_media": float(c[1]),
            }
            for i, c in enumerate(centroides)
        ],
    }


# -----------------------------------------------------------
# RODAR DIRETO (opcional)
# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

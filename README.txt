#Execução::
#Abrir CMD > uvicorn main:app --reload
#Abrir CMD > python -m http.server 8001

#Abrir navegador > http://127.0.0.1:8001/index.html


#-------------------------------------------------#

# SIMTRA – Sistema Inteligente de Mobilidade e Tráfego Rodoviário  
### Projeto Integrador IV – Aplicações de Inteligência Artificial

##  Descrição
O SIMTRA é um sistema web com backend em FastAPI e modelos de Inteligência Artificial aplicados ao dataset oficial do DNIT.  
O projeto permite analisar, prever e classificar o tráfego rodoviário brasileiro, oferecendo estatísticas avançadas e uma interface web interativa.

## Funcionalidades
- Listagem de UFs e BRs
- Consulta de trechos por UF/BR
- Estatísticas de VMDA por estado
- Previsão de tráfego (Regressão Linear)
- Classificação de intensidade (Árvore de Decisão)
- Agrupamento de trechos (K-Means)
- Interface web integrada à API

## IA Utilizada
- Regressão Linear
- Decision Tree Classifier
- K-Means Clustering

##  Estrutura


# app.py

import streamlit as st
import json
import pandas as pd
import os
#from sklearn.metrics import classification_report, confusion_matrix

LABELS = [
    {"label": "Bloqueio", "value": 1, "subpasta": "bloqueio", "tpOficio": "03"},
    {"label": "Não-Bloqueio", "value": 0, "subpasta": "nao_bloqueio", "tpOficio": "00"}
]
SUBPASTA_TO_LABEL = {lbl['subpasta']: lbl for lbl in LABELS}
FUNCIONALIDADES = ["CLASSIFICAR", "TESTAR MODELO", "RELATÓRIOS"]

st.set_page_config(page_title="Classificador de Ofícios - V4")


# -------------- ABA DE NAVEGAÇÃO ---------------
aba = st.sidebar.radio(
    "QUAL FUNCIONALIDADE DESEJA EXECUTAR?",
    FUNCIONALIDADES
)

# -------------- CLASSIFICAÇÃO DIÁRIA ---------------
if aba == FUNCIONALIDADES[0]:
    st.markdown("<h1 style='text-align: center;'>Descubra qual tipo o seu Ofício pertence:</h1>", unsafe_allow_html=True)
   

# -------------- TESTE DO MODELO (com upload de arquivos rotulados) ---------------
if aba == FUNCIONALIDADES[1]:
    st.markdown("<h1 style='text-align: center;'>Traga novos dados para testar o desempenho do modelo!</h1>", unsafe_allow_html=True)

if aba == FUNCIONALIDADES[2]:
# Carrega experimentos já ajustados
    with open("relatorio_experimentos.json", "r", encoding="utf-8") as f:
        data = json.load(f)

        # CONTEXTO RESUMIDO
        contexto = """
        O processo de atendimento a ordens judiciais de bloqueio financeiro é operacionalizado pela JD, que atua na estruturação e análise de ofícios encaminhados por diversos tribunais. O desafio envolve lidar com documentos em PDF altamente heterogêneos, com grande variedade de formatos, vocabulário jurídico complexo e estrutura textual não padronizada. Atualmente, todo o fluxo é majoritariamente manual, com participação intensa de advogados especializados. A empresa buscou soluções baseadas em IA para aumentar a produtividade e reduzir o tempo de processamento, testando modelos capazes de identificar e classificar automaticamente ofícios, em busca de um desempenho que permitisse automatização confiável do processo.
        """

        st.title("Relatório Executivo — Plataforma de IA JD para Classificação de Ofícios Jurídicos")
        st.markdown(f"##### Contexto do Problema")
        st.markdown(contexto)

        # SUMÁRIO
        st.markdown("### Sumário")
        st.markdown("""
        1. Estratégias e Soluções Testadas
        2. Pipeline de Automação
        3. Resultados: Números e Diagnóstico
        """)

        st.header("1. Estratégias e Soluções Testadas")
        st.markdown("A seguir, apresentamos os métodos avaliados para solucionar o desafio da JD. Cada solução traz sua abordagem, principais resultados e orientações para evolução do projeto:")

        # Agrupando por grupo/modelo
        grupos = {}
        for exp in data['experimentos']:
            if exp['grupo'] not in grupos:
                grupos[exp['grupo']] = []
            grupos[exp['grupo']].append(exp)

        for grupo, exps in grupos.items():
            with st.expander(f"🧩 {grupo}", expanded=True):
                for idx, exp in enumerate(exps):
                    with st.container():
                        st.markdown(f"##### {exp['nome']}")
                        st.markdown(f"*<b>Descrição:</b> {exp['observacoes']}*", unsafe_allow_html=True)

                        # Pipeline visual
                        with st.expander("🔎 Pipeline da solução", expanded=False):
                            st.markdown(" > ".join([f"**{step}**" for step in exp['pipeline']]), unsafe_allow_html=True)
                            if exp['hiperparametros']:
                                st.markdown(f"**Hiperparâmetros:** `{exp['hiperparametros']}`")

                        # Métricas em tabela
                        metricas = exp['resultados']
                        tabela = pd.DataFrame([{
                            "Sensibilidade": metricas['sensibilidade'],
                            "Especificidade": metricas['especificidade'],
                            "Precisão": metricas['precisao'],
                            "Acurácia": metricas['acuracia'],
                            "F1 Score": metricas['f1score']
                        }])
                        st.dataframe(tabela.style.format("{:.2%}"), use_container_width=True)

                        # Orientações
                        st.markdown("**Orientações por métrica:**")
                        for metrica, orientacoes in exp['orientacoes'].items():
                            st.markdown(f"**{metrica.capitalize()}:**")
                            st.markdown("\n".join([f"- {o}" for o in orientacoes]))
                        st.markdown(f"**Impacto no negócio:** {exp['impacto']}")

                        st.markdown("---")

        st.header("2. Pipeline de Automação")
        st.markdown("""
        O pipeline foi estruturado desde a extração do texto até a classificação final, abrangendo:
        - Entrada: PDF original do tribunal.
        - Extração e limpeza do texto (OCR, normalização, stopwords, tokenização, embeddings, compactação).
        - Extração de features: palavras, posições, palavras-chave.
        - Construção do dataset: treino/teste balanceado.
        - Treinamento do modelo (Random Forest, SVM, LogisticRegression, MLP, XGBoost).
        - Validação do modelo: sensibilidade, especificidade, precisão, acurácia e F1Score.
        """)

        st.header("3. Resultados: Números e Diagnóstico")
        tabela_resultados = []
        for exp in data['experimentos']:
            tabela_resultados.append({
                "Modelo": exp['nome'],
                "Acurácia (%)": f"{exp['resultados']['acuracia']*100:.0f}",
                "Precisão (%)": f"{exp['resultados']['precisao']*100:.0f}",
                "F1-score (%)": f"{exp['resultados']['f1score']*100:.0f}"
            })

        st.table(pd.DataFrame(tabela_resultados))

        # Diagnóstico final (visual: generalização vs memorização)
        st.markdown("#### Diagnóstico: Generalização x Memorização")
        st.markdown("""
        O modelo apresenta **alto desempenho na base conhecida**, mas sua habilidade de generalizar para novas massas ainda é limitada. 
        O maior risco reside na especialização excessiva nos padrões da base original — é fundamental evoluir o modelo para cenários mais diversos e robustos.

        **Diagnóstico visual:**  
        `Generalização` &nbsp;&nbsp;&nbsp; <span style='color:#4caf50;font-weight:bold;'>─────────────●─────────────</span> &nbsp;&nbsp;&nbsp; `Memorização`

        """, unsafe_allow_html=True)
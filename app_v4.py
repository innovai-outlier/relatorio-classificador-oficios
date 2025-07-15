# app.py

import streamlit as st
import json
import pandas as pd
import os
from inference import load_model, classify_oficio
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = [
    {"label": "Bloqueio", "value": 1, "subpasta": "bloqueio", "tpOficio": "03"},
    {"label": "Não-Bloqueio", "value": 0, "subpasta": "nao_bloqueio", "tpOficio": "00"}
]
SUBPASTA_TO_LABEL = {lbl['subpasta']: lbl for lbl in LABELS}
FUNCIONALIDADES = ["CLASSIFICAR", "TESTAR MODELO", "RELATÓRIOS"]

st.set_page_config(page_title="Classificador de Ofícios - V4")

@st.cache_resource
def get_model():
    return load_model()

modelo = get_model()

def listar_arquivos_pdf_com_rotulo(pasta_base):
    arquivos = []
    for root, dirs, files in os.walk(pasta_base):
        # identifica subpasta (bloqueio, nao_bloqueio, ...)
        for subpasta, info in SUBPASTA_TO_LABEL.items():
            if f"/{subpasta}" in root.replace("\\", "/").lower():
                for fname in files:
                    if fname.lower().endswith('.pdf'):
                        arquivos.append({
                            "caminho": os.path.join(root, fname),
                            "arquivo": fname,
                            "rotulo": info['value'],
                            "tpOficio": info['tpOficio'],
                            "descricao": info['label']
                        })
    return arquivos

# -------------- ABA DE NAVEGAÇÃO ---------------
aba = st.sidebar.radio(
    "QUAL FUNCIONALIDADE DESEJA EXECUTAR?",
    FUNCIONALIDADES[2]
)

# -------------- CLASSIFICAÇÃO DIÁRIA ---------------
if aba == FUNCIONALIDADES[0]:
    st.markdown("<h1 style='text-align: center;'>Descubra qual tipo o seu Ofício pertence:</h1>", unsafe_allow_html=True)
    st.write("Selecione um ou mais arquivos de ofício (PDF):")
    uploaded_files = st.file_uploader(
        "Escolha os arquivos PDF do ofício",
        type=['pdf'],
        accept_multiple_files=True
    )
    if uploaded_files:
        results = []
        with st.spinner('Classificando seu(s) ofício(s)...'):
            for arquivo in uploaded_files:
                resultado = classify_oficio(arquivo.read(), model=modelo)
                results.append({
                    "arquivo": arquivo.name,
                    "tpOficio": resultado["tpOficio"]
                })
        st.success(f'Classificação concluída! Total de arquivos: {len(results)}')
        st.markdown("**Resultado (cole este JSON onde quiser):**")
        st.code(json.dumps(results, ensure_ascii=False, indent=2), language='json')

# -------------- TESTE DO MODELO (com upload de arquivos rotulados) ---------------
if aba == FUNCIONALIDADES[1]:
    st.markdown("<h1 style='text-align: center;'>Traga novos dados para testar o desempenho do modelo!</h1>", unsafe_allow_html=True)
    st.markdown("### Estrutura esperada da massa de dados:")
    st.code(
        """
        raiz (.zip)
        ├── tpOficio_1 ("bloqueio" ou "03")
        │   ├── oficio_1.pdf
        │   ├── oficio_2.pdf
        │   └── oficio_U.pdf
        ├── tpOficio_2
        │   ├── oficio_1.pdf
        │   ├── oficio_2.pdf
        │   └── oficio_V.pdf
        └── tpOficio_N
            ├── oficio_1.pdf
            ├── oficio_2.pdf
            └── oficio_W.pdf
        """, language="text")
    # OBS: O st.file_uploader não permite selecionar pastas, por isso oriente a subir um arquivo ZIP
    uploaded_zip = st.file_uploader(
        "Envie um arquivo .ZIP contendo a estrutura de pastas e PDFs rotulados conforme acima.",
        type=['zip'],
        key="zip_files"
    )

    import tempfile
    import zipfile

    if uploaded_zip is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            # Acha o primeiro diretório do ZIP (dados/)
            for root, dirs, files in os.walk(tmpdir):
                if len(dirs) > 0:
                    pasta_base = os.path.join(root, dirs[0])
                    break
            arquivos = listar_arquivos_pdf_com_rotulo(pasta_base)

            if len(arquivos) == 0:
                st.warning("Nenhum PDF encontrado na estrutura esperada!")
            else:
                st.info(f"{len(arquivos)} arquivos encontrados. Clique abaixo para testar o modelo.")
                if st.button("Testar modelo", key="run_test_model"):
                    y_true = []
                    y_pred = []
                    nomes_arquivos = []
                    desc_esperado = []
                    desc_predito = []
                    tp_oficio_esperado = []
                    tp_oficio_predito = []

                    with st.spinner('Processando arquivos e avaliando...'):
                        for arqinfo in arquivos:
                            try:
                                with open(arqinfo["caminho"], "rb") as f:
                                    pdf_bytes = f.read()
                                resultado = classify_oficio(pdf_bytes, model=modelo)
                                pred_rotulo = 1 if resultado["tpOficio"] == "03" else 0
                                pred_desc = "Bloqueio" if resultado["tpOficio"] == "03" else "Não-Bloqueio"

                                nomes_arquivos.append(arqinfo["arquivo"])
                                y_true.append(arqinfo["rotulo"])
                                desc_esperado.append(arqinfo["descricao"])
                                tp_oficio_esperado.append(arqinfo["tpOficio"])
                                desc_predito.append(pred_desc)
                                tp_oficio_predito.append(resultado["tpOficio"])
                                y_pred.append(pred_rotulo)
                            except Exception as e:
                                st.warning(f"Erro ao processar {arqinfo['arquivo']}: {e}")

                    # Exibe resultados por arquivo
                    resultado_df = pd.DataFrame({
                        "Ofício": nomes_arquivos,
                        "Esperado": desc_esperado,
                        "Predito": desc_predito,
                        "tpOficio_esperado": tp_oficio_esperado,
                        "tpOficio_predito": tp_oficio_predito
                    })
                    st.markdown("#### Resultados individuais (por ofício):")
                    st.dataframe(resultado_df, use_container_width=True)

                    min_amostras_classe = 5
                    contagem_por_classe = pd.Series(y_true).value_counts()
                    
                    y_true_int = [int(x) for x in y_true]
                    y_pred_int = [int(x) for x in y_pred]

                    # Calculando métricas
                    # Classe positiva é bloqueio (1), negativa é não bloqueio (0)
                    sensibilidade = recall_score(y_true_int, y_pred_int, pos_label=1)
                    especificidade = recall_score(y_true_int, y_pred_int, pos_label=0)
                    precisao = precision_score(y_true_int, y_pred_int, pos_label=1)
                    f1 = f1_score(y_true_int, y_pred_int, pos_label=1)

                    st.markdown("#### Métricas de Classificação:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Sensibilidade", 
                            value=f"{sensibilidade*100:.2f}%",
                            border=True
                        )
                        with st.expander("O que é Sensibilidade?"):
                            st.markdown(
                                "Sensibilidade (Recall) é a probabilidade do modelo acertar que aquele ofício é um bloqueio de fato, ou seja, de todos os bloqueios reais, quantos o modelo acertou."
                            )

                        st.metric(
                            label="Precisão", 
                            value=f"{precisao*100:.2f}%",
                            border=True
                        )
                        with st.expander("O que é Precisão?"):
                            st.markdown(
                                "Precisão é, de todos os ofícios que o modelo inferiu ser bloqueio, qual o percentual que realmente eram bloqueios."
                            )

                    with col2:
                        st.metric(
                            label="Especificidade", 
                            value=f"{especificidade*100:.2f}%",
                            border=True
                        )
                        with st.expander("O que é Especificidade?"):
                            st.markdown(
                                "Especificidade é a probabilidade do modelo acertar que aquele ofício NÃO é um bloqueio de fato, ou seja, de todos os não bloqueios reais, quantos o modelo acertou."
                            )
                        st.metric(
                            label="F1 Score", 
                            value=f"{f1*100:.2f}%",
                            border=True
                        )
                        with st.expander("O que é F1 Score?"):
                            st.markdown(
                                "F1 Score é a média harmônica entre Precisão e Sensibilidade. Mede o equilíbrio entre acertar os bloqueios e não gerar muitos falsos positivos."
                            )
                        # Matriz de confusão
                    if contagem_por_classe.min() < min_amostras_classe:
                        st.warning("A matriz de confusão pode não ser representativa devido ao baixo número de amostras em uma ou mais classes.")
                    else:
                        st.markdown("#### Matriz de Confusão")
                        with st.expander("Visualizar..."):
                            labels = sorted(list(set(y_true_int) | set(y_pred_int)))
                            cm = confusion_matrix(y_true_int, y_pred_int, labels=labels)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
                            ax.set_xlabel('Predito')
                            ax.set_ylabel('Esperado')
                            st.pyplot(fig)

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
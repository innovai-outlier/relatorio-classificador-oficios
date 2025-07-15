import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

#st.set_page_config(layout="wide")

# Para ícones e temas, defina um dicionário de ícones para cada métrica
icon_metric = {
    "Sensibilidade": "🟦",
    "Especificidade": "🟩",
    "Precisão": "🟨",
    "Acurácia": "🟧",
    "F1 Score": "🟪"
}

def colored_metric_text(value, label):
    # Ajuste os thresholds conforme suas necessidades
    if value >= 0.85:
        color = "#4BB543"   # verde
    elif value >= 0.7:
        color = "#FFD700"   # amarelo
    else:
        color = "#FF6666"   # vermelho
    return f"<span style='color:{color}; font-weight:600'>{label}: {value:.0f}%</span>"

# Supondo relatorio já carregado como dict
# relatorio = ...
# Carrega o arquivo JSON
with open('relatorio_experimentos.json', 'r', encoding='utf-8') as f:
    relatorio = json.load(f)

    st.markdown("<h1 style='text-align: center; color: #345;'>📄 Relatório Executivo – Classificação de Ofícios Jurídicos com IA</h1>", unsafe_allow_html=True)

    # Contexto do problema
    st.markdown("""
    <div style='background-color: #f6f6f6; border-radius: 6px; padding: 14px 18px; margin-bottom:16px;'>
    <b>Contexto:</b>  
    No projeto JD, buscou-se automatizar a triagem e classificação de ofícios jurídicos com Inteligência Artificial, visando acelerar o fluxo, reduzir falhas humanas e aumentar a eficiência. O desafio: alto volume, diversidade de formatos e a necessidade de decisões rápidas e confiáveis.
    </div>
    """, unsafe_allow_html=True)

    # Sumário
    with st.expander("## 📋 Sumário", expanded=True):
        st.markdown("""
        1. Estratégias e Soluções Testadas  
        2. Pipeline de Automação  
        3. Principais Experimentos e Métricas  
        4. Teste Real com Massa Nova  
        5. Diagnóstico e Próximos Passos
        """)

    # 1. Estratégias e Soluções Testadas
    with st.expander("## 💡 Estratégias e Soluções Testadas"):
        st.info("""
        O projeto se  iniciou em arquiteturas generativas (multi-agentes e LLMs) e evoluiu para modelos clássicos de Machine Learning, sempre buscando o melhor equilíbrio entre simplicidade, desempenho e robustez. Foram avaliadas técnicas de extração de texto (OCR), processamento linguístico, modelos e hiperparâmetros, refinando continuamente as métricas.
        """)

    # 2. Pipeline de Automação
    with st.expander("## 🛠️ Pipeline de Automação"):    
        st.markdown("""
        O fluxo contempla:
        - 📥 Extração (OCR) 
        - 🧹 Limpeza & Normalização
        - 🧮 Vetorização (TF-IDF/Embeddings)
        - 🏷️ Feature Engineering
        - 🏗️ Dataset Split (Train/Test)
        - 🤖 Treinamento (RF, SVM, MLP, XGBoost, LLMs)
        - 📊 Validação (Sensibilidade, Especificidade, Precisão, Acurácia, F1)
        """)

    # 3. Principais Experimentos e Métricas
    with st.expander("## 📊 Principais Experimentos e Métricas"):
        # --- TABELA PRINCIPAL ---
        with st.expander("### 🔬 Tabela Comparativa dos Principais Experimentos"):
            tabela_resultados = []
            for exp in relatorio['experimentos']:
                if exp['resultados']['acuracia'] > 0:
                    tabela_resultados.append({
                        "🧪 Experimento": exp['nome'],
                        "Acurácia (%)": f"{exp['resultados']['acuracia']*100:.0f}",
                        "Sensibilidade (%)": f"{exp['resultados'].get('sensibilidade',0)*100:.0f}",
                        "Especificidade (%)": f"{exp['resultados'].get('especificidade',0)*100:.0f}",
                        "Precisão (%)": f"{exp['resultados'].get('precisao',0)*100:.0f}",
                        "F1-score (%)": f"{exp['resultados'].get('f1score',0)*100:.0f}"
                    })
            st.dataframe(pd.DataFrame(tabela_resultados), use_container_width=True, height=260, hide_index=True)

        # --- EXPANDERS POR GRUPO/MODELO ---
        with st.expander("### Resultados por grupo de modelos:"):
            grupos = {}
            for exp in relatorio['experimentos']:
                grupo = exp['grupo']
                if grupo not in grupos:
                    grupos[grupo] = []
                grupos[grupo].append(exp)

            for grupo, experiments in grupos.items():
                if grupo in "Teste Real com Massa Nova":
                    continue
                with st.expander(f"🗂️ {grupo}", expanded=False):
                    for exp in experiments:
                        with st.expander(f"{exp['nome']}", expanded=False):
                            st.markdown(f"**Modelos Utilizados:** {', '.join(exp['modelo'])}")
                            st.markdown("**Pipeline:**")
                            st.markdown(" ➡️ ".join([f"<b>{step}</b>" for step in exp['pipeline']]), unsafe_allow_html=True)

                            # Tabela de Métricas e Orientações
                            metricas = ['sensibilidade', 'especificidade', 'precisao', 'acuracia', 'f1score']
                            nomes_metricas = {
                                "sensibilidade": "Sensibilidade",
                                "especificidade": "Especificidade",
                                "precisao": "Precisão",
                                "acuracia": "Acurácia",
                                "f1score": "F1 Score"
                            }
                            resultados = exp['resultados']
                            orientacoes = exp['orientacoes']
                            dados = []
                            for m in metricas:
                                metric_icon = icon_metric[nomes_metricas[m]]
                                dados.append({
                                    "Métrica": f"{metric_icon} {nomes_metricas[m]}",
                                    "Resultado": f"{resultados.get(m, 0)*100:.0f}%",
                                    "Orientação": orientacoes[m] if isinstance(orientacoes[m], str) else " ".join(orientacoes[m])
                                })
                            st.dataframe(pd.DataFrame(dados), hide_index=True)
                            st.markdown(f"<b>Observações:</b> {exp['observacoes']}", unsafe_allow_html=True)
                            st.markdown(f"<b>Impacto:</b> <span style='color:#1464a5'>{exp['impacto']}</span>", unsafe_allow_html=True)

    # 4. Teste Real com Massa Nova
    teste_real = next((exp for exp in relatorio['experimentos'] if "Teste Real" in exp['grupo']), None)
    if teste_real:
        with st.expander("## 🧪 Teste Real com Massa Nova"):
            st.markdown("O melhor modelo foi testado em <b>100 novos ofícios</b> nunca vistos, balanceados entre bloqueio e não-bloqueio.", unsafe_allow_html=True)
            # Tabela única de métricas
            dados_teste = []
            for k, v in teste_real['resultados'].items():
                metric_icon = icon_metric[k.capitalize()] if k.capitalize() in icon_metric else ""
                dados_teste.append({
                    "Métrica": f"{metric_icon} {k.capitalize()}",
                    "Resultado": f"{v*100:.0f}%",
                    "Orientação": teste_real['orientacoes'][k] if isinstance(teste_real['orientacoes'][k], str) else " ".join(teste_real['orientacoes'][k])
                })
            st.dataframe(pd.DataFrame(dados_teste), hide_index=True)
            st.markdown(f"**Diagnóstico:** <span style='color:#e85757'>{teste_real['observacoes']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Impacto:** <span style='color:#1464a5'>{teste_real['impacto']}</span>", unsafe_allow_html=True)

            # Comparativo visual (Treino vs Teste Real)
            st.markdown("### 📊 Comparativo Visual – Desempenho Treino vs Teste Real")
            labels = ['Acurácia', 'Sensibilidade', 'Especificidade', 'Precisão', 'F1 Score']
            val_train = [0.85, 0.90, 0.81, 0.82, 0.86]
            val_teste = [0.64, 0.48, 0.90, 0.79, 0.59]
            fig, ax = plt.subplots()
            bar_width = 0.35
            bar1 = ax.bar([i-bar_width/2 for i in range(len(labels))], val_train, bar_width, label='Treino', color='#3b8eea')
            bar2 = ax.bar([i+bar_width/2 for i in range(len(labels))], val_teste, bar_width, label='Teste Real', color='#e85757')
            ax.set_ylabel('Score')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

    # 5. Diagnóstico e Próximos Passos
    with st.expander("## 🩺 Diagnóstico e Próximos Passos"):
        st.warning("""
        Após a validação real, o modelo apresentou redução na sensibilidade para detectar bloqueios, embora mantenha precisão e especificidade elevadas. Sugerem-se ações para ampliar a base de dados, explorar novos métodos de extração e ajustes finos, além de revisar amostras de maior risco para otimizar a generalização.
        """)
        st.markdown("""
        **Principais ToDos:**  
        - 🔎 Analisar casos de erro e falsos negativos  
        - ➕ Ampliar o dataset com novos exemplos reais  
        - 🛠️ Testar alternativas de OCR e embeddings  
        - 🔄 Reajustar hiperparâmetros e revalidar  
        - 📈 Relatar avanços e impactos práticos
        """)
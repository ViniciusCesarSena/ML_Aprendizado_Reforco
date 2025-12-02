# Contexto do Trabalho de Aprendizado por Reforço em Marketing

## Descrição geral
Este projeto utiliza o dataset `digital_marketing_campaign_dataset.csv` (8 mil registros e 20 atributos) para simular decisões de marketing digital. Cada linha representa um cliente, o canal utilizado (Email, PPC, Referral, SEO ou Social Media), custos da campanha e indicadores de engajamento/conversão. O briefing do documento **"Aprendizado por Reforço.docx"** solicitou a criação de um ambiente de simulação e de um agente que maximize métricas de negócio.

## Objetivos acadêmicos
- Aplicar conceitos de Aprendizado por Reforço (AR) vistos em aula, com foco em Q-Learning.
- Transformar dados reais em um ambiente contextualizado (bandit) que permita testar políticas de marketing.
- Comparar uma política aprendida com um baseline aleatório e interpretar as diferenças em termos de métricas do negócio.

## Metodologia adotada
1. **Exploração e visualização**: análise das distribuições de idade, canais e relação gasto vs. taxa de conversão para entender o comportamento do conjunto.
2. **Engenharia de estado/recompensa**: discretização de idade/renda/gênero para compor o estado e definição de uma recompensa que soma conversão, CTR e penalização de gasto.
3. **Ambiente de simulação**: construção de um contextual bandit que amostra experiências históricas por par estado-ação.
4. **Treinamento e avaliação**: Q-Learning com 600 episódios e comparação com política aleatória (valores médios de recompensa e taxa de conversão).

## Principais resultados observados
- O agente ganancioso aprendeu a priorizar PPC (27,7%) e SEO (23,8%), canais que apresentaram melhor relação investimento–conversão.
- A recompensa média subiu de 2.928 para 3.004 pontos (+75) e a taxa média de conversão aumentou de 0,879 para 0,901 (+0,022).
- A curva de recompensa acumulada estabilizou após ~400 episódios, indicando convergência do treinamento tabular.

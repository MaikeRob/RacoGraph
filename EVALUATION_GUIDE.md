# 📊 Guia de Avaliação do RacoGraph

Este guia explica como usar o sistema de avaliação offline para medir a qualidade das recomendações geradas pelo RacoGraph.

---

## 🎯 O que é Avaliação Offline?

A avaliação offline permite testar o sistema de recomendação **sem usuários reais**, usando dados históricos. O processo é:

1. **Dividir dados**: Separa avaliações em TREINO e TESTE
2. **Treinar**: Constrói o grafo usando apenas dados de TREINO
3. **Recomendar**: Gera recomendações para cada usuário
4. **Avaliar**: Compara recomendações com o que o usuário **realmente** fez no TESTE

---

## 🚀 Como Usar

### Uso Básico

```bash
python eval.py
```

Executa avaliação padrão:
- Top-10 recomendações
- 1000 caminhadas aleatórias
- Último filme de cada usuário no teste
- Nota mínima de 3.0

### Exemplos Práticos

#### 1️⃣ Avaliar Top-20 recomendações
```bash
python eval.py --k 20
```

#### 2️⃣ Aumentar precisão (mais caminhadas)
```bash
python eval.py --num-walks 5000
```

#### 3️⃣ Caminhadas mais longas
```bash
python eval.py --walk-length 15
```

#### 4️⃣ Considerar apenas notas altas
```bash
python eval.py --min-user-rating 4.0
```

#### 5️⃣ Split aleatório (20% teste)
```bash
python eval.py --split random --test-frac 0.2
```

#### 6️⃣ Avaliação completa otimizada
```bash
python eval.py --k 10 --num-walks 2000 --walk-length 15 --min-user-rating 3.5
```

---

## 📊 Entendendo as Métricas

### 🎯 Precision@K
**O que mede:** "Das K recomendações, quantas foram relevantes?"

**Fórmula:**
```
Precision@10 = Nº de hits / 10
```

**Exemplo:**
- Sistema recomenda 10 filmes
- Usuário assistiu 2 deles no conjunto de teste
- **Precision@10 = 0.20 (20%)**

**Interpretação:**
- ✅ `> 0.10`: Bom
- ⚠️ `0.05-0.10`: Moderado
- ❌ `< 0.05`: Ruim

---

### 🎯 Recall@K
**O que mede:** "Dos filmes que o usuário gostou, quantos foram recomendados?"

**Fórmula:**
```
Recall@10 = Nº de hits / Total de filmes relevantes
```

**Exemplo:**
- Usuário tem 5 filmes no teste
- Sistema acertou 2 deles no top-10
- **Recall@10 = 0.40 (40%)**

**Interpretação:**
- ✅ `> 0.30`: Bom
- ⚠️ `0.15-0.30`: Moderado
- ❌ `< 0.15`: Ruim

---

### 🎯 MAP@K (Mean Average Precision)
**O que mede:** "Quão bem os itens relevantes estão posicionados no ranking?"

**Por que é importante:** É a métrica mais importante! Não basta recomendar certo, precisa recomendar certo nas **primeiras posições**.

**Exemplo:**
```
Cenário A:
1. Matrix ✓
2. Inception ✓
MAP = Alto (relevantes no topo)

Cenário B:
1. Filme X ✗
2. Filme Y ✗
...
9. Matrix ✓
10. Inception ✓
MAP = Baixo (relevantes no fim)
```

**Interpretação:**
- ✅ `> 0.15`: Excelente
- ⚠️ `0.10-0.15`: Bom
- ❌ `< 0.10`: Precisa melhorar

---

### 🎯 NDCG@K (Normalized Discounted Cumulative Gain)
**O que mede:** "Quão próximo o ranking está do ideal?"

**Conceito:** Itens em posições mais baixas têm desconto logarítmico.

**Interpretação:**
- ✅ `> 0.25`: Excelente
- ⚠️ `0.15-0.25`: Bom
- ❌ `< 0.15`: Precisa melhorar

---

### 🎯 HitRate@K
**O que mede:** "Percentual de usuários que receberam pelo menos 1 recomendação relevante"

**Interpretação:**
- ✅ `> 0.70`: Ótimo (70% dos usuários têm hits)
- ⚠️ `0.50-0.70`: Moderado
- ❌ `< 0.50`: Ruim

---

### 🎯 Coverage
**O que mede:** "Diversidade do catálogo recomendado"

**Fórmula:**
```
Coverage = Nº de filmes recomendados / Total de filmes
```

**Interpretação:**
- ✅ `> 0.20`: Boa diversidade
- ⚠️ `0.10-0.20`: Moderada
- ❌ `< 0.10`: Muito focado em blockbusters

---

## 🔬 Experimentos Sugeridos

### 1. Impacto do Número de Caminhadas
```bash
python eval.py --num-walks 500
python eval.py --num-walks 1000
python eval.py --num-walks 2000
python eval.py --num-walks 5000
```

**Hipótese:** Mais caminhadas = maior MAP, mas mais lento

---

### 2. Impacto do Comprimento da Caminhada
```bash
python eval.py --walk-length 5
python eval.py --walk-length 10
python eval.py --walk-length 15
python eval.py --walk-length 20
```

**Hipótese:** Caminhadas mais longas exploram mais o grafo

---

### 3. Impacto da Nota Mínima
```bash
python eval.py --min-user-rating 2.5
python eval.py --min-user-rating 3.0
python eval.py --min-user-rating 3.5
python eval.py --min-user-rating 4.0
```

**Hipótese:** Notas mais altas = recomendações mais precisas

---

### 4. Comparação de Modos de Split
```bash
# Temporal (último filme)
python eval.py --split last --holdout 1

# Aleatório (20%)
python eval.py --split random --test-frac 0.2
```

**Hipótese:** Split temporal é mais realista

---

## 📈 Interpretando Resultados

### Exemplo de Saída

```
============================================================
  RESULTADOS DA AVALIACAO - RacoGraph (Random Walk)
============================================================

📋 Configuração:
   Top-K               : 10
   Num Walks           : 1000
   Walk Length         : 10
   Min User Rating     : 3.0

📊 Métricas de Qualidade:

   Estatísticas Básicas:
   users_evaluated               : 610
   users_with_recs               : 598
   unique_movies_recommended     : 412

   Métricas de Ranking:
   Precision@10                  : 0.0845
   Recall@10                     : 0.4521
   MAP@10                        : 0.1289
   NDCG@10                       : 0.2145
   HitRate@10                    : 0.6721

   Outras Métricas:
   Coverage                      : 0.2187

============================================================

💡 Interpretação:
   ✅ MAP: Excelente - modelo ranqueia bem itens relevantes
   ⚠️  HitRate: Moderado - muitos usuários sem hits
   ✅ Coverage: Boa diversidade no catálogo
```

---

## 🎯 Boas Práticas

### ✅ DO (Faça)

1. **Execute múltiplas vezes** para ter certeza dos resultados
2. **Compare configurações** de forma sistemática
3. **Documente parâmetros** usados em cada experimento
4. **Analise trade-offs**: precisão vs. tempo de execução
5. **Considere Coverage**: diversidade é importante!

### ❌ DON'T (Não Faça)

1. **Não avalie apenas 1 métrica**: use MAP + HitRate + Coverage
2. **Não use apenas split aleatório**: temporal é mais realista
3. **Não ignore tempo**: 10000 walks pode ser muito lento
4. **Não otimize apenas para Precision**: pode sacrificar diversidade
5. **Não compare com dados de treino**: sempre use teste separado!

---

## 🔧 Troubleshooting

### Problema: "MAP muito baixo (< 0.05)"
**Solução:**
- Aumente `--num-walks`
- Aumente `--walk-length`
- Reduza `--min-user-rating`

### Problema: "HitRate muito baixo (< 0.40)"
**Solução:**
- Verifique se grafo está bem conectado
- Aumente `--k` (avaliar top-20 em vez de top-10)
- Considere usar split aleatório

### Problema: "Coverage muito baixa (< 0.10)"
**Solução:**
- Sistema está enviesado para filmes populares
- Considere adicionar diversificação no algoritmo
- Aumente `--walk-length` para explorar mais

### Problema: "Avaliação muito lenta"
**Solução:**
- Reduza `--num-walks` (teste com 500)
- Reduza `--walk-length` (teste com 5)
- Avalie subset de usuários para testes rápidos

---

## 📚 Referências

- **Precision/Recall**: Métricas padrão de Recuperação de Informação
- **MAP**: Padrão em sistemas de ranqueamento
- **NDCG**: Microsoft Research, 2000
- **Random Walk**: Personalized PageRank (Page et al., 1998)

---

## 💡 Próximos Passos

Após avaliar o sistema:

1. **Identifique pontos fracos** (qual métrica está baixa?)
2. **Teste hipóteses** (o que pode melhorar?)
3. **Ajuste parâmetros** (num_walks, walk_length, etc.)
4. **Re-avalie** e compare resultados
5. **Documente** configuração final escolhida

---

**Dúvidas?** Consulte o código em `eval.py` para detalhes de implementação.

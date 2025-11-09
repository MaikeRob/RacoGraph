# 🎯 Quick Start - Avaliação RacoGraph

## Comandos Essenciais

### 1. Avaliação Rápida (para testar)
```bash
python eval.py --k 5 --num-walks 200
```
⏱️ Tempo: ~30 segundos

### 2. Avaliação Padrão (recomendado)
```bash
python eval.py
```
⏱️ Tempo: ~2-3 minutos
📊 Configuração: k=10, num_walks=1000, walk_length=10

### 3. Avaliação de Alta Qualidade
```bash
python eval.py --num-walks 5000 --walk-length 15
```
⏱️ Tempo: ~10-15 minutos
📊 Melhor precisão

### 4. Ver todas as opções
```bash
python eval.py --help
```

---

## 📊 Como Interpretar os Resultados

### Resultado BOM ✅
```
MAP@10        : > 0.15
HitRate@10    : > 0.70
Coverage      : > 0.20
```

### Resultado MODERADO ⚠️
```
MAP@10        : 0.10 - 0.15
HitRate@10    : 0.50 - 0.70
Coverage      : 0.10 - 0.20
```

### Resultado RUIM ❌
```
MAP@10        : < 0.10
HitRate@10    : < 0.50
Coverage      : < 0.10
```

---

## 🔧 Como Melhorar Resultados

Se MAP está baixo:
```bash
# Aumente caminhadas
python eval.py --num-walks 3000

# Aumente comprimento
python eval.py --walk-length 15

# Combine ambos
python eval.py --num-walks 3000 --walk-length 15
```

Se Coverage está baixo:
```bash
# Use caminhadas mais longas para explorar mais
python eval.py --walk-length 20
```

Se HitRate está baixo:
```bash
# Avalie mais recomendações
python eval.py --k 20

# Reduza nota mínima
python eval.py --min-user-rating 2.5
```

---

## 📖 Documentação Completa

Consulte `EVALUATION_GUIDE.md` para:
- Explicação detalhada de cada métrica
- Guia de experimentos
- Troubleshooting
- Boas práticas

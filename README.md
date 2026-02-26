# 🚀 Crypto Trading Bot – Optimisation par Algorithme Génétique 🧬💹

Bienvenue dans mon **bot de trading automatique pour le Bitcoin** !  
Ce projet utilise **Python**, **pandas**, **matplotlib** et un **algorithme génétique** pour optimiser des stratégies de trading basées sur des indicateurs techniques.

---

## 🔥 Fonctionnalités

- 📊 **Téléchargement automatique des données BTC-USD**
- 📈 **Calcul d’indicateurs techniques** :
  - Moyennes mobiles (MA)
  - RSI
  - Volatilité
- 🤖 **Backtesting robuste**
  - Gestion du capital
  - Calcul du Sharpe Ratio
  - Analyse du drawdown
- 🧬 **Optimisation automatique des paramètres**
  - Algorithme génétique
  - Mutation et reproduction
- 🎨 **Visualisation interactive**
  - Evolution de l’équité du portefeuille
  - Distribution de la population
  - Drawdowns et performance

---

## 📷 Aperçu du Bot

### Évolution du portefeuille
![Equity Curve](images/equity_curve.png)

### Distribution de la fitness
![Fitness Distribution](images/fitness_distribution.png)

### Meilleurs paramètres trouvés
| Paramètre | Valeur |
|-----------|--------|
| MA Short  | 23     |
| MA Long   | 192    |
| RSI p     | 21     |
| RSI OB    | 70     |
| RSI OS    | 45     |
| Volatility| 0.009  |

---

## 🛠️ Installation

```bash
git clone https://github.com/theorick/algo_genetique_trading.git
cd crypto-trading-bot
pip install -r requirements.txt

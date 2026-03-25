import re

log_path = '/home/jasan/.pm2/logs/03-V30-Trainer-out-4.log'

# We want to extract blocks of Coliseum evaluations
# Example lines might be:
# 🏆 Champion:    0.1234 avg PnL
# ⚔️  Challenger:  0.1500 avg PnL | 55.0% WR
# 🛡️ Champion holds title.
# OR
# 🚀 PROMOTION! Challenger A (GPU:0) defeats Champion!

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

evals = []
current_eval = {}

for line in lines:
    if "🏆 Champion:" in line:
        current_eval = {'champ': line.strip()}
    elif "⚔️  Challenger" in line and current_eval:
        current_eval['challenger'] = line.strip()
    elif "🛡️ Champion holds" in line and current_eval:
        current_eval['result'] = "Defended"
        evals.append(current_eval)
        current_eval = {}
    elif "🚀 PROMOTION!" in line and current_eval:
        current_eval['result'] = "Promoted"
        current_eval['promo_line'] = line.strip()
        evals.append(current_eval)
        current_eval = {}

# Print the last 15 evaluations to form the table
for i, ev in enumerate(evals[-15:]):
    print(f"Eval {len(evals) - 15 + i + 1}:")
    print("  " + ev.get('champ', ''))
    print("  " + ev.get('challenger', ''))
    if ev.get('result') == 'Promoted':
        print("  Status: " + ev.get('promo_line', ''))
    else:
        print("  Status: 🛡️ Defended")
    print("-" * 40)

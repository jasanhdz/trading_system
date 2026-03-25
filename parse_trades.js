const fs = require('fs');
const files = ['/home/jasan/.pm2/logs/01-Trading-Bot-out.log', '/home/jasan/.pm2/logs/01-Trading-Bot-out-0.log'];

files.forEach(file => {
    try {
        const data = fs.readFileSync(file, 'utf8');
        const lines = data.split('\n');
        
        console.log(`--- Parse Results for ${file} ---`);
        
        lines.forEach(line => {
            if (line.includes('Position closed') || 
                line.includes('MACRO PANIC') || 
                line.includes('Entry signal detected') ||
                line.includes('Stop Loss Triggered')) {
                // filter noisy entry signals out unless they have high confidence
                if (line.includes('Entry signal detected') && !line.includes('confidence=8') && !line.includes('confidence=9') && !line.includes('confidence=7')) return;
                console.log(line);
            }
        });
    } catch (e) {
        console.log(`Could not read ${file}`);
    }
});

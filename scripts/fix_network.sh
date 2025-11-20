#!/bin/bash
# Script para diagnosticar y arreglar conectividad de red

echo "════════════════════════════════════════════════════════════════"
echo "  Network Diagnostic & Fix Script"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  Some fixes require root. Run with: sudo bash $0"
    echo ""
fi

echo "Step 1: Checking network interfaces..."
echo "──────────────────────────────────────────────────────────────"
ip link show lan0
echo ""

echo "Step 2: Checking current routes..."
echo "──────────────────────────────────────────────────────────────"
ip route show
echo ""

echo "Step 3: Testing gateway connectivity..."
echo "──────────────────────────────────────────────────────────────"
if ping -c 2 -W 2 192.168.1.1 > /dev/null 2>&1; then
    echo "✅ Gateway 192.168.1.1 is REACHABLE!"
    echo ""
    echo "Gateway is responding but internet may still have issues."
    echo "Testing internet connectivity..."
    if ping -c 2 -W 2 8.8.8.8 > /dev/null 2>&1; then
        echo "✅ Internet is WORKING!"
        echo ""
        echo "Your network is fine. Try running the script again:"
        echo "  python scripts/update_ml_candles.py"
        exit 0
    else
        echo "❌ Gateway responds but no internet access"
        echo "   Issue: Router can't reach internet (ISP problem?)"
    fi
else
    echo "❌ Gateway 192.168.1.1 is NOT REACHABLE"
    echo ""

    if [ "$EUID" -eq 0 ]; then
        echo "Step 4: Attempting to fix..."
        echo "──────────────────────────────────────────────────────────────"

        # Option A: Try to renew DHCP
        echo "→ Releasing DHCP lease..."
        dhclient -r lan0 2>/dev/null
        sleep 2

        echo "→ Requesting new DHCP lease..."
        dhclient lan0 2>/dev/null
        sleep 3

        echo "→ Testing gateway again..."
        if ping -c 2 -W 2 192.168.1.1 > /dev/null 2>&1; then
            echo "✅ Gateway is now REACHABLE!"
            echo ""
            echo "Testing internet..."
            if ping -c 2 -W 2 8.8.8.8 > /dev/null 2>&1; then
                echo "✅ Internet is WORKING!"
                exit 0
            fi
        else
            echo "❌ Still cannot reach gateway"
            echo ""
            echo "Possible causes:"
            echo "  1. Ethernet cable disconnected"
            echo "  2. Router/switch powered off"
            echo "  3. Router changed IP (no longer 192.168.1.1)"
            echo "  4. Network port/switch failure"
        fi
    else
        echo "Run with sudo to attempt automatic fixes:"
        echo "  sudo bash $0"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Alternative Solution: Use Tailscale Exit Node"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "If gateway cannot be fixed, route internet through Tailscale:"
echo ""
echo "1. On your MacBook (jasans-macbook-air):"
echo "   tailscale set --advertise-exit-node"
echo ""
echo "2. Approve exit node at: https://login.tailscale.com/admin/machines"
echo ""
echo "3. On this server:"
echo "   tailscale set --exit-node=100.97.222.46"
echo ""
echo "4. Test:"
echo "   curl https://api.binance.com/api/v3/ping"
echo ""

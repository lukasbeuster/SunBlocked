#!/bin/bash
# Point-Based Shadow Ray Casting Prototype Runner
# ==============================================

echo "🚀 POINT-BASED SHADOW RAY CASTING PROTOTYPE"
echo "============================================"
echo ""

# Check Python version
python3 --version || { echo "Python3 not found!"; exit 1; }

# Check if in correct directory
if [ ! -f "prototype_demo.py" ]; then
    echo "❌ Please run this script from the prototype_ray_casting directory"
    exit 1
fi

echo "🎮 Running synthetic demo..."
python3 prototype_demo.py

echo ""
echo "🧪 Running real data integration test..."
python3 test_with_real_data.py

echo ""
echo "✅ Prototype testing complete!"

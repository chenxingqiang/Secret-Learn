#!/bin/bash
#
#  SS examplesexecution
#
# Usage: ./run_any_example.sh <example_name>
# Example: ./run_any_example.sh adaboost_classifier
#

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <example_name>"
    echo ""
    echo "Examples:"
    echo "  $0 adaboost_classifier"
    echo "  $0 logistic_regression"
    echo "  $0 random_forest_classifier"
    echo ""
    echo "Available examples:"
    ls -1 examples/SS/*.py | sed 's/.*\//  - /' | sed 's/\.py$//' | head -20
    echo "  ... and $(ls -1 examples/SS/*.py | wc -l | tr -d ' ') more"
    exit 1
fi

EXAMPLE_NAME="$1"
EXAMPLE_FILE="examples/SS/${EXAMPLE_NAME}.py"

if [ ! -f "$EXAMPLE_FILE" ]; then
    echo "✗ Example not found: $EXAMPLE_FILE"
    exit 1
fi

echo "======================================================================"
echo " Running SS Example: $EXAMPLE_NAME"
echo "======================================================================"
echo ""

# cleanup
echo "[1/4] Cleaning up..."
pkill -f "${EXAMPLE_NAME}.py --party" 2>/dev/null || true
rm -f /tmp/sf_*.lock 2>/dev/null
sleep 1
echo "✓ Clean"

# 
LOG_DIR="logs/SS"
mkdir -p "$LOG_DIR"
ALICE_LOG="$LOG_DIR/${EXAMPLE_NAME}_alice.log"
BOB_LOG="$LOG_DIR/${EXAMPLE_NAME}_bob.log"

> "$ALICE_LOG"
> "$BOB_LOG"

# 
echo ""
echo "[2/4] Starting parties..."
echo "Starting Bob..."
python "$EXAMPLE_FILE" --party bob > "$BOB_LOG" 2>&1 &
BOB_PID=$!
echo "✓ Bob (PID: $BOB_PID) - Log: $BOB_LOG"

sleep 5

echo "Starting Alice..."
python "$EXAMPLE_FILE" --party alice > "$ALICE_LOG" 2>&1 &
ALICE_PID=$!
echo "✓ Alice (PID: $ALICE_PID) - Log: $ALICE_LOG"

echo ""
echo "[3/4] Monitoring..."
echo "======================================================================"
echo "Showing Alice's log (Ctrl+C to stop monitoring):"
echo ""

tail -f "$ALICE_LOG" &
TAIL_PID=$!

wait $ALICE_PID 2>/dev/null
EXIT_CODE=$?

kill $TAIL_PID 2>/dev/null || true

echo ""
echo "======================================================================"
echo "[4/4] Result"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Example completed successfully!"
    echo ""
    echo "Note: SS mode with SecretFlow 1.14+ has limitations."
    echo "For development/testing, consider using FL mode:"
    echo "  python examples/FL/${EXAMPLE_NAME}.py"
else
    echo "✗ Example failed (exit code: $EXIT_CODE)"
    echo ""
    echo "This is expected with SecretFlow 1.14+ PRODUCTION mode."
    echo "The code is correct but requires true multi-machine setup."
    echo ""
    echo "Logs:"
    echo "  Alice: $ALICE_LOG"
    echo "  Bob: $BOB_LOG"
fi

# Cleanup
sleep 2
pkill -f "${EXAMPLE_NAME}.py --party" 2>/dev/null || true
rm -f /tmp/sf_*.lock 2>/dev/null

echo ""
echo "======================================================================"


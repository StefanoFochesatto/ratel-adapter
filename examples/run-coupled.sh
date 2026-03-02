#!/bin/bash
#
# Run Ratel-Dummy coupled simulation
# Launches both participants and waits for completion
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
PRECICE_CONFIG="${SCRIPT_DIR}/precice-config-ratel-dummy.xml"
RATEL_YAML="/home/stefano/ratel/examples/ymls/ex02/neo-hookean-current-beam.yml"
DUMMY_SOLVER="/home/stefano/precice-3.3.1/examples/solverdummies/c/solverdummy"
RATEL_EXAMPLE="${SCRIPT_DIR}/../build/examples/ex02-quasistatic-precice"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Ratel-Dummy Coupled Simulation${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Check if executables exist
if [ ! -f "$RATEL_EXAMPLE" ]; then
    echo -e "${RED}Error: Ratel example not found at $RATEL_EXAMPLE${NC}"
    echo "Please build the adapter and example first:"
    echo "  cd /home/stefano/ratel/ratel-adapter/build"
    echo "  make ex02-quasistatic-precice"
    exit 1
fi

if [ ! -f "$DUMMY_SOLVER" ]; then
    echo -e "${RED}Error: Dummy solver not found at $DUMMY_SOLVER${NC}"
    echo "Please build the dummy solver first:"
    echo "  cd /home/stefano/precice-3.3.1/examples/solverdummies/c"
    echo "  make"
    exit 1
fi

if [ ! -f "$PRECICE_CONFIG" ]; then
    echo -e "${RED}Error: preCICE config not found at $PRECICE_CONFIG${NC}"
    echo "Make sure you're running from the examples directory"
    exit 1
fi

# Clean up any previous preCICE runs
echo -e "${YELLOW}Cleaning up previous preCICE runs...${NC}"
rm -rf precice-run/

echo ""
echo -e "${GREEN}Starting coupled simulation...${NC}"
echo "  Config: $PRECICE_CONFIG"
echo "  Time windows: 5 steps of 0.1s"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ -n "$RATEL_PID" ]; then
        kill $RATEL_PID 2>/dev/null || true
    fi
    if [ -n "$DUMMY_PID" ]; then
        kill $DUMMY_PID 2>/dev/null || true
    fi
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT SIGTERM

# Create log directory
mkdir -p logs

# Start SolverOne (acceptor - must start first)
echo -e "${BLUE}[1/2] Starting SolverOne (Ratel - acceptor)...${NC}"
$RATEL_EXAMPLE \
    -precice_config $PRECICE_CONFIG \
    -options_file $RATEL_YAML \
    -quiet \
    > logs/solverone.log 2>&1 &
RATEL_PID=$!
echo "  PID: $RATEL_PID"
echo "  Log: logs/solverone.log"

# Give SolverOne a moment to create the socket
sleep 2

# Start SolverTwo (connector)
echo -e "${BLUE}[2/2] Starting SolverTwo (Dummy - connector)...${NC}"
$DUMMY_SOLVER \
    $PRECICE_CONFIG \
    SolverTwo \
    > logs/solvertwo.log 2>&1 &
DUMMY_PID=$!
echo "  PID: $DUMMY_PID"
echo "  Log: logs/solvertwo.log"

echo ""
echo -e "${GREEN}Both participants running. Waiting for completion...${NC}"
echo "  Press Ctrl+C to stop"
echo ""

# Wait for both processes to complete
wait $RATEL_PID
RATEL_EXIT=$?
wait $DUMMY_PID
DUMMY_EXIT=$?

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Simulation Complete${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Check exit status
if [ $RATEL_EXIT -eq 0 ] && [ $DUMMY_PID -eq 0 ]; then
    echo -e "${GREEN}Success! Both participants completed normally.${NC}"
    echo ""
    echo "Output logs:"
    echo "  SolverOne:  logs/solverone.log"
    echo "  SolverTwo:  logs/solvertwo.log"
    exit 0
else
    echo -e "${RED}Error: One or more participants failed.${NC}"
    echo "  SolverOne exit code: $RATEL_EXIT"
    echo "  SolverTwo exit code: $DUMMY_EXIT"
    echo ""
    echo "Check the log files for details:"
    echo "  SolverOne:  logs/solverone.log"
    echo "  SolverTwo:  logs/solvertwo.log"
    exit 1
fi

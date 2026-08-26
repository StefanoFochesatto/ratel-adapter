#!/usr/bin/env bash
set -e -u

# Tutorial logging setup
# (Assuming script is moved to tutorials/elastic-tube-3d-absolute/solid-ratel/)
if [ -f ../../tools/log.sh ]; then
    . ../../tools/log.sh
    exec > >(tee --append "$LOGFILE") 2>&1
fi

# Configuration
# (Update this path to point to your ratel-adapter build)
RATEL_ADAPTER_DIR="/home/stefano/ratel-adapter"
SOLVER_BIN="${RATEL_ADAPTER_DIR}/build/examples/ex03-dynamic-precice"
PRECICE_CONFIG="../precice-config.xml"
RATEL_CONFIG="tube.yml"

# Colors for output
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Ratel-preCICE Elastic Tube 3D${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check if executable exists
if [ ! -f "$SOLVER_BIN" ]; then
    echo -e "${RED}Error: Ratel-preCICE solver not found at $SOLVER_BIN${NC}"
    echo "Please build the ratel-adapter first."
    exit 1
fi

# 1. Generate mesh if needed
if [ ! -f "tube.msh" ] || [ "tube.geo" -nt "tube.msh" ]; then
    echo -e "${YELLOW}Generating 3D mesh from tube.geo...${NC}"
    gmsh -3 tube.geo -o tube.msh
fi

# 2. Run the Solid participant
echo -e "${BLUE}Starting Solid participant (Ratel)...${NC}"
 $SOLVER_BIN \
    -precice_config $PRECICE_CONFIG \
    -options_file $RATEL_CONFIG \
    -precice_participant Solid \
    -precice_mesh Solid-Mesh \
    -precice_read_data Force \
    -precice_write_data DisplacementDelta \
    -precice_boundary_label "Face Sets" \
    -precice_boundary_value 2 \
    -dim 3 \
    -dm_plex_gmsh_interpolate true \
    -ts_type alpha2 \
    -ts_alpha_alpha_m 0.5 \
    -ts_alpha_alpha_f 0.5 \
    -ts_alpha_gamma 0.5 \
    -ts_alpha_beta 0.25 \
    -snes_rtol 1e-6 \
    "$@"

if [ -f ../../tools/log.sh ]; then
    close_log
fi

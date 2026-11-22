#!/bin/sh
# Regenerate using shaderc 2026.1 and SPIRV-Tools 2026. Vulkan 1.0 requires only core compute.
set -eu
cd "$(dirname "$0")"
for shader in *.comp; do
    output=${shader%.comp}.spv
    glslc --target-env=vulkan1.0 -O -fshader-stage=compute "$shader" -o "$output"
    spirv-val --target-env vulkan1.0 "$output"
done

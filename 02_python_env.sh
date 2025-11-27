#!/bin/bash
set -e

GREEN='\033[0;32m'
NC='\033[0m'

# Nombre del directorio del proyecto
PROJECT_DIR="reconocimiento_facial"

echo -e "${GREEN}=== INICIANDO INSTALACIÓN DE ENTORNO PYTHON ===${NC}"

# 1. Crear directorio del proyecto
if [ ! -d "$PROJECT_DIR" ]; then
  echo -e "Creando directorio $PROJECT_DIR..."
  mkdir "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# 2. Crear entorno virtual
if [ ! -d "venv" ]; then
  echo -e "${GREEN}[1/3] Creando entorno virtual (venv)...${NC}"
  python3 -m venv venv
else
  echo -e "${GREEN}[1/3] El entorno virtual ya existe.${NC}"
fi

# Definir la ruta al pip del entorno virtual
PIP_CMD="./venv/bin/pip"

# 3. Actualizar pip y herramientas básicas
echo -e "${GREEN}[2/3] Actualizando pip, setuptools y wheel...${NC}"
$PIP_CMD install --upgrade pip setuptools wheel

# 4. Instalar librerías pesadas (Orden específico para evitar errores)
echo -e "${GREEN}[3/3] Instalando librerías de IA (Esto puede tardar 10-20 minutos)...${NC}"

echo "Instalando Numpy y Cython..."
$PIP_CMD install numpy cython

echo "Instalando OnnxRuntime..."
# Intentamos instalar la versión estándar. Si falla en tu Pi específica,
# habría que buscar el wheel específico, pero suele funcionar en Pi OS 64-bit.
$PIP_CMD install onnxruntime

echo "Instalando InsightFace (Paciencia, compilando C++)..."
$PIP_CMD install insightface

echo "Instalando OpenCV y utilidades..."
$PIP_CMD install opencv-python

echo -e "${GREEN}=== ¡INSTALACIÓN COMPLETADA! ===${NC}"
echo -e "Para empezar a trabajar, usa los siguientes comandos:"
echo -e "${GREEN}cd $PROJECT_DIR${NC}"
echo -e "${GREEN}source venv/bin/activate${NC}"

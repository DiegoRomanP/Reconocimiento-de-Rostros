#!/bin/bash
set -e # Detener el script si hay algún error

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== INICIANDO CONFIGURACIÓN DE SISTEMA RASPBERRY PI ===${NC}"

# 1. Verificar si se está ejecutando como root
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}Por favor, ejecuta este script con sudo:${NC} sudo ./01_system_setup.sh"
  exit
fi

# 2. Actualizar el sistema
echo -e "${GREEN}[1/4] Actualizando lista de paquetes y sistema...${NC}"
apt update && apt upgrade -y

# 3. Instalar dependencias de sistema (necesarias para compilar InsightFace y OpenCV)
echo -e "${GREEN}[2/4] Instalando dependencias de compilación y librerías de imagen...${NC}"
apt install -y build-essential cmake pkg-config libjpeg-dev libtiff5-dev libpng-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev python3-venv

# 4. Aumentar memoria SWAP (Vital para compilar en Pi)
echo -e "${GREEN}[3/4] Configurando memoria SWAP a 2048MB...${NC}"
# Hacemos backup del archivo original
cp /etc/dphys-swapfile /etc/dphys-swapfile.bak

# Cambiamos el tamaño de swap usando sed
sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile

# Reiniciamos el servicio de swap
/etc/init.d/dphys-swapfile stop
/etc/init.d/dphys-swapfile start

echo -e "${GREEN}[4/4] Configuración de sistema completada.${NC}"
echo -e "${YELLOW}NOTA: Recuerda restaurar el SWAP a 100MB cuando termines todo el proyecto para cuidar tu SD.${NC}"
echo -e "Ahora ejecuta el segundo script como usuario normal (sin sudo)."

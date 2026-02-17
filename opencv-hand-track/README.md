Hand Detection & Gesture Recognition (MediaPipe Tasks)

Este projeto utiliza a nova arquitetura MediaPipe Tasks do Google para realizar o rastreamento de mãos em tempo real via webcam e identificar gestos através da geometria das mãos com Python e OpenCV.
🚀 Funcionalidades

    Detecção de até 2 mãos simultaneamente.

    Classificação de lateralidade (Mão Esquerda / Direita).

    Desenho manual de landmarks e conexões sem dependências legado.

    Estrutura de código orientada a objetos (Classe DetectHands).

🛠️ Pré-requisitos

Antes de rodar, instale as bibliotecas necessárias:
Bash

pip install opencv-python mediapipe numpy

📦 Configuração do Modelo

Diferente das versões antigas, a API de Tasks exige o download manual do modelo de IA:

    Baixe o arquivo hand_landmarker.task.

    Certifique-se de que o arquivo esteja na raiz do projeto (mesma pasta do script .py).

💻 Como usar

Basta executar o arquivo principal:
Bash

python hand-tracking.py

    Pressione 'q' para fechar a janela da câmera.

🖐️ Gestos Implementados

O projeto identifica os seguintes estados de dedos, e com alguns testes comentados:

    [x] Polegar

    [x] Indicador

    [x] Médio

    [x] Anelar

    [x] Mínimo

# Reinforcement Learning - Snake game

Este proyecto implementa un agente de aprendizaje por refuerzo que juega el juego de la serpiente (Snake). El agente utiliza una red neuronal para aprender a jugar y mejorar su puntaje.

## Ejecución
Para ejecutar el programa, se debe ejecutar el siguiente comando en la línea de comandos:

```console
python agent.py
```

## Librerías requeridas
Este proyecto requiere las siguientes librerías:

[PyTorch](https://pytorch.org/)
[torchvision](https://pytorch.org/vision/)
[matplotlib](https://matplotlib.org/)
[ipython](https://ipython.org/)
[pygame](https://www.pygame.org/)

Para instalar estas librerías, se pueden usar los siguientes comandos pip:

```console
pip install torch torchvision matplotlib ipython pygame
```

Se recomienda crear un entorno virtual antes de instalar las librerías. Por ejemplo, con venv se puede crear un entorno virtual de la siguiente manera:

```console
python3 -m venv env
source env/bin/activate  # en Linux o macOS
.\env\Scripts\activate  # en Windows
```

## Uso
Al ejecutar agent.py, el agente comenzará a jugar automáticamente. Durante el juego, se mostrará la puntuación actual del agente y se actualizará a medida que la serpiente come más alimentos. Si el agente pierde, se reiniciará automáticamente para comenzar un nuevo juego.

Se puede ejecutar el juego manual que está en la carpeta snake_manual_game con el comando

```console
python snake_game.py
```

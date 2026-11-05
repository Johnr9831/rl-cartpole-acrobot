# Proyecto individual: CartPole + Acrobot

## Herramientas
Python 3.11 de 64 bits, terminal PowerShell y opcionalmente VS Code y Git.
No necesitas CUDA, Colab, cuentas de servicios ni una API key. CPU es el valor predeterminado.
Extrae el ZIP y abre una terminal dentro de esta carpeta (donde está run.py).

## Instalación en Windows
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
En Linux: python3 -m venv .venv; usa .venv/bin/python en lugar del ejecutable de Windows.
Los rangos de dependencias permiten instalación; cada ejecución guarda requirements_lock.txt con las versiones exactas. Para reproducir usa ese archivo en un entorno limpio.

## Verificación corta (no es evidencia científica)
```powershell
.\.venv\Scripts\python.exe run.py --smoke
.\.venv\Scripts\python.exe plot.py --root results_smoke
.\.venv\Scripts\python.exe video.py results_smoke/cart_dqn/seed_11
```
Ejecuta las diez configuraciones con tres semillas y solo 1200 pasos. Aunque aprenda poco, prueba actualizaciones, escritura, evaluación y generación de figuras. Puede tardar minutos por las evaluaciones.

## Experimentos completos
```powershell
.\.venv\Scripts\python.exe run.py --suite tier1
.\.venv\Scripts\python.exe run.py --suite tier2
.\.venv\Scripts\python.exe plot.py
.\.venv\Scripts\python.exe video.py results/cart_dqn/seed_11
.\.venv\Scripts\python.exe video.py results/ac_base/seed_11
```
También puedes ejecutar ambos tiers con `run.py --suite all`.
Para una configuración: `run.py --experiment ac_base` (tres semillas).
Para una medición inicial: `run.py --experiment ac_base --seeds 11`.
Si luego ejecutas tier2, la ejecución completa existente se omite.
Reproduce cada figura: tier1 requiere --suite tier1; las otras tres requieren --suite tier2; luego plot.py. Para reproducir desde cero usa --out results_repeat y plot.py --root results_repeat.

## Diseño experimental
- Semillas 11,22,33 para Python, NumPy, PyTorch y entorno.
- CartPole: 200000 pasos por ejecución en ambos algoritmos; bins 6,10,20 por dimensión. Rangos: posición ±2.4, velocidad ±3, ángulo ±0.2095 y velocidad angular ±3.5. Valores externos caen en los bins extremos; esta saturación es una limitación a discutir.
- Acrobot: 300000 pasos por ejecución; es un presupuesto inicial, no garantía de éxito.
- DQN: dos capas de 64 unidades, ReLU, Adam, Huber, buffer 50000, minibatch 64, gamma .99, una actualización cada cuatro pasos tras 1000 pasos iniciales.
- Red objetivo copiada cada 1000 pasos. Ablación: se usa la red online para calcular el objetivo, sin gradiente; todo lo demás permanece igual.
- Exploración epsilon lineal de 1 a .05 durante la mitad del presupuesto y luego constante. No se presenta como GLIE.
- Estudio de learning rate: .0003,.001,.003. Estudio de fracción de decaimiento: .25,.5,.75. Una variable por vez y base compartida.
- Evaluación cada 10000 pasos con 10 episodios greedy; entorno separado y semillas fijas de evaluación para comparabilidad. Evaluación final con 100 episodios y semillas distintas.
- Curvas: media ± una desviación estándar ENTRE semillas; no es intervalo de confianza.
- first_threshold_10ep es solo primer cruce provisional basado en 10 episodios. La columna success verifica el umbral en los 100 episodios finales; no prueba mantenimiento del rendimiento.
- Umbrales del enunciado: CartPole 475; Acrobot -100. Los retornos de entrenamiento y evaluación se guardan separados.
- Reiniciar omite ejecuciones terminadas, pero una ejecución interrumpida se reinicia desde cero. Los modelos finales no son checkpoints de reanudación.
- Cambia presupuestos y parámetros en configs() de run.py ANTES de lanzar; usa otra carpeta --out para un diseño nuevo. No mezcles resultados de protocolos diferentes.

## Archivos producidos
results/<experimento>/seed_<n>/ contiene config.json, training.csv, evaluation.csv, summary.json, model.pt o q.npy. video.py añade agent.gif.
results/figures contiene cuatro gráficos; summary.csv agrega resultados, dispersión, tiempo y umbrales.
summary.json conserva los 100 retornos finales, tiempos total/evaluación/entrenamiento y semilla.
hardware.json registra sistema y dispositivo; agrega manualmente modelo de CPU, RAM y modo energético si el sistema no los identifica.
El tiempo de entrenamiento excluye evaluaciones; wall_seconds incluye evaluación final y guardado del modelo. Registra ambos al comparar.

## Qué esperar
El programa muestra nombre, semilla, pasos y retorno medio. No abre ventanas durante entrenamiento. CartPole puede mejorar hacia 500 y Acrobot hacia valores menos negativos; las curvas pueden oscilar y ciertas configuraciones pueden fallar. Ningún umbral está garantizado.
Los 30 entrenamientos suman 7.8 millones de pasos de entrenamiento, más evaluación. Mide primero una ejecución para estimar el tiempo real. Los tiempos anteriormente discutidos son orientativos.
No selecciones solo la mejor semilla para las tablas. Los GIF usan una semilla fija ilustrativa y no sustituyen evaluaciones.

## Entrega académica pendiente
Este paquete proporciona infraestructura experimental, NO un artículo con resultados ya obtenidos.
Después de entrenar faltan interpretar resultados, escribir el PDF de 6–8 páginas a doble columna, análisis teórico de 1–2 páginas, declaración de IA y apéndice con prompts, repositorio Git y defensa.
Estructura del artículo: título, resumen, introducción, pertinencia, métodos, configuración experimental, resultados, discusión, análisis de acciones continuas/píxeles, limitaciones, declaración IA y referencias verificadas.
No inventes cifras ni resultados. Guarda los prompts utilizados, incluyendo la solicitud de este paquete.
La defensa exige explicar discretización, Bellman, epsilon, replay, red objetivo, terminal vs truncamiento, evaluación y bandas entre semillas.

## Referencias para estudiar
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://gymnasium.farama.org/environments/classic_control/cart_pole/
https://gymnasium.farama.org/environments/classic_control/acrobot/

## Diagnóstico de CartPole y gráficos revisados

Ejecutar desde la raíz del proyecto con el entorno virtual activo.
El primer comando reproduce el diagnóstico de una semilla con la tasa reducida.
El segundo reproduce el ajuste con tres semillas y copia de la red objetivo cada 10 000 pasos.

```powershell
python -c "from pathlib import Path; from run import configs, run; c=configs()['cart_dqn']; c['lr']=0.0001; run('cart_dqn_lr0001', c, 11, Path('results_diagnostico'), False)"
python -c "from pathlib import Path; from run import configs, run; c=configs()['cart_dqn']; c['lr']=0.0001; c['target_every']=10000; [run('cart_dqn_lr0001_target10000', c, seed, Path('results_diagnostico'), False) for seed in [11,22,33]]"
python plot_revision.py
```

Los gráficos revisados y la tabla se guardan en `resultados_revisados/`.
El informe en inglés está en `report/`, con `main.tex` y las figuras para compilarlo con pdfLaTeX.
Las demostraciones de CartPole y Acrobot están en `evidencias/`.

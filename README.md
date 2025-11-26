## Instalación
### Entorno virtual (ubuntu)
```bash
# Instalar módulo de entronos virtuales
sudo apt install python3-venv
# Crear entorno virtual (posicionarse en la raís del proyecto)
python3 -m venv .venv
# Activar entorno virtual
source .venv/bin/activate
# Instalar requerimentos (verificar que torch y torchvision estén comentados)
pip install -r requirements.txt
```

### Instalar Torch
Buscar comando [acá](https://pytorch.org/get-started/locally/). Por ejemplo para linux usando "CPU" como plataforma:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```


## Ejecución

### Ejecutar preprocesamiento
```bash
# Habilitar permisos de ejecución para el archivo
chmod +x run_preprocess.sh
# Ejecutar
./run_preprocess.sh
```


### Ejecutar entrenamiento
Modificar el campo `device` si hace falta en el [archivo de configuración](/configs/baselineRNN.yaml) para que se corresponda con el *device* de *torch* seleccionado en la instalación.

```yaml
device: cpu
```

Se puede ejecutar con o sin los **checkpoints del entrenamiento**, modificando la variable `store_checkpoints` en el [archivo de configuración](/configs/baselineRNN.yaml). Tener en cuenta que unos cinco checkpoints pueden llegar a ocupar un total de 1GB, por lo tanto entrenar 100 *epochs* podría llegar a escribir **20GB en disco**. 

```yaml
store_checkpoints: False
```

```bash
# Habilitar permisos de ejecución para el archivo
chmod +x run_train.sh
# Ejecutar
./run_train.sh
```



## Convenciones de etiquetado
- Puntuacion inicial: {'¿', None} --> 0/1
- Puntuacion final: {None, ',', '.', '?'} --> ints{0, 1, 2, 3}
- Capitalizacion: {todo minuscula, primera letra en mayuscula, algunas pero no todas en mayuscula, todo en mayuscula} --> {0, 1, 2, 3}


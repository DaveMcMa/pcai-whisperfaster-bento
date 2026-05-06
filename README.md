This is how I created this bento:

1st step - create environment -

    python -m venv ~/.venvs/example
    source ~/.venvs/example/bin/activate
    pip install --upgrade pip
    pip install [requirements.txt]
    python -m ipykernel install --user --name examplekernel --display-name "Python (examplekernel)"

2nd step - create bento artifacts - 
    requirements.txt
    service.py
    bentofile.yaml

3rd step - build & export bento file - 
    bentoml build
    bentoml export

4th step - upload the exported bento to local S3 & serve (see s3upload notebook)

5th step - deploy from MLIS - configure with following env variables (need to ensure mmpeg is available at runtime):
    AIOLI_COMMAND_OVERRIDE : export LD_LIBRARY_PATH=$(find /mnt/models -name "libcublas.so.12" -exec dirname {} \; 2>/dev/null | head -1):$(find /mnt/models -name "libcudnn.so.9" -exec dirname {} \; 2>/dev/null | head -1):$LD_LIBRARY_PATH
    HF_TOKEN : [your token here - it's a gated repo]
    LD_LIBRARY_PATH : /mnt/models/rundmc-whisper-faster-bento.vX/.virtualenv/lib/python3.11/site-packages/nvidia/cublas/lib:/mnt/models/rundmc-whisper-faster-bento.vX/.virtualenv/lib/python3.11/site-packages/nvidia/cudnn/lib

6th step - test using notebook

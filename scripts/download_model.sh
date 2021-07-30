if [ $# -eq 0 ]; then
    echo "Please provide model name. Possibilities are:"
    for dataset in imagenet1k imagenet21k chexpert chexpert_mimic chexpert_mimic_nih chexpert_mimic_nih_padchest;do
        for model in bit50x1 bit152x4;do
            echo "${dataset}_${model}"
        done
    done
    exit 1
fi
model_name=$1
url="https://fz-juelich.sciebo.de/s/9kqbf6yBDyJnQfd/download?path=%2F${model_name}&files=model.pth.tar"
mkdir -p pretrained_models/$model_name
wget $url -O pretrained_models/$model_name/model.pth.tar

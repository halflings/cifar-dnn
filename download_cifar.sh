DATA_DIR="data"
CIFAR_10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_PATH=$DATA_DIR/cifar-10.tar.gz
# curl -o $TAR_PATH $CIFAR_10_URL
mkdir -p $DATA_DIR
tar -C $DATA_DIR -xzf $TAR_PATH
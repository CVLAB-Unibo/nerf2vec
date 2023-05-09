from classification.train_nerf2vec import Nerf2vecTrainer

def train_nerf2vec():
    nerf2vec = Nerf2vecTrainer()
    nerf2vec.train()

if __name__ == '__main__':
    train_nerf2vec()

.PHONY: train test ipt clp show clean train-clean

mode = CIFAR100
epochs = 20
latent_dim = 16
hidden_dim = 16
batch_size = 128

train:
	python main.py --do_train --num_epochs $(epochs) --latent_dim $(latent_dim) --generator_hidden_dim $(hidden_dim) --discriminator_hidden_dim $(hidden_dim) --batch_size $(batch_size) --mode $(mode) --notes $(mode)

test:
	python main.py --latent_dim $(latent_dim) --generator_hidden_dim $(hidden_dim) --discriminator_hidden_dim $(hidden_dim) --batch_size $(batch_size) --mode $(mode) --notes $(mode)

ipt:
	python main.py --interpolation --latent_dim $(latent_dim) --generator_hidden_dim $(hidden_dim) --discriminator_hidden_dim $(hidden_dim) --batch_size $(batch_size) --mode $(mode) --notes $(mode)

clp:
	python main.py --collapse --latent_dim $(latent_dim) --generator_hidden_dim $(hidden_dim) --discriminator_hidden_dim $(hidden_dim) --batch_size $(batch_size) --mode $(mode) --notes $(mode)

show:
	tensorboard --logdir="./log"

clean:
	rm ./log/**/events.*

train-clean:
	rm -rf ./train/**/
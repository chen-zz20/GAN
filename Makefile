.PHONY: clean show

show:
	tensorboard --logdir="./log"

clean:
	rm ./log/train/events.* ./log/test/events.*
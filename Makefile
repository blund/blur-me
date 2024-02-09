
pre:
	OPENCV_OPENCL_RUNTIME=null python pre.py
train:
	python train.py
run:
	python run.py
predict:
	python predict.py
db:
	python -i db.py

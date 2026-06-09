# Build TensorRT Engine By TensorRT Python API

When you want to build engine by API. You should generate the pickle weights parameters first.

```shell
python3 gen_pkl.py -w yolov8s.pt -o yolov8s.pkl
```

You will get a `yolov8s.pkl` which contain the operators' parameters.

And you can rebuild `yolov8s` model in TensorRT api.

```
python3 build.py \
--weights yolov8s.pkl \
--iou-thres 0.65 \
--conf-thres 0.25 \
--topk 100 \
--fp16  \
--input-shape 1 3 640 640 \
--device cuda:0
```

***Notice !!!***

Now we only support static input shape model build by TensorRT api.
You'd best give the legal `input-shape`.

***Notice !!!***

Now we don't support YOLOv8-seg model building by API. It will be supported later.

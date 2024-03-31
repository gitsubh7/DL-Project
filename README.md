# Bird's Eye View



## Requirements

Please run the following code to install the Requirements.

`pip install -r requirements.txt`


## Preparation and Run the code:

1. Download the models from [here](https://docs.google.com/uc?export=download&id=1EaBmCzl4xnuebfoQnxU1xQgNmBy7mWi2) and place them under `weights/`.
2. Test on your sample video and see the demo results using the command mentioned in the next line:
```bash
$ python3 main.py --source test_video.mp4 [--view] [--save]
```
3. If you use the `--save` argument, the output will be saved in the `inference/output` folder.

# Door open and close detect
## Run the Code
```
/
├── Tests/               (The 5 tests videos)
├── solution/output.json (The output we produce)
├── requirements.txt     (Environment setting)
├── straight_github.py   (The main code to produce output)
├── train.ipynb          (Train model 'yolonano.pt')
├── yolonano.pt          (The model we use in straight_github.py)

```

## Environment
1. MacOS (M2 chip)  
2. python = 3.12.0
3. The recommended way to install the environment is by running the command below:

```
conda create -n {env name} python=3.12
conda activate {env name} 
pip install -r requirements.txt
(or pip3 install -r requirements.txt)
```
4. Check there are any error messages or not!

Now, you have finished environment setting!

## Run the Code
1. Please fill in the input video path in straight_github.py (line464), for example:

```
input_videos = ['Tests/01.mp4',
                'Tests/03.mp4',
                'Tests/05.mp4',
                'Tests/07.mp4',
                'Tests/09.mp4']
```
2. run straight_github.py (this may take 5-10 minutes because of inference)
```
python3 straight_github.py
```
3. Then you can get output.json 

## Output
Put output.json into a file named solution.
Upload solution.zip to codalalab to display the ranking.

solution/
├── output.json

## About our Way
Door frame detect (yolo) + open/close detect (houghline detect)
## Model Training
1. over 3000 datasets with bus door
2. pretrain model = yolo v8x.pt 
3. You can train the model by train.ipynb


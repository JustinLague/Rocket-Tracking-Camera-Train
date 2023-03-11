from ultralytics import YOLO
from roboflow import Roboflow
import os 
import shutil
from dotenv import load_dotenv

def dowload_dataset(cwd, dataset_version):
    if(os.path.exists(cwd + "\\datasets\\RockETS-Rockets-Flying-" + str(dataset_version))):
        print("Dataset already downloaded")
        return

    print("Downloading dataset...")
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')    
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("whowho").project("rockets-rockets-flying")
    dataset = project.version(dataset_version).download("yolov8")
    shutil.move(dataset.location, cwd + "\\datasets\\" + dataset.location.split("\\")[-1])

def train(model, cwd, dataset_version):
    model.train(data=cwd + "\\datasets\\RockETS-Rockets-Flying-" + str(dataset_version) + "\\data.yaml", epochs=100, imgsz=640, workers=4, lr0=0.05, device=0)

def test(model):
    results = model.predict("test/aarluk-III-LCE.jpg")
    printResults(results)

def printResults(results):
    if(results is None):
        return

    for result in results:
        boxes = result.boxes
        print(boxes)


if __name__ == "__main__":
    cwd = os.getcwd() 
    load_dotenv()
    dataset_version = 2
    model = YOLO("yolov8m.yaml")
    #model = YOLO(cwd + "\\runs\\detect\\train20\\weights\\last.pt")
    #model.train(resume=True)

    dowload_dataset(cwd, dataset_version)
    train(model, cwd, dataset_version)
    #test(model)

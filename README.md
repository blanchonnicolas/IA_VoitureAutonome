# Project "Future vision transport"

Context: Autonomous Car
Goals: Solutions for video stream images interpretation
Repository of OpenClassrooms project 8' [AI Engineer path](https://openclassrooms.com/fr/paths/188)

## Future vision transport

Our role is to participate to the conception and development of the autonomous car.
To do so, we need to :
 - Train a segmentation model, able to differentiate human, road, vehicle, ... from Images (8 Classes)
 - Develop API that predict 8 classes segmentation from Image input
 - Deploy solution on Cloud

To do so, we rely on [Cityscapes dataset](https://www.cityscapes-dataset.com/dataset-overview/).

You can see the results here :

-   [Presentation](https://github.com/blanchonnicolas/)

-   [Technical Note](https://github.com/blanchonnicolas/)

-   [Vidéo]()

-   [Notebook 1 : datagenerator](https://github.com/blanchonnicolas/IA_Project8_Openclassrooms_IA_VoitureAutonome/blob/master/data_generator.ipynb)
    - Data Analysis: Visualisation, Modification using [OpenCV](https://opencv.org/)
    - Data Generator: Batch [step by step](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) ; [Custom Data Generator - En Français](https://deeplylearning.fr/cours-pratiques-deep-learning/realiser-son-propre-generateur-de-donnees/)
    - Masks Colors and 8 classes Labels: [Source GitHub](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)
    - Augmentation des Images : [albumentation](https://albumentations.ai/docs/#introduction-to-image-augmentation)

-   [Notebook 2 : Trained Models](https://github.com/blanchonnicolas/IA_Project8_Openclassrooms_IA_VoitureAutonome/blob/master/Model_UNET.ipynb)
    - UNET Architecture - Neural Networks : Keras Tensorflow, CNN, Convolution, Max Pooling, GPU
    - Transfer Learning & Bacbones : [Librairy Segmentation Models](https://github.com/qubvel/segmentation_models)
    - Loss and Metrics evolution : [DICE vs IOU](https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou) ; [Segmentation Metrics](https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html)

-   [Best Models](https://github.com/blanchonnicolas/IA_Project8_Openclassrooms_IA_VoitureAutonome/tree/master/Models) 
 
-   [Web Application Deployment - Prototype](xxx)
    - [FastAPI Framework](https://fastapi.tiangolo.com/): uvicorn, Server, Swagger Interactive interface
    - [Streamlit](https://streamlit.io/)
    - [Heroku](https://dashboard.heroku.com/apps)

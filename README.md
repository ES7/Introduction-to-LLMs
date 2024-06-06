# Introduction-to-LLMs
In this repository I have explained the application of Large Language Models (LLMs). Starting from how to use LLMs in our own application till how to build a LLM. Most of the knowledge I have gained is from **`DeepLearning.AI`**. A big thanks to Andrew Ng and the team at DeepLearning.AI for providing amazing courses on ML. https://learn.deeplearning.ai 

## 1. How to utilize a model via API
Application Programming Interface (API) is a way to interact with a remote application programmatically. Instead of going to any website and typing the prompt we can do this via API. so in python we type the prompt and with this API we can send requests and receive a response from the model.<br>
Using API we can customize the system messages, we can adjust the input parameters like max response length, no of responses and temperature, process images and other files, extract helpful word embeddings for downstream tasks, input audio for transcription and translation, model fine-tuning functionality.<br>
**Code :-** **`1. Using_Gemini_API.ipynb`**

## 2. How to use Open Source Models from Hugging Face
There are thousands of open source pre-trained ML models available for free. The datasets repository which are also available for free to train our own models or for fine-tuning. Hugging Face Spaces is a platform to build and deploy ML models.<br>
Transformers is a python library which makes downloading and training ML models super easy. Initially it was only for NLP; now it can be used for any domain. We can use its pipeline function to perform many tasks.<br>
On Hugging Face there are several models we can specify which task we want to accomplish, which dataset we need, and for commercial use cases we can also select the license. We can use Gradio library to build a UI for our application and instead of hosting it locally we can host it on HF Spaces which is a git repos hosted by Hugging Face that allows us to make ML applications.<br>

### 2.1. Natural Language Processing
NLP is a field of linguistics and ML, and it is focused on everything related to human language. The significant improvement in this field is due to the transformer architecture, from a well known paper "Attention Is All You Need" paper in 2017.<br>
And since then this architecture is the core of many state-of-art ML models nowadays. 
We will try to create our own chatbot and for this we will use Facebook's blenderbot model as it is very small and only needs 1.6GB to load it. We can’t use LLAMA2 or any other model since we won’t be able to load it as it exceeds 4GB. HF also provides a leaderboard where we can compare models.<br>
**Code :-** **`2.1. Intro_to_NLP.ipynb`**

### 2.2. Embeddings
Here we will try to measure sentence similarity, it measures how close two pieces of text are. For example ‘I like kittens’ and ‘We love cats’ have similar meanings. Sentence similarity is particularly used for information retrieval and clustering or grouping. The sentence similarity models convert input text into vectors or so-called embeddings. These embeddings capture semantic information.<br>
**Code :-** **`2.2. Embeddings.ipynb`**

### 2.3. Zero-Shot Audio Classification
Zero-Shot is an alternate method to avoid fine-tuning. Here we will use the Hugging Face model to solve Audio Classification problem.<br>
**Code :-** **`2.3. Zero_Shot_Audio_Classification.ipynb`**

### 2.4. Automatic Speech Recognition (ASR)
ASR is a task that involves transcribing speech audio recording into text. Eg: meeting notes or automatic video subtitle. For this task we will learn the Whisper model by OpenAI. This model was trained on vast quantity of labeled audio transcription data 680,000 hours to be precise. 117,000 hours of this pre-training data is multilingual or non-english. This results in checkpoints that can be applied to over 96 languages.<br>
**Code :-** **`2.4. Automatic_Speech_Recognition.ipynb`**

### 2.5. Text to Speech
It is a challenging task because it is a one-to-many problem. In classification we have one correct label and maybe a few. In ASR there is one correct transcription for a given utterance. However there are an infinite number of ways to say the same sentence. Each person has a different way of speaking but they are all valid and correct. Think about different voices, dialects, speaking styles and so on. Despite these challenges there are open-source models that can handle this task really well.<br>
**Code :-** **`2.5. Text_to_Speech.ipynb`**

### 2.6. Object Detection
The task of object detection simply consists of detecting objects of interest in a specific image. Object detection combines two subtasks, which are classification but also localization. Because for each object that we detect in an image, we also have to provide the label of the instance, but also the localization of the detected object.<br>
**Code :-** **`2.6. Object_Detection.ipynb`**

### 2.7. Image Segmentation
Here we will perform image segmentation and something called visual prompting. We will specify a point in the picture and the segmentation model will then identify a segmented object of interest.<br>
**Code :-** **`2.7. Image_Segmentation.ipynb`**

### 2.8. Multimodal Models
Here we will work with multimodal models to perform image-text matching, image captioning, question-answering and zero-shot image classification using the open-source models Bleep from salesforce for the first three task and Clip from OpenAI for the last task.
* **Image-Text Retrival :-** So if we pass an image along with a text it will compare how similar they both are.
* **Image Captioning :-** Passing an image and the model will give a description (caption) of the image.
* **Visual QnA :-** Pass an image with a question, the model will answer the question in context on the image.
* **Zero-shot Image Classification :-** Pass an image with a list of labels the model will choose the appropriate label from the list.<br>
**Code :-** **`2.8. Multimodal_Models.ipynb`**

### 2.9. Deployment
Uptill now we know that different types of tasks can be achieved within the Hugging Face ecosystem. In most cases, for hosting demos and practical applications it will be nice to have our application running without leaving our computer on. In other words offload the whole compute requirements outside our local machine. Here we will leverage Hugging Face Spaces to deploy our demos and use them as an API.<br>
**Code :-** **`2.9. Deployment.ipynb`**

## 3. Prompt Engineering (ChatGPT)
Prompt engineering is the process of designing and refining prompts to guide the output of large language models. It involves crafting prompts that elicit the desired responses and fine-tuning them iteratively to improve performance. Effective prompt engineering can significantly influence the quality and relevance of the model's outputs.<br>

### 3.1. Basics of Prompt Engineering
In this notebook we will se how to prompt any LLMs by three basic methods.<br>
**Code :-** **`3.1. Types_of_Prompt.ipynb`**

### 3.2. Guidlines of Prompting
In this notebook we will se how the two basic principles for prompting and limitations of LLMs.<br>
**Code :-** **`3.2. Guidlines_for_Prompting.ipynb`**

### 3.3. Iterative Prompt Development
In this notebook we will se iterative process of prompting to achieve better results from LLMs.<br>
**Code :-** **`3.3. Iterative_Prompt_Development.ipynb`**

### 3.4. Summarization Task
In this notebook we will see how to write prompts for summarization tasks.<br>
**Code :-** **`3.4. Prompt_for_Summarization_Task.ipynb`**

### 3.5. MultiTasking
In this notebook I have explained how we can use single model on multiple tasks just by changing the prompts.<br>
**Code :-** **`3.5. MultiTasking.ipynb`**

### 3.6. Transforming
In this notebook I have explained how we transform our text into someting another. LLMs are pretty good at this task for example language translation.<br>
**Code :-** **`3.6. Transforming.ipynb`**

### 3.7. Expanding
In this notebook I have explained how we use a parameter **temperature** to change the response of the model.<br>
**Code :-** **`3.7. Expanding.ipynb`**

### 3.8. ChatBot
In this notebook we will see how to use ChatGPT to build a ChatBot.<br>
**Code :-** **`3.7. Expanding.ipynb`**

## 4. Prompt Engineering for Vision Models
In 2023, "prompt engineering" emerged in machine learning, extending beyond text prompts for Large Language Models (LLMs) to include images, audio, and video. This approach uses prompts as inputs that guide the model's output distribution, with data converted into numerical representations and processed into embeddings.
Visual prompting involves providing instructions with relevant image data to pre-trained models for specific tasks, with prompts specifying tasks and being part of the total input data fed into the model.<br>

### 4.1. Prompt for Image Segmentation
In this notebook we will se how to prompt Segment Anything Model (SAM) for image segmentation task.<br>
**Code :-** **`4.1. Prompt_for_Image_Segmentation.ipynb`**

### 4.2. Prompt for Object Detection
In this notebook we will se how to use **natural language** to prompt a **zero-shot object detection model OWL-ViT** where ViT stands for Vision Transformer. We will see how to create a pipeline that uses the output of this model as an input to SAM Model.<br>
**Code :-** **`4.2. Prompt_for_Object_Detection.md`**

### 4.3. Prompt for Image Generation
In this notebook we will see how we can prompt stable diffusion with **images** and **masks**. Additionally, we can tune some hyperparameters such as **guidance scale, strength** and **the number of inference step** to better the performance the diffusion process.<br>
**Code :-** **`4.3. Prompt_for_Image_Generation.ipynb`**

### 4.4. Fine Tuning
Diffusion models are powerful tools for image generation, but sometimes no amount of prompt engineering or hyperparameter optimization will yield the results we are looking for. When prompt engineering isn't enough. We may opt for fine-tuning instead. Fine-tuning is extremely resource intensive, but thankfully there is a method of fine-tuning stable diffusion that uses far fewer resources, called Dreambooth.<br>
**Code :-** **`4.3. Fine_Tuning.ipynb`**

## 5. Fine Tuning
Writing a prompt is good to give a direction to the LLM but that won't be enough. Fine Tuning helps LLMs to work well on those data that it has not seen or not been trained on. With the help of fine tuning we can enhance the performance of LLMs on our own private data.

### 5.1. Comparision
In this notebook we will see what is finetuning, why it is important and a comparision between finetuned model and non-finetuned model.<br>
**Code :-** **`5.1. Comparision.ipynb`**

### 5.2. Process of Finetuning
In this notebook we will see what the difference between pre-trianing and finetuning and how to prepare the data for finetuning.<br>
**Code :-** **`5.2. Where_Finetuning_Fits.ipynb`**

### 5.3. Instruct Finetuning
In this notebook we will see how to perform instruct finetuning.<br>
**Code :-** **`5.3. Instruction_Finetuning.ipynb`**

### 5.4. Data Perparation
In this notebook we will see how to prepare the data, what are the important factors to keep in mind while collecting the data.<br>
**Code :-** **`5.4. Data_Preparation.ipynb`**

### 5.5. Training
In this notebook we will see how to train an LLM on our dataset.<br>
**Code :-** **`5.5. Training.ipynb`**

### 5.6. Evaluation
In this notebook we will see how to evaluate our trained LLM on benchmarks.<br>
**Code :-** **`5.6. Evaluation.ipynb`**

## 6. Reinforcement Learning with Human Feedback (RLHF)
It is a technique used to improve ML models by incorporating feedback from human evaluators or labellers. The process involves training a model using reinforcement learning algorithms, where the reward signal is derived from human feedback on the model's outputs. This approach helps align the model's behavior with human preferences and ethical considerations, enhancing its performance and reliability in real-world applications.

### 6.1 Dataset
In this notebook we will see what is RLHF and how to create dataset for it.<br>
**Code :-** **`6.1. Dataset_for_RLHF.ipynb`**

### 6.2 Tuning the LLM
In this notebook we will see how can we use the dataset to tune the LLM via (VertexAI) pipeline.<br>
**Code :-** **`6.2. Tune_LLM.ipynb`**

### 6.3 Evaluate the Tuned Model
In this notebook we will see how to evaluate the tuned LLM.<br>
**Code :-** **`6.3. Evaluation.ipynb`**

### 6.4 Google Cloud Setup
In this notebook we will see how to setup Google Cloud to use Vertex AI.<br>
**Code :-** **`6.4. Google_Cloud_Setup.ipynb`**

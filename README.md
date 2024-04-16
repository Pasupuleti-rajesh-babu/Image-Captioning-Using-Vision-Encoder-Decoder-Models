# Image-Captioning-Using-Vision-Encoder-Decoder-Models and the Flickr8k Dataset

<h2>Abstract</h2>
Image captioning, a crucial intersection of computer vision and natural
language processing, involves the automatic generation of textual descriptions
for images. This study aims to enhance the capabilities of image captioning
systems by leveraging advanced deep learning models, specifically Vision
Transformers and GPT-2, and a diverse dataset, Flickr8k. The Flickr8k dataset
consists of 8,000 images, each associated with five distinct human-generated
captions, providing a rich resource for training and evaluation. The main
objective of this study is to successfully implement and evaluate a Vision
Encoder Decoder Model that effectively converts 2D visual data into 1D
sequential data, a crucial step in generating relevant textual descriptions.
<h2>1. Introduction</h2>
This study resides at the intersection of natural language processing and
computer vision, with a focus on image captioning. The task of generating
descriptive sentences for images presents a significant challenge that requires
capabilities in both image understanding and language generation. By
successfully implementing a Vision Encoder Decoder Model, this study aims
to advance automated image interpretation. The potential applications of this
work span various fields, such as accessibility services for visually impaired
individuals, security and surveillance systems, and more.
<h2>2. Project Description</h2>
The study revolves around the development of an automatic image caption
generator. A Vision Encoder Decoder Model is used, combining a Vision
Transformer (ViT) for image feature extraction and GPT-2 for generating
corresponding captions. The model "sees" and "describes" the images,
emulating human-like understanding and expression.
<h2>3. Tasks Summary</h2>
<h2>The study was divided into several key tasks:</h2>

<h3>1. Data Acquisition and Preprocessing:</h3> 
<p>The Flickr8k dataset, which provides
a diverse collection of images and associated captions, was used. After
acquiring the dataset, preprocessing steps such as image resizing and caption
tokenization were performed.</p>
  
<h3>2. Model Creation:</h3> 
<p>A Vision Encoder Decoder Model was implemented.
This model uses a Vision Transformer (ViT) as the encoder and GPT-2 as
the decoder.</p>

<h3>3. Model Training: </h3>
<p>The model was trained using the Seq2SeqTrainer from
the HuggingFace transformers library.</p>

<h3>4. Model Evaluation: </h3>
<p>The model's performance was evaluated based on the
Rouge2 metric, which measures the quality of the generated captions.</p>

<h2>4. Data Source</h2>
<p>The dataset powering our research is Flickr8k, a publicly accessible resource
commonly implemented in image captioning studies. Flickr8k distinguishes
itself with its variety and applicability, encompassing 8,000 images, each
associated with five distinct human-generated captions. This culminates in an
aggregate of 40,000 unique image-caption combinations. The collection of
images spans a broad spectrum of scenarios and motifs, reflecting daily life
situations. The dataset is obtainable from the official Flickr8k database (please
insert the correct website URL here). The comprehensive size of the dataset,
which includes both images and their corresponding captions, is
approximately 1GB, although this can vary based on image quality and
resolution.</p>

<img width=300 src= "https://github.com/Pasupuleti-rajesh-babu/Image-Captioning-Using-Vision-Encoder-Decoder-Models/blob/main/readme_doc/pic1.png " >
<li>1. A black dog and a spotted dog are fighting.
<li>2. A black dog and a tri-colored dog playing with each
other on the road.</li>
<li>3. A black dog and a white dog with brown spots are
staring at each other in the street.</li>
<li>4. Two dogs of different breeds looking at each other
on the road.</li>
<li>5. Two dogs on pavement moving toward each other.</li>

<h2>5. Proposed Method</h2>

<p>The proposed method for this study combines the 'google/vit-base-patch16-
224-in21k' encoder and 'gpt2' decoder. The encoder, a type of Vision
Transformer (ViT), transforms the image into a sequence of patches,
effectively converting 2D visual data into 1D sequential data. This approach
enables the model to grasp the semantic content of the image better. The 'gpt2'
decoder, renowned for its capabilities in language generation, sequentially
generates the image caption, ensuring a coherent and contextually accurate
description.</p>
<p>The Vision Transformer (ViT) is an innovative model that applies the
transformer architecture, originally designed for text data, to visual data. By
dividing the image into patches and flattening them into a sequence of vectors,
the ViT can process visual data in a way similar to how transformers process
text data. This approach allows the model to capture complex patterns and
dependencies in the image data./p>
<p>The GPT-2 decoder, known for its proficiency in language generation, takes
the encoded image data and generates a sequence of tokens, forming the image
caption. The GPT-2 model uses a transformer-based architecture, which
allows it to generate coherent and contextually accurate sentences by
capturing the dependencies between the tokens in the sequence.</p>

<h2>6. Implementation Details</h2>

<p>The implementation of the study involved several steps, each demanding a
deep understanding of the tools and techniques involved.</p>

<h3>1. Data Preprocessing:</h3> 
<p>One of the initial challenges was to prepare the data
from the Flickr8k dataset in a form that could be efficiently used by the
models. This process involved resizing the images to match the input
requirements of the Vision Transformer, and tokenizing the captions using the
GPT-2 tokenizer. A critical part of this process was ensuring that the data was
correctly formatted and ready for input into the VisionEncoderDecoderModel.</p>


<h3>2. Model Creation:</h3>
<p>Creating a model that could effectively handle both the
visual and textual aspects of the problem was a significant challenge. The
solution was found in the VisionEncoderDecoderModel provided by the
HuggingFace transformers library. This model allows for seamless integration
of encoder and decoder models—in this case, a Vision Transformer (ViT) and
GPT-2.</p>
<h3>
3. Model Training:</h3>

<p>Once the model was created, the next step was to train it
on the prepared dataset. The Seq2SeqTrainer from the HuggingFace
transformers library was employed for this purpose. However, training a deep
learning model is not a straightforward task. Selecting the right
hyperparameters, such as learning rate and batch size, and designing a training
loop that effectively leverages GPU resources were critical aspects of this
step.</p>
<h3>4. Model Evaluation: </h3>
<p>The model was evaluated using the Rouge2 metric,
which measures the quality of the generated captions. This metric provides an
objective measure of the model's performance, helping to identify areas for
improvement.</p>
<p>A challenge encountered during implementation was the out-of-memory
errors during training. This common issue can occur when the model or the
dataset is too large to fit into GPU memory. The problem was resolved by
reducing the batch size during training, which decreased the memory
requirements. However, this also increased the training time, as fewer
examples were processed at each step.</p>

<h2>7. Results</h2>
<p>
The performance of the model was evaluated based on the Rouge2 metric.
The training epochs are shown in the below figure. The model showed
promising improvements. The model was trained on 1000 images from 8000
images. Due to limited resources, the model was trained for 5 epochs which
took around 6 hours to train. The NVIDIA 940MX GPU was used for the</p>
training. The attached figures are the test image with its output.
<img width=600 src= "https://github.com/Pasupuleti-rajesh-babu/Image-Captioning-Using-Vision-Encoder-Decoder-Models/blob/main/readme_doc/pic2.png " >

<img width=300 src= "https://github.com/Pasupuleti-rajesh-babu/Image-Captioning-Using-Vision-Encoder-Decoder-Models/blob/main/readme_doc/pic1.png " >
<ul>
<li>1. Two dogs are playing in the pond</li>
<li>2. Dog is swimming in the pond</li>
<li>3. Two dogs are running through a lake in the snow </li>
</ul>
<h2>8. Conclusion</h2>
<p>In conclusion, this study demonstrates the potential of using advanced deep
learning techniques for automatic image captioning. The combination of
Vision Transformer (ViT) and GPT-2 models, along with the diverse and rich
Flickr8k dataset, allowed the development of a system capable of generating
descriptive sentences for images. The study provides a solid foundation for
future work in this area and could have wide-reaching implications in various
fields.</p>
  
<p>Future work will involve refining the model, experimenting with different
architectures and parameters, and potentially expanding the system to handle
more complex scenarios and a broader range of input data.</p>

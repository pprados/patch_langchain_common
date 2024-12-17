LayoutParser: A Unified Toolkit for Deep
Learning Based Document Image Analysis
Zejiang Shen
1
(), Ruochen Zhang
2
, Melissa Dell
3
, Benjamin Charles Germain
Lee
4
, Jacob Carlson
3
, and Weining Li
5
1 Allen Institute for AI
shannons@allenai.org 2 Brown University
ruochen zhang@brown.edu 3 Harvard University
{melissadell,jacob carlson}@fas.harvard.edu
4 University of Washington
bcgl@cs.washington.edu 5 University of Waterloo
w422li@uwaterloo.ca
Abstract. Recent advances in document image analysis (DIA) have been
primarily driven by the application of neural networks. Ideally, research
outcomes could be easily deployed in production and extended for further
investigation. However, various factors like loosely organized codebases
and sophisticated model configurations complicate the easy reuse of important innovations by a wide audience. Though there have been on-going
efforts to improve reusability and simplify deep learning (DL) model
development in disciplines like natural language processing and computer
vision, none of them are optimized for challenges in the domain of DIA.
This represents a major gap in the existing toolkit, as DIA is central to
academic research across a wide range of disciplines in the social sciences
and humanities. This paper introduces LayoutParser, an open-source
library for streamlining the usage of DL in DIA research and applications. The core LayoutParser library comes with a set of simple and
intuitive interfaces for applying and customizing DL models for layout detection, character recognition, and many other document processing tasks.
To promote extensibility, LayoutParser also incorporates a community
platform for sharing both pre-trained models and full document digitization pipelines. We demonstrate that LayoutParser is helpful for both
lightweight and large-scale digitization pipelines in real-word use cases.
The library is publicly available at https://layout-parser.github.io.
Keywords: Document Image Analysis· Deep Learning· Layout Analysis
· Character Recognition· Open Source library· Toolkit.
1 Introduction
Deep Learning(DL)-based approaches are the state-of-the-art for a wide range of
document image analysis (DIA) tasks including document image classification [11,
arXiv:2103.15348v2 [cs.CV] 21 Jun 2021

2 Z. Shen et al.
37], layout detection [38, 22], table detection [26], and scene text detection [4].
A generalized learning-based framework dramatically reduces the need for the
manual specification of complicated rules, which is the status quo with traditional
methods. DL has the potential to transform DIA pipelines and benefit a broad
spectrum of large-scale document digitization projects.
However, there are several practical difficulties for taking advantages of recent advances in DL-based methods: 1) DL models are notoriously convoluted
for reuse and extension. Existing models are developed using distinct frameworks like TensorFlow [1] or PyTorch [24], and the high-level parameters can
be obfuscated by implementation details [8]. It can be a time-consuming and
frustrating experience to debug, reproduce, and adapt existing models for DIA,
and many researchers who would benefit the most from using these methods lack
the technical background to implement them from scratch. 2) Document images
contain diverse and disparate patterns across domains, and customized training
is often required to achieve a desirable detection accuracy. Currently there is no
full-fledged infrastructure for easily curating the target document image datasets
and fine-tuning or re-training the models. 3) DIA usually requires a sequence of
models and other processing to obtain the final outputs. Often research teams use
DL models and then perform further document analyses in separate processes,
and these pipelines are not documented in any central location (and often not
documented at all). This makes it difficult for research teams to learn about how
full pipelines are implemented and leads them to invest significant resources in
reinventing the DIA wheel.
LayoutParser provides a unified toolkit to support DL-based document image
analysis and processing. To address the aforementioned challenges, LayoutParser
is built with the following components:
1. An off-the-shelf toolkit for applying DL models for layout detection, character
recognition, and other DIA tasks (Section 3)
2. A rich repository of pre-trained neural network models (Model Zoo) that
underlies the off-the-shelf usage
3. Comprehensive tools for efficient document image data annotation and model
tuning to support different levels of customization
4. A DL model hub and community platform for the easy sharing, distribution, and discussion of DIA models and pipelines, to promote reusability,
reproducibility, and extensibility (Section 4)
The library implements simple and intuitive Python APIs without sacrificing
generalizability and versatility, and can be easily installed via pip. Its convenient
functions for handling document image data can be seamlessly integrated with
existing DIA pipelines. With detailed documentations and carefully curated
tutorials, we hope this tool will benefit a variety of end-users, and will lead to
advances in applications in both industry and academic research.
LayoutParser is well aligned with recent efforts for improving DL model
reusability in other disciplines like natural language processing [8, 34] and computer vision [35], but with a focus on unique challenges in DIA. We show
LayoutParser can be applied in sophisticated and large-scale digitization projects

LayoutParser: A Unified Toolkit for DL-Based DIA 3
that require precision, efficiency, and robustness, as well as simple and lightweight document processing tasks focusing on efficacy and flexibility (Section 5).
LayoutParser is being actively maintained, and support for more deep learning
models and novel methods in text-based layout analysis methods [37, 34] is
planned.
The rest of the paper is organized as follows. Section 2 provides an overview
of related work. The core LayoutParser library, DL Model Zoo, and customized
model training are described in Section 3, and the DL model hub and community platform are detailed in Section 4. Section 5 shows two examples of how
LayoutParser can be used in practical DIA projects, and Section 6 concludes.
2 Related Work
Recently, various DL models and datasets have been developed for layout analysis
tasks. The dhSegment [22] utilizes fully convolutional networks [20] for segmentation tasks on historical documents. Object detection-based methods like Faster
R-CNN [28] and Mask R-CNN [12] are used for identifying document elements [38]
and detecting tables [30, 26]. Most recently, Graph Neural Networks [29] have also
been used in table detection [27]. However, these models are usually implemented
individually and there is no unified framework to load and use such models.
There has been a surge of interest in creating open-source tools for document
image processing: a search of document image analysis in Github leads to 5M
relevant code pieces 6; yet most of them rely on traditional rule-based methods
or provide limited functionalities. The closest prior research to our work is the
OCR-D project7, which also tries to build a complete toolkit for DIA. However,
similar to the platform developed by Neudecker et al. [21], it is designed for
analyzing historical documents, and provides no supports for recent DL models.
The DocumentLayoutAnalysis project8focuses on processing born-digital PDF
documents via analyzing the stored PDF data. Repositories like DeepLayout9
and Detectron2-PubLayNet10 are individual deep learning models trained on
layout analysis datasets without support for the full DIA pipeline. The Document
Analysis and Exploitation (DAE) platform [15] and the DeepDIVA project [2]
aim to improve the reproducibility of DIA methods (or DL models), yet they
are not actively maintained. OCR engines like Tesseract [14], easyOCR11 and
paddleOCR12 usually do not come with comprehensive functionalities for other
DIA tasks like layout analysis.
Recent years have also seen numerous efforts to create libraries for promoting
reproducibility and reusability in the field of DL. Libraries like Dectectron2 [35],
6 The number shown is obtained by specifying the search type as ‘code’.
7 https://ocr-d.de/en/about
8 https://github.com/BobLd/DocumentLayoutAnalysis
9 https://github.com/leonlulu/DeepLayout
10 https://github.com/hpanwar08/detectron2
11 https://github.com/JaidedAI/EasyOCR
12 https://github.com/PaddlePaddle/PaddleOCR

4 Z. Shen et al.
Efficient Data Annotation
Customized Model Training
Model Customization
DIA Model Hub
DIA Pipeline Sharing
Community Platform
Layout Detection Models
Document Images 
The Core LayoutParser Library
OCR Module Layout Data Structure Storage & Visualization
Fig. 1: The overall architecture of LayoutParser. For an input document image,
the core LayoutParser library provides a set of off-the-shelf tools for layout
detection, OCR, visualization, and storage, backed by a carefully designed layout
data structure. LayoutParser also supports high level customization via efficient
layout annotation and model training functions. These improve model accuracy
on the target samples. The community platform enables the easy sharing of DIA
models and whole digitization pipelines to promote reusability and reproducibility.
A collection of detailed documentation, tutorials and exemplar projects make
LayoutParser easy to learn and use.
AllenNLP [8] and transformers [34] have provided the community with complete
DL-based support for developing and deploying models for general computer
vision and natural language processing problems. LayoutParser, on the other
hand, specializes specifically in DIA tasks. LayoutParser is also equipped with a
community platform inspired by established model hubs such as Torch Hub [23]
and TensorFlow Hub [1]. It enables the sharing of pretrained models as well as
full document processing pipelines that are unique to DIA tasks.
There have been a variety of document data collections to facilitate the
development of DL models. Some examples include PRImA [3](magazine layouts),
PubLayNet [38](academic paper layouts), Table Bank [18](tables in academic
papers), Newspaper Navigator Dataset [16, 17](newspaper figure layouts) and
HJDataset [31](historical Japanese document layouts). A spectrum of models
trained on these datasets are currently available in the LayoutParser model zoo
to support different use cases.
3 The Core LayoutParser Library
At the core of LayoutParser is an off-the-shelf toolkit that streamlines DLbased document image analysis. Five components support a simple interface
with comprehensive functionalities: 1) The layout detection models enable using
pre-trained or self-trained DL models for layout detection with just four lines
of code. 2) The detected layout information is stored in carefully engineered

LayoutParser: A Unified Toolkit for DL-Based DIA 5
Table 1: Current layout detection models in the LayoutParser model zoo
Dataset Base Model1 Large Model Notes
PubLayNet [38] F / M M Layouts of modern scientific documents
PRImA [3] M - Layouts of scanned modern magazines and scientific reports
Newspaper [17] F - Layouts of scanned US newspapers from the 20th century
TableBank [18] F F Table region on modern scientific and business document
HJDataset [31] F / M - Layouts of history Japanese documents
1 For each dataset, we train several models of different sizes for different needs (the trade-off between accuracy
vs. computational cost). For “base model” and “large model”, we refer to using the ResNet 50 or ResNet 101
backbones [13], respectively. One can train models of different architectures, like Faster R-CNN [28] (F) and Mask
R-CNN [12] (M). For example, an F in the Large Model column indicates it has a Faster R-CNN model trained
using the ResNet 101 backbone. The platform is maintained and a number of additions will be made to the model
zoo in coming months.
layout data structures, which are optimized for efficiency and versatility. 3) When
necessary, users can employ existing or customized OCR models via the unified
API provided in the OCR module. 4) LayoutParser comes with a set of utility
functions for the visualization and storage of the layout data. 5) LayoutParser
is also highly customizable, via its integration with functions for layout data
annotation and model training. We now provide detailed descriptions for each
component.
3.1 Layout Detection Models
In LayoutParser, a layout model takes a document image as an input and
generates a list of rectangular boxes for the target content regions. Different
from traditional methods, it relies on deep convolutional neural networks rather
than manually curated rules to identify content regions. It is formulated as an
object detection problem and state-of-the-art models like Faster R-CNN [28] and
Mask R-CNN [12] are used. This yields prediction results of high accuracy and
makes it possible to build a concise, generalized interface for layout detection.
LayoutParser, built upon Detectron2 [35], provides a minimal API that can
perform layout detection with only four lines of code in Python:
1 import layoutparser as lp
2 image = cv2 . imread (" image_file ") # load images
3 model = lp . Detectron2LayoutModel (
4 "lp :// PubLayNet / faster_rcnn_R_50_FPN_3x / config ")
5 layout = model . detect ( image )
LayoutParser provides a wealth of pre-trained model weights using various
datasets covering different languages, time periods, and document types. Due to
domain shift [7], the prediction performance can notably drop when models are applied to target samples that are significantly different from the training dataset. As
document structures and layouts vary greatly in different domains, it is important
to select models trained on a dataset similar to the test samples. A semantic syntax
is used for initializing the model weights in LayoutParser, using both the dataset
name and model name lp://<dataset-name>/<model-architecture-name>.

6 Z. Shen et al.
Fig. 2: The relationship between the three types of layout data structures.
Coordinate supports three kinds of variation; TextBlock consists of the coordinate information and extra features like block text, types, and reading orders;
a Layout object is a list of all possible layout elements, including other Layout
objects. They all support the same set of transformation and operation APIs for
maximum flexibility.
Shown in Table 1, LayoutParser currently hosts 9 pre-trained models trained
on 5 different datasets. Description of the training dataset is provided alongside
with the trained models such that users can quickly identify the most suitable
models for their tasks. Additionally, when such a model is not readily available,
LayoutParser also supports training customized layout models and community
sharing of the models (detailed in Section 3.5).
3.2 Layout Data Structures
A critical feature of LayoutParser is the implementation of a series of data
structures and operations that can be used to efficiently process and manipulate
the layout elements. In document image analysis pipelines, various post-processing
on the layout analysis model outputs is usually required to obtain the final
outputs. Traditionally, this requires exporting DL model outputs and then loading
the results into other pipelines. All model outputs from LayoutParser will be
stored in carefully engineered data types optimized for further processing, which
makes it possible to build an end-to-end document digitization pipeline within
LayoutParser. There are three key components in the data structure, namely
the Coordinate system, the TextBlock, and the Layout. They provide different
levels of abstraction for the layout data, and a set of APIs are supported for
transformations or operations on these classes.



Coordinate
(x1, y1)
(X1, y1)
(x2,y2)
APIS
x-interval
tart
end
Quadrilateral
operation
Rectangle
y-interval
ena
(x2, y2)
(x4, y4)
(x3, y3)
and
textblock
Coordinate
transformation
+
Block
Block
Reading
Extra features
Text
Type
Order
coordinatel
textblockl
layout
 same
textblock2
layoutl
The
A list of the layout elements

LayoutParser: A Unified Toolkit for DL-Based DIA 7
Coordinates are the cornerstones for storing layout information. Currently,
three types of Coordinate data structures are provided in LayoutParser, shown
in Figure 2. Interval and Rectangle are the most common data types and
support specifying 1D or 2D regions within a document. They are parameterized
with 2 and 4 parameters. A Quadrilateral class is also implemented to support
a more generalized representation of rectangular regions when the document
is skewed or distorted, where the 4 corner points can be specified and a total
of 8 degrees of freedom are supported. A wide collection of transformations
like shift, pad, and scale, and operations like intersect, union, and is_in,
are supported for these classes. Notably, it is common to separate a segment
of the image and analyze it individually. LayoutParser provides full support
for this scenario via image cropping operations crop_image and coordinate
transformations like relative_to and condition_on that transform coordinates
to and from their relative representations. We refer readers to Table 2 for a more
detailed description of these operations13.
Based on Coordinates, we implement the TextBlock class that stores both
the positional and extra features of individual layout elements. It also supports
specifying the reading orders via setting the parent field to the index of the parent
object. A Layout class is built that takes in a list of TextBlocks and supports
processing the elements in batch. Layout can also be nested to support hierarchical
layout structures. They support the same operations and transformations as the
Coordinate classes, minimizing both learning and deployment effort.
3.3 OCR
LayoutParser provides a unified interface for existing OCR tools. Though there
are many OCR tools available, they are usually configured differently with distinct
APIs or protocols for using them. It can be inefficient to add new OCR tools into
an existing pipeline, and difficult to make direct comparisons among the available
tools to find the best option for a particular project. To this end, LayoutParser
builds a series of wrappers among existing OCR engines, and provides nearly
the same syntax for using them. It supports a plug-and-play style of using OCR
engines, making it effortless to switch, evaluate, and compare different OCR
modules:
1 ocr_agent = lp . TesseractAgent ()
2 # Can be easily switched to other OCR software
3 tokens = ocr_agent . detect ( image )
The OCR outputs will also be stored in the aforementioned layout data
structures and can be seamlessly incorporated into the digitization pipeline.
Currently LayoutParser supports the Tesseract and Google Cloud Vision OCR
engines.
LayoutParser also comes with a DL-based CNN-RNN OCR model [6] trained
with the Connectionist Temporal Classification (CTC) loss [10]. It can be used
like the other OCR modules, and can be easily trained on customized datasets.
13 This is also available in the LayoutParser documentation pages.

8 Z. Shen et al.
Table 2: All operations supported by the layout elements. The same APIs are
supported across different layout element classes including Coordinate types,
TextBlock and Layout.
Operation Name Description
block.pad(top, bottom, right, left) Enlarge the current block according to the input
block.scale(fx, fy) Scale the current block given the ratio
in x and y direction
block.shift(dx, dy) Move the current block with the shift
distances in x and y direction
block1.is in(block2) Whether block1 is inside of block2
block1.intersect(block2) Return the intersection region of block1 and block2.
Coordinate type to be determined based on the inputs.
block1.union(block2) Return the union region of block1 and block2.
Coordinate type to be determined based on the inputs.
block1.relative to(block2) Convert the absolute coordinates of block1 to
relative coordinates to block2
block1.condition on(block2) Calculate the absolute coordinates of block1 given
the canvas block2’s absolute coordinates
block.crop image(image) Obtain the image segments in the block region
3.4 Storage and visualization
The end goal of DIA is to transform the image-based document data into a
structured database. LayoutParser supports exporting layout data into different
formats like JSON, csv, and will add the support for the METS/ALTO XML
format 14 . It can also load datasets from layout analysis-specific formats like
COCO [38] and the Page Format [25] for training layout models (Section 3.5).
Visualization of the layout detection results is critical for both presentation
and debugging. LayoutParser is built with an integrated API for displaying the
layout information along with the original document image. Shown in Figure 3, it
enables presenting layout data with rich meta information and features in different
modes. More detailed information can be found in the online LayoutParser
documentation page.
3.5 Customized Model Training
Besides the off-the-shelf library, LayoutParser is also highly customizable with
supports for highly unique and challenging document analysis tasks. Target
document images can be vastly different from the existing datasets for training layout models, which leads to low layout detection accuracy. Training data
14 https://altoxml.github.io

LayoutParser: A Unified Toolkit for DL-Based DIA 9
Fig. 3: Layout detection and OCR results visualization generated by the
LayoutParser APIs. Mode I directly overlays the layout region bounding boxes
and categories over the original image. Mode II recreates the original document
via drawing the OCR’d texts at their corresponding positions on the image
canvas. In this figure, tokens in textual regions are filtered using the API and
then displayed.
can also be highly sensitive and not sharable publicly. To overcome these challenges, LayoutParser is built with rich features for efficient data annotation and
customized model training.
LayoutParser incorporates a toolkit optimized for annotating document layouts using object-level active learning [32]. With the help from a layout detection
model trained along with labeling, only the most important layout objects within
each image, rather than the whole image, are required for labeling. The rest of
the regions are automatically annotated with high confidence predictions from
the layout detection model. This allows a layout dataset to be created more
efficiently with only around 60% of the labeling budget.
After the training dataset is curated, LayoutParser supports different modes
for training the layout models. Fine-tuning can be used for training models on a
small newly-labeled dataset by initializing the model with existing pre-trained
weights. Training from scratch can be helpful when the source dataset and
target are significantly different and a large training set is available. However, as
suggested in Studer et al.’s work[33], loading pre-trained weights on large-scale
datasets like ImageNet [5], even from totally different domains, can still boost
model performance. Through the integrated API provided by LayoutParser,
users can easily compare model performances on the benchmark datasets.



Option 1: Display Token Bounding Box
Anonymous ICDAR 2021 Submission
AnonymousICDAR2021 Submission
Figure
Model Customization
Document Images
Community Platform
Efficient Data Annotation
DIA Model Hub
CustomizedModel Training
Layout Detection Models
DIA Pipeline Sharing
←
OCR Module
Layout Data Structure
Storage&Visualization
The Core LayoutParser Library
Text
The overallarchitecture ofLayoutParser.For animputdocumentimage
Fig.
The
overall
architecture
of
LayoutParser.
For
input
document
image,
thecore LayoutParser library provides a set of off-the-shelf tools forlayout
the
LayoutParser
library
provides
setbf off-the-shelf
toolsfor
layout
detection,OCR,visualization, and storage,backed by a carefully designed layout
detection,
OCR,
visualization,
andstorage,
backedbycarefully
designed
layout
data structure.LayoutParser also supports high level of customization via
data
structure.
LayoutParser
also
supports
high level
via
efficientlayout data annotation and model training functions thatimproves
efficient
layout
Hataannotation
pome
training
functions
that
improves
model accuracy on the target samples. The community platform enables the
model
accuracy
the
target
samples.
The
community
platform
enablesthe
easy share of DIA models and even whole digitization pipelines to promote
share
1bf DIA
modelsand
wholedigitization
easy
pipelines
to
promote
reusability and reproducibility.A collection of detailed documentations,tutorials
reusability
and
reproducibility.
collectionbf detailed
documentations,
tutorials
andexemplarprojectsmakesLayoutParsereasytolearn anduse.
andexemplar
projects
makes
LayoutParser
easy
tlearnand
LayoutParser
 alsohighly
customizable,
integrated
withfunctions
forlayout
data annotation andmodel training.Weprovidedetaileddescriptionsfor each
data
annotation
andmodel
training.
Weprovidedetaileddescriptions
foreach
component as follows.
component
Follows.
3.1Titletyout Detection Models
3.1
Layout
Detection
Models
Option 2: Hide Token Bounding Box
In LayoutParser,
layout
model
takesadocument
image
inputand
generates a list of rectangular boxes for the target content regions.Different fron
generates
a list of rectangular
boxes for the target content regions.
Different
tfrom
traditional methods,it relies on deep convolutional neural networks rather thar
traditional
methods,i
it relies on deep convolutional
neuralnetworks rather than
manually curatedrulesfor identifying the contentregions.It is formulated as
manually
curated
1rules for identifyingthe content regions.It is formulated
SB
an object detection problem and state of the art models like Fast RCNNs [22
an object
detectionproblem and state of the art models like Fast RCNNs
[22]
and Mask RCNN [11] are being used. Not only it yields prediction results O
and Mask RCNN
[1ll are being used. Not only it yields prediction
results
of
nigh accuracy,but also makes it possible tobuild a concise while generalizec
high accuracy,
but also makes it possible to build a concise while generalized
nterface for using the layout detection models.In fact,built upon Detectron2 [28]
LayoutParser provides a minimal API that one can perform layout detectior
LayoutParser
vith only four lines of code in Python:
with only four lines of code in Python:
dT se 1asiednot
Listport
import
layoutparser
aslp
image
cv2.imread("image_file"） # load images
2 image
cv2.imread("image_file")
#load
images
model = lp.Detectron2LayoutModel(
model
lp.Detectron2LayoutModel
"lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")
4
"lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")
layout = model.detect(image)
5 layout
model.detect
(image)
Mode I: Showing Layout on the Original Image
Mode Il: Drawing OCR'd Text at the Correspoding Position

10 Z. Shen et al.
Fig. 4: Illustration of (a) the original historical Japanese document with layout
detection results and (b) a recreated version of the document image that achieves
much better character recognition recall. The reorganization algorithm rearranges
the tokens based on the their detected bounding boxes given a maximum allowed
height.
4 LayoutParser Community Platform
Another focus of LayoutParser is promoting the reusability of layout detection
models and full digitization pipelines. Similar to many existing deep learning
libraries, LayoutParser comes with a community model hub for distributing
layout models. End-users can upload their self-trained models to the model hub,
and these models can be loaded into a similar interface as the currently available
LayoutParser pre-trained models. For example, the model trained on the News
Navigator dataset [17] has been incorporated in the model hub.
Beyond DL models, LayoutParser also promotes the sharing of entire document digitization pipelines. For example, sometimes the pipeline requires the
combination of multiple DL models to achieve better accuracy. Currently, pipelines
are mainly described in academic papers and implementations are often not publicly available. To this end, the LayoutParser community platform also enables
the sharing of layout pipelines to promote the discussion and reuse of techniques.
For each shared pipeline, it has a dedicated project page, with links to the source
code, documentation, and an outline of the approaches. A discussion panel is
provided for exchanging ideas. Combined with the core LayoutParser library,
users can easily build reusable components based on the shared pipelines and
apply them to solve their unique problems.
5 Use Cases
The core objective of LayoutParser is to make it easier to create both large-scale
and light-weight document digitization pipelines. Large-scale document processing



Intra-columnreadingorder
設立
天株主
代大下食保一
决算期六月
設立
設備
取引银行滋貨（九條）
年商内高一四千六百万圆内外
悠巢具
大株主
代西村永治郎
資本额
目的
設立二昭和洲年十
取引银行三和（西陣）
大口出餐者
盛江吟次郎
决算期八月配當無
目易的
設立
Token Categories
大山中錦造
代光村
决算期
餐本额二百二十萬圆（一千四
装具
西村永治郎
快算期
年商内高
夜野
株榮光商店
中央信金（强榮）
大下食八重
英昌
0O年
永和化成工業林
江局喜
百口）
本金百二十万具
江
Column reading order
百楼
天
下食知惠八烟市郎
西陣物製造服
徽雅服飾品银造贩
一昭和廿八年四月一
一人織物製造贩
昭和廿八年七月
化學药品製造
一昭租廿五年九月
寻西人電四局二八
上京區元誓寺通泽
Title
東部一
中京區西）京左馬察町
南區吉祥院落合町六四
白郎荒川正太郎
竹野郡溺榮町和田野
三月配當一
（株主数七）
（株主数一二>
和雄
五月
百万
电
百膜具（二千棵）
脱水機三、其他六
Address
四
三電五局六三三七
郎
織
一千八百萬冒内外
郎吉圖
（出資者敷一O
圆（二千株）
配当無
瑞吉田
配富無
江局
Text
翻植
電八四局三六九
一百万冒内外
物有
蜀谷七七
物株
Number
国
建物三
七00株
四00株
次郎
·二割
Variable
一男
千
Company Type
(a) Illustration of the original Japanese document with detected layout elements highlighted in colored boxes
Column Categories
Title
Maximum Allowed Height 
附大下會保一郎大下會留藏
OO坪、脱水機三、其他六
南區吉祥院落合六四
上京區元誓寺通澄漏
中京區西）京左馬寮
地四O、建物三
寺西人龙四局二八三二
東部
永和化成工菜
竹野郡彌榮町和田野
人物裂造賣
英昌物
江島
微釉服飾品製造贩賣
野专村儀一郎吉网
四千六百万冒内外
四千二百万闻内外
西陣物製造
二千八百万圆内外
京都（岭山)丹後有
昭和廿八年四月
昭和廿八年七月
百萬圆（二千株）
昭和廿五年九月目的
（出者数一O）
百万圆（二千楼）
昭和廿年十一月
中央信金（荣）山中
電五局六三三七目的
江局吟大郎目的
代江局喜一郎
化學藥品製造
大下合保一郎
代西村永治郎
配當一三割
（株主数一二七）
荒川正太郎二四
溝谷七七立夜野
大下八重
三和（西陣）
滋贺（九條）
大下會知惠
百二十萬园
大口出资者
西村永治郎
百二十万圆
（株主数七）资术
年商内高资本额
取引银行
年
取引銀行配當無
年商内高决期
二千二决算期
（千四
业具
代光村大株主
业具
Address
引银行
商内高
一0電公局三六九
Text
业員
八烟
配當無
决算期
大株主
織
設備筛造
资本金
费本
SectionHeader
市郎
白郎
和雄
百株）
吉田
决算期
六有
八月
設立
重雄
三月
壽男
目的
江局
配當無
物
楠
五月
設立
一五
株
毛
次郎
立
百口）
株
株
(b) Illustration of the recreated document with dense text structure for better OCR performance

LayoutParser: A Unified Toolkit for DL-Based DIA 11
focuses on precision, efficiency, and robustness. The target documents may have
complicated structures, and may require training multiple layout detection models
to achieve the optimal accuracy. Light-weight pipelines are built for relatively
simple documents, with an emphasis on development ease, speed and flexibility.
Ideally one only needs to use existing resources, and model training should be
avoided. Through two exemplar projects, we show how practitioners in both
academia and industry can easily build such pipelines using LayoutParser and
extract high-quality structured document data for their downstream tasks. The
source code for these projects will be publicly available in the LayoutParser
community hub.
5.1 A Comprehensive Historical Document Digitization Pipeline
The digitization of historical documents can unlock valuable data that can shed
light on many important social, economic, and historical questions. Yet due to
scan noises, page wearing, and the prevalence of complicated layout structures, obtaining a structured representation of historical document scans is often extremely
complicated.
Fig. 5: Illustration of how LayoutParser
helps with the historical document digitization pipeline.
In this example, LayoutParser was
used to develop a comprehensive
pipeline, shown in Figure 5, to generate high-quality structured data from
historical Japanese firm financial tables with complicated layouts. The
pipeline applies two layout models to
identify different levels of document
structures and two customized OCR
engines for optimized character recognition accuracy.
As shown in Figure 4 (a), the
document contains columns of text
written vertically 15, a common style
in Japanese. Due to scanning noise
and archaic printing technology, the
columns can be skewed or have variable widths, and hence cannot be easily identified via rule-based methods.
Within each column, words are separated by white spaces of variable size,
and the vertical positions of objects
can be an indicator of their layout
type.
15 A document page consists of eight rows like this. For simplicity we skip the row
segmentation discussion and refer readers to the source code when available.



Active Learning Layout
Annotate Layout Dataset
Annotation Toolkit
Deep Learning Layout
Layout Detection
Model Training & Inference
Handy Data Structures &
Post-processing
APls for Layout Data
Default and Customized
Text Recognition
OCR Models
Layout Structure
Visualization & Export
Visualization & Storage
The Japanese Document
Helpful LayoutParser
Digitization Pipeline
Modules

12 Z. Shen et al.
To decipher the complicated layout
structure, two object detection models have been trained to recognize individual
columns and tokens, respectively. A small training set (400 images with approximately 100 annotations each) is curated via the active learning based annotation
tool [32] in LayoutParser. The models learn to identify both the categories and
regions for each token or column via their distinct visual features. The layout
data structure enables easy grouping of the tokens within each column, and
rearranging columns to achieve the correct reading orders based on the horizontal
position. Errors are identified and rectified via checking the consistency of the
model predictions. Therefore, though trained on a small dataset, the pipeline
achieves a high level of layout detection accuracy: it achieves a 96.97 AP [19]
score across 5 categories for the column detection model, and a 89.23 AP across
4 categories for the token detection model.
A combination of character recognition methods is developed to tackle the
unique challenges in this document. In our experiments, we found that irregular
spacing between the tokens led to a low character recognition recall rate, whereas
existing OCR models tend to perform better on densely-arranged texts. To
overcome this challenge, we create a document reorganization algorithm that
rearranges the text based on the token bounding boxes detected in the layout
analysis step. Figure 4 (b) illustrates the generated image of dense text, which is
sent to the OCR APIs as a whole to reduce the transaction costs. The flexible
coordinate system in LayoutParser is used to transform the OCR results relative
to their original positions on the page.
Additionally, it is common for historical documents to use unique fonts
with different glyphs, which significantly degrades the accuracy of OCR models
trained on modern texts. In this document, a special flat font is used for printing
numbers and could not be detected by off-the-shelf OCR engines. Using the highly
flexible functionalities from LayoutParser, a pipeline approach is constructed
that achieves a high recognition accuracy with minimal effort. As the characters
have unique visual structures and are usually clustered together, we train the
layout model to identify number regions with a dedicated category. Subsequently,
LayoutParser crops images within these regions, and identifies characters within
them using a self-trained OCR model based on a CNN-RNN [6]. The model
detects a total of 15 possible categories, and achieves a 0.98 Jaccard score16 and
a 0.17 average Levinstein distances17 for token prediction on the test set.
Overall, it is possible to create an intricate and highly accurate digitization
pipeline for large-scale digitization using LayoutParser. The pipeline avoids
specifying the complicated rules used in traditional methods, is straightforward
to develop, and is robust to outliers. The DL models also generate fine-grained
results that enable creative approaches like page reorganization for OCR.
16 This measures the overlap between the detected and ground-truth characters, and
the maximum is 1.
17 This measures the number of edits from the ground-truth text to the predicted text,
and lower is better.

LayoutParser: A Unified Toolkit for DL-Based DIA 13
Fig. 6: This lightweight table detector can identify tables (outlined in red) and
cells (shaded in blue) in different locations on a page. In very few cases (d), it
might generate minor error predictions, e.g, failing to capture the top text line of
a table.
5.2 A light-weight Visual Table Extractor
Detecting tables and parsing their structures (table extraction) are of central importance for many document digitization tasks. Many previous works [26, 30, 27]
and tools 18 have been developed to identify and parse table structures. Yet they
might require training complicated models from scratch, or are only applicable
for born-digital PDF documents. In this section, we show how LayoutParser can
help build a light-weight accurate visual table extractor for legal docket tables
using the existing resources with minimal effort.
The extractor uses a pre-trained layout detection model for identifying the
table regions and some simple rules for pairing the rows and the columns in the
PDF image. Mask R-CNN [12] trained on the PubLayNet dataset [38] from the
LayoutParser Model Zoo can be used for detecting table regions. By filtering
out model predictions of low confidence and removing overlapping predictions,
LayoutParser can identify the tabular regions on each page, which significantly
simplifies the subsequent steps. By applying the line detection functions within
the tabular segments, provided in the utility module from LayoutParser, the
pipeline can identify the three distinct columns in the tables. A row clustering
method is then applied via analyzing the y coordinates of token bounding boxes in
the left-most column, which are obtained from the OCR engines. A non-maximal
suppression algorithm is used to remove duplicated rows with extremely small
gaps. Shown in Figure 6, the built pipeline can detect tables at different positions
on a page accurately. Continued tables from different pages are concatenated,
and a structured table representation has been easily created.
18 https://github.com/atlanhq/camelot, https://github.com/tabulapdf/tabula



CM/ECF - District of Mi
urts.gov/cgi-bin/DktRpt.pl%637175371107253-L_9.
CM/ECF - District of Minnesota - Live - Docket Rcport 
epuurjso(/:sd
arts.gov/cgi-bin/DktRpt.pl7637175371107253-L_9
CM/ECF - District of Minn
sota - Live - Docket Report
https:/ecf.mnd.usc
Case: 7:97-v-2974
As of: 09/29/2012 12:31 PM EDT8 of 13
ATTORNEY TO BE NOTICED
04724/2002
PETTTION AND ORDER for admission pro hac vice of govermment atty. ( Clerk
attachmenl(s) added on 10/26/2004 (ak1). (Entered: 12/11/2003)
1007_A m
bs flad umtil I
16_1907_Dic
Richard D. Sletten ) on behalf of plaintiffby Rosemary J. Fox Ipg(s) (JMR)
Modifioed on 04/29/2002 (Entered: 04/29/2002)
01/13/2004
52NOTICE OF ASSIGNMENT OF CASES FOR TRIAL (Senior Judge David S.
stfrdtifusifadlthaJ
ses to sach interrogatories shall be served within 30 days thercafter. First
(HLL) (Entered: 01/13/2004)
Doty / 1/12/04)jury trial set for 9:00 a.m. on 4/26/04 . 3 pg(s) (cc: all counsel)
05/03/2002
Summons - RETURN OF SERVICE executed upon defendant Minnesota Beef
997.Depsostoled byDe599Ayfr
Defendant
ndustries on 5/1/02 2pg(s) (JMR) (Entered: 05/09/2002)
eogatrildingpetngareberedlathn
01/13/2004
NOTICE OF FINAL SETTLEMENT CONFERENCE (Magistrate Judge Susan
Nembe15.1997AlldisoveryistbcompletbyDecmbr31.1997
Minnesota Beef Industries, Inc,
represented by william J Egan
05/14/2002
4ANSWER by defendant 4pg(s) (JMR) (Entered: 05/17/2002)
R. Nelson / 1/13/04) final selement conference set for 9:30 a.m. on 4/8/04. 2
Echngfxpwiadbm997f
Oppenier,Wolf&Doelly
pg(s) (cc: all counsel) (HLL) Additional attachment(s) added on 10/26/2004
lassficimbyFeur899: shllyfb
assln9yd
Suite 3300, PLaza VII
45 South Seventh Street
05/21/2002
NOTICE OF PRETRIAL CONFERENCE (Magistrate Judge Susan R. Nelson/
(akl). (Entered: 01/15/2004)
Mach15998ytninsmayjdmbyhy
5/20/02) : pretrial conferene set for 2:30 6/21/02: Rule report d set for 6/10/02
mustarhdbvas
Minneapolis, MN 55402
pg(s) (cc: alcounsel) (JMR) (Entered: 05/23/2002)
02/26/2004
TRIAL NOTICE Jury Trial set for 4/26/2004 09:00 AM in Minneapolis -
onference Spem9.19978:45.signedbyJdChrsLBrient);
612-607-7509
Courtroom 14W before Senior Judge David S Doty. (PJM) (Entered: 02/26/2004)
Copies mailed. (ec) Modified on 05/19/1997 (Entered: 05/19/1997)
Fax: 612-607-7100
05/29/2002
6
Amended Notice Of Pretrial Conference (Magistrate Judge Susan R. Nelson /
MinteEny for prcdnheldbfeMagJe Su RNsFnal
06/26/1997
ANSWER o Complaint by The County of Wesch, AndrewJ.ORourke (Atomey
Email: wegan@oppenheimer.com
5/28/02) ; pretrial conference set for 1:30 6/18/02 ; Rule report d set for 6/10/02
04/08/2004
55
MathewT.Miklave),:Fim of:EstnBecker&Grn by atmey Mathew T
LEAD ATTORNEY
pg(s) (cc: allcounsel) (JMR) (Entered: 05/30/2002)
Settlement Conference held on 4/8/2004. No Settlement Reached (HLL)
Mikave for defendant Andrew J. ORourke (ds) (Entered: 07/01/1997)
ATTORNEY TO BE NOTICED
06/04/2002
REPORT OF RULE 26(f) MEETING by plaintiff, defendant 6pgs (JMR)
(Entered: 04/09/2004)
L661/20L0
5
y Mathew Miklave (ds) (Emtered: 07/07/1997)
NOTICE of ey appeae fTe County of Wesch, Andew J. ORouk
Interpleader
(Entered: 06/05/2002)
04/14/2004
56
Minute Entry for proceedings held before Mag.Judge Susan R Nelson:
06/20/2002
Pretrial SCHEDULING ORDER (Magistrate Judge Susan R. Nelson/ 6/19:02) :
Telephone Conference re videotaped discovery of meat-pcking plant to take
6661/91/00
Sheila Kutz
represented by Celeste E Culberth
place at 10:00 a.m. on 4/20/04 held on 4/14/2004. (HLL) (Entered: 04/15/2004)
y Mathew T. Miklave. Esq. (kz) (Entered: 07/15/1997)
Culberth & Lienemann, LLP
set for4/15/03: dispsitivemotions sefor6//03rcady fr trialset for8/1/03
amd complaint set for 8/1/02 ; discovery set foe 3/17/03 non-dispositive motions
1002/L1/90
07/22/1997
STIPULATION and ORDERitishereby stipandagred tht the time forboh
444 Cedar St Ste 1050
STIPULATION AND ORDER for Dismissal With Prejudice. Signed by Senior
St Paul, MN 55101
pg(s) (cc: counsel) (JMR) (Entered: 06/25/2002)
Judge David S Doty on 5/17/04. (HLL) (Entered: 05/19/2004)
efnshedndiugus9997P
651-290-9300
8/02/2002
9
MOTIONbymovant SheilaKut forleae tointvene as plaintiff 
05/17/2004
Consent Decree. Signod by Senior Judge David S Doty on 5/17/04. (HLL)
thReusfrPrdctionfocmensshalbxdedandnlig
Fax: 651-290-9305
Email: culberth@clslawyers.com
Magistrate Judge Susan R. Nelson ). 2 pg(s) (JVN) (Entered: 08/09/2002)
(Entered: 05/19/2004)
o and inclding Sembr 2, 1997. sied by Jde Charles L Brient ). (e)
LEAD ATTORNEY
08/02/2002
10DECLARATION of Celeste E.Culberth remotion for leave tointervene as
Entered: 07/22/1997)
ATTORNEY TO BE NOTICED
plaintiff. [9-1] 15 pg(s) (JVN) (Entered: 08/09/2002)
09/19/1997
Case Managm Conference heldby Jude Bricant. Transrip taken by
Adrienne Mignano (jac) (Enered: 09/22/1997)
Leslie L Lienemann
08/19/2002
11
Amended NOTICE by movant Sheila Kutz of hcaring seting hearing for motion
r leave to intervene as plaintiff to Magistrate Judge Susan R. Nelsn ) [9-1] af
PACER Service Center
10/23/1997
8
Letter filed dated Oetober 21, 1997 to Judge Brieant from atty Robert David
Culberth & Lieneman, LLP
444 Cedar St Ste 1050
(200/1/80 puu) (4r) (s)ds 20/7/6 02
Transsction Receipt
St Paul, MN 55101
06/01/2007 13:02:08
etained by Westcheser County, (ec) (Entered: 10/23/1997)
651-290-9300
08/21/2002
12RESPONSE by plaintiff to Sheila Kutz's motion to intervene [9-1] 1pg(s) (JMR)
PACER
Client
L661/20/11
ase Maem Confern held by Jude Brient Trsrip akenb Su
Fax: 651-290-9305
(Entered: 08/27/2002)
Login:
hs0328
Cede:
eeoc
Email: Ilienemann@clslawyers.com
09/18/2002
13STIPULATIONAND ORDER( Magistrate JudeSusan RNelsn）granting
6661/20/11
9tterled dat 1/97 toJdge Brict frm aty MathwT.Miklave re w
LEAD ATTORNEY
motion that Sheila Kutzmy intervene as plaintiffintervenr [9-1 11pg(s)ce: al
Destription:
Docket
Seareh
Report
Criteria:
0:02-cv-00810-DSD-SRN
write inresponse to pltfs Oet.2i, 1997 leter to this Court. ec) (Entered
ATTORNEY TO BE NOTICED
oumsel) (JMR) (Entered: 09/26/2002)
1/07/1997)
14AMENDED COMPLAINT [1-1] by plaintif, jury demand. 8pg(s) (VEM)
Billable
4
Cest:
0.32
11/13/1997
10eer to USDJ Brieant from atyh MahewMidae filed byThe County of
09/26/2002
Pages:
hdisdsond
WestchAndrewJ.ORorke dated Nov.121997e th paiesartial resolution
Date Filed
Docket Text
(Entered: 09/26/2002)
SUMMONS issued as to Minnesota Beef Ind (VEM) (Entered: 09/26/2002)
Entered: 11/13/1997)
09/26/2002
11/13/1997
11
er tat ca b rfed theClerk ofCot frassmn ta Maist
04/17/2002
COMPLAINT - Summons issued Assigned to Senior Judge David S. Doty pe
09/27/2002
15 OFFER OF JUDGMENT filed by defl. (3pg(s) (DDB) (Entered: 10/03/2002)
dgefoAllpuoesitd blw(sig byJeChalsBt）
Civil Rights list and refered to Magistrate Judge Susan R. Nelson 5pg(s) (
Referred to Magistrae Jodgc Mark D. Fox ds) (Entered: 1/13/1997)
Entered: 04/17/2002)
10/09/2002
16 AMENDMENT by defendant to offer of judgment [15-1]: re against dft jointly
1/21/1997
12RDERIdfcil thpgres ofpetrial disy f this litigat
y pltf and intervenor fo total amount of$40,000, cluing costs,
jst eedyandivemer,nsucmi a
04/17/2002
NOTICE given to atty admission clerk to mail a Govt PHV to out of state
isbursements, attomey fees 2pg(s) (SJH) (Entered: 10/17/2002)
adpaiddisry
ttomey: Gwendolyn Young Reams and Jean P. Kamp on bchalf of the pltf. (
follwigprocere willbefollwed frthresltonofdisoverydise
10/18/2002
17
(200//01 pouu) (Hr) (s)8d, pu ag mos suap q 3ASN
Fox ): Copies mailed ds) (Entered: 1/24/1997)
See document for details) . So Ordered; ( signed by Magistrate Judge Mark D.
Entered: 04/17/2002)
12/20/2002
18 RULE 7.1 DISCLOSURE STATEMENT by Minnesota Beef Ind that none exist
8661/L1/70
13
Transcript ofrecord of proceedings filed for dates of November 21, 1997 ()
Zpe(s) (DFL) (Entered: 01/03/2003)
: 032/17/1998)
2 of6
N65-21007/1/9
3 of 6
6/1/2007 12:59 PM
6 of 6
6/1/2007 12:59 PM
Partial tableatthebottom
(b) Full page table
(c)Partial table at the top
(d)Mis-detected textline

14 Z. Shen et al.
6 Conclusion
LayoutParser provides a comprehensive toolkit for deep learning-based document
image analysis. The off-the-shelf library is easy to install, and can be used to
build flexible and accurate pipelines for processing documents with complicated
structures. It also supports high-level customization and enables easy labeling and
training of DL models on unique document image datasets. The LayoutParser
community platform facilitates sharing DL models and DIA pipelines, inviting
discussion and promoting code reproducibility and reusability. The LayoutParser
team is committed to keeping the library updated continuously and bringing
the most recent advances in DL-based DIA, such as multi-modal document
modeling [37, 36, 9] (an upcoming priority), to a diverse audience of end-users.
Acknowledgements We thank the anonymous reviewers for their comments
and suggestions. This project is supported in part by NSF Grant OIA-2033558
and funding from the Harvard Data Science Initiative and Harvard Catalyst.
Zejiang Shen thanks Doug Downey for suggestions.
References
[1] Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado,
G.S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A.,
Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg,
J., Man´e, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J.,
Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V.,
Vi´egas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., Zheng,
X.: TensorFlow: Large-scale machine learning on heterogeneous systems (2015),
https://www.tensorflow.org/, software available from tensorflow.org
[2] Alberti, M., Pondenkandath, V., W¨ursch, M., Ingold, R., Liwicki, M.: Deepdiva: a
highly-functional python framework for reproducible experiments. In: 2018 16th
International Conference on Frontiers in Handwriting Recognition (ICFHR). pp.
423–428. IEEE (2018)
[3] Antonacopoulos, A., Bridson, D., Papadopoulos, C., Pletschacher, S.: A realistic
dataset for performance evaluation of document layout analysis. In: 2009 10th
International Conference on Document Analysis and Recognition. pp. 296–300.
IEEE (2009)
[4] Baek, Y., Lee, B., Han, D., Yun, S., Lee, H.: Character region awareness for text
detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 9365–9374 (2019)
[5] Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A Large-Scale
Hierarchical Image Database. In: CVPR09 (2009)
[6] Deng, Y., Kanervisto, A., Ling, J., Rush, A.M.: Image-to-markup generation with
coarse-to-fine attention. In: International Conference on Machine Learning. pp.
980–989. PMLR (2017)
[7] Ganin, Y., Lempitsky, V.: Unsupervised domain adaptation by backpropagation.
In: International conference on machine learning. pp. 1180–1189. PMLR (2015)

LayoutParser: A Unified Toolkit for DL-Based DIA 15
[8] Gardner, M., Grus, J., Neumann, M., Tafjord, O., Dasigi, P., Liu, N., Peters,
M., Schmitz, M., Zettlemoyer, L.: Allennlp: A deep semantic natural language
processing platform. arXiv preprint arXiv:1803.07640 (2018)
[9] Lukasz Garncarek, Powalski, R., Stanis lawek, T., Topolski, B., Halama, P.,
Grali´nski, F.: Lambert: Layout-aware (language) modeling using bert for information extraction (2020)
[10] Graves, A., Fern´andez, S., Gomez, F., Schmidhuber, J.: Connectionist temporal
classification: labelling unsegmented sequence data with recurrent neural networks.
In: Proceedings of the 23rd international conference on Machine learning. pp.
369–376 (2006)
[11] Harley, A.W., Ufkes, A., Derpanis, K.G.: Evaluation of deep convolutional nets for
document image classification and retrieval. In: 2015 13th International Conference
on Document Analysis and Recognition (ICDAR). pp. 991–995. IEEE (2015)
[12] He, K., Gkioxari, G., Doll´ar, P., Girshick, R.: Mask r-cnn. In: Proceedings of the
IEEE international conference on computer vision. pp. 2961–2969 (2017)
[13] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.
In: Proceedings of the IEEE conference on computer vision and pattern recognition.
pp. 770–778 (2016)
[14] Kay, A.: Tesseract: An open-source optical character recognition engine. Linux J.
2007(159), 2 (Jul 2007)
[15] Lamiroy, B., Lopresti, D.: An open architecture for end-to-end document analysis
benchmarking. In: 2011 International Conference on Document Analysis and
Recognition. pp. 42–47. IEEE (2011)
[16] Lee, B.C., Weld, D.S.: Newspaper navigator: Open faceted search for 1.5
million images. In: Adjunct Publication of the 33rd Annual ACM Symposium on User Interface Software and Technology. p. 120–122. UIST
’20 Adjunct, Association for Computing Machinery, New York, NY, USA
(2020). https://doi.org/10.1145/3379350.3416143, https://doi-org.offcampus.
lib.washington.edu/10.1145/3379350.3416143
[17] Lee, B.C.G., Mears, J., Jakeway, E., Ferriter, M., Adams, C., Yarasavage, N.,
Thomas, D., Zwaard, K., Weld, D.S.: The Newspaper Navigator Dataset: Extracting
Headlines and Visual Content from 16 Million Historic Newspaper Pages in
Chronicling America, p. 3055–3062. Association for Computing Machinery, New
York, NY, USA (2020), https://doi.org/10.1145/3340531.3412767
[18] Li, M., Cui, L., Huang, S., Wei, F., Zhou, M., Li, Z.: Tablebank: Table benchmark
for image-based table detection and recognition. arXiv preprint arXiv:1903.01949
(2019)
[19] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P.,
Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference
on computer vision. pp. 740–755. Springer (2014)
[20] Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic
segmentation. In: Proceedings of the IEEE conference on computer vision and
pattern recognition. pp. 3431–3440 (2015)
[21] Neudecker, C., Schlarb, S., Dogan, Z.M., Missier, P., Sufi, S., Williams, A., Wolstencroft, K.: An experimental workflow development platform for historical document
digitisation and analysis. In: Proceedings of the 2011 workshop on historical
document imaging and processing. pp. 161–168 (2011)
[22] Oliveira, S.A., Seguin, B., Kaplan, F.: dhsegment: A generic deep-learning approach
for document segmentation. In: 2018 16th International Conference on Frontiers
in Handwriting Recognition (ICFHR). pp. 7–12. IEEE (2018)

16 Z. Shen et al.
[23] Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z.,
Desmaison, A., Antiga, L., Lerer, A.: Automatic differentiation in pytorch (2017)
[24] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen,
T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style,
high-performance deep learning library. arXiv preprint arXiv:1912.01703 (2019)
[25] Pletschacher, S., Antonacopoulos, A.: The page (page analysis and ground-truth
elements) format framework. In: 2010 20th International Conference on Pattern
Recognition. pp. 257–260. IEEE (2010)
[26] Prasad, D., Gadpal, A., Kapadni, K., Visave, M., Sultanpure, K.: Cascadetabnet:
An approach for end to end table detection and structure recognition from imagebased documents. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition Workshops. pp. 572–573 (2020)
[27] Qasim, S.R., Mahmood, H., Shafait, F.: Rethinking table recognition using graph
neural networks. In: 2019 International Conference on Document Analysis and
Recognition (ICDAR). pp. 142–147. IEEE (2019)
[28] Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object
detection with region proposal networks. In: Advances in neural information
processing systems. pp. 91–99 (2015)
[29] Scarselli, F., Gori, M., Tsoi, A.C., Hagenbuchner, M., Monfardini, G.: The graph
neural network model. IEEE transactions on neural networks 20(1), 61–80 (2008)
[30] Schreiber, S., Agne, S., Wolf, I., Dengel, A., Ahmed, S.: Deepdesrt: Deep learning
for detection and structure recognition of tables in document images. In: 2017 14th
IAPR international conference on document analysis and recognition (ICDAR).
vol. 1, pp. 1162–1167. IEEE (2017)
[31] Shen, Z., Zhang, K., Dell, M.: A large dataset of historical japanese documents
with complex layouts. In: Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition Workshops. pp. 548–549 (2020)
[32] Shen, Z., Zhao, J., Dell, M., Yu, Y., Li, W.: Olala: Object-level active learning
based layout annotation. arXiv preprint arXiv:2010.01762 (2020)
[33] Studer, L., Alberti, M., Pondenkandath, V., Goktepe, P., Kolonko, T., Fischer,
A., Liwicki, M., Ingold, R.: A comprehensive study of imagenet pre-training for
historical document image analysis. In: 2019 International Conference on Document
Analysis and Recognition (ICDAR). pp. 720–725. IEEE (2019)
[34] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P.,
Rault, T., Louf, R., Funtowicz, M., et al.: Huggingface’s transformers: State-ofthe-art natural language processing. arXiv preprint arXiv:1910.03771 (2019)
[35] Wu, Y., Kirillov, A., Massa, F., Lo, W.Y., Girshick, R.: Detectron2. https://
github.com/facebookresearch/detectron2 (2019)
[36] Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florencio, D., Zhang, C.,
Che, W., et al.: Layoutlmv2: Multi-modal pre-training for visually-rich document
understanding. arXiv preprint arXiv:2012.14740 (2020)
[37] Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., Zhou, M.: Layoutlm: Pre-training of
text and layout for document image understanding (2019)
[38] Zhong, X., Tang, J., Yepes, A.J.: Publaynet: largest dataset ever for document layout analysis. In: 2019 International Conference on Document
Analysis and Recognition (ICDAR). pp. 1015–1022. IEEE (Sep 2019).
https://doi.org/10.1109/ICDAR.2019.00166

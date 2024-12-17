1
2
0
2

n
u
J

1
2

]

V
C
.
s
c
[

2
v
8
4
3
5
1
.
3
0
1
2
:
v
i
X
r
a

LayoutParser: A Uniﬁed Toolkit for Deep
Learning Based Document Image Analysis

Zejiang Shen1 ((cid:0)), Ruochen Zhang2, Melissa Dell3, Benjamin Charles Germain
Lee4, Jacob Carlson3, and Weining Li5

1 Allen Institute for AI
shannons@allenai.org
2 Brown University
ruochen zhang@brown.edu
3 Harvard University
{melissadell,jacob carlson}@fas.harvard.edu
4 University of Washington
bcgl@cs.washington.edu
5 University of Waterloo
w422li@uwaterloo.ca

Abstract. Recent advances in document image analysis (DIA) have been
primarily driven by the application of neural networks. Ideally, research
outcomes could be easily deployed in production and extended for further
investigation. However, various factors like loosely organized codebases
and sophisticated model conﬁgurations complicate the easy reuse of im-
portant innovations by a wide audience. Though there have been on-going
eﬀorts to improve reusability and simplify deep learning (DL) model
development in disciplines like natural language processing and computer
vision, none of them are optimized for challenges in the domain of DIA.
This represents a major gap in the existing toolkit, as DIA is central to
academic research across a wide range of disciplines in the social sciences
and humanities. This paper introduces LayoutParser, an open-source
library for streamlining the usage of DL in DIA research and applica-
tions. The core LayoutParser library comes with a set of simple and
intuitive interfaces for applying and customizing DL models for layout de-
tection, character recognition, and many other document processing tasks.
To promote extensibility, LayoutParser also incorporates a community
platform for sharing both pre-trained models and full document digiti-
zation pipelines. We demonstrate that LayoutParser is helpful for both
lightweight and large-scale digitization pipelines in real-word use cases.
The library is publicly available at https://layout-parser.github.io.

Keywords: Document Image Analysis · Deep Learning · Layout Analysis
· Character Recognition · Open Source library · Toolkit.

1

Introduction

Deep Learning(DL)-based approaches are the state-of-the-art for a wide range of
document image analysis (DIA) tasks including document image classiﬁcation [11,

 
 
 
 
 
 

2

Z. Shen et al.

37], layout detection [38, 22], table detection [26], and scene text detection [4].
A generalized learning-based framework dramatically reduces the need for the
manual speciﬁcation of complicated rules, which is the status quo with traditional
methods. DL has the potential to transform DIA pipelines and beneﬁt a broad
spectrum of large-scale document digitization projects.

However, there are several practical diﬃculties for taking advantages of re-
cent advances in DL-based methods: 1) DL models are notoriously convoluted
for reuse and extension. Existing models are developed using distinct frame-
works like TensorFlow [1] or PyTorch [24], and the high-level parameters can
be obfuscated by implementation details [8]. It can be a time-consuming and
frustrating experience to debug, reproduce, and adapt existing models for DIA,
and many researchers who would beneﬁt the most from using these methods lack
the technical background to implement them from scratch. 2) Document images
contain diverse and disparate patterns across domains, and customized training
is often required to achieve a desirable detection accuracy. Currently there is no
full-ﬂedged infrastructure for easily curating the target document image datasets
and ﬁne-tuning or re-training the models. 3) DIA usually requires a sequence of
models and other processing to obtain the ﬁnal outputs. Often research teams use
DL models and then perform further document analyses in separate processes,
and these pipelines are not documented in any central location (and often not
documented at all). This makes it diﬃcult for research teams to learn about how
full pipelines are implemented and leads them to invest signiﬁcant resources in
reinventing the DIA wheel.

LayoutParser provides a uniﬁed toolkit to support DL-based document image
analysis and processing. To address the aforementioned challenges, LayoutParser
is built with the following components:

1. An oﬀ-the-shelf toolkit for applying DL models for layout detection, character

recognition, and other DIA tasks (Section 3)

2. A rich repository of pre-trained neural network models (Model Zoo) that

underlies the oﬀ-the-shelf usage

3. Comprehensive tools for eﬃcient document image data annotation and model

tuning to support diﬀerent levels of customization

4. A DL model hub and community platform for the easy sharing, distribu-
tion, and discussion of DIA models and pipelines, to promote reusability,
reproducibility, and extensibility (Section 4)

The library implements simple and intuitive Python APIs without sacriﬁcing
generalizability and versatility, and can be easily installed via pip. Its convenient
functions for handling document image data can be seamlessly integrated with
existing DIA pipelines. With detailed documentations and carefully curated
tutorials, we hope this tool will beneﬁt a variety of end-users, and will lead to
advances in applications in both industry and academic research.

LayoutParser is well aligned with recent eﬀorts for improving DL model
reusability in other disciplines like natural language processing [8, 34] and com-
puter vision [35], but with a focus on unique challenges in DIA. We show
LayoutParser can be applied in sophisticated and large-scale digitization projects


LayoutParser: A Uniﬁed Toolkit for DL-Based DIA

3

that require precision, eﬃciency, and robustness, as well as simple and light-
weight document processing tasks focusing on eﬃcacy and ﬂexibility (Section 5).
LayoutParser is being actively maintained, and support for more deep learning
models and novel methods in text-based layout analysis methods [37, 34] is
planned.

The rest of the paper is organized as follows. Section 2 provides an overview
of related work. The core LayoutParser library, DL Model Zoo, and customized
model training are described in Section 3, and the DL model hub and commu-
nity platform are detailed in Section 4. Section 5 shows two examples of how
LayoutParser can be used in practical DIA projects, and Section 6 concludes.

2 Related Work

Recently, various DL models and datasets have been developed for layout analysis
tasks. The dhSegment [22] utilizes fully convolutional networks [20] for segmen-
tation tasks on historical documents. Object detection-based methods like Faster
R-CNN [28] and Mask R-CNN [12] are used for identifying document elements [38]
and detecting tables [30, 26]. Most recently, Graph Neural Networks [29] have also
been used in table detection [27]. However, these models are usually implemented
individually and there is no uniﬁed framework to load and use such models.

There has been a surge of interest in creating open-source tools for document
image processing: a search of document image analysis in Github leads to 5M
relevant code pieces 6; yet most of them rely on traditional rule-based methods
or provide limited functionalities. The closest prior research to our work is the
OCR-D project7, which also tries to build a complete toolkit for DIA. However,
similar to the platform developed by Neudecker et al. [21], it is designed for
analyzing historical documents, and provides no supports for recent DL models.
The DocumentLayoutAnalysis project8 focuses on processing born-digital PDF
documents via analyzing the stored PDF data. Repositories like DeepLayout9
and Detectron2-PubLayNet10 are individual deep learning models trained on
layout analysis datasets without support for the full DIA pipeline. The Document
Analysis and Exploitation (DAE) platform [15] and the DeepDIVA project [2]
aim to improve the reproducibility of DIA methods (or DL models), yet they
are not actively maintained. OCR engines like Tesseract [14], easyOCR11 and
paddleOCR12 usually do not come with comprehensive functionalities for other
DIA tasks like layout analysis.

Recent years have also seen numerous eﬀorts to create libraries for promoting
reproducibility and reusability in the ﬁeld of DL. Libraries like Dectectron2 [35],

6 The number shown is obtained by specifying the search type as ‘code’.
7 https://ocr-d.de/en/about
8 https://github.com/BobLd/DocumentLayoutAnalysis
9 https://github.com/leonlulu/DeepLayout
10 https://github.com/hpanwar08/detectron2
11 https://github.com/JaidedAI/EasyOCR
12 https://github.com/PaddlePaddle/PaddleOCR


4

Z. Shen et al.

Fig. 1: The overall architecture of LayoutParser. For an input document image,
the core LayoutParser library provides a set of oﬀ-the-shelf tools for layout
detection, OCR, visualization, and storage, backed by a carefully designed layout
data structure. LayoutParser also supports high level customization via eﬃcient
layout annotation and model training functions. These improve model accuracy
on the target samples. The community platform enables the easy sharing of DIA
models and whole digitization pipelines to promote reusability and reproducibility.
A collection of detailed documentation, tutorials and exemplar projects make
LayoutParser easy to learn and use.

AllenNLP [8] and transformers [34] have provided the community with complete
DL-based support for developing and deploying models for general computer
vision and natural language processing problems. LayoutParser, on the other
hand, specializes speciﬁcally in DIA tasks. LayoutParser is also equipped with a
community platform inspired by established model hubs such as Torch Hub [23]
and TensorFlow Hub [1]. It enables the sharing of pretrained models as well as
full document processing pipelines that are unique to DIA tasks.

There have been a variety of document data collections to facilitate the
development of DL models. Some examples include PRImA [3](magazine layouts),
PubLayNet [38](academic paper layouts), Table Bank [18](tables in academic
papers), Newspaper Navigator Dataset [16, 17](newspaper ﬁgure layouts) and
HJDataset [31](historical Japanese document layouts). A spectrum of models
trained on these datasets are currently available in the LayoutParser model zoo
to support diﬀerent use cases.

3 The Core LayoutParser Library

At the core of LayoutParser is an oﬀ-the-shelf toolkit that streamlines DL-
based document image analysis. Five components support a simple interface
with comprehensive functionalities: 1) The layout detection models enable using
pre-trained or self-trained DL models for layout detection with just four lines
of code. 2) The detected layout information is stored in carefully engineered

Efficient Data AnnotationCustomized Model TrainingModel CustomizationDIA Model HubDIA Pipeline SharingCommunity PlatformLayout Detection ModelsDocument Images The Core LayoutParser LibraryOCR ModuleStorage & VisualizationLayout Data Structure
LayoutParser: A Uniﬁed Toolkit for DL-Based DIA

5

Table 1: Current layout detection models in the LayoutParser model zoo

Dataset

Base Model1 Large Model Notes

PubLayNet [38]
PRImA [3]
Newspaper [17]
TableBank [18]
HJDataset [31]

F / M
M
F
F
F / M

M
-
-
F
-

Layouts of modern scientiﬁc documents
Layouts of scanned modern magazines and scientiﬁc reports
Layouts of scanned US newspapers from the 20th century
Table region on modern scientiﬁc and business document
Layouts of history Japanese documents

1 For each dataset, we train several models of diﬀerent sizes for diﬀerent needs (the trade-oﬀ between accuracy
vs. computational cost). For “base model” and “large model”, we refer to using the ResNet 50 or ResNet 101
backbones [13], respectively. One can train models of diﬀerent architectures, like Faster R-CNN [28] (F) and Mask
R-CNN [12] (M). For example, an F in the Large Model column indicates it has a Faster R-CNN model trained
using the ResNet 101 backbone. The platform is maintained and a number of additions will be made to the model
zoo in coming months.

layout data structures, which are optimized for eﬃciency and versatility. 3) When
necessary, users can employ existing or customized OCR models via the uniﬁed
API provided in the OCR module. 4) LayoutParser comes with a set of utility
functions for the visualization and storage of the layout data. 5) LayoutParser
is also highly customizable, via its integration with functions for layout data
annotation and model training. We now provide detailed descriptions for each
component.

3.1 Layout Detection Models

In LayoutParser, a layout model takes a document image as an input and
generates a list of rectangular boxes for the target content regions. Diﬀerent
from traditional methods, it relies on deep convolutional neural networks rather
than manually curated rules to identify content regions. It is formulated as an
object detection problem and state-of-the-art models like Faster R-CNN [28] and
Mask R-CNN [12] are used. This yields prediction results of high accuracy and
makes it possible to build a concise, generalized interface for layout detection.
LayoutParser, built upon Detectron2 [35], provides a minimal API that can
perform layout detection with only four lines of code in Python:

1 import layoutparser as lp
2 image = cv2 . imread ( " image_file " ) # load images
3 model = lp . De t e c tro n2 Lay outM odel (

" lp :// PubLayNet / f as t er _ r c nn _ R _ 50 _ F P N_ 3 x / config " )

4
5 layout = model . detect ( image )

LayoutParser provides a wealth of pre-trained model weights using various
datasets covering diﬀerent languages, time periods, and document types. Due to
domain shift [7], the prediction performance can notably drop when models are ap-
plied to target samples that are signiﬁcantly diﬀerent from the training dataset. As
document structures and layouts vary greatly in diﬀerent domains, it is important
to select models trained on a dataset similar to the test samples. A semantic syntax
is used for initializing the model weights in LayoutParser, using both the dataset
name and model name lp://<dataset-name>/<model-architecture-name>.


6

Z. Shen et al.

Fig. 2: The relationship between the three types of layout data structures.
Coordinate supports three kinds of variation; TextBlock consists of the co-
ordinate information and extra features like block text, types, and reading orders;
a Layout object is a list of all possible layout elements, including other Layout
objects. They all support the same set of transformation and operation APIs for
maximum ﬂexibility.

Shown in Table 1, LayoutParser currently hosts 9 pre-trained models trained
on 5 diﬀerent datasets. Description of the training dataset is provided alongside
with the trained models such that users can quickly identify the most suitable
models for their tasks. Additionally, when such a model is not readily available,
LayoutParser also supports training customized layout models and community
sharing of the models (detailed in Section 3.5).

3.2 Layout Data Structures

A critical feature of LayoutParser is the implementation of a series of data
structures and operations that can be used to eﬃciently process and manipulate
the layout elements. In document image analysis pipelines, various post-processing
on the layout analysis model outputs is usually required to obtain the ﬁnal
outputs. Traditionally, this requires exporting DL model outputs and then loading
the results into other pipelines. All model outputs from LayoutParser will be
stored in carefully engineered data types optimized for further processing, which
makes it possible to build an end-to-end document digitization pipeline within
LayoutParser. There are three key components in the data structure, namely
the Coordinate system, the TextBlock, and the Layout. They provide diﬀerent
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
LayoutParser: A Uniﬁed Toolkit for DL-Based DIA

7

Coordinates are the cornerstones for storing layout information. Currently,
three types of Coordinate data structures are provided in LayoutParser, shown
in Figure 2. Interval and Rectangle are the most common data types and
support specifying 1D or 2D regions within a document. They are parameterized
with 2 and 4 parameters. A Quadrilateral class is also implemented to support
a more generalized representation of rectangular regions when the document
is skewed or distorted, where the 4 corner points can be speciﬁed and a total
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
specifying the reading orders via setting the parent ﬁeld to the index of the parent
object. A Layout class is built that takes in a list of TextBlocks and supports
processing the elements in batch. Layout can also be nested to support hierarchical
layout structures. They support the same operations and transformations as the
Coordinate classes, minimizing both learning and deployment eﬀort.

3.3 OCR

LayoutParser provides a uniﬁed interface for existing OCR tools. Though there
are many OCR tools available, they are usually conﬁgured diﬀerently with distinct
APIs or protocols for using them. It can be ineﬃcient to add new OCR tools into
an existing pipeline, and diﬃcult to make direct comparisons among the available
tools to ﬁnd the best option for a particular project. To this end, LayoutParser
builds a series of wrappers among existing OCR engines, and provides nearly
the same syntax for using them. It supports a plug-and-play style of using OCR
engines, making it eﬀortless to switch, evaluate, and compare diﬀerent OCR
modules:

1 ocr_agent = lp . TesseractAgent ()
2 # Can be easily switched to other OCR software
3 tokens = ocr_agent . detect ( image )

The OCR outputs will also be stored in the aforementioned layout data
structures and can be seamlessly incorporated into the digitization pipeline.
Currently LayoutParser supports the Tesseract and Google Cloud Vision OCR
engines.

LayoutParser also comes with a DL-based CNN-RNN OCR model [6] trained
with the Connectionist Temporal Classiﬁcation (CTC) loss [10]. It can be used
like the other OCR modules, and can be easily trained on customized datasets.

13 This is also available in the LayoutParser documentation pages.


8

Z. Shen et al.

Table 2: All operations supported by the layout elements. The same APIs are
supported across diﬀerent layout element classes including Coordinate types,
TextBlock and Layout.

Operation Name

Description

block.pad(top, bottom, right, left) Enlarge the current block according to the input

block.scale(fx, fy)

block.shift(dx, dy)

Scale the current block given the ratio
in x and y direction

Move the current block with the shift
distances in x and y direction

block1.is in(block2)

Whether block1 is inside of block2

block1.intersect(block2)

block1.union(block2)

block1.relative to(block2)

block1.condition on(block2)

Return the intersection region of block1 and block2.
Coordinate type to be determined based on the inputs.

Return the union region of block1 and block2.
Coordinate type to be determined based on the inputs.

Convert the absolute coordinates of block1 to
relative coordinates to block2

Calculate the absolute coordinates of block1 given
the canvas block2’s absolute coordinates

block.crop image(image)

Obtain the image segments in the block region

3.4 Storage and visualization

The end goal of DIA is to transform the image-based document data into a
structured database. LayoutParser supports exporting layout data into diﬀerent
formats like JSON, csv, and will add the support for the METS/ALTO XML
format 14 . It can also load datasets from layout analysis-speciﬁc formats like
COCO [38] and the Page Format [25] for training layout models (Section 3.5).
Visualization of the layout detection results is critical for both presentation
and debugging. LayoutParser is built with an integrated API for displaying the
layout information along with the original document image. Shown in Figure 3, it
enables presenting layout data with rich meta information and features in diﬀerent
modes. More detailed information can be found in the online LayoutParser
documentation page.

3.5 Customized Model Training

Besides the oﬀ-the-shelf library, LayoutParser is also highly customizable with
supports for highly unique and challenging document analysis tasks. Target
document images can be vastly diﬀerent from the existing datasets for train-
ing layout models, which leads to low layout detection accuracy. Training data

14 https://altoxml.github.io


LayoutParser: A Uniﬁed Toolkit for DL-Based DIA

9

Fig. 3: Layout detection and OCR results visualization generated by the
LayoutParser APIs. Mode I directly overlays the layout region bounding boxes
and categories over the original image. Mode II recreates the original document
via drawing the OCR’d texts at their corresponding positions on the image
canvas. In this ﬁgure, tokens in textual regions are ﬁltered using the API and
then displayed.

can also be highly sensitive and not sharable publicly. To overcome these chal-
lenges, LayoutParser is built with rich features for eﬃcient data annotation and
customized model training.

LayoutParser incorporates a toolkit optimized for annotating document lay-
outs using object-level active learning [32]. With the help from a layout detection
model trained along with labeling, only the most important layout objects within
each image, rather than the whole image, are required for labeling. The rest of
the regions are automatically annotated with high conﬁdence predictions from
the layout detection model. This allows a layout dataset to be created more
eﬃciently with only around 60% of the labeling budget.

After the training dataset is curated, LayoutParser supports diﬀerent modes
for training the layout models. Fine-tuning can be used for training models on a
small newly-labeled dataset by initializing the model with existing pre-trained
weights. Training from scratch can be helpful when the source dataset and
target are signiﬁcantly diﬀerent and a large training set is available. However, as
suggested in Studer et al.’s work[33], loading pre-trained weights on large-scale
datasets like ImageNet [5], even from totally diﬀerent domains, can still boost
model performance. Through the integrated API provided by LayoutParser,
users can easily compare model performances on the benchmark datasets.

Option 1: Display Token Bounding Box
Anonymous ICDAR 2021 Submission
Anonymous[CDAR2021 Submission
Figure
Model Customization
Document Images
CommunityPlatform
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
:The overall architectureofLayoutParser.Foraninputdocumentimage
Fig.
The
bverall
architecture
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
setbfoffthe-shelf
tools
for
layout
detection,OCR,visualization, and storage,backed by a carefully designed layout
detection,
OCR,
visualization,
and
storage,
backedbycarefullydesigned
layout
data structure.LayoutParser also supports high level of customization via
datastructure.
LayoutParser
also
supports
highlevelbf customization
via
efficientlayout data annotation and model training functions thatimproves
efficient
ayout
Hataannotation
and
model
training
functionsthatmproves
model accuracy on the target samples.The community platform enables the
model
target
samples.
The
community
platform
enahles
the
easy share of DIA models and even whole digitization pipelines to promote
accuracy
share
models
and
whole
digitization
pipelines
to
promote
reusability and reproducibility.A collection of detailed documentations,tutorials
reusability
and
reproducibility.
collection
fdetailed
Hocumentations,
tutorials
andexemplarprojectsmakesLayoutParsereasytolearn anduse.
andexemplar
projects
makes
LavoutParser
easy learn and
LayoutParser
alsohighly
customizable,
integrated
withfunctions
for
layout
data annotation andmodel training.Weprovidedetaileddescriptionsfor each
dlata
annotation
and
model
training.
We
provide
HetailedHescriptions
for
each
component as follows.
component
Follows.
3.1Titleryout Detection Models
3.1
Layout
Detection
Models
Option 2: Hide Token Bounding Box
In LayoutParser,
layout
model
takes
adocument
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
neuralnetworksrather than
manually curatedrulesfor identifying the contentregions.It is formulated as
manually
curated
1rulesfor identifying
the content regions.It is formulated
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
nighaccuracy,but also makes it possible tobuild a concise while generalizec
high accuracy,
but also makes it possible to build a concise while generalized
nterface for using thelayout detection models. In fact,built upon Detectron2 [28]
LayoutParser provides a minimal API that one can perform layout detection
LayoutParser
vith only four lines of code in Python:
with only four lines of code in Python:
Listport
dT se 1osiedanoAet 1
import
layoutparser
aslp
image
cv2.imread("image_file"） # load images
2 image
cv2.imread("image_file")
#load
images
model：
=lp.Detectron2LayoutModel(
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
10

Z. Shen et al.

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

Beyond DL models, LayoutParser also promotes the sharing of entire doc-
ument digitization pipelines. For example, sometimes the pipeline requires the
combination of multiple DL models to achieve better accuracy. Currently, pipelines
are mainly described in academic papers and implementations are often not pub-
licly available. To this end, the LayoutParser community platform also enables
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
最引银行京都（峰山）丹
天株主
代大下食保一
决算期六月
毅立
設備
取引银行滋貨（九條）
年商内高一四千六百万圆内外
悠巢具
大株主
代西村永治郎
資本额
目的
毅立昭和州年十
取引银行三和（西陣）
大口出餐者
盛江局吟次郎
决算期八月配當無
目
設立
Token Categories
大山中錦造
代光村
决算期
餐本额二百二十萬具（千四
装具
株榮光商店
西村永治郎
快算期
中央信金（榮）
年商丙富
夜野
天下食八重
英
口O年
尿和化成工業淋
江局喜
百口）
江
Column reading order
天
人下詹知惠八烟市郎
西陣物製造服
搬雅服飾品製造贩颤
一昭和十八年四月一
一人織物造贩
昭和廿八年七月
一化學药品製造
一昭租廿五年九月
寻西人電四局二
上京區元誓寺通净
Title
東部
中京温西）京左馬察町
昌
南區吉祥院落合六四
竹野都斓榮町和田野
三月配富
【株主数七）
（株主数一一）
五月
百万凰（二千株）
和雄
百莫冒（二千株）
脱水機三、其他六
四千
Address
四
郎
織
一千八百萬冒内外
識
郎吉圖
電五局六三三七
出餐者敷
荒川正太郎
配富無
瑞吉田
配當無
江局
Text
翻植
電八四局三六九
百萬貢内外
物有
電满谷七七
Number
物株
国
建物三
·割
七00糕
四00株
一次郎
Variable
一男
Company Type
(a) llustration of the original Japanese document with detected layout elements highlighted in colored boxes 
Column Categories
Title
Maximum Allowed Height 
附大下會保一郎大下會留藏
OO坪、脱水機三、其他六
南區吉祥院落合六四
上京區元誓寺通澄漏
中京温西京左馬寮町
敷地四のO坪
寺西人龙四局二八三二
東部
永和化成工業
竹野郡彌榮町和田野
人物裂造服賣
微釉服飾品製造贩賣
野专村儀一郎吉网
四千六百万圆内外
四千二百万闻内外
西陣物製造
二千八百万圆内外
京都（岭山）丹後有
昭和廿八年四月
昭和廿八年七月
百万圆（二千株）
昭和廿五年九月目的
（出资者数一O）
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
溝谷七七設立夜野
大下八重
三和（西陣）
滋贺（九條）
大下會知惠
百二十萬
大口出资者
一
西村永治郎
百二十万圆
（株主数七）资本
年商内高资本额
取引银行
取
年
取引銀行配當無
年商内高决算期
二千二决算期
（千四
业具
代光村大株主
Address
引银行
商内高
一0電公局三六九
Text
业具
八烟
配當無
决算期
大株主
織
资本金
资本
SectionHeader
設備
市郎
白郎
和雄
百株）
吉田
决算期
建物三
有
八月
設立
重雄
三月
壽男
目的
江局
設立
配當無
物
一五
楠
五月
毛
筛造
次郎
立
百口
株
株
株
(b) Illustration of the recreated document with dense text structure for better OCR performance
LayoutParser: A Uniﬁed Toolkit for DL-Based DIA

11

focuses on precision, eﬃciency, and robustness. The target documents may have
complicated structures, and may require training multiple layout detection models
to achieve the optimal accuracy. Light-weight pipelines are built for relatively
simple documents, with an emphasis on development ease, speed and ﬂexibility.
Ideally one only needs to use existing resources, and model training should be
avoided. Through two exemplar projects, we show how practitioners in both
academia and industry can easily build such pipelines using LayoutParser and
extract high-quality structured document data for their downstream tasks. The
source code for these projects will be publicly available in the LayoutParser
community hub.

5.1 A Comprehensive Historical Document Digitization Pipeline

The digitization of historical documents can unlock valuable data that can shed
light on many important social, economic, and historical questions. Yet due to
scan noises, page wearing, and the prevalence of complicated layout structures, ob-
taining a structured representation of historical document scans is often extremely
complicated.
In this example, LayoutParser was
used to develop a comprehensive
pipeline, shown in Figure 5, to gener-
ate high-quality structured data from
historical Japanese ﬁrm ﬁnancial ta-
bles with complicated layouts. The
pipeline applies two layout models to
identify diﬀerent levels of document
structures and two customized OCR
engines for optimized character recog-
nition accuracy.

As shown in Figure 4 (a), the
document contains columns of text
written vertically 15, a common style
in Japanese. Due to scanning noise
and archaic printing technology, the
columns can be skewed or have vari-
able widths, and hence cannot be eas-
ily identiﬁed via rule-based methods.
Within each column, words are sepa-
rated by white spaces of variable size,
and the vertical positions of objects
can be an indicator of their layout
type.

Fig. 5: Illustration of how LayoutParser
helps with the historical document digi-
tization pipeline.

15 A document page consists of eight rows like this. For simplicity we skip the row

segmentation discussion and refer readers to the source code when available.

Active Learning Layout
Annotate Layout Dataset
Annotation Toolkit
Deep Learning Layout
Layout Detection
Model Training &Inference
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
12

Z. Shen et al.

To decipher the complicated layout

structure, two object detection models have been trained to recognize individual
columns and tokens, respectively. A small training set (400 images with approxi-
mately 100 annotations each) is curated via the active learning based annotation
tool [32] in LayoutParser. The models learn to identify both the categories and
regions for each token or column via their distinct visual features. The layout
data structure enables easy grouping of the tokens within each column, and
rearranging columns to achieve the correct reading orders based on the horizontal
position. Errors are identiﬁed and rectiﬁed via checking the consistency of the
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
sent to the OCR APIs as a whole to reduce the transaction costs. The ﬂexible
coordinate system in LayoutParser is used to transform the OCR results relative
to their original positions on the page.

Additionally, it is common for historical documents to use unique fonts
with diﬀerent glyphs, which signiﬁcantly degrades the accuracy of OCR models
trained on modern texts. In this document, a special ﬂat font is used for printing
numbers and could not be detected by oﬀ-the-shelf OCR engines. Using the highly
ﬂexible functionalities from LayoutParser, a pipeline approach is constructed
that achieves a high recognition accuracy with minimal eﬀort. As the characters
have unique visual structures and are usually clustered together, we train the
layout model to identify number regions with a dedicated category. Subsequently,
LayoutParser crops images within these regions, and identiﬁes characters within
them using a self-trained OCR model based on a CNN-RNN [6]. The model
detects a total of 15 possible categories, and achieves a 0.98 Jaccard score16 and
a 0.17 average Levinstein distances17 for token prediction on the test set.

Overall, it is possible to create an intricate and highly accurate digitization
pipeline for large-scale digitization using LayoutParser. The pipeline avoids
specifying the complicated rules used in traditional methods, is straightforward
to develop, and is robust to outliers. The DL models also generate ﬁne-grained
results that enable creative approaches like page reorganization for OCR.

16 This measures the overlap between the detected and ground-truth characters, and

the maximum is 1.

17 This measures the number of edits from the ground-truth text to the predicted text,

and lower is better.


LayoutParser: A Uniﬁed Toolkit for DL-Based DIA

13

Fig. 6: This lightweight table detector can identify tables (outlined in red) and
cells (shaded in blue) in diﬀerent locations on a page. In very few cases (d), it
might generate minor error predictions, e.g, failing to capture the top text line of
a table.

5.2 A light-weight Visual Table Extractor

Detecting tables and parsing their structures (table extraction) are of central im-
portance for many document digitization tasks. Many previous works [26, 30, 27]
and tools 18 have been developed to identify and parse table structures. Yet they
might require training complicated models from scratch, or are only applicable
for born-digital PDF documents. In this section, we show how LayoutParser can
help build a light-weight accurate visual table extractor for legal docket tables
using the existing resources with minimal eﬀort.

The extractor uses a pre-trained layout detection model for identifying the
table regions and some simple rules for pairing the rows and the columns in the
PDF image. Mask R-CNN [12] trained on the PubLayNet dataset [38] from the
LayoutParser Model Zoo can be used for detecting table regions. By ﬁltering
out model predictions of low conﬁdence and removing overlapping predictions,
LayoutParser can identify the tabular regions on each page, which signiﬁcantly
simpliﬁes the subsequent steps. By applying the line detection functions within
the tabular segments, provided in the utility module from LayoutParser, the
pipeline can identify the three distinct columns in the tables. A row clustering
method is then applied via analyzing the y coordinates of token bounding boxes in
the left-most column, which are obtained from the OCR engines. A non-maximal
suppression algorithm is used to remove duplicated rows with extremely small
gaps. Shown in Figure 6, the built pipeline can detect tables at diﬀerent positions
on a page accurately. Continued tables from diﬀerent pages are concatenated,
and a structured table representation has been easily created.

18 https://github.com/atlanhq/camelot, https://github.com/tabulapdf/tabula

CM/ECF - District of Mi
urts.gov/cgi-bin/DktRpt.,pl7%37175371107253-L_9.
CM/ECF - District of Minne
sota - Live - Docket Rcport
rts.gov/cgi-bin/DktRpt.pl?%637175371107253-L_9.
CM/ECF - District of Min
ttps://ect.mnd
Case: 7:97-cv-2974
As of: 09/29/2012 12:31 PM EDT8 of 13
ATTORNEY TO BE NOTICED
4724/2002
PETITION AND ORDER for admmission pro hac vice of govemment atty. ( Clerk
attachmen(s) added on 10/26/2004 (akl). (Entered: 12/11/2003)
Richard D. Sletten ) on behalf of plaintiffby Rosemary J. Fox Ipg(s) (JMR)
DAm
Modified on 04/29/2002 (Entered: 04/29/2002)
01/13/2004
NOTICE OF ASSIGNMENT OF CASES FOR TRIAL (Senior Judge David S.
Doty / 1/12/04) jury trial set for 9:00 a.m. on 4/26/04 . 3 pg(s) (c: all counsel)
cspo
stfrdtifusifabdltha
ses to sach interrogatories shall be served within 30 days thercafter. First
V.
Defendant
05/03/2002
Summons - RETURN OF SERVICE executed upon defendant Minnesota Beef
(HLL) (Entered: 01/13/2004)
ndustries on 5/1/02 2pg(s) (JMR) (Entered: 05/09/2002)
eogatriecldingxpertinrgatoresbeervedlatthn
01/13/2004
国
NOTICE OF FINAL SETTLEMENT CONFERENCE (Magistrate Judge Susan
Nembe15.1997AlldisoveryistbcompletbyDecmbr31.1997
Minnesota Beef Industries, Inc.
represented by william J Egan
05/14/2002
4ANSWER by defendant 4pg(s) (JMR) (Entered: 05/17/2002)
R. Nelson / 1/13/04) final selement conference set for 9:30 a.m. on 4/8/04. 2
Ehngfxiadbm997
Oppener,Wolf&Doelly
pg(s) (cc: all counsel) (HLL) Additional attachment(s) added on 10/26/2004
assin9yD
Suite 3300, PLaza VII
05/21/2002
NOTICE OF PRETRIAL CONFERENCE (Magistrate Judge Susan R. Nelson/
lass cerificati motionbyFebury 28,1998: plf shall submit reply if any by
45 South Seventh Street
5/20/02) : pretrial conferene set for 2:30 6/21/02 ; Rule report dl set for 6/10/02
(akl). (Entered: 01/15/2004)
stadhlndavas
Mach15998yttninsmayjdmbyhy
Minneapolis, MN 55402
5pg(s) (cc: allcounsel) (JMR) (Entered: 05/23/2002)
02/26/2004
TRIAL NOTICE Jury Trial set for 4/26/2004 09:00 AM in Minneapolis -
onference Spem 9.19978:45.(signedby JdChresLBrient);
612-607-7509
Courtroom 14W before Senior Judge David S Doty. (PJM) (Entered: 02/26/2004)
Copies mailed. (ec) Modified on 05/19/1997 (Entered: 05/19/1997)
Fax: 612-607-7100
05/29/2002
6
Amended Notice Of Pretrial Conference (Magistrate Judge Susan R. Nelson/
MinteEy fo pcdneldbfeMagJ Su RNsnl
06/26/1997
4
ANSWER o Complaint by The County of Wesch, Andrew J. ORourke (Atoey
Email: wegan@oppenheimer.com
5/28/02) ; pretrial conference set for 1:30 6/18/02 ; Rule report dl set for 6/10/02
04/08/2004
55
MathewT.Miklave),:Fim ofEstnBecker&Grcen by ateyMathwT
LEAD ATTORNEY
pg(s) (cc: allcounsel) (JMR) (Entered: 05/30/2002)
Settlement Conference held on 4/8/2004. No Settlement Reached. (HLL)
Mikave for defendant Andrew J. ORourke (ds) (Entered: 07/01/1997)
ATTORNEY TO BE NOTICED
06/04/2002
7REPORT OF RULE 26(f) MEETING by plaintiff, defendant 6pgs (JMR)
(Entered: 04/09/2004)
L661/20/0
y Mathew Miklave (ds) (Entered: 07/07/1997)
NOTICE of somey appeaanee fo The Conty of Wesch, Andrew J. ORourke
Interpleader
(Entered: 06/05/2002)
04/14/2004
Minute Entry for proccedings held before Mag. Judge Susan R Nelson :
Pretrial SCHEDULING ORDER (Magistrate Judge Susan R. Nelson/ 6/19:02) :
Telephone Conference re videotaped discovery of meat-packing plant to take
L661/S1/0
06/20/2002
6
Sheila Kutz
represented by Celeste E Culberth
place at 10:00 a.m. on 4/20/04 held on 4/14/2004. (HLL) (Entered: 04/15/2004)
y Mathew T. Miklave. Esq. (kz) (Entend: 07/15/1997)
Culberth & Lienemann, LLP
set for4/15/03: dispsitivemotions sefor6//03rcady fr trialset for 8/1/03
amd complaint set for 8/1/02 ; discovery set foe 3/17/03 non-dispositive motions
007/L1/90
07/22/1997
STIPULATION AND ORDER for Dismissal With Prejudice. Signed by Senior
STIPULATION and ORDERitishereby stip.andsgred tht the time for boh
444 Cedar St Ste 1050
pg(s) (cc: counsel) (JMR) (Entered: 06/25/2002)
St Paul, MN 55101
Judge David S Doty on 5/17/04. (HLL) (Entered: 05/19/2004)
651-290-9300
MOTIONbymovant SheilaKut forleae tointvene as laintiff( 
Def'sresponse shallbxndedoandmuinAugust91997Psespm
8/02/2002
05/17/2004
国
Consent Decree. Signod by Senior Judge David S Doty on 5/17/04. (HLL)
thReusfrPrdctionfocmensshalbxdedandnlig
Fax: 651-290-9305
Email: culberth@clslawyers.com
Magistrate Judge Susan R. Nelson ). 2 pg(s) (JVN) (Entered: 08/09/2002)
(Entered: 05/19/2004)
Entered: 07/22/1997)
o and inclding Sembr 2, 1997. sied by Jde Charles L Brient ). (e)
LEAD ATTORNEY
08/02/2002
10DECLARATION of Celeste E. Culberth re motion for leave to intervene as
ATTORNEY TO BE NOTICED
plaintiff. [9-1] 15 pg(s) (JVN) (Entered: 08/09/2002)
09/19/1997
Case Managm Conference held by Jude Bricant. Transrip taken by
Adrienne Mignano (jac) (Enered: 09/2/1997)
Leslie L Lienemann
08/19/2002
11
Amended NOTICE by movant Sheila Kutz of hcaring seting hearing for motion
r leave to intervene as plaintiff to Magistrate Judge Susan R. Nelsn ) [9-1] af
PACER Service Center
10/23/1997
Letter filed dated Oetober 21, 1997 to Judge Brieant from atty Robert David
Culberth & Lieneman, LLP
Becker&GPCisintloprthsactinrhthiwsroy
GoodstitissthClswtsf
444 Cedar St Ste 1050
(200/1/80 puu) (4r) (s)ds 20/7/6 0c2
Transaetion Receipt
06/01/2007 13:02:08
elainod by Wesichester County, (ec) (Enered: 10/23/1997)
St Paul, MN 55101
651-290-9300
08/21/2002
12RESPONSE by plaintiffto Sheila Kutz's motion to intervene [9-1] 1pg(s) (JMR)
PACER
Client
L661/20/11
Case Maaem Conferne held by Judge Brient Trsrip akenby Su
Fax: 651-290-9305
(Entered: 08/27/2002)
Login:
hs0328
Cede:
eeoc
Ghorayeb (jac) (Entered: 11/07/1997)
Email: Ilienemann@clslawyers.com
09/18/2002
Docket
Sesreh
1107/1997
9Ltter filed dated 1/6/97 toJdge Bricnt frm aty Mathew T. Miklave re we
LEAD ATTORNEY
motion that Sheila Kutz my intervene as plaintiffintervenr [9-1 11pg(s)c: all
Deseription:
Report
Criteria:
0:02-cv-00810-DSD-SRN
Write inresponse to plfs Oet. 2, 1997 letter to this Cou. (ec) (Enteed:
ATTORNEY TO BE NOTICED
Counsel) (JMR) (Entered: 09/26/2002)
1/07/1997)
14AMENDED COMPLAINT [1-1] by plaintiff, jury demand. 8pg(s) (VEM)
Billable
4
Cest:
0.32
11/13/1997
10Leer to USDJ Brieant from atyh MahewMidae filed byThe County of
09/26/2002
Pages:
hdisdsond
WestchAndrewJ.Rorke dated Nov.121997 th paiesartial resolution
Docket Text
(Entered: 09/26/2002)
Entered: 11/13/1997)
09/26/2002
SUMMONS issued as to Minnesota Beef Ind (VEM) (Entered: 09/26/2002)
11/13/1997
11
dgefoAllpuoesitd blw(sig byJeChalsBt
ner tat case befmd ttheClerk of Co fr asigment a Magistt
04/17/2002
COMPLAINT - Summons issued. Assigned to Senior Judge David S. Doty pe
09/27/2002
15 OFFER OF JUDGMENT filed by defl. (3pg(s) (DDB) (Entered: 10/03/2002)
Civil Rights list and referred to Magistate Judge Susan R. Nelson 5g(s) (
Referred to Magistrate Judge Mark D. Fox (ds) (Entered: 11/13/1997)
Entered: 04/17/2002)
10/09/2002
16AMENDMENT by defendant to offer of judgment [15-1: re against dft jointly
11/21/1997
12RER In oder to facilitae the progress of petrial disovey of thislitigatin
by pltf and intervenor for total amount of $40,000, including costs,
jstsdyandiesivemaer,tsuecomiei as
04/17/2002
NOTICE given to atty admission clerk to mail a Govt PHV to out of state
isbursements, attomey fees 2pg(s) (SJH) (Entered: 10/17/2002)
daidiry
followingprocereswillbefollwedforthresoltionofdiscoverydiste
ttomey: Gwendolyn Young Reams and Jean P. Kamp on bchalf of the pltf. (
10/18/2002
17
(200//01 u) (Hr) (s)8d, p g mos spp q 3ASN
Fox ): Copies muiled (ds) (Emtered: 11/24/1997)
See document for details) . So Ordered; ( signed by Magistrate Judge Mark D.
Entered: 04/17/2002)
12/20/2002
18 RULE 7.1 DISCLOSURE STATEMENT by Minnesota Beef Ind that none exist
8661/L1/20
Transcript ofrecord of proceedings filed for dates of November 21, 1997 (II)
2pe(s) (DFL) (Entered: 01/03/2003)
2 of6
N465-21007/19
3 of 6
6/1/2007 12:59 PM
6 of 6
6/1/2007 12:59 PM
Partial tableatthebottom
(b)Full page table
(c)Partial table at thetop
(d)Mis-detected textline
14

Z. Shen et al.

6 Conclusion

LayoutParser provides a comprehensive toolkit for deep learning-based document
image analysis. The oﬀ-the-shelf library is easy to install, and can be used to
build ﬂexible and accurate pipelines for processing documents with complicated
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
https://www.tensorflow.org/, software available from tensorﬂow.org

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
coarse-to-ﬁne attention. In: International Conference on Machine Learning. pp.
980–989. PMLR (2017)

[7] Ganin, Y., Lempitsky, V.: Unsupervised domain adaptation by backpropagation.
In: International conference on machine learning. pp. 1180–1189. PMLR (2015)


LayoutParser: A Uniﬁed Toolkit for DL-Based DIA

15

[8] Gardner, M., Grus, J., Neumann, M., Tafjord, O., Dasigi, P., Liu, N., Peters,
M., Schmitz, M., Zettlemoyer, L.: Allennlp: A deep semantic natural language
processing platform. arXiv preprint arXiv:1803.07640 (2018)
(cid:32)Lukasz Garncarek, Powalski, R., Stanis(cid:32)lawek, T., Topolski, B., Halama, P.,
Grali´nski, F.: Lambert: Layout-aware (language) modeling using bert for in-
formation extraction (2020)

[9]

[10] Graves, A., Fern´andez, S., Gomez, F., Schmidhuber, J.: Connectionist temporal
classiﬁcation: labelling unsegmented sequence data with recurrent neural networks.
In: Proceedings of the 23rd international conference on Machine learning. pp.
369–376 (2006)

[11] Harley, A.W., Ufkes, A., Derpanis, K.G.: Evaluation of deep convolutional nets for
document image classiﬁcation and retrieval. In: 2015 13th International Conference
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
million images. In: Adjunct Publication of the 33rd Annual ACM Sym-
posium on User
Interface Software and Technology. p. 120–122. UIST
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

[21] Neudecker, C., Schlarb, S., Dogan, Z.M., Missier, P., Suﬁ, S., Williams, A., Wolsten-
croft, K.: An experimental workﬂow development platform for historical document
digitisation and analysis. In: Proceedings of the 2011 workshop on historical
document imaging and processing. pp. 161–168 (2011)

[22] Oliveira, S.A., Seguin, B., Kaplan, F.: dhsegment: A generic deep-learning approach
for document segmentation. In: 2018 16th International Conference on Frontiers
in Handwriting Recognition (ICFHR). pp. 7–12. IEEE (2018)


16

Z. Shen et al.

[23] Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z.,
Desmaison, A., Antiga, L., Lerer, A.: Automatic diﬀerentiation in pytorch (2017)
[24] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen,
T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style,
high-performance deep learning library. arXiv preprint arXiv:1912.01703 (2019)
[25] Pletschacher, S., Antonacopoulos, A.: The page (page analysis and ground-truth
elements) format framework. In: 2010 20th International Conference on Pattern
Recognition. pp. 257–260. IEEE (2010)

[26] Prasad, D., Gadpal, A., Kapadni, K., Visave, M., Sultanpure, K.: Cascadetabnet:
An approach for end to end table detection and structure recognition from image-
based documents. In: Proceedings of the IEEE/CVF Conference on Computer
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
Rault, T., Louf, R., Funtowicz, M., et al.: Huggingface’s transformers: State-of-
the-art natural language processing. arXiv preprint arXiv:1910.03771 (2019)
[35] Wu, Y., Kirillov, A., Massa, F., Lo, W.Y., Girshick, R.: Detectron2. https://

github.com/facebookresearch/detectron2 (2019)

[36] Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florencio, D., Zhang, C.,
Che, W., et al.: Layoutlmv2: Multi-modal pre-training for visually-rich document
understanding. arXiv preprint arXiv:2012.14740 (2020)

[37] Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., Zhou, M.: Layoutlm: Pre-training of

text and layout for document image understanding (2019)

[38] Zhong, X., Tang, J., Yepes, A.J.: Publaynet:

layout analysis.

ument
Analysis and Recognition (ICDAR). pp. 1015–1022.
https://doi.org/10.1109/ICDAR.2019.00166

largest dataset ever for doc-
In: 2019 International Conference on Document
IEEE (Sep 2019).


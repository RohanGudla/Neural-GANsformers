# Curated List of Fascinating GAN Applications

![banner](https://github.com/nashory/gans-awesome-applications/blob/master/jpg/gans.jpg)

## gans-awesome-applications

This is a carefully curated list of remarkable applications and demonstrations of Generative Adversarial Networks (GANs). Please note that this list prioritizes applications over general GAN papers focused on simple image generation, such as DCGAN and BEGAN.

## Landmark Papers I Respect

I have taken inspiration from these landmark GAN papers for personal and educational purposes:

- Generative Adversarial Networks, [paper](https://arxiv.org/abs/1406.2661), [github](https://github.com/goodfeli/adversarial)
- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1511.06434), [github](https://github.com/soumith/dcgan.torch)
- Improved Techniques for Training GANs, [paper](https://arxiv.org/pdf/1606.03498.pdf), [github](https://github.com/openai/improved-gan)
- BEGAN: Boundary Equilibrium Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1703.10717), [github](https://github.com/carpedm20/BEGAN-tensorflow)

## Contents

- [Applications using GANs](#applications-using-gans)
- [Did not use GAN but still interesting applications](#did-not-use-gan-but-still-interesting-applications)
- [GAN tutorials with easy and simple example code for starters](#gan-tutorials-with-easy-and-simple-example-code-for-starters)
- [Implementations of various types of GANs collection](#implementations-of-various-types-of-gans-collection)
- [Trendy AI-application Articles](#trendy-ai-application-articles)

### Applications using GANs

#### Font generation
- Learning Chinese Character style with conditional GAN, [blog](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html), [github](https://github.com/kaonashi-tyc/zi2zi)
- Artistic Glyph Image Synthesis via One-Stage Few-Shot Learning, [paper](http://arxiv.org/abs/1910.04987), [github](https://github.com/hologerry/AGIS-Net)
- Attribute2Font: Creating Fonts You Want From Attributes, [paper](https://arxiv.org/abs/2005.07865), [github](https://github.com/hologerry/Attr2Font)

#### Anime character generation
- Towards the Automatic Anime Characters Creation with Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1708.05509)
- [Project] A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing, [github](https://github.com/jayleicn/animeGAN)
- [Project] A simple, clean TensorFlow implementation of Generative Adversarial Networks with a focus on modeling illustrations, [github](https://github.com/tdrussell/IllustrationGAN)
- [Project] Keras-GAN-Animeface-Character, [github](https://github.com/forcecore/Keras-GAN-Animeface-Character)
- [Project] A DCGAN to generate anime faces using a custom mined dataset, [github](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras)

#### Interactive Image generation
- Generative Visual Manipulation on the Natural Image Manifold, [paper](https://arxiv.org/pdf/1609.03552), [github](https://github.com/junyanz/iGAN)
- Neural Photo Editing with Introspective Adversarial Networks, [paper](http://arxiv.org/abs/1609.07093), [github](https://github.com/ajbrock/Neural-Photo-Editor)

#### Text2Image (text to image)
- TAC-GAN – Text Conditioned Auxiliary Classifier Generative Adversarial Network, [paper](https://arxiv.org/pdf/1703.06412.pdf), [github](https://github.com/dashayushman/TAC-GAN)
- StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1612.03242.pdf), [github](https://github.com/hanzhanggit/StackGAN)
- Generative Adversarial Text to Image Synthesis, [paper](https://arxiv.org/pdf/1605.05396.pdf), [github](https://github.com/paarthneekhara/text-to-image), [github](https://github.com/reedscot/icml2016)
- Learning What and Where to Draw, [paper](http://www.scottreed.info/files/nips2016.pdf), [github](https://github.com/reedscot/nips2016)

#### 3D Object generation
- Parametric 3D Exploration with Stacked Adversarial Networks, [github](https://github.com/maxorange/pix2vox), [youtube](https://www.youtube.com/watch?v=ITATOXVvWEM)
- Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling, [paper](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf), [github](https://github.com/zck119/3dgan-release), [youtube](https://www.youtube.com/watch?v=HO1LYJb818Q)
- 3D Shape Induction from 2D Views of Multiple Objects, [paper](https://arxiv.org/pdf/1612.05872.pdf)
- Fully Convolutional Refined Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes, [github](https://github.com/yunishi3/3D-FCR-alphaGAN), [blog](https://becominghuman.ai/3d-multi-object-gan-7b7cee4abf80)

#### Image Editing
- Invertible Conditional GANs for image editing, [paper](https://arxiv.org/abs/1611.06355), [github](https://github.com/Guim3/IcGAN)
- Image De-raining Using a Conditional Generative Adversarial Network, [paper](https://arxiv.org/abs/1701.05957), [github](https://github.com/hezhangsprinter/ID-CGAN)

#### Face Aging
- Age Progression/Regression by Conditional Adversarial Autoencoder, [paper](https://arxiv.org/pdf/1702.08423), [github](https://github.com/ZZUTK/Face-Aging-CAAE)
- CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms, [paper](https://arxiv.org/pdf/1706.07068.pdf)
- FACE AGING WITH CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS

, [paper](https://arxiv.org/pdf/1702.01983.pdf)

#### Human Pose Estimation
- Joint Discriminative and Generative Learning for Person Re-identification, [paper](https://arxiv.org/abs/1904.07223), [github](https://github.com/NVlabs/DG-Net), [video](https://www.youtube.com/watch?v=ubCrEAIpQs4)
- Pose Guided Person Image Generation, [paper](https://arxiv.org/abs/1705.09368)

#### Domain-transfer (e.g., style-transfer, pix2pix, sketch2image)
- Image-to-Image Translation with Conditional Adversarial Networks, [paper](https://arxiv.org/pdf/1611.07004), [github](https://github.com/phillipi/pix2pix), [youtube](https://www.youtube.com/watch?v=VVqxbmUJorQ)
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, [paper](https://arxiv.org/pdf/1703.10593.pdf), [github](https://github.com/junyanz/CycleGAN), [youtube](https://www.youtube.com/watch?v=JzgOfISLNjk)
- Learning to Discover Cross-Domain Relations with Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1703.05192.pdf), [github](https://github.com/carpedm20/DiscoGAN-pytorch)
- Unsupervised Creation of Parameterized Avatars, [paper](https://arxiv.org/pdf/1704.05693.pdf)
- UNSUPERVISED CROSS-DOMAIN IMAGE GENERATION, [paper](https://openreview.net/pdf?id=Sk2Im59ex)
- Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks, [paper](http://arxiv.org/abs/1604.04382), [github](https://github.com/chuanli11/MGANs)
- Pixel-Level Domain Transfer, [paper](https://arxiv.org/pdf/1603.07442), [github](https://github.com/fxia22/PixelDTGAN)
- TextureGAN: Controlling Deep Image Synthesis with Texture Patches, [paper](https://arxiv.org/pdf/1706.02823.pdf), [demo](https://github.com/varunagrawal/t-gan-demo)
- Vincent AI Sketch Demo Draws In Throngs at GTC Europe, [blog](https://blogs.nvidia.com/blog/2017/10/11/vincent-ai-sketch-demo-draws-in-throngs-at-gtc-europe/), [youtube](https://www.youtube.com/watch?v=kIcqXTUMwps)
- Deep Photo Style Transfer, [paper](https://arxiv.org/pdf/1703.07511.pdf), [github](https://github.com/luanfujun/deep-photo-styletransfer)

#### Image Inpainting (hole filling)
- Context Encoders: Feature Learning by Inpainting, [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf), [github](https://github.com/pathak22/context-encoder)
- Semantic Image Inpainting with Perceptual and Contextual Losses, [paper](https://arxiv.org/abs/1607.07539), [github](https://github.com/bamos/dcgan-completion.tensorflow)
- SEMI-SUPERVISED LEARNING WITH CONTEXT-CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS, [paper](https://arxiv.org/pdf/1611.06430v1.pdf)
- Generative Face Completion, [paper](https://drive.google.com/file/d/0B8_MZ8a8aoSeenVrYkpCdnFRVms/edit), [github](https://github.com/Yijunmaverick/GenerativeFaceCompletion)

#### Super-resolution
- Image super-resolution through deep learning, [github](https://github.com/david-gpu/srez)
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, [paper](https://arxiv.org/abs/1609.04802), [github](https://github.com/leehomyc/Photo-Realistic-Super-Resoluton)
- High-Quality Face Image Super-Resolution Using Conditional Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1707.00737.pdf)
- Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network, [paper](https://arxiv.org/pdf/1811.00344.pdf), [github](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw)

#### Image Blending
- GP-GAN: Towards Realistic High-Resolution Image Blending, [paper](https://arxiv.org/abs/1703.07195), [github](https://github.com/wuhuikai/GP-GAN)

#### High-resolution image generation (large-scale image)
- Generating Large Images from Latent Vectors, [blog](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/), [github](https://github.com/hardmaru/cppn-gan-vae-tensorflow)
- PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION, [paper](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf), [github](https://github.com/tkarras/progressive_growing_of_gans)

#### Adversarial Examples (Defense vs Attack) 
- SafetyNet: Detecting and Rejecting Adversarial Examples Robustly, [paper](https://arxiv.org/abs/1704.00103)
- ADVERSARIAL EXAMPLES FOR GENERATIVE MODELS, [paper](https://arxiv.org/pdf/1702.06832.pdf)
- Adversarial Examples Generation and Defense Based on Generative Adversarial Network, [paper](http://cs229.stanford.edu/proj2016/report/LiuXia-AdversarialExamplesGenerationAndDefenseBasedOnGenerativeAdversarialNetwork-report.pdf)

#### Visual Saliency Prediction (attention prediction)
- SalGAN: Visual Saliency Prediction with Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1701.01081), [github](https://github.com/imatge-upc/saliency-salgan-2017)

#### Object Detection/Recognition
- Perceptual Generative Adversarial Networks for Small Object Detection, [paper](https://arxiv.org/pdf/1706.05274)
- Adversarial Generation of Training Examples for Vehicle License Plate Recognition, [paper](https://arxiv.org/pdf/1707.03124.pdf)

#### Robotics
- Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1612.05424.pdf)

github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/pix2pix)
- DualGAN: Unsupervised Dual Learning for Image-to-Image Translation, [paper](https://arxiv.org/pdf/1704.02510), [github](https://github.com/duxingren14/DualGAN)

#### Face Generation
- Perceptual Face Hallucination with Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1706.01353.pdf), [github](https://github.com/1991viet/PerceptualFaceHallucination)
- FACE AGING WITH CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS, [paper](https://arxiv.org/pdf/1702.01983.pdf)

#### Sketch-to-Image
- A Diverse Handwriting Generation Model, [paper](https://arxiv.org/pdf/1609.07843.pdf), [github](https://github.com/tegg89/Sketch-RNN-VAE-Decoder)
- sketch-rnn: a generative model for vector drawings, [github](https://github.com/magenta/magenta/tree/master/magenta/models/sketch_rnn)

#### Video-to-Video Synthesis
- Video-to-Video Synthesis, [paper](https://tcwang0509.github.io/pix2pixHD/), [github](https://github.com/NVIDIA/vid2vid)
- High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs, [paper](https://arxiv.org/pdf/1711.11585), [github](https://github.com/NVIDIA/pix2pixHD)

#### Medical Image Synthesis
- Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery, [paper](https://arxiv.org/pdf/1703.05921.pdf)
- Conditional Generative Adversarial Nets for Convolutional Electron Energy-Loss Spectroscopy Image Spectrum Mapping, [paper](https://arxiv.org/pdf/1701.07847)
- SpectGAN: Deep generative model for synthetic spectral generation via conditional GAN, [paper](https://arxiv.org/pdf/1705.02538.pdf)

#### Image-to-Image Translation in PyTorch
- Image-to-Image Translation with Conditional Adversarial Networks, [github](https://github.com/phillipi/pix2pix)
- PyTorch implementation of Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), [github](https://github.com/wiseodd/generative-models)
- Official PyTorch implementation of "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", [github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- An implementation of Cycle-GAN and Pix2pix in PyTorch, [github](https://github.com/aitorzip/PyTorch-CycleGAN)

#### Video Generation
- Video Generation from Text, [paper](https://arxiv.org/pdf/1702.07860), [github](https://github.com/batzner/tensorflow-videogan)

#### Semi-Supervised Learning
- Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks, [paper](https://arxiv.org/pdf/1611.06430)

#### GANs for Anomaly Detection
- AnoGAN: Anomaly Detection with GAN, [paper](https://arxiv.org/abs/1703.05921), [github](https://github.com/LeeDoYup/AnoGAN)

#### Neural Network Compression
- Learning Compact Recurrent Neural Networks with Block-Term Tensor Decomposition, [paper](https://arxiv.org/pdf/1708.01009.pdf)

#### GANs for Molecular Chemistry
- ChemGAN challenge for drug discovery: can AI reproduce natural chemical diversity?, [paper](https://www.nature.com/articles/s41598-017-06451-3)

#### Anomaly Detection
- GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training, [paper](https://arxiv.org/pdf/1805.06725.pdf), [github](https://github.com/samet-akcay/ganomaly)

#### Recommendation Systems
- Adversarial Personalized Ranking for Recommendation, [paper](https://arxiv.org/pdf/1808.03908.pdf)

#### 3D Object Generation and Manipulation
- Learning to Reconstruct 3D Non-Rigidly Deforming Objects with Temporally-Consistent Pose, [paper](https://arxiv.org/pdf/1811.10136.pdf)

#### Image Style Transfer
- Arbitrary Style Transfer with Style-Attentional Networks, [paper](https://arxiv.org/abs/1812.02342), [github](https://github.com/CompVis/taming-transformers)

#### Image-to-Image Translation with Fewer Examples
- Few-Shot Unsupervised Image-to-Image Translation, [paper](https://arxiv.org/abs/1812.01849), [github](https://github.com/NVlabs/FUNIT)

#### Music Generation
- MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment, [paper](https://arxiv.org/pdf/1709.06298.pdf), [github](https://github.com/salu133445/musegan)

#### Object Detection
- Improved Adversarial Systems for 3D Object Detection, [paper](https://arxiv.org/pdf/1907.03244.pdf)

#### Multi-Agent Cooperation
- Training Multi-Agent Collaboration with Adversarial Reinforcement Learning, [paper](https://arxiv.org/pdf/1911.00401.pdf), [github](https://github.com/openai/multiagent-competition)

#### Image Enhancement
- GAN-based Image Enhancement with Spatial and Channel-wise Attention Mechanisms, [paper](https://arxiv.org/abs/2001.04519), [github](https://github.com/SeungjunNah/EEGAN)

#### Image Synthesis for Virtual Try-On
- VITON: An Image-based Virtual Try-on Network, [paper](https://arxiv.org/pdf/1711.08447.pdf), [github](https://github.com/xthan/VITON)

#### 3D Human Pose Estimation
- Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis, [paper](https://arxiv.org/pdf/2003.04095.pdf)

#### Deep Reinforcement Learning
- Continuous control with deep reinforcement learning, [paper](https://arxiv.org/pdf/1509.02971.pdf), [github](https://github.com/rll/rllab)

#### Realistic Caricature Generation
- Realistic Caricature Generation with Progressive Self-Supervised Learning, [paper](https://arxiv.org/pdf/2007.01253.pdf)

#### Music Style Transfer
- MUSGAN: Unsupervised Music Style Transfer with StyleGAN, [paper](https://arxiv.org/pdf/2004.11149.pdf)

#### Image-to-Image Translation with Style and Structure
- Artbreeder, [website](https://www.artbreeder.com/)

#### Speech

 Enhancement
- A UNIVERSAL NOISE SUPPRESSION ARCHITECTURE FOR SELF-SUPERVISED PERCEPTUAL LEARNING, [paper](https://arxiv.org/pdf/2006.09534.pdf)

# gans-awesome-applications
Curated list of awesome GAN applications and demonstrations.

__Note: General GAN papers targeting simple image generation such as DCGAN, BEGAN etc. are not included in the list. I mainly care about applications.__

## The landmark papers that I respect.
+ Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1406.2661), [[github]](https://github.com/goodfeli/adversarial)
+ Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1511.06434), [[github]](https://github.com/soumith/dcgan.torch)
+ Improved Techniques for Training GANs, [[paper]](https://arxiv.org/pdf/1606.03498.pdf), [[github]](https://github.com/openai/improved-gan)
+ BEGAN: Boundary Equilibrium Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1703.10717), [[github]](https://github.com/carpedm20/BEGAN-tensorflow)

...

#### Image Enhancement
- GAN-based Image Enhancement with Spatial and Channel-wise Attention Mechanisms, [paper](https://arxiv.org/abs/2001.04519), [github](https://github.com/SeungjunNah/EEGAN) (Credit goes to the original authors; this information is recreated for personal understanding and educational purposes.)

...

#### Music Style Transfer
- MUSGAN: Unsupervised Music Style Transfer with StyleGAN, [paper](https://arxiv.org/pdf/2004.11149.pdf) (Credit goes to the original authors; this information is recreated for personal understanding and educational purposes.)

...

Please note that the field of Generative Adversarial Networks (GANs) is rapidly evolving, and new research papers and implementations are being published regularly. I have recreated this list for personal understanding and educational purposes, and credit goes to the original authors of the papers and repositories mentioned.

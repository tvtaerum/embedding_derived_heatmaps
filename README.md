## cGAN_diff: more than pretty faces
### Exploring where differences come from.  

We make substantial progress when we go from discriminating between images to actually generating images.  We make even more progress when we are able to "make sense" of differences between images.  It's like asking a child, 1. tell me if this is a mom or a dad, 2. draw a mom and a dad for me, and 3. point out the differences between a drawing of a mom and a dad.    

<p align="center">
<img src="/images/attractiveFaces.png" width="650" height="135">
</p>

While we quickly recognize if a face is typical female or male, we often have difficulties describing the differences.  Can we produce visual representations of these differences?  Similarily, most nonmedical people cannot distinguish between the x-rays of the lungs of healthy children and those with viral or bacterial pneumonia.  Further, it's one thing for a neural net to distinguish between healthy children and those with pneumonia, it demonstrates even greater insight when it can generate images of healthy lungs or lungs with pneumonia.  It's an even larger step when a neural net can highlight the differences it has depended on.  Are the differences the same as what an expert would consider, are the differences based on artifacts, or do the highlighted differences suggest additional insights.

Some explanation is required.  A challenge with making image comparisons between x-rays of lungs is we cannot line up images of lungs with different kinds of disease.  In most cases, x-rays of lungs with pneumonia and healthy lungs are from different patients with different sizes of ribs and lungs, where angles are different, scopes are different, and shadings are different.  Making comparisons would not be possible unless we are able to generate x-rays using GANs where the only differences are due to embeddings.  

As we observed in https://github.com/tvtaerum/cGANs_housekeeping and https://github.com/tvtaerum/xray_housekeeping, we are able to generate images which are based on the same weights modified only by an embedding ("attractive male" vs "attractive female with high cheeks bones") or ("x-rays of healthy lungs" vs "viral pneumonia" vs "bacterial pneumonia").  What happens when we apply the same processes to x-ray images of healthy lungs and those with bacterial and viral pneumonia. Are the predictions sufficiently strong that we can visually distinguish between healthy, viral pneumonia, and bacterial pneumonia based on generated images?    

The first thing we have to do is prove the technique using images that we understand.  We are able to understand and graphically display the source of differences between typical male and female faces using the results from the cGANs housekeeping display.  Once we have proven we can do that, we need to investigate if we can generate images of healthy lungs versus lungs with pneumonia.  And then can we display the source of the differences in the same manner we did with generated faces.  

I thank Jason Brownlee for his work and tutorials at https://machinelearningmastery.com (citations below in project) and Wojciech Łabuński for his excellent application of image resizing and histogram equilization at https://www.kaggle.com/wojciech1103/x-ray-classification-and-visualisation.  Data collected from: https://data.mendeley.com/datasets/rscbjbr9sj/2 License: CC BY 4.0 Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

### Motivation for identifying differences between xrays of healthy lungs and those with pneumonia:
Considerable effort has been applied to building neural nets to discriminate between patients who are healthy and those patients with pneumonia based on x-rays.  An avenue which hasn't been explored as much is identifying where these differences exist.  In some cases, the discriminator is run, 90% accuracy is achieved, but no test that I am aware of tells the observer what the differences are and whether or not the apparent differences may partly be an artifact.  

As a reminder of what was previously established, we can see in the faces above, that the https://github.com/tvtaerum/cGANs_housekeeping program did a good job of creating images that are obviously "attractive females with high cheek bones" in contrast to "attractive males".  The question now is, "can we generate images which make apparent the differences between typical female and male faces?".  Once we have done that, can we apply those same techniques in comparing images healthy lungs, lungs with viral pneumonia, and lungs with bacterial pneumonia.  
</p>

The following is a screenshot illustrating our abiliity to indicate what regions of a face allows us to determine if it is and "attractive female with high cheek bones" or an "attractive male".   
<p align="center">
<img src="/images/Female&MaleEmbeddings.png" width="650" height="290">
</p>
In particular, we have four rows of figures:  female, male, delta female, and delta male faces.  In order to generalize what we observe with faces to x-rays, we need to first understand what we're seeing with respect to faces.  The first two rows are generated images of x-rays for females and males.  The next two rows show the differences between the generated images.  Yellow identifies large additions required to turn an image into a female (female delta) or turn an image into a male (male delta), green represents moderate additions, and purple represents small additions.  To clarify, "additions" refers to something which is "added".  So, for instance, a beard or shadow is "added" to make "male"; higher eyebrows are "added" to make "female".  The definition of what is "added" is arbitrary but operationally "added" is defined as something which makes a part of a face look darker.  In the last row, for instance, we can see the "addition" of facial hair to make an image "male" .    
<br/><br/>
In the screen shot below, the first three rows are cGAN generations of healthy lungs, lungs with viral pneumonia, and lungs with bacterial pneumonia.  Below that we see three sets of two comparisons:  healthy vs viral, healthy vs bacterial, and viral vs bacterial.  Each comparison raises interesting questions.  Is there evidence of artifacts?  Are there detectable differences between pneumonia due to virus and pneumonia due to bacteria.  

<p align="center">
<img src="/images/healthy_viral_bacterial_pneumonia.png" width="650" height="590">
</p>
It's worth noting that the "sense" of contrasts for x-rays is quite different than a regular photo.  In x-rays, it is whiter that matters and not darker.  In x-rays, it is lighter pixels (e.g. of bone and possibly inflammation) which are of interest.    

### Citations:
<dl>
<dt> Jason Brownlee, How to Develop a Conditional GAN (cGAN) From Scratch,</dt><dd> Available from https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch, accessed January 4th, 2020. </dd>
<dt>Jason Brownlee, How to Explore the GAN Latent Space When Generating Faces, </dt><dd>Available from https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network, accessed January 13th, 2020. </dd>
<dt>Iván de Paz Centeno, MTCNN face detection implementation for TensorFlow, as a PIP package,</dt><dd> Available from https://github.com/ipazc/mtcnn, accessed February, 2020. </dd>
<dt>Jeff Heaton, Jeff Heaton's Deep Learning Course,</dt><dd> Available from https://www.heatonresearch.com/course/, accessed February, 2020. </dd>
<dt>Wojciech Łabuński, X-ray - classification and visualisation</dt>  <dd> Available from 
https://www.kaggle.com/wojciech1103/x-ray-classification-and-visualisation, accessed March, 2020</dd>
<dt>Tory Walker, Histogram equalizer, </dt> <dd>Available from 
https://github.com/torywalker/histogram-equalizer, accessed March, 2020</dd>
</dl>

### Deliverables:
  1.  description of issues identified and resolved within specified limitations
  2.  code fragments illustrating the core of how an issue was resolved
  3.  two Python programs which vectorize face and x-ray images and compare these images producing contrasts

### Limitations and caveates:

  1.  stream:  refers to the overall process of streaming/moving data through input, algorithms, and output of data and its evaluation.
  2.  convergence:  since there are no unique solutions in GAN, convergence is sufficient when there are no apparent improvements in a subjective evaluation of clarity of images being generated.   
  3.  limited applicability:  the methods described work for a limited set of data and cGan problems.
  4.  bounds of model loss:  there is an apparent relationship between mode collapse and model loss - when model loss is extreme (too high or too low) then there is mode collapse.  
  
### Software and hardware requirements:
    - Python version 3.7.3
        - Numpy version 1.17.3
        - Tensorflow with Keras version 2.0.0
        - Matplotlib version 3.0.3
    - GPU is highly recommended
    - Operating system used for development and testing:  Windows 10

#### The process:

 Creating a cGAN as illustration, I provide limited working solutions to the following problems:

<ol type="1">
  <li>can we generate images of female and male faces based solely on embedding labels</li>
  <li>can we create images which point out the differences between typical female and male faces</li>
  <li>can we generate images of x-rays differentiating between healthy lungs and those with bacterial and viral pneumonia</li>
  <li>can we create images which point out the differneces betweeen healthy lungs and those with bacterial and viral pneumonia</li>
  <li>cGan streams and data sources</li>
</ol>


### 1.  can we generate images of female and male faces by alternating only embeddings:

As we saw in https://github.com/tvtaerum/cGANs_housekeeping, it is possible to both create and vertorize images where male versus female faces can be created simply by selecting a corresponding label/embedding.  

### 2. can we create images which point out the differences between typical female and male faces:
In making comparisons between female and male faces, there is considerable advantage to the fact the same weights can be used to create a male face and a female face and the only difference is the label/embedding.  

### 3.  can we generate images of x-rays differentiating between healthy lungs and those with bacterial and viral pneumonia based solely on alternating embeddings?
As we saw in https://github.com/tvtaerum/xray_housekeeping, it is possible to both create and vertorize images where healthy lungs versus viral pneumonia lungs versus bacterial pneumonia lungs can be created simply by selecting a corresponding label/embedding.  

### 4.  can we create images which point out the differences betweeen healthy lungs and those with bacterial and viral pneumonia?
In making comparisons between healthy lungs and lungs with viral or bacterial pneumonia, there is considerable advantage to the fact that the same weights can be used to create the different images and the only difference is the label/embedding.  

###  5.  cGan streams and data sources:
The following is an outline of the programming steps and Python code used to create the results observed in this repository.  There are two Python programs which are unique to this repository and five modelling (.h5) files.   

The recommended folder structure looks as follows:
<ul>
    <li>embedding_derived_heatmaps-master (or any folder name)</li>
	<ul>
       <li> files (also contains two Python programs - program run from here)</li>
	<ul>
		<li> <b>celeb</b></li>
		<ul>
			<li> <b>label_results</b> (contains five .h5 generator model files)</li>
		</ul>
		<li> <b>xray</b></li>
		<ul>
			<li> <b>label_results</b> (contains five .h5 generator model files)</li>
		</ul>
		<li> <b>cgan</b> (contains images from summary analysis of models)</li>
	</ul>
       <li> images (contains images for README file)</li>
	</ul>
</ul>
Those folders which are in <b>BOLD</b> need to be created. 
All Python programs must be run from within the "file" directory. 

#### LICENSE  <a href="/LICENSE">MIT license</a>

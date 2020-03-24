## cGAN_diff: more than a pretty face
### Exploring where differences come from.  

We make substantial progress when we go from discriminating between images to actually generating images.  We make even more progress when we are able to "make sense" of differences between images.  It's like asking a child, 1. tell me if this is a mom or a dad, 2. draw a mom and a dad for me, and 3. point out the differences between a drawing of a mom and a dad.    

<p align="center">
<img src="/images/attractiveFaces.png" width="650" height="135">
</p>

While we quickly recognize if a face is typical female or male, we often have difficulties describing the differences.  Can we produce visual representations of these differences?  Similarily, most nonmedical people cannot distinguish between the x-rays of the lungs of healthy children and those with viral or bacterial pneumonia.  Again, it's one thing for a neural net to distinguish between healthy children and those with pneumonia, it demonstrates even greater insight when it can generate images of healthy lungs or lungs with pneumonia.  It's an even larger step when a neural net can highlight the differences it has depended on.  Are the differences the same as what an expert would consider, are the differences based on artifacts, or do the highlighted differences provide additional insights.   
 
As we observed in https://github.com/tvtaerum/cGANs_housekeeping, we are able to generate images which are based on the same weights modified only by an embedding label (e.g. "attractive male" vs "attractive female with high cheeks bones").  What happens when we apply the same processes to x-ray images of healthy lungs and those with bacterial and viral pneumonia. Are the predictions sufficiently strong that we can visually distinguish between healthy, viral pneumonia, and bacterial pneumonia based on generated images?    

The first thing we have to do is prove the technique - that we are able to identify and graphically display the source of differences between typical male and female faces using the results from the cGANs housekeeping display.  Once we have proven we can do that, we need to investigate if we can generate images of healthy lungs and lungs with pneumonia.  And then can we display the source of the differences in the same manner we did with generated faces.  

I thank Jason Brownlee for his work and tutorials at https://machinelearningmastery.com (citations below in project) and Wojciech Łabuński for his excellent application of image resizing and histogram equilization at https://www.kaggle.com/wojciech1103/x-ray-classification-and-visualisation.  Data collected from: https://data.mendeley.com/datasets/rscbjbr9sj/2 License: CC BY 4.0 Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

### Motivation for housekeeping with xrays of children with pneumonia:
Considerable effort has been applied to creating discriminators between patients who are healthy and those patients with pneumonia based on x-rays, an avenue which hasn't been explored as much is identifying where these differences exist.  In some cases, the discriminator is run, 90% accuracy is achieved, but no test that I am aware of tells the observer what the differences are and whether or not the apparent differences may partly be an artifact.  There are many indirect tests but I want to know which regions in an image are helpful in discrimination.  

As a reminder of what was previously established, we can see in the faces above, that the https://github.com/tvtaerum/cGANs_housekeeping program did a good job of creating images that are obviously "attractive females with high cheek bones" in contrast to "attractive males".  The question now is, "can we generate images which make apparent the differences between typical female and male faces?" and "can we also generate images which make apparent the differences between "healthy lungs", "viral pneumonia" and "bacterial pneumonia?".   
</p>

The following is a screenshot illustrating our abiliity to indicate what regions of a face allows us to determine if it is and "attractive female with high cheek bones" or an "attractive male".   
<p align="center">
<img src="/images/Female&MaleEmbeddings.png" width="650" height="290">
</p>
In particular, we have four rows of figures:  female, male, delta female, and delete male faces.  Yellow identifies large additions to make an image female or male, green represents moderate additions, and purple represents small additions.  To clarify, most often "addition" refers to something which is "added".  So, for instance, a beard or shadow is "added"; higher eyebrows are "added"; wider eyes are "added".  The definition is somewhat arbitrary but is detected by looking for darker regions.  In the last row, for instance, we can see, for instance, the addition of a shadow due to facial hair.  

In the screen shot below, the first three rows are cGAN generations of healthy lungs, lungs with viral pneumonia, and lungs with bacterial pneumonia.  Below that we see three sets of two comparisons:  healthy vs viral, healthy vs bacterial, and viral vs bacterial.  <p align="center">
<img src="/images/healthy_viral_bacterial_pneumonia.png" width="650" height="590">
</p>
It's worth noting that the "sense" of contrasts for x-rays is quite different than a regular photo.  In x-rays, it is whiter that matters and not darker.  
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
  3.  a Python program to prepare images for selection and training
  4.  a cGan Python program with embedding
  5.  a Python program which vectorizes images generated with embedding

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


### 1.  can we generate images of female and male faces based solely on embedding label:

Even with reasonable learning rates, convergence can slide into "mode collapse" and require a manual restart.  The stream provides one way of giving intial estimates multiple but limited opportunities to halt it's slide towards mode collapse.  The process also allows the stream to retain whatever progress it has made towards convergence while recovering from mode collapse.     

There are three critical measures of loss:
<ol>
	<li>dis_loss, _ = d_model.train_on_batch([X_real, labels_real], y_real)</li>
	<li>gen_loss, _ = d_model.train_on_batch([X_fake, labels], y_fake)</li>
	<li>gan_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)</li>
</ol>
Before examining the screen shot which comes below, I define the measures used to determine when mode collapse is imminent and recovery is necessary:
<table style="width:100%">
  <tr> <th> Column </th>    <th> measure </th>      <th> example </th>  </tr>
  <tr> <td> 1 </td>  <td> epoch/max_epochs </td>    <td> 1/100 </td>  </tr>
  <tr> <td> 2 </td>  <td> iteration/max_iterations  </td>    <td> 125/781 </td>  </tr>
  <tr> <td> 3 </td>  <td> discriminator loss </td>    <td> d1(dis)=0.020 </td>  </tr>
  <tr> <td> 4 </td>  <td> generator loss </td>    <td> d2(gen)=0.114 </td>  </tr>
  <tr> <td> 5 </td>  <td> gan loss </td>    <td> g(gan)=2.368 </td>  </tr>
  <tr> <td> 6 </td>  <td> run time (seconds) </td>   <td> secs=142 </td>  </tr>
  <tr> <td> 7 </td>  <td> number of restarts </td>    <td> tryAgain=0 </td>  </tr>
  <tr> <td> 8 </td>  <td> number of restarts using same base </td>    <td> nTripsOnSameSavedWts=0 </td>  </tr>
  <tr> <td> 9 </td>  <td> number of weight saves </td>    <td> nSaves=2 </td>  </tr>
</table>
There are three parts in the screen shots below: 
<p align="center">
<img src="/images/escapingModeCollapse.png" width="850" height="225">
</p>

### 2. can we create images which point out the differences between typical female and male faces:
There is nothing quite as problematic as running a program and six days later the process is interrupted when it appears to be 90% complete.  Like many others, I have run streams for over 21 days using my GPU before something goes wrong and I am unable to restart the process.  Progress is measured in "epochs".  There is no guarantee but with a bit of good fortune and cGAN steams which are properly set up, every epoch brings an improvement in clarity.  The images which follow illustrate observed improvements over epochs.  


### 3.  can we generate images of x-rays differentiating between healthy lungs and those with bacterial and viral pneumonia?
While the use of normal like distributions may be useful, there is no reason to believe that other distributions will not work.  A small investigation on my part suggested that leptokurtic distributions were poorest in generating good images.  For most of the results discussed here, I use a uniform distribution in a bounded 100-dimensional space.   
```Python
def generate_latent_points(latent_dim, n_samples, cumProbs, n_classes=4):
	# print("generate_latent_points: ", latent_dim, n_samples)
	initX = -3.0
	rangeX = 2.0*abs(initX)
	stepX = rangeX / (latent_dim * n_samples)
	x_input = asarray([initX + stepX*(float(i)) for i in range(0,latent_dim * n_samples)])
	shuffle(x_input)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	randx = random(n_samples)
	labels = np.zeros(n_samples, dtype=int)
	for i in range(n_classes):
		labels = np.where((randx >= cumProbs[i]) & (randx < cumProbs[i+1]), i, labels)
	return [z_input, labels]
```
Substantially, the routine divides the range of values from -3.0 to +3.0 into equal intervals and then randomizes the values by a shuffle.  The process works - I'm still examining whether it accelerates convergence with images.  
 
### 4.  can we create images which point out the differneces betweeen healthy lungs and those with bacterial and viral pneumonia?
In my attempts to improve the results of the generations, I initially overlooked a critical factor - what does the transformed data going into the cGAN look like.  When the data going into a stream is a derivative of another process, as in this case, it is critical to examine the quality of the input data before declaring the results to be useful or invalid.  

The code to examine the data going into the cGAN is trivial and is included in the final stream.  

![real faces rows](images/sampleRealImagesRows.png)


###  5.  cGan streams and data sources:
The following is an outline of the programming steps and Python code used to create the results observed in this repository.  There are three Python programs which are unique to this repository.  The purpose of the code is to assist those who struggled like I struggled to understand the fundamentals of Generative Adversarial Networks and to generate interesting and useful results beyond number and fashion generation.  My edits are not elegant... it purports to do nothing more than resolve a few issues which I imagine many novices to the field of Generative Adversarial Networks face.  If you know of better ways to do something, feel free to demonstrate it.  If you know of others who have found better ways to resolve these issues, feel free to point us to them.  

The recommended folder structure looks as follows:
<ul>
    <li>cGANs_housekeeping-master (or any folder name)</li>
	<ul>
       <li> files (also contains Python programs - program run from here)</li>
	<ul>
		<li> <b>celeb</b></li>
		<ul>
			<li> <b>img_align_celeba</b> (contains about 202,599 images for data input)</li>
			<li> <b>real_plots</b> (contains arrays of real images for inspection)</li>
			<li> <b>results</b> (contains generated png images of faces and and h5 files for models saved by program)</li>
		</ul>
		<li> <b>cgan</b> (contains images from summary analysis of models)</li>
	</ul>
       <li> images (contains images for README file)</li>
	</ul>
</ul>
Those folders which are in <b>BOLD</b> need to be created. 
All Python programs must be run from within the "file" directory.  

#### a. download celebrity images from https://www.kaggle.com/jessicali9530/celeba-dataset
#### b. select out subset of images with attractive faces and compress <a href="/files/images_convert_mtcnn_attractive_faces.py">MTCNN convert attractive faces</a>

When executing, you will get the following output:  
<p align="left">
<img src="/images/LoadingAndCompressing50000Images.png" width="200" height="100">
</p>  

It will create two files:
    ids_align_celeba_attractive.npz
    image_align_celeba_attractive.npz
    
#### c. cGan stream <a href="/files/tutorial_latent_space_embedding_cgan.py">cGan embedding</a>

Refer back to Python coding fragments for explanation on restarting program.

#### d. vectorize images <a href="/files/images_run_thru_models_1_restart_cgan.py">run thru faces using embedding</a> 

The list of images for visual examination depends on the lstEpochs variable included in the code fragment below.  In the example below, epochs 5, 15, 25... 145, 150 are displayed.  If you have fewer than 150 epochs saved then you'll need to adjust the lstEpochs list.    
```Python
directory = 'celeb/results/'
iFile = 0
for idx, filename in enumerate(listdir(directory)):
    if ".h5" in filename and not("_gan" in filename) and not("_dis" in filename):
        iFile += 1
        lstEpochs = [5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,150]
        if iFile in lstEpochs: 
            model = load_model(directory + filename)
            gen_weights = array(model.get_weights())
```
#### LICENSE  <a href="/LICENSE">MIT license</a>

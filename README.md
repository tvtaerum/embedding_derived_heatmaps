## cGANs: more than a pretty face
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

As a reminder of what was previously established, we can see in the faces above, that the https://github.com/tvtaerum/cGANs_housekeeping program did a good job of creating images that are obviously "attractive females with high cheek bones" in contrast to "attractive males".  Can we generate images which make apparent the differences between typical female and male faces.  Can we also generate images which make apparent the differences between "healthy lungs", "viral pneumonia" and "bacterial pneumonia".  
</p>

The following is a screenshot illustrating our abiliity to indicate what regions of a face allows us to determine if it is and "attractive female with high cheek bones" or an "attractive male".   
<p align="center">
<img src="/images/Female&MaleEmbeddings.png" width="650" height="290">
</p>
In particular, we have four rows of figures:  female, male, delta female, and delete male faces.  Yellow identifies large additions to make an image female or male, green represents moderate additions, and purple represents small additions.  To clarify, most often "addition" refers to something which is "added".  So, for instance, a beard or shadow is "added"; higher eyebrows are "added"; wider eyes are "added".  The definition is somewhat arbitrary but is detected by looking for darker regions.   

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
</ol>


### 1.  is there an automatic way to recover from some "mode collapse"?:

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

In layer 1 of the screen shot above, we can see at epoch 1/100 and iteration 126/781, the discriminator loss has dropped to near zero and the gan loss is beginning to escalate.  Left to itself, the discriminator loss would drop to zero and we would see mode collapse.  In this case, the saved discriminator weights (d_weights) are loaded back in and the stream recovers.  

In layer 2, we see proof of recovery at the end of epoch 1 with discriminator loss at 0.459 and gan loss at 1.280.  At this point, the accuracy for "real" is 77% and fake is 93%.  These values may not sound impessive until we look at the generated faces from epoch 1.

In layer 3, we see a screen shot of the generated faces from epoch 1 out of 100 epoches.  

So how can we recover from a mode collapse?  The syntax below illustrates the core of the process:  

```Python
		if (d_loss1 < 0.001 or d_loss1 > 2.0) and ijSave > 0:
			print("RELOADING d_model weights",j+1," from ",ijSave)
			d_model.set_weights(d_trainable_weights)
		if (d_loss2 < 0.001 or d_loss2 > 2.0) and ijSave > 0:
			print("RELOADING g_model weights",j+1," from ",ijSave)
			g_model.set_weights(g_trainable_weights)
		if (g_loss < 0.010 or g_loss > 4.50) and ijSave > 0:
			print("RELOADING gan_models weights",j+1," from ",ijSave)
			gan_model.set_weights(gan_trainable_weights)
```
The previous programming fragment illustrates an approach which often prevents a stream from mode collapse.  It depends on having captured disciminator weights, generator weights, and gan weights either during initialization or later in the process when all model losses are within bounds.  The definition of model loss bounds are arbitrary but reflect expert opinion about when losses are what might be expected and when they are clearly much too high or much too low.  Reasonable discriminator and generator losses are between 0.1 and 1.0, and their arbitrary bounds are set to between 0.001 and 2.0.  Reasonable gan losses are between 0.2 and 2.0 and their arbitrary bounds are set to 0.01 and 4.5.  

What happens then is discriminator, generator, and gan weights are collected when all three losses are "reasonable".  When an individual model's loss goes out of bounds, then the last collected weights for that particular model are replaced, leaving the other model weights are they are, and the process moves forward.  The process stops when mode collapse appears to be unavoidable even when model weights are replaced.  This is identified when a particular set of model weights continue to be reused but repeatedly result in out of bound model losses.   

The programming fragment for saving the weights are:

```Python
	if d_loss1 > 0.30 and d_loss1 < 0.95 and d_loss2 > 0.25 and d_loss2 < 0.95 and g_loss > 0.40 and g_loss < 1.50:
		d_trainable_weights = np.array(d_model.get_weights())
		g_trainable_weights = np.array(g_model.get_weights())
		gan_trainable_weights = np.array(gan_model.get_weights())
```
Needless to say, there are a few additional requirements which can be found in the Python program available at the end of this README document.  For instance, if your stream goes into mode collapse just after saving your trainable weights, there is little likelihood that the most recently saved weights will save the recovery.  

It's important to note that a critical aspect of this stream is to help the novice get over the difficult challenge of making the first GAN program work.  As such, its focus is not simply on automatic ways to recover from mode collapse and methods of restarting streams, but on the debugging process that may be required.  To do this, we need constant reporting.  As we observe in the screen shot below, not every execution results in a requirement to load in most recent working trainable weights.  However, we do see information which may be helpful in understanding what is going on.  
<p align="center">
<img src="/images/nonEscapingModeCollapse.png" width="500" height="150">
</p>
Typically, the situation for loss is reported every five iterations.  As illustrated in the area in the red blocked area, when the program appears to be drifting into mode collapse, losses are reported on every iteration.  In the blue blocked area, we can see the generative loss beginning to incease beyond reasonable limits.  In the green blocked area, we see the tendency for when the discriminator or generator losses move beyond reasonable limits, the gans losses move out of range.  And finally, in the brown blocked area, we see a counter of the number of times weights have been saved to be used later in recovery.  

### 2.  is there a way to restart a cGAN which has not completed convergence:
There is nothing quite as problematic as running a program and six days later the process is interrupted when it appears to be 90% complete.  Like many others, I have run streams for over 21 days using my GPU before something goes wrong and I am unable to restart the process.  Progress is measured in "epochs".  There is no guarantee but with a bit of good fortune and cGAN steams which are properly set up, every epoch brings an improvement in clarity.  The images which follow illustrate observed improvements over epochs.  
<p align="center">
<img src="/images/improvedImagesOverEpochs.png" width="650" height="500">
</p>
  
The numbers on the left side are epochs which produced the observed results.  We can see the faint images of faces by epoch 5, good impressions of faces by epoch 45, details of faces by epoch 165 and small improvements by epoch 205.  We want to do better than being stuck at epoch 45 and we want to be able to continue from epoch 45 if the process is interrupted.  We are, in a sense, mapping from a 100-dimensional space to images of faces and it takes time to complete the mapping from representative parts of the 100-dimensional space.      
    
Needless to say, the steam needs to be prepared for interruptions.  Even with preparation, attempts to restart can result in warnings about model and/or layers being trainable=False, dimensions of weights being incompatable for discriminate, generative, and gan models, and optimizations that collapse.  It's important to note that cGAN will not properly restart unless you resolve the issues of what is trainable, what are the correct dimensions, and what are viable models. If your only interest is in examining weights and optimization, then warning messages can often be ignored.  If you wish to restart from where you left off, then you ignore warning messages at considerable risk.   
 
Once issues with dimensions and what is trainable are resolved, there are then problems where models suffer from mode collapse when attempts are made to restart the cGAN.  What happened?  If you wish to continue executing the program, my experience is you need to handle the GAN model as a new instance using the loaded discriminator and generator models.  After all, the GAN model is there only to constrain and make the discriminator and generator work together.  
 
Restarting a cGAN requires saving models and their optimizations in case they are required after each epoch.  When saving a model, the layers that get saved are those which are trainable.  It's worth recalling that the discriminator model is set to trainable=False within the gan model.  Depending on the requirements, there may also be layers which are set to trainable=False.  In order to save the models, and recover the fixed weights, the weights must temporarily be set to trainable=True.  The following code fragment is required when saving the discriminator model:  
```Python
	filename = 'celeb/results/generator_model_dis%03d.h5' % (epoch+1)
	d_model.trainable = True
	for layer in d_model.layers:
		layer.trainable = True
	d_model.save(filename)
	d_model.trainable = False
	for layer in d_model.layers:
		layer.trainable = False
```
And when loading:
```Python
	filename = 'celeb/results/generator_model_dis%03d.h5' % (ist_epochs)
	d_model = load_model(filename, compile=True)
	d_model.trainable = True
	for layer in d_model.layers:
		layer.trainable = True
	d_model.summary()
```
Setting the layers on an individual basis may seem overly detailed but it is a reminder that, in some circumstances, there are layers which may need to be set to trainable-False. 

Three parameters need to be changed in order to restart the process:  qRestart, epochs_done, epochs_goal.  These parameters are found near the beginning of the Python program.  
```Python
#    INDICATE IF STARTING OR CONTINUING FROM PREVIOUS RUN
qRestart = False
if qRestart:
    epochs_done = 105
    epochs_goal = 115
else:
    epochs_done = 0
    epochs_goal = 100
```
qRestart is set to True indicating the program needs to start from where it left off.
"epochs_done" refers to the number of epochs already completed.  
"epochs_goal" refers to how many epochs you think you'd like to complete.  


### 3.  are there different kinds of random initialization processes that can be helpful in accelerating convergence?
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
 
### 4.  how important is the source material (original images of faces)?
In my attempts to improve the results of the generations, I initially overlooked a critical factor - what does the transformed data going into the cGAN look like.  When the data going into a stream is a derivative of another process, as in this case, it is critical to examine the quality of the input data before declaring the results to be useful or invalid.  

The code to examine the data going into the cGAN is trivial and is included in the final stream.  

![real faces rows](images/sampleRealImagesRows.png)

It's worth remembering that the GAN process sees the images at the convoluted pixel level - it sees every spot and wrinkle, every imperfection.   
![real faces](images/sampleRealImages.png)

In spite of all the imperfections in individual images, my belief is the final results are impressive.  Selecting out only faces featured as attractive helped in obtaining results which had considerable clarity.  


```Python
            n_classes = 4     
            latent_dim = 100                  # 100 dimensional space
            pts, labels_input = generate_latent_points(latent_dim, n_samples, cumProbs)
            results = None
            for i in range(n_samples):        # interpolate points in latent space
                interpolated = interpolate_points(pts[2*i], pts[2*i+1])
                for j in range(n_classes):    # run each class (embedding label)
                    labels = np.ones(10,dtype=int)*j
                    X = model.predict([interpolated, labels])  # predict image based on latent points & label
                    X = (X + 1) / 2.0         # scale from [-1,1] to [0,1]
                    if results is None:
                        results = X
                    else:
                        results = vstack((results, X))   # stack the images for display
            plot_generated(filename, results, labels_input, 10, n_samples, n_classes)   #generate plot
```
###  8.  cGan streams and data sources:
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

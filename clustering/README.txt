The clustering folderis the experiment of clustering.

Before training, you should change the data path of MNIST at 26 line of "MNIST_loader.py" 
and the saving place of files at 252 line of "MNIST_train.py" and 7 line of "print.py"

Followings are the introduction of each file.

(1)"MNIST_train.py": 
The main training function. 
If you select the Wasserstein distance, you can change the function at 32 line to the function at 47 line.

To change the tau to try defferent bounds, you could set the "tau" at 132-134 line to whatever you want.

If you want to change the clustering set, you could change the "num_list" at 191 line to select the number of 
each class.

"reg" at 206 line --- the epsilon in sinkhorn
"numClass" at 207 line --- the number of class you want to cluster
"numIter" at 208 line --- the iteration number of updating the centroids

(2)"print.py":
this file is used to draw the picture of your training result. It's not necessary to use. You can write 
your own file to show the image by matplotlib.

7-15 line: the data path of the result you want to draw.

"numList" at 18 line needs to be the same as "num_list" in "MNIST_train.py"

"numTau" at 22 line --- depends on how many kinds of taus that you get the result after training.
"numClass" at 23 line --- needs to be the same as "numClass" in "MNIST_train.py"
"tau_list" at 25 line --- depends on which tau you use in training

You need to change the details if you want to use it.

(3)"sinkhorn.py"
It stores the realization of barycenter, sinkhorn, and our DBOT. You don't need to change anything.

(4)"MNIST_loader.py"
It's used to load the MNIST dataset and get the subset of MNIST as the "num_list" you input in "MNIST_train.py"
You don't need to change anything.

(5)"plot_OT_1D.py", "plot_OT_2D_picture.py", "plot_OT_2D_sample.py"
These three files is the pre-experiment of clustering.

"plot_OT_1D.py" is the attempt of sinkhorn on mixed gauss of 1 dimension and the figure we use in the report.

"plot_OT_2D_picture.py", "plot_OT_2D_sample.py" are the clustering attempts of 150 points of 2 dimension 
gaussian scatters in 5 class and also used to draw figure.

These three files are not necessary to run.



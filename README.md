# EthereumDeepLearning

I would like to thank [Prof. Vadim Sokolov](http://vsokolov.org/) for mentoring me through various stages of this project. I would like to thank [Cuneyt Gurcan Akcora](https://www.linkedin.com/in/cuneyt-gurcan-akcora-97272421) and [Yulia R. Gel](https://www.utdallas.edu/math/809/yulia-gel/) for helping me gain a baseline understanding of the problem. 


### What is Ethereum ? 

Ethereum describes itself as : <i>A decentralized platform that runs smart contracts: applications that run exactly as programmed without any possibility of downtime, censorship, fraud or third-party interference.</i> This platform runs on a custom built blockchain, an infrastructure that allows value transfer and an accurate representation of value ownership. [Source](https://www.ethereum.org/)
___

### What is the Ethereum blockchain ? 

A blockchain is defined as a cryptographically secure transactional singleton machine with shared-state [(source)](https://github.com/ethereum/yellowpaper). Let us try and understand the three terms mentioned here: 
<b> Cryptographically secure: </b> The creation of digital currency (like Ether) is secured by an algorithm that is hard to break (not impossible, just that it would take a lot of computation power to break it). 
<b>Transaction singleton machine: </b>There is a single instance of the machine that is being used to create , process and store all the transactions that are occuring. 
<b>Shared state: </b>The state that is stored on the machine is accessible to everyone. 

Ethereum implements a version of this paradigm. 

Any blockchain has two parts - a  header and  transactions . The transactions store information pertaining to each transfer of currency (to, from, value , timestamp etc) and the  header stores metadata for the transactions  (timestamp, miner information etc). The header and the transactions are contained in a "block". The blocks are chained in a sequential order, with each new block being chained to its previou block. This image by [Louis Pinsard](https://blog.theodo.fr/2018/01/deploy-first-ethereum-smart-contract-blockchain/) explains the concept beautifully. 
![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/2.png). 

If you are looking to understand how the Ethereum blockchain works in detail , I recommend this article by [Preethi Kasireddy](https://medium.com/preethikasireddy/how-does-ethereum-work-anyway-22d1df506369). On the other hand , if you are more of a visual person , this image by Lee Thomas is pretty neat.  ![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/3.jpg) 
<i>[high res version](https://i.stack.imgur.com/afWDt.jpg)</i>
___

### What am I looking to answer ? 

Can the log returns for Ethereum be predicted by using the network subgraph characterestics of the Ethereum transaction graph 

___

### Data collection 

I will split this into two major sections - Gathering data which relates to the Ethereum transactions and gathering data which relates to the price of Ethereum. My philosophy for gathering data has always been that of a hoarder - gather everything you can at the fastest possible rate, then use whatever you need from it. Storage is cheap these days, and I get a kick out of figuring out how to get around rate limits :).

<b>Pricing data </b> 

There are many marketplaces and websites which display the price of Ethereum. Almost all of them have beauiful interfaces and display prices that are accurate. A lot of them however do not allow you to extract the data into a queryable format. 

Considerations :

a. I was looking to gather price information at an hourly interval. 

b. I wanted the price from when Ethereum began- July 30, 2015

c. I did not want to rely on a single marketplace but an aggregated data source. 

Solution: 

[Coindesk!](www.coindesk.com) . It is possible to extract price on an hour by hour basis. It has data from when Ethereum began. It provides aggregated pricing information. Lastly, it is a pretty trusted source in the crypto community, having been active since May 2013. Coindesk allows you to search for the price of Ethereum on a day by day basis and download a CSV file of the data ( which is in an hour by hour format. From late 2017 , the results do break down into lower time intervals, but to ensure consistency, I stuck to the one hour time interval).
![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/1.JPG)

I wanted to automate the process of querying the information I need for each day. I chose to work with two python libraries - selenium and pyautogui. The intention was to mimic the workflow that a human would follow and replicate the process for every day. 

Process : 

1. Open the webpage using Selenium

2. Type the start date and end date as done by a human. 

3. Click the export button , rename the file with the date and save it , as done by a human. 

4. Repeat the process for all dates.

(Note : for devices with different screen settings , might want to consider changing the coordinates to ensure the same process occurs). 


After all the CSV files are gathered , the idea is to delete the header line in each CSV file and concatenate all of the information into a single file. This can be accomplished by a few simple commands in the windows command prompt (Mac and linux have similar options as well). 

To delete the first line in each CSV file, use:  
```
MORE /E +1 original.csv > new.csv
```
Initializing all of the filenames as strings in a list and looping through the list by running CMD through the subprocess library in Python does the trick.

To concatenate the output files into a single file , get to the folder where all the output files are stored and run : 
```
for %f in (*.csv) do type "%f" >> localpath\output.csv
```
Open the output file , add a header line and you're good to go. 


<b>Ethereum data </b>

To get Ethereum data , the following steps have to be followed: 

1. Run a local Ethereum node . This would involve downloading the entire chain data. 

2. Use the [Ethereum web3.js API](https://github.com/ethereum/web3.js/) to query information about each block header and the transactions contained in the block. Save this information locally. 


Running a local Ethereum node : 

(Note: this process is for Windows machines. Mac and Linux have a similar process) 

1. Download,unzip and install geth. Make sure to add the location of geth.exe to your environment variables.  [Geth](https://geth.ethereum.org/downloads/)

Geth is the program that communicates with the Ethereum Network and acts as the a relay between your computer, its hardware and the rest of the ethereum network computers so if a block is mined by another computer your Geth program will pick it up and then pass on the new information onto your GPU or CPU for mining.

2. Open command prompt , and create your userID and password combination by typing: 
```
geth account new
```

3. Now , get geth to communicate with the rest of the Ethereum network. This can be done by typing 

```
geth --rpc
```

If you're looking for a slightly faster processing time , try adding  "--fast" and "--cache=x" flags. For example , 
```
geth --rpc --fast --cache=1024
```
This should start downloading the entire blockchain. It takes anywhere from an hour to 24 hours depending on your internet speed and firewall configuration. Sometimes, the download gets stuck / stops abruptly for no apparent reason. The best solution is to delete the entire chaindata folder and restart the download. 
If you are looking to check the status of your download, open up another command prompt and type in the following commands : 

```
geth attach 
eth.syncing
```
This should return information about the current block and the highest block , which gives you an idea on how much longer it will take. 
If it returns a False , you either are in sync or have not found a peer to start syncing with. 


Using the web3.JS API : 

I have used an [R implementation](https://github.com/BSDStudios/ethr) of the JSON-RPC API to query information I need. 

Process : 

1. Query header information for the specific block number and store that information in a dataframe. 
2. Get the transactions contained in the specific block and save all features to another dataframe. 
3. After every <i>n</i> number of blocks decided by how huge you want each dataframe to be, output the dataframes as CSV files. 
Outputting each block's information as a CSV file is time intensive. Repeated writing of data to a hard drive / solid state drive can be slow. During this stage , you also might want to consider deleting a few runtime variables to speed up processing time. 
4. Repeat steps 1-3 , starting from block # 0 and continue until the current block. You might want to consider running multiple instances of the same script with different block number start and end values for downloading the data in parallel.

___

### Data processing 
As anyone who has ever dealt with real world data will know, data is ugly and needs to be properly formatted before any kind of algorithm can be run on it. I relied heavily on Pandas for this stage of the project. 

Pricing data :

1. Convert the  data to a timestamp format . 
2. Aggregate data on an hourly basis. If more than one record exists for a single time delta , consider the average price during that time delta. 

Output : 

![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/4.JPG)


Ethereum data :

1. Join header and transaction data files by header information's primary key. 
2. Delete unnecessary columns. 
3. Convert data columns from hexadecimal to human readable format. Convert date from unix timestamp to datetime format. 

Output: 

![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/5.JPG)

___

### Feature extraction 

Pricing data :

1. Get the log return and the percentage change in value over time from the Ethereum price information. Log return is being predicted instead of price, because it reduces the variation of the time series making it easier to fit a model to it. 

Ethereum data : 

To extract the network features, all the transactions that are in a time delta are considered. For each transaction , the sender and receiver are considered as nodes, with an edge existing between them having a weight equal to the value of the transaction. 

Process : 

1. Consider all the transactions happening in an hour. 
2. Extract the transactions count as a feature. 
3. Represent the transactions as a network graph , with the sender and receiver acting as nodes with an edge having a weight equal to the value of the Ether transferred. 
4. Remove the transactions that do not have any edges in the graph. 
5. Extract the following features from  the graph - edge count , node count , degree , transitivity , centrality , average clustering , number of connected components and density.  
6. Apply a bincount  get the number of nodes which have 1 edge , 2 edges , 3 edges ... and append the counts as features. (Zeroes can be appended to ensure that we have an equal size of all arrays once we append these lists to the final dataset )
7. Repeat 1 through 6 for every hour and concatenate all the values into a single dataset. 

![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/6.jpg)
___

### Baseline models 

Consider Naive forecast , simple exponential smoothing and ARIMA to be the baseline models. All of the baseline models provide a flat forecast. The flat forecast gives an RMSE of 0.012 , but it would not be prudent to attach too much importance to this as a flat forecast does not lead to any meaningful predictions. 

Naive forecast 

![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/naive.png)


Simple exponential smoothing 

![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/ses.png)


ARIMA 

![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/arima.png)
___

### Deep Learning using RNN LSTMs 

Recurrent neural networks have been used extensively for sequence learning tasks - language modelling , machine translation , image captioning etc. They are good at understanding the temporal aspects of the data, which is why they have also been used for price prediction problems. 

The task is to predict the log return using a sequential RNN LSTM model. Keras is used as the backend. 

Process : 
1. Load the dataset. 
2. Fill null values (if any) with 0 and convert all columns to float32 datatype. 
3. Scale the columns in the dataset. Experiment with sklearn's StandardScaler, MinMaxScaler,MaxAbsScaler and RobustScaler. The reason why multiple scaling methods are being tried is because some of the log returns are extremely large and therefore act as outliers. For this problem, ignoring the outliers does not make any sense, which is why we try to reduce the effect of their impact. 
4. Specify the number of lag hours and convert the dataset to a supervised learning problem by adding previous timestamp information to each data entry. Experiment with the number of lag hours to see if there is a difference. 
5. Split the dataset into train_X, train_y, test_X, test_y. 
6. Initialize a sequential model with the first LSTM layer having a linear activation function. 
7. Experiment with different dropout rates. Also experiment with different cells - GRU , peephole LSTMs etc. 
8. Specify the output layer as a dense layer with 1 neuron and a linear activation function. 
9. Experiment with different loss and optimizer functions and their parameters. 
10. Compile the model. 
11. Fit the model to training data. Experiment with different batch sizes, epoch counts and shuffle options. 
12. Plot the training and test loss. 
13. Make a prediction for the test set by inverse transforming the test values. Plot the actual vs predicted log returns. Calculate the RMSE as well. 
14. Save the model. 
___

### Results 

RNN LSTMs provide a forecast that is meaningful and not just a flat forecast. GRU cells have a better convergence rate when compared to plain LSTM cells. 

___

Future work 

1. Implement an RNN LSTM model with a lower time delta ( it is possible to get Ethereum pricing information for a 5 minute time interval ). 
2. Implement custom loss functions 
3. Use "network motifs", i.e., instead of simply sticking to node and edge features , try to enumerate different types of specific subgraphs, as mentioned [here](http://www.cs.unibo.it/babaoglu/courses/cas06-07/resources/tutorials/motifs.pdf) , [here](https://dl.acm.org/citation.cfm?id=2920564) and [here](https://europepmc.org/abstract/med/27734973)



<<<<<<< HEAD
=======

>>>>>>> origin/master

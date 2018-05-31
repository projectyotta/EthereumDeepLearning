# EthereumDeepLearning

Project name 

Description 

Table of contents 

Installation 

USage 

Credits 

License 


####What is Ethereum ? 

Ethereum describes itself as : <i>A decentralized platform that runs smart contracts: applications that run exactly as programmed without any possibility of downtime, censorship, fraud or third-party interference.</i> This platform runs on a custom built blockchain, an infrastructure that allows value transfer and an accurate representation of value ownership. [Source](https://www.ethereum.org/)

####What is the Ethereum blockchain ? 

A blockchain is defined as a cryptographically secure transactional singleton machine with shared-state [(source)](https://github.com/ethereum/yellowpaper). Let us try and understand the three terms mentioned here: 
<b> Cryptographically secure: </b> The creation of digital currency (like Ether) is secured by an algorithm that is hard to break (not impossible, just that it would take a lot of computation power to break it). 
<b>Transaction singleton machine: </b>There is a single instance of the machine that is being used to create , process and store all the transactions that are occuring. 
<b>Shared state: </b>The state that is stored on the machine is accessible to everyone. 

Ethereum implements a version of this paradigm. 

Any blockchain has two parts - a  header and  transactions . The transactions store information pertaining to each transfer of currency (to, from, value , timestamp etc) and the  header stores metadata for the transactions  (timestamp, miner information etc). The header and the transactions are contained in a "block". The blocks are chained in a sequential order, with each new block being chained to its previou block. This image by [Louis Pinsard](https://blog.theodo.fr/2018/01/deploy-first-ethereum-smart-contract-blockchain/) explains the concept beautifully. 
![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/2.png). 

If you are looking to understand how the Ethereum blockchain works in detail , I recommend this article by [Preethi Kasireddy](https://medium.com/preethikasireddy/how-does-ethereum-work-anyway-22d1df506369). On the other hand , if you are more of a visual person , this image by Lee Thomas is pretty neat.  ![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/3.jpg) 
<i>[high res version](https://i.stack.imgur.com/afWDt.jpg)</i>

####What am I looking to answer ? 

Can the log returns for Ethereum be predicted by using the network subgraph characterestics of the Ethereum transaction graph 


####Data collection 

I will split this into two major sections - Gathering data which relates to the Ethereum transactions and gathering data which relates to the price of Ethereum. My philosophy for gathering data has always been that of a hoarder - gather everything you can at the fastest possible rate, then use whatever you need from it. Storage is cheap these days, and I get a kick out of figuring out how to get around rate limits :).

<b>Pricing data </b> 

There are many marketplaces and websites which display the price of Ethereum. Almost all of them have beauiful interfaces and display prices that are accurate. A lot of them however do not allow you to extract the data into a queryable format. 

Considerations :

a. I was looking to gather price information at the lowest possible time interval (minute by minute, or even lower, if possible)

b. I wanted the price from when Ethereum began- July 30, 2015

c. I did not want to rely on a single marketplace but an aggregated data source. 

Solution: 

[Coindesk!](www.coindesk.com) . It is possible to extract price on an hour by hour basis. It has data from when Ethereum began. It provides aggregated pricing information. Lastly, it is a pretty trusted source in the crypto community, having been active since May 2013. One consideration that I was not able to satisfy is data being available at a lower time interval. If anyone reading this happens to know of a data source that provides pricing information at a lower time interval, I'd love to know. 

Process: 

Coindesk allows you to search for the price of Ethereum on a day by day basis and download a CSV file of the data ( which is in an hour by hour format. From late 2017 , the results do break down into lower time intervals, but to ensure consistency, I stuck to the one hour time interval).
![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/1.JPG)

I wanted to automate the process of querying the information I need for each day. I chose to work with two python libraries - selenium and pyautogui. The intention was to mimic the workflow that a human would follow and replicate the process for every day. 

Steps : 

1. Open the webpage using selenium

2. Type the start date and end date as done by a human. 

3. Click the export button , rename the file with the date and save it , as done by a human. 

4. Repeat the process for all dates.

(Note : for devices with different screen settings , might want to consider changing the (x,y) coordinates to ensure the same process occurs) . 


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


Using the web3.JS API 

I have used an [R implementation](https://github.com/BSDStudios/ethr) of the JSON-RPC API to query information I need. 
The process : 


1. Query header information for the specific block number and store that information in a dataframe. 
2. Get the transactions contained in the specific block and save all features to another dataframe. 
3. After every <i>n</i> number of blocks decided by how huge you want each dataframe to be, output the dataframes as CSV files. 
Outputting each block's information as a CSV file is time intensive. Repeated writing of data to a hard drive / solid state drive can be slow. 
4. Repeat steps 1-3 , starting from block # 0 and continue until the current block. You might want to consider running multiple instances of the same script with different block number start and end values for downloading the data in parallel.




 




Additional links 

How hard is it to break the EDCSA algorithm used by Ethereum [Source](https://pdfs.semanticscholar.org/5646/d266fcd0472bd188b913f9a7a420d82e8859.pdf)
How hard is it to break the SHA-256 algorithm used by Bitcoin [Source](https://eprint.iacr.org/2016/992)
Implementing a simple blockchain in Python [Source](https://github.com/satwikkansal/ibm_blockchain)




I would like to thank the following people who have helped me immensely with my work : 1. Vsokolov 
2. Dallas


# EthereumDeepLearning

Project name 

Description 

Table of contents 

Installation 

USage 

Credits 

License 


1. What is Ethereum ? 
Ethereum describes itself as : <i>A decentralized platform that runs smart contracts: applications that run exactly as programmed without any possibility of downtime, censorship, fraud or third-party interference.</i> This platform runs on a custom built blockchain, an infrastructure that allows value transfer and an accurate representation of value ownership. [Source](https://www.ethereum.org/)

2. What is the Ethereum blockchain ? 
A blockchain is defined as a cryptographically secure transactional singleton machine with shared-state [(source)](https://github.com/ethereum/yellowpaper). Let us try and understand the three terms mentioned here: 
<b> Cryptographically secure: </b> The creation of digital currency (like Ether) is secured by an algorithm that is hard to break (not impossible, just that it would take a lot of computation power to break it). 
<b>Transaction singleton machine: </b>There is a single instance of the machine that is being used to create , process and store all the transactions that are occuring. 
<b>Shared state: </b>The state that is stored on the machine is accessible to everyone. 

Ethereum implements a version of this paradigm. 

Any blockchain has two parts - a  header and  transactions . The transactions store information pertaining to each transfer of currency (to, from, value , timestamp etc) and the  header stores metadata for the transactions  (timestamp, miner information etc). The header and the transactions are contained in a "block". The blocks are chained in a sequential order, with each new block being chained to its previou block. This image by [Louis Pinsard](https://blog.theodo.fr/2018/01/deploy-first-ethereum-smart-contract-blockchain/) explains the concept beautifully. 
![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/2.PNG). 

If you are looking to understand how the Ethereum blockchain works in detail , I recommend this article by [Preethi Kasireddy](https://medium.com/preethikasireddy/how-does-ethereum-work-anyway-22d1df506369). On the other hand , if you are more of a visual person , this image by Lee Thomas is pretty neat.  ![](https://github.com/saurabh-rao/EthereumDeepLearning/blob/master/images/3.JPG) 
<i>[high res version](https://i.stack.imgur.com/afWDt.jpg)</i>

3. What am I looking to answer ? 
Can the log returns for Ethereum be predicted by using the network subgraph characterestics of the Ethereum transaction graph 


4. Data collection 
I will split this into two major sections - Gathering data which relates to the Ethereum transactions and gathering data which relates to the price of Ethereum. My philosophy for gathering data has always been that of a hoarder - gather everything you can at the fastest possible rate, then use whatever you need from it. Storage is cheap these days, and I get a kick out of figuring out how to get around rate limits :).


4.1. Pricing data 
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
###NOTE TO SELF : ADD A GIF HERE SHOWING WHAT IS HAPPENING 

```python
from datetime import date, timedelta
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys
import pyautogui as pag
import time
from selenium.webdriver.chrome.options import Options

d1 = date(2015, 9, 1)  # start date
d2 = date(2018, 1, 25)  # end date

delta = d2 - d1         # define timedelta

for i in range(delta.days + 1):
	j = i + 1 
	x=(d1 + timedelta(days=i))
	y=(d1 + timedelta(days=j))

	x= str(x)
	y=str(y)
	
	print(x)
	print(y)
	print("___")

	options = Options() 
	options.add_argument("--start-maximized") 

	driver = webdriver.Chrome("localpath\\chromedriver.exe", chrome_options=options)
	driver.get("https://www.coindesk.com/ethereum-price/")
	time.sleep(3)

	pag.click(x=713, y=470)

	pag.keyDown('ctrl')
	pag.press('a')
	pag.keyUp('ctrl')
	pag.press('backspace')

	pag.typewrite(x)
	pag.press('tab')
	time.sleep(1)
	pag.keyDown('ctrl')
	pag.press('a')
	pag.keyUp('ctrl')
	pag.press('backspace')	
	pag.typewrite(y)
	time.sleep(2)
	pag.press('enter')
	time.sleep(1)
	pag.click(x=928, y=471)
	time.sleep(0.5)
	pag.click(x=861, y=710)

	# avoid potential rate limiting issues 	
	time.sleep(15)
	driver.close()
```

After all the CSV files are gathered , the idea is to delete the header line in each CSV file and concatenate all of the information into a single file. This can be accomplished by a few simple commands in the windows command prompt (I am sure mac and linux have similar options as well). 

To delete the first line in each CSV file, use:  
```
MORE /E +1 original.csv > new.csv
```
Initializing all of the filenames as strings in a list and looping through the list by running CMD through the subprocess library in Python does the trick.

To concatenate


4.2 Chain data 

 




Additional links 
How hard is it to break the EDCSA algorithm used by Ethereum [Source](https://pdfs.semanticscholar.org/5646/d266fcd0472bd188b913f9a7a420d82e8859.pdf)
How hard is it to break the SHA-256 algorithm used by Bitcoin [Source](https://eprint.iacr.org/2016/992)
Implementing a simple blockchain in Python [Source](https://github.com/satwikkansal/ibm_blockchain)




I would like to thank the following people who have helped me immensely with my work : 1. Vsokolov 
2. Dallas


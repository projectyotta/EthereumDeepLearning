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

	driver = webdriver.Chrome("C:\\Users\\raosa\\Desktop\\PROJECTS\\chromedriver.exe", chrome_options=options)
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





